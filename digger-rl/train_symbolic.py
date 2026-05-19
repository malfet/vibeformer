"""PPO trained on symbolic GameState observations.

The pixel-based agents (PPO, Dreamer) stalled around avg_ret ~80-225 because
most training cost goes to learning the visual representation. With
SymbolicDiggerEnv handing the agent a 6-channel 10x15 grid where each
channel is a perfect mask (dirt, emerald, digger, monster, bag, cherry),
representation learning is free -- the agent only has to learn the policy.

    python train_symbolic.py --total-timesteps 100000 --num-envs 1

Tiny network: 3 conv layers over (6, 10, 15) -> flatten -> actor/critic.
~50k params. Should converge in a few minutes for level-1 strategy.
"""

from __future__ import annotations

import argparse
import collections
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from symbolic_env import OBS_CHANNELS, OBS_SHAPE, SymbolicDiggerEnv
from train_ppo import layer_init, select_device

REPO = Path(__file__).parent.resolve()
CKPT_DIR = REPO / "data" / "checkpoints"


@dataclass
class Config:
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.02
    ent_coef_final: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    frame_skip: int = 4
    episodic_life: bool = True
    death_penalty: float = 0.0
    shaping_coef: float = 0.0    # potential-based shaping toward nearest emerald
    run_name: str = "symbolic"
    save_every: int = 50
    seed: int = 1
    device: str = "auto"


class SymbolicAgent(nn.Module):
    """Small CNN over the (C, MHEIGHT, MWIDTH) grid + actor/critic heads.

    With a 10x15 grid we don't need pooling. Two 3x3 convs preserve
    spatial dims, then flatten and project to a 128-d feature.
    """

    NUM_ACTIONS = SymbolicDiggerEnv.NUM_ACTIONS

    def __init__(self, in_channels: int = OBS_CHANNELS):
        super().__init__()
        h = OBS_SHAPE[1]; w = OBS_SHAPE[2]
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 3, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, padding=1)), nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * h * w, 128)), nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(128, self.NUM_ACTIONS), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)

    def encode(self, x):
        return self.encoder(x)

    def value(self, x):
        return self.critic(self.encode(x)).squeeze(-1)

    def act(self, x, action=None):
        z = self.encode(x)
        logits = self.actor(z)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(z).squeeze(-1)


def env_step_skipped(env: SymbolicDiggerEnv, action: int, skip: int):
    """Hold action across skip emulator frames; return last obs + summed reward."""
    total_r = 0.0
    info: dict = {}
    obs = None
    done = False
    for _ in range(skip):
        obs, r, done_, info = env.step(action)
        total_r += r
        if done_:
            done = True
            break
    return obs, total_r, done, info


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=Config.total_timesteps)
    p.add_argument("--lr", type=float, default=Config.learning_rate)
    p.add_argument("--num-steps", type=int, default=Config.num_steps)
    p.add_argument("--ent-coef", type=float, default=Config.ent_coef)
    p.add_argument("--ent-coef-final", type=float, default=Config.ent_coef_final)
    p.add_argument("--frame-skip", type=int, default=Config.frame_skip)
    p.add_argument("--episodic-life", action="store_true", default=Config.episodic_life)
    p.add_argument("--no-episodic-life", action="store_false", dest="episodic_life")
    p.add_argument("--death-penalty", type=float, default=Config.death_penalty)
    p.add_argument("--shaping-coef", type=float, default=Config.shaping_coef,
                   help="potential-based reward per step for moving toward the "
                        "nearest emerald (try 0.5-2.0)")
    p.add_argument("--save-every", type=int, default=Config.save_every)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--run-name", type=str, default=Config.run_name)
    a = p.parse_args()
    return Config(
        total_timesteps=a.total_timesteps, learning_rate=a.lr,
        num_steps=a.num_steps, ent_coef=a.ent_coef, ent_coef_final=a.ent_coef_final,
        frame_skip=a.frame_skip, episodic_life=a.episodic_life,
        death_penalty=a.death_penalty, shaping_coef=a.shaping_coef,
        save_every=a.save_every, seed=a.seed,
        device=str(select_device(a.force_cpu)), run_name=a.run_name,
    )


def main() -> None:
    cfg = parse_args()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    ckpt_dir = CKPT_DIR / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{cfg.run_name}] device={device} cfg={cfg}", flush=True)

    env = SymbolicDiggerEnv(
        shaping_coef=cfg.shaping_coef,
        max_steps=10**9, episodic_life=cfg.episodic_life,
        death_penalty=cfg.death_penalty)
    agent = SymbolicAgent().to(device)
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"[{cfg.run_name}] agent: {n_params:,} params", flush=True)
    optim = Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    N = cfg.num_steps
    obs_buf = torch.zeros(N, *OBS_SHAPE, device=device)
    act_buf = torch.zeros(N, dtype=torch.long, device=device)
    logp_buf = torch.zeros(N, device=device)
    rew_buf = torch.zeros(N, device=device)
    done_buf = torch.zeros(N, device=device)
    val_buf = torch.zeros(N, device=device)

    obs_np = env.reset()
    obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)
    done_t = torch.zeros(1, device=device)

    global_step = 0
    update = 0
    ep_return = 0.0
    ep_length = 0
    ep_returns: collections.deque[float] = collections.deque(maxlen=20)
    t0 = time.monotonic()

    while global_step < cfg.total_timesteps:
        update += 1
        frac = max(0.0, 1.0 - global_step / cfg.total_timesteps)
        if cfg.anneal_lr:
            optim.param_groups[0]["lr"] = frac * cfg.learning_rate
        current_ent = cfg.ent_coef_final + frac * (cfg.ent_coef - cfg.ent_coef_final)

        # ---- Rollout ----
        for step in range(N):
            obs_buf[step] = obs_t[0]
            done_buf[step] = done_t[0]
            with torch.no_grad():
                action, logp, _, value = agent.act(obs_t)
            act_buf[step] = action[0]
            logp_buf[step] = logp[0]
            val_buf[step] = value[0]
            obs_np, reward, done, info = env_step_skipped(
                env, int(action[0].item()), cfg.frame_skip)
            global_step += cfg.frame_skip
            rew_buf[step] = reward
            ep_return += reward; ep_length += cfg.frame_skip
            if done:
                ep_returns.append(ep_return)
                print(f"[{cfg.run_name}]   ep_end return={ep_return:.0f} length={ep_length} "
                      f"score={info.get('score', 0)} lives={info.get('lives', 0)}",
                      flush=True)
                ep_return = 0.0; ep_length = 0
                obs_np = env.reset()
                done_t = torch.ones(1, device=device)
            else:
                done_t = torch.zeros(1, device=device)
            obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)

        # ---- GAE ----
        with torch.no_grad():
            next_value = agent.value(obs_t)
            advantages = torch.zeros_like(rew_buf)
            lastgae = torch.zeros((), device=device)
            for t in reversed(range(N)):
                if t == N - 1:
                    next_nonterminal = 1.0 - done_t[0]
                    next_v = next_value[0]
                else:
                    next_nonterminal = 1.0 - done_buf[t + 1]
                    next_v = val_buf[t + 1]
                delta = rew_buf[t] + cfg.gamma * next_v * next_nonterminal - val_buf[t]
                lastgae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * lastgae
                advantages[t] = lastgae
            returns = advantages + val_buf

        # ---- PPO update ----
        b_inds = np.arange(N)
        clipfracs = []
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            mb_size = N // cfg.num_minibatches
            for start in range(0, N, mb_size):
                mb = b_inds[start:start + mb_size]
                _, new_logp, entropy, new_val = agent.act(obs_buf[mb], act_buf[mb])
                ratio = (new_logp - logp_buf[mb]).exp()
                mb_adv = advantages[mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                if cfg.clip_vloss:
                    v_un = (new_val - returns[mb]) ** 2
                    v_cl = val_buf[mb] + torch.clamp(
                        new_val - val_buf[mb], -cfg.clip_coef, cfg.clip_coef)
                    v_clloss = (v_cl - returns[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_un, v_clloss).mean()
                else:
                    v_loss = 0.5 * ((new_val - returns[mb]) ** 2).mean()
                ent = entropy.mean()
                loss = pg_loss - current_ent * ent + cfg.vf_coef * v_loss
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optim.step()
                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

        # ---- Log ----
        avg_ret = (sum(ep_returns) / len(ep_returns)) if ep_returns else float("nan")
        elapsed = time.monotonic() - t0
        sps = global_step / max(elapsed, 1e-6)
        # Action distribution + entropy diagnostic on rollout buffer
        with torch.no_grad():
            buf_logits = agent.actor(agent.encode(obs_buf))
            buf_log_probs = F.log_softmax(buf_logits, dim=-1)
            buf_probs = buf_log_probs.exp()
            buf_entropy = -(buf_probs * buf_log_probs).sum(-1).mean().item()
        act_counts = torch.bincount(act_buf, minlength=agent.NUM_ACTIONS)
        top_idx = int(act_counts.argmax().item())
        top_frac = act_counts.max().item() / N
        print(f"[{cfg.run_name}] upd {update:>3d}  step {global_step:>7d}  "
              f"pg {pg_loss.item():+.3f}  v {v_loss.item():.3f}  ent {ent.item():.3f}  "
              f"buf_ent {buf_entropy:.2f}  top {top_idx}@{top_frac:.0%}  "
              f"avg_ret {avg_ret:.1f}  sps {sps:.0f}",
              flush=True)

        if cfg.save_every and update % cfg.save_every == 0:
            ckpt = ckpt_dir / f"symbolic_step{global_step:08d}.pt"
            torch.save({"agent": agent.state_dict(), "step": global_step,
                        "config": cfg.__dict__}, ckpt)
            print(f"[{cfg.run_name}]   saved {ckpt}", flush=True)

    env.close()
    final = ckpt_dir / "symbolic_final.pt"
    torch.save({"agent": agent.state_dict(), "step": global_step,
                "config": cfg.__dict__}, final)
    print(f"[{cfg.run_name}] done. final ckpt {final}", flush=True)


if __name__ == "__main__":
    main()
