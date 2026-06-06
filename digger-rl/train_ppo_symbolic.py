"""PPO on the symbolic tile-grid observation.

Same PPO recipe as `train_ppo.py` but the agent sees the (C, 10, 15)
symbolic mask grid from SymbolicDiggerEnv instead of raw 84x84 RGB
frames. Two reasons:

  1. Smaller state space -> sample-efficient learning. The pixel
     PPO runs at exp1/exp4/exp7 plateaued at avg_ret ~50 because
     the policy collapsed onto one action while still in the
     "policy gradient = 0, clip_frac = 0" regime. The symbolic obs
     gives the encoder a one-hot starting point: monsters here,
     emeralds there, dirt there. No representation problem.

  2. SymbolicDiggerEnv already supports potential-based shaping
     via `shaping_coef`. The default reward signal (game score)
     gives +25/+100/etc. on emerald/bag/monster events; the shaping
     adds a small per-step term proportional to the Manhattan-
     distance reduction toward the nearest emerald, which gives PPO
     a non-zero gradient *every* step instead of waiting for sparse
     pickups.

Single env loop (DOSBox is one process per emulator instance).
frame_skip applied via env_step_skipped, same as DAGGER.

Run:
    python train_ppo_symbolic.py --total-timesteps 500000 \\
        --shaping-coef 0.5 --frame-stack 4 \\
        --run-name ppo_sym_v1
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
from torch.optim import Adam

from tools.game_state import MHEIGHT, MWIDTH
from tools.symbolic_env import BASE_OBS_CHANNELS, SymbolicDiggerEnv
from train_dagger import env_step_skipped
from train_ppo import layer_init, select_device

REPO = Path(__file__).parent.resolve()
CKPT_DIR = REPO / "data" / "checkpoints"


@dataclass
class Config:
    total_timesteps: int = 500_000        # emulator frames (incl. skip)
    learning_rate: float = 2.5e-4
    num_steps: int = 256                  # policy steps per rollout
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    ent_coef_final: float | None = 0.005  # gentler anneal than pixel PPO
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    frame_skip: int = 4
    frame_stack: int = 4
    shaping_coef: float = 0.5
    episodic_life: bool = True
    force_cpu: bool = False
    log_every: int = 1
    save_every: int = 100
    seed: int = 1
    run_name: str = "ppo_sym"


class SymbolicAgent(nn.Module):
    """Shared 3-conv body (same as PolicyNet in train_dagger), separate
    actor / critic heads. ~150k params at BASE_OBS_CHANNELS*frame_stack=24.

    Why share encoder: PPO needs both pi(a|s) and V(s); a shared trunk
    keeps value-error gradients from re-learning the same features.
    """

    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        h, w = MHEIGHT, MWIDTH
        self.body = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 3, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, padding=1)), nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * h * w, 128)), nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(128, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self.encode(x)).squeeze(-1)

    def act(self, x: torch.Tensor, action: torch.Tensor | None = None):
        z = self.encode(x)
        logits = self.actor(z)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), \
            self.critic(z).squeeze(-1)


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-timesteps", type=int, default=Config.total_timesteps)
    p.add_argument("--lr", type=float, default=Config.learning_rate)
    p.add_argument("--num-steps", type=int, default=Config.num_steps)
    p.add_argument("--no-anneal-lr", action="store_true")
    p.add_argument("--gamma", type=float, default=Config.gamma)
    p.add_argument("--ent-coef", type=float, default=Config.ent_coef)
    p.add_argument("--ent-coef-final", type=float, default=Config.ent_coef_final)
    p.add_argument("--clip-coef", type=float, default=Config.clip_coef)
    p.add_argument("--frame-skip", type=int, default=Config.frame_skip)
    p.add_argument("--frame-stack", type=int, default=Config.frame_stack)
    p.add_argument("--shaping-coef", type=float, default=Config.shaping_coef,
                   help="potential-based shaping for emerald-direction. "
                        "Per step, adds shaping_coef * delta_manhattan to "
                        "the reward. 0 disables.")
    p.add_argument("--episodic-life", default=Config.episodic_life,
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--save-every", type=int, default=Config.save_every)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--run-name", type=str, default=Config.run_name)
    a = p.parse_args()
    return Config(
        total_timesteps=a.total_timesteps,
        learning_rate=a.lr, num_steps=a.num_steps,
        anneal_lr=not a.no_anneal_lr, gamma=a.gamma,
        ent_coef=a.ent_coef, ent_coef_final=a.ent_coef_final,
        clip_coef=a.clip_coef,
        frame_skip=a.frame_skip, frame_stack=a.frame_stack,
        shaping_coef=a.shaping_coef,
        episodic_life=a.episodic_life,
        force_cpu=a.force_cpu,
        save_every=a.save_every, seed=a.seed,
        run_name=a.run_name,
    )


def main() -> None:
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = select_device(cfg.force_cpu)
    ckpt_dir = CKPT_DIR / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = f"[{cfg.run_name}]"
    print(f"{tag} device={device}  cfg={cfg}", flush=True)

    env = SymbolicDiggerEnv(max_steps=10**9,
                             episodic_life=cfg.episodic_life,
                             frame_stack=cfg.frame_stack,
                             shaping_coef=cfg.shaping_coef)
    in_ch = BASE_OBS_CHANNELS * cfg.frame_stack
    num_actions = env.NUM_ACTIONS
    agent = SymbolicAgent(in_channels=in_ch, num_actions=num_actions).to(device)
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"{tag} agent params={n_params:,}  in_ch={in_ch}  "
          f"H={MHEIGHT} W={MWIDTH}", flush=True)
    optim = Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    N = cfg.num_steps
    obs_buf = torch.zeros(N, in_ch, MHEIGHT, MWIDTH, device=device)
    act_buf = torch.zeros(N, dtype=torch.long, device=device)
    logp_buf = torch.zeros(N, device=device)
    rew_buf = torch.zeros(N, device=device)
    done_buf = torch.zeros(N, device=device)
    val_buf = torch.zeros(N, device=device)

    obs_np = env.reset()
    obs_t = torch.from_numpy(obs_np).to(device)
    done_t = torch.zeros((), device=device)

    global_step = 0
    update = 0
    ep_return = 0.0
    ep_length = 0
    ep_returns: collections.deque[float] = collections.deque(maxlen=20)
    # Track the raw game score *at episode end* (info["score"] is cumulative
    # from libretro). Summing per-step info["score_reward"] would
    # under-count because env_step_skipped overwrites info each sub-step.
    ep_scores: collections.deque[float] = collections.deque(maxlen=20)
    t0 = time.monotonic()

    while global_step < cfg.total_timesteps:
        update += 1
        frac_remaining = max(0.0, 1.0 - global_step / cfg.total_timesteps)
        if cfg.anneal_lr:
            optim.param_groups[0]["lr"] = frac_remaining * cfg.learning_rate
        if cfg.ent_coef_final is not None:
            current_ent_coef = (cfg.ent_coef_final
                                + frac_remaining * (cfg.ent_coef - cfg.ent_coef_final))
        else:
            current_ent_coef = cfg.ent_coef

        # ---- Rollout ----
        for step in range(N):
            obs_buf[step] = obs_t
            done_buf[step] = done_t
            with torch.no_grad():
                action, logp, _, value = agent.act(obs_t.unsqueeze(0))
            a = int(action.item())
            act_buf[step] = action.squeeze(0)
            logp_buf[step] = logp.squeeze(0)
            val_buf[step] = value.squeeze(0)

            next_obs, total_r, done, info = env_step_skipped(
                env, a, cfg.frame_skip)
            global_step += cfg.frame_skip
            rew_buf[step] = float(total_r)
            ep_return += float(total_r)
            ep_length += cfg.frame_skip

            if done:
                final_score = float(info.get("score", 0))
                ep_returns.append(ep_return)
                ep_scores.append(final_score)
                print(f"{tag}   ep_end return={ep_return:.1f} "
                      f"score={int(final_score)} length={ep_length} "
                      f"lives={info.get('lives', 0)}", flush=True)
                ep_return = 0.0
                ep_length = 0
                next_obs = env.reset()
                done_t = torch.ones((), device=device)
            else:
                done_t = torch.zeros((), device=device)
            obs_t = torch.from_numpy(next_obs).to(device)

        # ---- Diagnostics on the rollout buffer ----
        flat_obs = obs_buf
        flat_act = act_buf
        with torch.no_grad():
            buf_logits = agent.actor(agent.encode(flat_obs))
            buf_log_probs = F.log_softmax(buf_logits, dim=-1)
            buf_probs = buf_log_probs.exp()
            buf_entropy = -(buf_probs * buf_log_probs).sum(-1)
            buf_spread = (buf_logits.max(-1).values - buf_logits.min(-1).values)
        ent_buf_mean = buf_entropy.mean().item()
        ent_buf_p10 = buf_entropy.quantile(0.1).item()
        spread_mean = buf_spread.mean().item()
        act_counts = torch.bincount(flat_act, minlength=num_actions).float()
        act_dist = act_counts / act_counts.sum()
        top_act_idx = int(act_dist.argmax().item())
        top_act_frac = act_dist.max().item()

        # ---- GAE ----
        with torch.no_grad():
            next_value = agent.value(obs_t.unsqueeze(0)).squeeze(0)
            advantages = torch.zeros_like(rew_buf)
            lastgae = torch.zeros((), device=device)
            for t in reversed(range(N)):
                if t == N - 1:
                    next_nonterminal = 1.0 - done_t
                    next_v = next_value
                else:
                    next_nonterminal = 1.0 - done_buf[t + 1]
                    next_v = val_buf[t + 1]
                delta = rew_buf[t] + cfg.gamma * next_v * next_nonterminal - val_buf[t]
                lastgae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * lastgae
                advantages[t] = lastgae
            returns = advantages + val_buf

        # ---- PPO update ----
        flat_logp = logp_buf
        flat_val = val_buf
        flat_adv = advantages
        flat_ret = returns
        b_inds = np.arange(N)
        clipfracs = []
        approx_kls = []
        for _ in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            mb_size = N // cfg.num_minibatches
            for start in range(0, N, mb_size):
                mb = b_inds[start:start + mb_size]
                _, new_logp, entropy, new_val = agent.act(
                    flat_obs[mb], flat_act[mb])
                ratio = (new_logp - flat_logp[mb]).exp()
                mb_adv = flat_adv[mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                if cfg.clip_vloss:
                    v_unclipped = (new_val - flat_ret[mb]) ** 2
                    v_clipped = flat_val[mb] + torch.clamp(
                        new_val - flat_val[mb], -cfg.clip_coef, cfg.clip_coef)
                    v_clipped_loss = (v_clipped - flat_ret[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped_loss).mean()
                else:
                    v_loss = 0.5 * ((new_val - flat_ret[mb]) ** 2).mean()
                ent = entropy.mean()
                loss = pg_loss - current_ent_coef * ent + cfg.vf_coef * v_loss
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optim.step()
                with torch.no_grad():
                    clipfracs.append(
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())
                    approx_kls.append(
                        ((ratio - 1) - (new_logp - flat_logp[mb])).mean().item())

        # ---- Log ----
        if update % cfg.log_every == 0:
            avg_ret = (sum(ep_returns) / len(ep_returns)) if ep_returns else float("nan")
            avg_score = (sum(ep_scores) / len(ep_scores)) if ep_scores else float("nan")
            elapsed = time.monotonic() - t0
            sps = global_step / max(elapsed, 1e-6)
            lr_now = optim.param_groups[0]["lr"]
            print(f"{tag} upd {update:>4d}  step {global_step:>8d}  "
                  f"pg {pg_loss.item():+.3f}  v {v_loss.item():.3f}  "
                  f"ent {ent.item():.3f}  clip {np.mean(clipfracs):.3f}  "
                  f"kl {np.mean(approx_kls):+.4f}  "
                  f"avg_ret {avg_ret:.1f}  score {avg_score:.0f}  "
                  f"lr {lr_now:.2e}  entc {current_ent_coef:.3f}  "
                  f"sps {sps:.0f}", flush=True)
            print(f"      buf: ent_mean {ent_buf_mean:.2f}  "
                  f"ent_p10 {ent_buf_p10:.2f}  spread {spread_mean:5.2f}  "
                  f"top_a {top_act_idx}@{top_act_frac:.0%}", flush=True)

        if cfg.save_every and update % cfg.save_every == 0:
            ckpt = ckpt_dir / f"ppo_sym_step{global_step:08d}.pt"
            torch.save({"agent": agent.state_dict(), "step": global_step,
                        "config": cfg.__dict__}, ckpt)
            print(f"{tag}   saved {ckpt}", flush=True)

    env.close()
    final = ckpt_dir / "ppo_sym_final.pt"
    torch.save({"agent": agent.state_dict(), "step": global_step,
                "config": cfg.__dict__}, final)
    print(f"{tag} done. final checkpoint {final}", flush=True)


if __name__ == "__main__":
    main()
