"""PPO training for Digger, CleanRL-style: single file, no gymnasium.

Pipeline:
  - DiggerEnv (raw 640x400 RGBA frames) -- one env per process due to the
    LibretroCore singleton constraint
  - Per policy step: hold action for frame_skip=4 emulator frames, sum reward
  - Preprocess: drop alpha, grayscale, area-resize to 84x84, float32 in [0,1]
  - Frame-stack last 4 preprocessed frames -> (4, 84, 84) policy input
  - NatureCNN encoder + linear actor head (Discrete 6) + linear critic head
  - Standard PPO: GAE, clipped policy ratio, clipped value loss, entropy bonus,
    advantage normalization, orthogonal init, linear LR anneal

Defaults aim at a sane first run on Apple Silicon MPS in ~10-30 min wall clock.
Tune via --total-timesteps and friends.
"""

from __future__ import annotations

import argparse
import collections
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from digger_env import DiggerEnv

REPO = Path(__file__).parent.resolve()
CKPT_DIR = REPO / "data" / "checkpoints"


@dataclass
class Config:
    total_timesteps: int = 1_000_000  # emulator frames (NOT policy steps)
    learning_rate: float = 2.5e-4
    num_steps: int = 128              # policy steps per rollout
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    frame_skip: int = 4
    frame_stack: int = 4
    obs_size: int = 84
    clip_reward: bool = False
    episodic_life: bool = False
    log_every: int = 1                # updates
    save_every: int = 50              # updates
    device: str = "auto"              # resolved by select_device()
    seed: int = 1


def select_device(force_cpu: bool = False) -> torch.device:
    """Pick the best available accelerator, with a CPU override.

    Delegates to torch.accelerator so we don't enumerate device names here --
    whatever backend (CUDA, MPS, XPU, ...) the torch build supports is what
    we use.
    """
    if force_cpu:
        return torch.device("cpu")
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()
    return torch.device("cpu")


def layer_init(layer: nn.Module, std: float = np.sqrt(2),
               bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """NatureCNN trunk + linear actor / critic heads."""

    def __init__(self, num_actions: int, in_channels: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)), nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self.encode(x)).squeeze(-1)

    def act(self, x: torch.Tensor, action: torch.Tensor | None = None):
        z = self.encode(x)
        logits = self.actor(z)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(z).squeeze(-1)


def preprocess(obs_rgba: np.ndarray, size: int = 84) -> np.ndarray:
    """raw (H, W, 4) RGBA uint8 -> (size, size) float32 in [0, 1]."""
    rgb = obs_rgba[..., :3].astype(np.float32) * (1.0 / 255.0)
    # ITU-R 601-2 luma transform.
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    t = torch.from_numpy(gray)[None, None]
    t = F.interpolate(t, size=(size, size), mode="area")
    return t[0, 0].numpy()


class FrameStack:
    def __init__(self, k: int = 4, size: int = 84):
        self.k = k
        self.size = size
        self.frames: collections.deque[np.ndarray] = collections.deque(maxlen=k)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        for _ in range(self.k):
            self.frames.append(frame)
        return self._stacked()

    def push(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(frame)
        return self._stacked()

    def _stacked(self) -> np.ndarray:
        return np.stack(list(self.frames), axis=0)  # (k, size, size)


def env_step_skipped(env: DiggerEnv, action: int, skip: int):
    """Hold action across `skip` emulator frames; return last obs + summed reward."""
    total_r = 0.0
    info: dict = {}
    obs = None
    done = False
    for _ in range(skip):
        s = env.step(action)
        total_r += s.reward
        obs = s.obs
        info = s.info
        if s.done:
            done = True
            break
    return obs, total_r, done, info


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=Config.total_timesteps)
    p.add_argument("--force-cpu", action="store_true",
                   help="force CPU even if a CUDA/MPS accelerator is available")
    p.add_argument("--lr", type=float, default=Config.learning_rate)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--num-steps", type=int, default=Config.num_steps)
    p.add_argument("--frame-skip", type=int, default=Config.frame_skip)
    p.add_argument("--frame-stack", type=int, default=Config.frame_stack)
    p.add_argument("--save-every", type=int, default=Config.save_every)
    p.add_argument("--no-anneal-lr", action="store_true")
    p.add_argument("--clip-reward", action="store_true",
                   help="report sign(reward) to the agent instead of raw score delta")
    p.add_argument("--episodic-life", action="store_true",
                   help="emit done=True on every life loss, not just game over")
    p.add_argument("--ent-coef", type=float, default=Config.ent_coef,
                   help="entropy bonus weight (default 0.01; bump to 0.05-0.1 to fight collapse)")
    a = p.parse_args()
    return Config(
        total_timesteps=a.total_timesteps,
        device=str(select_device(a.force_cpu)),
        learning_rate=a.lr, seed=a.seed, num_steps=a.num_steps,
        frame_skip=a.frame_skip, frame_stack=a.frame_stack,
        save_every=a.save_every, anneal_lr=not a.no_anneal_lr,
        clip_reward=a.clip_reward, episodic_life=a.episodic_life,
        ent_coef=a.ent_coef,
    )


def main() -> None:
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"device={device} cfg={cfg}", flush=True)

    env = DiggerEnv(max_steps=10**9,  # don't truncate during training
                    clip_reward=cfg.clip_reward,
                    episodic_life=cfg.episodic_life)
    stack = FrameStack(k=cfg.frame_stack, size=cfg.obs_size)
    agent = Agent(num_actions=DiggerEnv.NUM_ACTIONS,
                  in_channels=cfg.frame_stack).to(device)
    optim = Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # Rollout storage
    N = cfg.num_steps
    obs_buf = torch.zeros(N, cfg.frame_stack, cfg.obs_size, cfg.obs_size, device=device)
    act_buf = torch.zeros(N, dtype=torch.long, device=device)
    logp_buf = torch.zeros(N, device=device)
    rew_buf = torch.zeros(N, device=device)
    done_buf = torch.zeros(N, device=device)
    val_buf = torch.zeros(N, device=device)

    raw = env.reset()
    obs_np = stack.reset(preprocess(raw, cfg.obs_size))
    obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)
    done_t = torch.zeros(1, device=device)

    global_step = 0
    update = 0
    ep_return = 0.0
    ep_length = 0
    ep_returns: collections.deque[float] = collections.deque(maxlen=20)
    t0 = time.monotonic()

    while global_step < cfg.total_timesteps:
        update += 1
        if cfg.anneal_lr:
            frac = max(0.0, 1.0 - global_step / cfg.total_timesteps)
            optim.param_groups[0]["lr"] = frac * cfg.learning_rate

        # ---- Rollout ----
        for step in range(N):
            obs_buf[step] = obs_t[0]
            done_buf[step] = done_t[0]
            with torch.no_grad():
                action, logp, _, value = agent.act(obs_t)
            act_buf[step] = action[0]
            logp_buf[step] = logp[0]
            val_buf[step] = value[0]

            raw, reward, done, info = env_step_skipped(
                env, int(action[0].item()), cfg.frame_skip)
            global_step += cfg.frame_skip
            rew_buf[step] = reward
            ep_return += reward
            ep_length += cfg.frame_skip

            if done:
                ep_returns.append(ep_return)
                print(f"  ep_end return={ep_return:.0f} length={ep_length} "
                      f"score={info.get('score', 0)} lives={info.get('lives', 0)}",
                      flush=True)
                ep_return = 0.0
                ep_length = 0
                raw = env.reset()
                obs_np = stack.reset(preprocess(raw, cfg.obs_size))
                done_t = torch.ones(1, device=device)
            else:
                obs_np = stack.push(preprocess(raw, cfg.obs_size))
                done_t = torch.zeros(1, device=device)
            obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)

        # ---- Pre-update diagnostics: encoder-collapse early-warning signals.
        # All computed on the full rollout buffer in a single forward pass so
        # we get a stable measurement that isn't biased to a single minibatch.
        with torch.no_grad():
            buf_logits = agent.actor(agent.encode(obs_buf))
            buf_log_probs = F.log_softmax(buf_logits, dim=-1)
            buf_probs = buf_log_probs.exp()
            buf_entropy = -(buf_probs * buf_log_probs).sum(-1)
            buf_spread = (buf_logits.max(-1).values - buf_logits.min(-1).values)

        ent_buf_mean = buf_entropy.mean().item()
        ent_buf_p10 = buf_entropy.quantile(0.1).item()
        spread_mean = buf_spread.mean().item()

        act_counts = torch.bincount(act_buf, minlength=DiggerEnv.NUM_ACTIONS).float()
        act_dist = act_counts / act_counts.sum()
        top_act_idx = int(act_dist.argmax().item())
        top_act_frac = act_dist.max().item()

        collapse_signals = []
        if top_act_frac > 0.85:
            collapse_signals.append(f"action {top_act_idx} = {top_act_frac:.0%}")
        if ent_buf_p10 < 0.05:
            collapse_signals.append(f"ent_p10 = {ent_buf_p10:.3f}")
        if spread_mean > 15.0:
            collapse_signals.append(f"logit_spread = {spread_mean:.1f}")

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
        approx_kls = []
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
                    v_unclipped = (new_val - returns[mb]) ** 2
                    v_clipped = val_buf[mb] + torch.clamp(
                        new_val - val_buf[mb], -cfg.clip_coef, cfg.clip_coef)
                    v_clipped_loss = (v_clipped - returns[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped_loss).mean()
                else:
                    v_loss = 0.5 * ((new_val - returns[mb]) ** 2).mean()

                ent = entropy.mean()
                loss = pg_loss - cfg.ent_coef * ent + cfg.vf_coef * v_loss

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optim.step()

                with torch.no_grad():
                    clipfracs.append(
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())
                    approx_kls.append(((ratio - 1) - (new_logp - logp_buf[mb])).mean().item())

        # ---- Logging ----
        if update % cfg.log_every == 0:
            avg_ret = (sum(ep_returns) / len(ep_returns)) if ep_returns else float("nan")
            elapsed = time.monotonic() - t0
            sps = global_step / max(elapsed, 1e-6)
            lr_now = optim.param_groups[0]["lr"]
            print(f"upd {update:>4d}  step {global_step:>8d}  "
                  f"pg {pg_loss.item():+.3f}  v {v_loss.item():.3f}  "
                  f"ent {ent.item():.3f}  clip {np.mean(clipfracs):.3f}  "
                  f"avg_ret {avg_ret:.1f}  lr {lr_now:.2e}  sps {sps:.0f}",
                  flush=True)
            print(f"     buf: ent_mean {ent_buf_mean:.2f}  ent_p10 {ent_buf_p10:.2f}  "
                  f"spread {spread_mean:5.2f}  top_a {top_act_idx}@{top_act_frac:.0%}",
                  flush=True)
            if collapse_signals:
                print(f"     ** collapse-warning: {' | '.join(collapse_signals)}",
                      flush=True)

        if cfg.save_every and update % cfg.save_every == 0:
            ckpt = CKPT_DIR / f"ppo_digger_step{global_step:08d}.pt"
            torch.save({"agent": agent.state_dict(), "step": global_step,
                        "config": cfg.__dict__}, ckpt)
            print(f"  saved {ckpt}", flush=True)

    env.close()
    final = CKPT_DIR / "ppo_digger_final.pt"
    torch.save({"agent": agent.state_dict(), "step": global_step,
                "config": cfg.__dict__}, final)
    print(f"done. final checkpoint {final}", flush=True)


if __name__ == "__main__":
    main()
