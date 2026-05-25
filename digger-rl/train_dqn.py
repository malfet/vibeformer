"""DQN on the symbolic GameState.

Why DQN here: our hypothesis after PPO/Dreamer was that the bottleneck
is the *stochasticity* of policy-gradient methods on a game where the
optimal policy is deterministic (the greedy heuristic is deterministic
and scores 1645 mean). DQN sidesteps that:
  - Inference policy is argmax Q(s, a) -- fully deterministic.
  - Off-policy training from a replay buffer means rare reward signals
    are revisited many times, helping with the sparse-reward issue
    that hurt symbolic-PPO.
  - epsilon-greedy can decay to a pure greedy policy.

Run:
    # Vanilla DQN from scratch
    python train_dqn.py --total-timesteps 200000

    # DQfD-style: pre-fill replay with heuristic transitions, supervised
    # pretrain Q to imitate the teacher, then proceed with TD learning.
    python train_dqn.py --total-timesteps 200000 \
        --bc-steps 5000 --bc-pretrain-epochs 10
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

from tools.heuristic_agent import GreedyEmerald
from tools.symbolic_env import (
    BASE_OBS_CHANNELS,
    MHEIGHT,
    MWIDTH,
    SymbolicDiggerEnv,
)
from train_ppo import layer_init, select_device

REPO = Path(__file__).parent.resolve()
CKPT_DIR = REPO / "data" / "checkpoints"


@dataclass
class Config:
    total_timesteps: int = 200_000   # emulator frames (NOT policy steps)
    learning_rate: float = 1e-4
    buffer_size: int = 50_000
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 0.005                # Polyak soft-update rate
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_frac: float = 0.5   # decay over the first half of training
    train_frequency: int = 4          # gradient steps per env step
    learning_starts: int = 1_000      # env steps before first gradient update
    target_update_frequency: int = 1  # soft-update target net per train step
    frame_skip: int = 4
    frame_stack: int = 1
    episodic_life: bool = True
    death_penalty: float = 0.0
    shaping_coef: float = 0.0
    # Optional DQfD-style heuristic warmup
    bc_steps: int = 0
    bc_pretrain_epochs: int = 0
    bc_pretrain_batch: int = 64
    seed: int = 1
    log_every: int = 500              # env steps between log lines
    save_every: int = 5000            # env steps between checkpoints
    run_name: str = "dqn"
    device: str = "auto"


class QNet(nn.Module):
    """Small CNN on the (C, MHEIGHT, MWIDTH) grid -> Q values for 6 actions.

    in_channels = BASE_OBS_CHANNELS * frame_stack; callers must pass the
    right value.
    """

    def __init__(self, in_channels: int = BASE_OBS_CHANNELS,
                 num_actions: int = SymbolicDiggerEnv.NUM_ACTIONS):
        super().__init__()
        h, w = MHEIGHT, MWIDTH
        self.body = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 3, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, padding=1)), nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * h * w, 128)), nn.ReLU(),
        )
        self.head = layer_init(nn.Linear(128, num_actions), std=0.01)

    def forward(self, x):
        return self.head(self.body(x))


class ReplayBuffer:
    """Uniform-sample ring buffer of (s, a, r, s', done)."""

    def __init__(self, capacity: int, obs_shape: tuple, device: torch.device):
        self.capacity = capacity
        self.idx = 0
        self.full = False
        self.obs       = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions   = np.zeros(capacity, dtype=np.int64)
        self.rewards   = np.zeros(capacity, dtype=np.float32)
        self.next_obs  = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones     = np.zeros(capacity, dtype=np.float32)
        self.device = device

    def add(self, s, a, r, ns, d):
        i = self.idx
        self.obs[i] = s
        self.actions[i] = a
        self.rewards[i] = r
        self.next_obs[i] = ns
        self.dones[i] = float(d)
        self.idx = (i + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, n: int, rng: np.random.Generator):
        idx = rng.integers(0, len(self), size=n)
        return (
            torch.from_numpy(self.obs[idx]).to(self.device),
            torch.from_numpy(self.actions[idx]).to(self.device),
            torch.from_numpy(self.rewards[idx]).to(self.device),
            torch.from_numpy(self.next_obs[idx]).to(self.device),
            torch.from_numpy(self.dones[idx]).to(self.device),
        )


def env_step_skipped(env: SymbolicDiggerEnv, action: int, skip: int):
    """Hold action for `skip` emulator frames; return last obs + summed reward."""
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


def epsilon_at(step: int, total_steps: int, cfg: Config) -> float:
    decay_steps = int(total_steps * cfg.epsilon_decay_frac)
    if step >= decay_steps:
        return cfg.epsilon_end
    frac = step / max(1, decay_steps)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def prefill_heuristic(env: SymbolicDiggerEnv, buffer: ReplayBuffer,
                      n_transitions: int, frame_skip: int) -> None:
    """Roll the greedy heuristic and dump its transitions into the buffer.

    Both prefills the replay (so DQN's first updates have non-random data)
    AND yields a labeled (obs, teacher_action) set we can later use for
    supervised Q-pretrain (DQfD-style).
    """
    obs = env.reset()
    chaser = GreedyEmerald()
    for _ in range(n_transitions):
        action = chaser(env._last_state)
        next_obs, r, done, info = env_step_skipped(env, action, frame_skip)
        buffer.add(obs, action, r, next_obs, done)
        if done:
            obs = env.reset()
            chaser.reset()
        else:
            obs = next_obs


def bc_pretrain_q(qnet: QNet, optim: Adam, buffer: ReplayBuffer,
                  epochs: int, batch_size: int, device: torch.device) -> None:
    """Supervised pretrain: nudge Q(s, teacher_a) above Q(s, other_a).

    DQfD's supervised loss: max-margin classification on the action
    chosen by the teacher. We use the simpler "Q-max-action cross-
    entropy" form: treat the Q vector as logits and minimise CE
    against the teacher's action.
    """
    n = len(buffer)
    if n == 0:
        return
    print(f"dqn-bc: {n} teacher transitions, {epochs} epochs, batch={batch_size}",
          flush=True)
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        perm = rng.permutation(n)
        ce_sum = acc_sum = 0.0
        nb = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            obs = torch.from_numpy(buffer.obs[idx]).to(device)
            acts = torch.from_numpy(buffer.actions[idx]).to(device)
            q = qnet(obs)
            ce = F.cross_entropy(q, acts)
            optim.zero_grad()
            ce.backward()
            nn.utils.clip_grad_norm_(qnet.parameters(), 5.0)
            optim.step()
            with torch.no_grad():
                acc_sum += (q.argmax(-1) == acts).float().mean().item()
            ce_sum += ce.item()
            nb += 1
        print(f"  dqn-bc epoch {epoch + 1}/{epochs}  "
              f"ce {ce_sum/nb:.3f}  acc {acc_sum/nb:.3f}", flush=True)


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=Config.total_timesteps)
    p.add_argument("--lr", type=float, default=Config.learning_rate)
    p.add_argument("--buffer-size", type=int, default=Config.buffer_size)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--gamma", type=float, default=Config.gamma)
    p.add_argument("--tau", type=float, default=Config.tau)
    p.add_argument("--eps-start", type=float, default=Config.epsilon_start)
    p.add_argument("--eps-end", type=float, default=Config.epsilon_end)
    p.add_argument("--eps-decay-frac", type=float, default=Config.epsilon_decay_frac)
    p.add_argument("--train-frequency", type=int, default=Config.train_frequency)
    p.add_argument("--learning-starts", type=int, default=Config.learning_starts)
    p.add_argument("--frame-skip", type=int, default=Config.frame_skip)
    p.add_argument("--frame-stack", type=int, default=Config.frame_stack,
                   help="symbolic-obs frame stack (1 = single frame)")
    p.add_argument("--episodic-life", default=Config.episodic_life,
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--death-penalty", type=float, default=Config.death_penalty)
    p.add_argument("--shaping-coef", type=float, default=Config.shaping_coef)
    p.add_argument("--bc-steps", type=int, default=Config.bc_steps,
                   help="prefill the replay buffer with N heuristic transitions")
    p.add_argument("--bc-pretrain-epochs", type=int, default=Config.bc_pretrain_epochs,
                   help="supervised pretrain Q against the teacher's actions")
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--run-name", type=str, default=Config.run_name)
    p.add_argument("--save-every", type=int, default=Config.save_every)
    a = p.parse_args()
    return Config(
        total_timesteps=a.total_timesteps, learning_rate=a.lr,
        buffer_size=a.buffer_size, batch_size=a.batch_size,
        gamma=a.gamma, tau=a.tau,
        epsilon_start=a.eps_start, epsilon_end=a.eps_end,
        epsilon_decay_frac=a.eps_decay_frac,
        train_frequency=a.train_frequency, learning_starts=a.learning_starts,
        frame_skip=a.frame_skip, frame_stack=a.frame_stack,
        episodic_life=a.episodic_life,
        death_penalty=a.death_penalty, shaping_coef=a.shaping_coef,
        bc_steps=a.bc_steps, bc_pretrain_epochs=a.bc_pretrain_epochs,
        seed=a.seed, run_name=a.run_name, save_every=a.save_every,
        device=str(select_device(a.force_cpu)),
    )


def main() -> None:
    cfg = parse_args()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)
    ckpt_dir = CKPT_DIR / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = f"[{cfg.run_name}] "
    print(f"{tag}device={device} cfg={cfg}", flush=True)

    env = SymbolicDiggerEnv(shaping_coef=cfg.shaping_coef,
                            max_steps=10**9,
                            episodic_life=cfg.episodic_life,
                            death_penalty=cfg.death_penalty,
                            frame_stack=cfg.frame_stack)
    qnet = QNet(in_channels=env.obs_channels).to(device)
    target = QNet(in_channels=env.obs_channels).to(device)
    target.load_state_dict(qnet.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    optim = Adam(qnet.parameters(), lr=cfg.learning_rate)
    buffer = ReplayBuffer(cfg.buffer_size, env.obs_shape, device)

    n_params = sum(p.numel() for p in qnet.parameters())
    print(f"{tag}qnet: {n_params:,} params", flush=True)

    # ---- Optional heuristic warmup ----
    if cfg.bc_steps > 0:
        print(f"{tag}prefilling buffer with {cfg.bc_steps} heuristic transitions",
              flush=True)
        prefill_heuristic(env, buffer, cfg.bc_steps, cfg.frame_skip)
        if cfg.bc_pretrain_epochs > 0:
            bc_pretrain_q(qnet, optim, buffer,
                          cfg.bc_pretrain_epochs, cfg.bc_pretrain_batch, device)
            target.load_state_dict(qnet.state_dict())

    obs = env.reset()
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)

    n_steps = cfg.total_timesteps // cfg.frame_skip
    ep_return = 0.0
    ep_length = 0
    ep_returns: collections.deque[float] = collections.deque(maxlen=30)
    last_action: int = 0
    last_q: float = 0.0
    losses: collections.deque[float] = collections.deque(maxlen=50)
    t0 = time.monotonic()
    global_step = 0

    for step in range(n_steps):
        # epsilon-greedy action selection
        epsilon = epsilon_at(step, n_steps, cfg)
        if rng.random() < epsilon:
            action = int(rng.integers(0, SymbolicDiggerEnv.NUM_ACTIONS))
        else:
            with torch.no_grad():
                q = qnet(obs_t)
                action = int(q.argmax(-1).item())
                last_q = float(q.max().item())

        next_obs, reward, done, info = env_step_skipped(env, action, cfg.frame_skip)
        buffer.add(obs, action, reward, next_obs, done)
        ep_return += reward; ep_length += cfg.frame_skip
        global_step += cfg.frame_skip
        last_action = action

        if done:
            ep_returns.append(ep_return)
            print(f"{tag}  ep_end return={ep_return:.0f} length={ep_length} "
                  f"score={info.get('score', 0)} lives={info.get('lives', 0)}",
                  flush=True)
            ep_return = 0.0; ep_length = 0
            next_obs = env.reset()

        obs = next_obs
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)

        # ---- Train step ----
        if step >= cfg.learning_starts and step % cfg.train_frequency == 0 \
                and len(buffer) >= cfg.batch_size:
            s, a, r, ns, d = buffer.sample(cfg.batch_size, rng)
            with torch.no_grad():
                next_q = target(ns).max(-1).values
                target_value = r + cfg.gamma * next_q * (1.0 - d)
            q_sa = qnet(s).gather(1, a.unsqueeze(-1)).squeeze(-1)
            loss = F.mse_loss(q_sa, target_value)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(qnet.parameters(), 10.0)
            optim.step()
            losses.append(loss.item())

            # Polyak target net update
            with torch.no_grad():
                for tp, p in zip(target.parameters(), qnet.parameters()):
                    tp.data.lerp_(p.data, cfg.tau)

        # ---- Logging ----
        if step > 0 and step % cfg.log_every == 0:
            avg_ret = (sum(ep_returns) / len(ep_returns)) if ep_returns else float("nan")
            elapsed = time.monotonic() - t0
            sps = global_step / max(elapsed, 1e-6)
            loss_avg = (sum(losses) / len(losses)) if losses else float("nan")
            print(f"{tag}step {step:>6d}/{n_steps}  frame {global_step:>7d}  "
                  f"eps {epsilon:.3f}  avg_ret {avg_ret:6.1f}  "
                  f"loss {loss_avg:7.3f}  last_q {last_q:+6.2f}  "
                  f"buf {len(buffer):>6d}  sps {sps:.0f}", flush=True)

        if cfg.save_every and step > 0 and step % cfg.save_every == 0:
            ckpt = ckpt_dir / f"dqn_step{global_step:08d}.pt"
            torch.save({"qnet": qnet.state_dict(), "target": target.state_dict(),
                        "step": global_step, "config": cfg.__dict__}, ckpt)
            print(f"{tag}  saved {ckpt}", flush=True)

    final = ckpt_dir / "dqn_final.pt"
    torch.save({"qnet": qnet.state_dict(), "target": target.state_dict(),
                "step": global_step, "config": cfg.__dict__}, final)
    print(f"{tag}done. final ckpt {final}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
