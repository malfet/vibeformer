"""Pixel DAGGER: imitation from frame-stacked RGB with a privileged
symbolic teacher.

Mirrors train_dagger.py (symbolic DAGGER) but the student consumes the
preprocessed 84x84 RGB frame stack -- raw pixels only, never the
extracted GameState. The teacher (GreedyEmerald) still reads the
symbolic state on the side and provides the label.

At each visited state we run extract_state_fast(raw_frame) to get the
teacher's chosen action, then train the pixel CNN against it. The
student has to implicitly learn whatever subset of the CV extractor
matters for the heuristic policy.

This is the cleanest test of whether pixel-only RL on this game can
match symbolic-state RL given identical supervision -- pixel PPO with
sparse rewards collapsed to a single action, so we're swapping the
optimisation problem for dense imitation supervision.

    python train_dagger_pixel.py --iterations 8 --run-name dagger_pixel
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

from digger_env import DiggerEnv, FrameStack, _env_step_skipped, preprocess_uint8
from tools.game_state import extract_state_fast
from tools.heuristic_agent import GreedyEmerald
from train_ppo import Agent, select_device

REPO = Path(__file__).parent.resolve()
CKPT_DIR = REPO / "data" / "checkpoints"


@dataclass
class Config:
    iterations: int = 8
    initial_steps: int = 5_000
    steps_per_iter: int = 3_000
    epochs_per_iter: int = 5
    initial_epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 3e-4
    eval_episodes: int = 3
    eval_max_steps: int = 4000
    frame_skip: int = 4
    frame_stack: int = 4
    obs_size: int = 84
    color: bool = True
    episodic_life: bool = True
    encoder_width: int = 1
    seed: int = 1
    run_name: str = "dagger_pixel"
    save_every_iter: bool = True
    device: str = "auto"
    # Aggregated dataset cap (FIFO eviction). 29k samples * 12 * 84 *
    # 84 uint8 ~ 2.4 GB, so the cap matters once we run beyond default
    # budgets. Default 50k is roomy for the default 8-iter schedule.
    max_dataset_size: int = 50_000


class PixelEnv:
    """DiggerEnv + raw-frame access + preprocessed frame stack.

    Standard (obs, reward, done, info) protocol, plus `raw_frame`
    exposing the most recent raw RGBA frame so the teacher can run CV
    on it without driving the emulator a second time.
    """

    def __init__(self, frame_stack: int = 4, obs_size: int = 84,
                 color: bool = True, episodic_life: bool = True):
        self._env = DiggerEnv(episodic_life=episodic_life, max_steps=10**9)
        self._stack = FrameStack(k=frame_stack, size=obs_size, color=color)
        self.obs_size = obs_size
        self.frame_stack = frame_stack
        self.color = color
        self.raw_frame: np.ndarray | None = None

    @property
    def obs_channels(self) -> int:
        return self.frame_stack * (3 if self.color else 1)

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        return (self.obs_channels, self.obs_size, self.obs_size)

    def reset(self) -> np.ndarray:
        raw = self._env.reset()
        self.raw_frame = raw
        return self._stack.reset(preprocess_uint8(raw, self.obs_size, self.color))

    def step_skipped(self, action: int, skip: int):
        raw, total_r, done, info = _env_step_skipped(self._env, action, skip)
        self.raw_frame = raw
        obs = self._stack.push(preprocess_uint8(raw, self.obs_size, self.color))
        return obs, total_r, done, info

    def set_episodic_life(self, on: bool) -> None:
        self._env.episodic_life = on

    def close(self) -> None:
        self._env.close()


def rollout_collect(env: PixelEnv, policy_fn, n_steps: int, frame_skip: int,
                    teacher: GreedyEmerald, teacher_acts: bool,
                    aggregated: collections.deque) -> None:
    """Roll policy in env for n_steps; label each visited state with the
    teacher. obs (the pixel stack) is what the student trains against;
    raw_frame is what the teacher reads CV from. Both refer to the same
    point in time.
    """
    obs = env.reset()
    teacher.reset()
    collected = 0
    while collected < n_steps:
        state = extract_state_fast(env.raw_frame)
        a_teacher = teacher(state)
        a_taken = a_teacher if teacher_acts else policy_fn(obs)
        aggregated.append((obs.copy(), a_teacher))
        obs, _r, done, _info = env.step_skipped(a_taken, frame_skip)
        collected += 1
        if done:
            obs = env.reset()
            teacher.reset()


def _obs_to_device(obs_u8: torch.Tensor, device: torch.device) -> torch.Tensor:
    """uint8 (..., C, H, W) -> float32 in [0, 1] on device."""
    return obs_u8.to(device).float().mul_(1.0 / 255.0)


def train_bc(policy: Agent, optim: Adam,
             aggregated: collections.deque,
             epochs: int, batch_size: int, device: torch.device,
             rng: np.random.Generator) -> tuple[float, float]:
    """Cross-entropy BC over the aggregated dataset; returns (ce, acc).

    Dataset is kept on CPU as uint8 (84672 B/sample) so 50k samples
    fit in ~4 GB. Each minibatch is copied to device and cast to
    float32/[0,1] just-in-time.
    """
    n = len(aggregated)
    obs_np = np.stack([a[0] for a in aggregated], axis=0)
    act_np = np.array([a[1] for a in aggregated], dtype=np.int64)
    obs_cpu = torch.from_numpy(obs_np)
    act_cpu = torch.from_numpy(act_np)
    ce_sum = acc_sum = 0.0
    total_batches = 0
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            mb = perm[start:start + batch_size]
            obs_t = _obs_to_device(obs_cpu[mb], device)
            act_t = act_cpu[mb].to(device)
            logits = policy.actor(policy.encode(obs_t))
            ce = F.cross_entropy(logits, act_t)
            optim.zero_grad()
            ce.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optim.step()
            with torch.no_grad():
                acc_sum += (logits.argmax(-1) == act_t).float().mean().item()
            ce_sum += ce.item()
            total_batches += 1
    return ce_sum / total_batches, acc_sum / total_batches


def evaluate(env: PixelEnv, policy: Agent, episodes: int, max_steps: int,
             frame_skip: int, device: torch.device) -> tuple[float, float, int]:
    """Argmax-policy rollouts under full game-over termination."""
    env.set_episodic_life(False)
    scores: list[int] = []
    lengths: list[int] = []
    for _ in range(episodes):
        obs = env.reset()
        score = 0
        steps = 0
        for _ in range(max_steps):
            obs_t = _obs_to_device(torch.from_numpy(obs), device).unsqueeze(0)
            with torch.no_grad():
                logits = policy.actor(policy.encode(obs_t))
                action = int(logits.argmax(-1).item())
            obs, _, done, info = env.step_skipped(action, frame_skip)
            score = int(info.get("score", score))
            steps += 1
            if done:
                break
        scores.append(score)
        lengths.append(steps)
    return float(np.mean(scores)), float(max(scores)), int(np.mean(lengths))


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=Config.iterations)
    p.add_argument("--initial-steps", type=int, default=Config.initial_steps)
    p.add_argument("--steps-per-iter", type=int, default=Config.steps_per_iter)
    p.add_argument("--epochs-per-iter", type=int, default=Config.epochs_per_iter)
    p.add_argument("--initial-epochs", type=int, default=Config.initial_epochs)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--lr", type=float, default=Config.learning_rate)
    p.add_argument("--eval-episodes", type=int, default=Config.eval_episodes)
    p.add_argument("--frame-skip", type=int, default=Config.frame_skip)
    p.add_argument("--frame-stack", type=int, default=Config.frame_stack)
    p.add_argument("--obs-size", type=int, default=Config.obs_size)
    p.add_argument("--encoder-width", type=int, default=Config.encoder_width,
                   help="NatureCNN width multiplier (1 = ~1.7M params, 2 = ~6M)")
    p.add_argument("--color", default=Config.color,
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--episodic-life", default=Config.episodic_life,
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--max-dataset-size", type=int, default=Config.max_dataset_size)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--run-name", type=str, default=Config.run_name)
    a = p.parse_args()
    return Config(
        iterations=a.iterations, initial_steps=a.initial_steps,
        steps_per_iter=a.steps_per_iter, epochs_per_iter=a.epochs_per_iter,
        initial_epochs=a.initial_epochs, batch_size=a.batch_size,
        learning_rate=a.lr, eval_episodes=a.eval_episodes,
        frame_skip=a.frame_skip, frame_stack=a.frame_stack,
        obs_size=a.obs_size, encoder_width=a.encoder_width,
        color=a.color, episodic_life=a.episodic_life,
        max_dataset_size=a.max_dataset_size, seed=a.seed,
        run_name=a.run_name, device=str(select_device(a.force_cpu)),
    )


def _recent_action_counts(aggregated: collections.deque, n: int) -> list[int]:
    """Action histogram over the last `n` aggregated samples."""
    n = min(n, len(aggregated))
    if n == 0:
        return [0] * DiggerEnv.NUM_ACTIONS
    # deque slicing isn't supported; islice from the right via list.
    tail = list(aggregated)[-n:]
    counts = np.bincount([a for _, a in tail],
                          minlength=DiggerEnv.NUM_ACTIONS)
    return counts.tolist()


def main() -> None:
    cfg = parse_args()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)
    ckpt_dir = CKPT_DIR / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = f"[{cfg.run_name}] "
    print(f"{tag}device={device} cfg={cfg}", flush=True)

    env = PixelEnv(frame_stack=cfg.frame_stack, obs_size=cfg.obs_size,
                   color=cfg.color, episodic_life=cfg.episodic_life)
    teacher = GreedyEmerald()
    policy = Agent(num_actions=DiggerEnv.NUM_ACTIONS,
                   in_channels=env.obs_channels,
                   width=cfg.encoder_width).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"{tag}policy: {n_params:,} params  obs_channels={env.obs_channels}",
          flush=True)
    optim = Adam(policy.parameters(), lr=cfg.learning_rate)

    aggregated: collections.deque = collections.deque(
        maxlen=cfg.max_dataset_size)

    def policy_argmax(obs_np: np.ndarray) -> int:
        x = _obs_to_device(torch.from_numpy(obs_np), device).unsqueeze(0)
        with torch.no_grad():
            return int(policy.actor(policy.encode(x)).argmax(-1).item())

    t0 = time.monotonic()

    # ---- Iter 0: teacher rollouts + initial BC ----
    print(f"{tag}== iter 0 ==  collecting {cfg.initial_steps} teacher transitions...",
          flush=True)
    env.set_episodic_life(cfg.episodic_life)
    rollout_collect(env, policy_argmax, cfg.initial_steps,
                    cfg.frame_skip, teacher, teacher_acts=True,
                    aggregated=aggregated)
    counts = _recent_action_counts(aggregated, cfg.initial_steps)
    ce, acc = train_bc(policy, optim, aggregated,
                        cfg.initial_epochs, cfg.batch_size, device, rng)
    print(f"{tag}iter 0: |D|={len(aggregated)}  "
          f"action dist (N/L/R/U/D/F) {counts}  "
          f"bc ce={ce:.3f} acc={acc:.3f}", flush=True)
    mean_s, max_s, mean_len = evaluate(
        env, policy, cfg.eval_episodes, cfg.eval_max_steps,
        cfg.frame_skip, device)
    print(f"{tag}iter 0 eval (argmax, {cfg.eval_episodes} eps): "
          f"mean {mean_s:.0f}  max {max_s:.0f}  len {mean_len}", flush=True)

    # ---- DAGGER iterations ----
    for it in range(1, cfg.iterations + 1):
        print(f"{tag}== iter {it} ==  rolling student for {cfg.steps_per_iter} steps...",
              flush=True)
        env.set_episodic_life(cfg.episodic_life)
        rollout_collect(env, policy_argmax, cfg.steps_per_iter,
                        cfg.frame_skip, teacher, teacher_acts=False,
                        aggregated=aggregated)
        counts = _recent_action_counts(aggregated, cfg.steps_per_iter)
        ce, acc = train_bc(policy, optim, aggregated,
                            cfg.epochs_per_iter, cfg.batch_size, device, rng)
        print(f"{tag}iter {it}: |D|={len(aggregated)}  "
              f"action dist (N/L/R/U/D/F) {counts}  "
              f"bc ce={ce:.3f} acc={acc:.3f}", flush=True)
        mean_s, max_s, mean_len = evaluate(
            env, policy, cfg.eval_episodes, cfg.eval_max_steps,
            cfg.frame_skip, device)
        elapsed = time.monotonic() - t0
        print(f"{tag}iter {it} eval (argmax, {cfg.eval_episodes} eps): "
              f"mean {mean_s:.0f}  max {max_s:.0f}  len {mean_len}  "
              f"wall {elapsed:.0f}s", flush=True)
        if cfg.save_every_iter:
            ckpt = ckpt_dir / f"dagger_pixel_iter{it:02d}.pt"
            torch.save({"policy": policy.state_dict(), "iter": it,
                        "config": cfg.__dict__}, ckpt)

    final = ckpt_dir / "dagger_pixel_final.pt"
    torch.save({"policy": policy.state_dict(),
                "iter": cfg.iterations,
                "config": cfg.__dict__}, final)
    print(f"{tag}done. final ckpt {final}  total wall {time.monotonic()-t0:.0f}s",
          flush=True)
    env.close()


if __name__ == "__main__":
    main()
