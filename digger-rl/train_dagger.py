"""DAGGER (Dataset Aggregation) imitation on the symbolic GameState.

Why DAGGER here:
  BC alone at 86-89% imitation accuracy doesn't reproduce a 1645-score
  deterministic teacher, because the ~13% mis-predictions push the
  student into states the teacher never visits -- where the BC dataset
  has zero coverage and the student picks even more wrong actions.
  Compounding error degrades a 1600-score teacher into a 150-score
  student. DAGGER closes the coverage gap by rolling the *student* in
  the env, querying the teacher for the correct action on each visited
  state, and aggregating those (state, teacher_action) pairs into the
  training set.

Schedule:
  iter 0:   collect N_init teacher rollouts, train policy via BC.
  iter k:   rollout student (argmax), label every visited state with
            the teacher's chosen action, append to dataset, retrain.

Eval at end of each iteration on K independent episodes (argmax,
no shaping) so we can see the curve as the aggregated dataset grows.

    python train_dagger.py --iterations 8 --steps-per-iter 3000
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
from tools.symbolic_env import OBS_CHANNELS, OBS_SHAPE, SymbolicDiggerEnv
from train_ppo import layer_init, select_device

REPO = Path(__file__).parent.resolve()
CKPT_DIR = REPO / "data" / "checkpoints"


@dataclass
class Config:
    iterations: int = 8                # outer DAGGER iterations
    initial_steps: int = 5_000         # teacher rollout to seed the dataset
    steps_per_iter: int = 3_000        # student rollout per DAGGER iter
    epochs_per_iter: int = 5           # BC training passes after each rollout
    initial_epochs: int = 20           # passes over the initial teacher dataset
    batch_size: int = 64
    learning_rate: float = 3e-4
    eval_episodes: int = 3             # argmax rollouts after each iter
    eval_max_steps: int = 4000
    frame_skip: int = 4
    episodic_life: bool = True
    seed: int = 1
    run_name: str = "dagger"
    save_every_iter: bool = True
    device: str = "auto"


class PolicyNet(nn.Module):
    """Same CNN body as SymbolicAgent; just a single action-logits head."""

    def __init__(self, in_channels: int = OBS_CHANNELS,
                 num_actions: int = SymbolicDiggerEnv.NUM_ACTIONS):
        super().__init__()
        h, w = OBS_SHAPE[1], OBS_SHAPE[2]
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


def env_step_skipped(env: SymbolicDiggerEnv, action: int, skip: int):
    total_r = 0.0
    obs = None
    info: dict = {}
    done = False
    for _ in range(skip):
        obs, r, done_, info = env.step(action)
        total_r += r
        if done_:
            done = True
            break
    return obs, total_r, done, info


def rollout_collect(env: SymbolicDiggerEnv, policy_fn, n_steps: int,
                    frame_skip: int, teacher: GreedyEmerald,
                    teacher_acts: bool):
    """Roll `policy_fn` in env for n_steps policy steps; label each state
    with teacher's choice (independent of who actually steps the env).

    teacher_acts=True forces the teacher to drive (used for iter 0).
    teacher_acts=False uses the student; this is the DAGGER step.

    Returns: list[(obs_np, teacher_action)] + total reward of the rollout.
    """
    pairs: list[tuple[np.ndarray, int]] = []
    obs = env.reset()
    teacher.reset()
    total_r = 0.0
    while len(pairs) < n_steps:
        # Teacher's correct action for the current state -> training label
        a_teacher = teacher(env._last_state)
        # Action actually taken -> what advances the env
        a_taken = a_teacher if teacher_acts else policy_fn(obs)
        pairs.append((obs.copy(), a_teacher))
        obs, r, done, _info = env_step_skipped(env, a_taken, frame_skip)
        total_r += r
        if done:
            obs = env.reset()
            teacher.reset()
    return pairs, total_r


def train_bc(policy: PolicyNet, optim: Adam,
             pairs: list[tuple[np.ndarray, int]],
             epochs: int, batch_size: int, device: torch.device,
             rng: np.random.Generator) -> tuple[float, float]:
    """Cross-entropy BC over the aggregated dataset; returns (ce, acc)."""
    n = len(pairs)
    obs_np = np.stack([p[0] for p in pairs], axis=0)
    act_np = np.array([p[1] for p in pairs], dtype=np.int64)
    obs_t = torch.from_numpy(obs_np).to(device)
    act_t = torch.from_numpy(act_np).to(device)
    ce_sum = acc_sum = 0.0
    total_batches = 0
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            mb = perm[start:start + batch_size]
            logits = policy(obs_t[mb])
            ce = F.cross_entropy(logits, act_t[mb])
            optim.zero_grad()
            ce.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optim.step()
            with torch.no_grad():
                acc_sum += (logits.argmax(-1) == act_t[mb]).float().mean().item()
            ce_sum += ce.item()
            total_batches += 1
    return ce_sum / total_batches, acc_sum / total_batches


def evaluate(env: SymbolicDiggerEnv, policy: PolicyNet,
             episodes: int, max_steps: int, frame_skip: int,
             device: torch.device) -> tuple[float, float, int]:
    """Argmax-policy rollouts. Returns (mean_score, max_score, mean_length)."""
    scores: list[int] = []
    lengths: list[int] = []
    for _ in range(episodes):
        env.reset()
        score = 0
        steps = 0
        for _ in range(max_steps):
            obs = env._last_state
            obs_tensor = torch.from_numpy(
                _state_to_obs(env)).unsqueeze(0).to(device)
            with torch.no_grad():
                action = int(policy(obs_tensor).argmax(-1).item())
            _, _, done, info = env_step_skipped(env, action, frame_skip)
            score = int(info.get("score", score))
            steps += 1
            if done:
                break
        scores.append(score)
        lengths.append(steps)
    return float(np.mean(scores)), float(max(scores)), int(np.mean(lengths))


def _state_to_obs(env: SymbolicDiggerEnv) -> np.ndarray:
    """Pull the current observation tensor without a step. We can't query
    env.step without advancing; instead reuse the cached state via the
    public state_to_tensor helper.
    """
    from tools.symbolic_env import state_to_tensor
    return state_to_tensor(env._last_state)


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
    p.add_argument("--episodic-life", default=Config.episodic_life,
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--run-name", type=str, default=Config.run_name)
    a = p.parse_args()
    return Config(
        iterations=a.iterations, initial_steps=a.initial_steps,
        steps_per_iter=a.steps_per_iter, epochs_per_iter=a.epochs_per_iter,
        initial_epochs=a.initial_epochs, batch_size=a.batch_size,
        learning_rate=a.lr, eval_episodes=a.eval_episodes,
        frame_skip=a.frame_skip, episodic_life=a.episodic_life,
        seed=a.seed, run_name=a.run_name,
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

    # Single env: LibretroCore is per-process singleton. We toggle
    # `_env.episodic_life` between collection (True) and eval (False)
    # phases via a reset-and-flip helper.
    env = SymbolicDiggerEnv(max_steps=10**9, episodic_life=cfg.episodic_life)
    teacher = GreedyEmerald()

    def reset_eval_mode():
        env._env.episodic_life = False

    def reset_train_mode():
        env._env.episodic_life = cfg.episodic_life

    policy = PolicyNet().to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"{tag}policy: {n_params:,} params", flush=True)
    optim = Adam(policy.parameters(), lr=cfg.learning_rate)

    def policy_argmax(obs_np: np.ndarray) -> int:
        x = torch.from_numpy(obs_np).unsqueeze(0).to(device)
        with torch.no_grad():
            return int(policy(x).argmax(-1).item())

    aggregated: list[tuple[np.ndarray, int]] = []
    t0 = time.monotonic()

    # ---- Iter 0: teacher rollouts + initial BC ----
    print(f"{tag}== iter 0 ==  collecting {cfg.initial_steps} teacher transitions...",
          flush=True)
    reset_train_mode()
    pairs, _ = rollout_collect(
        env, policy_argmax, cfg.initial_steps,
        cfg.frame_skip, teacher, teacher_acts=True)
    aggregated.extend(pairs)
    ce, acc = train_bc(policy, optim, aggregated,
                        cfg.initial_epochs, cfg.batch_size, device, rng)
    counts = np.bincount([p[1] for p in pairs], minlength=6)
    print(f"{tag}iter 0: |D|={len(aggregated)}  "
          f"action dist (N/L/R/U/D/F) {counts.tolist()}  "
          f"bc ce={ce:.3f} acc={acc:.3f}", flush=True)
    reset_eval_mode()
    mean_s, max_s, mean_len = evaluate(
        env, policy, cfg.eval_episodes, cfg.eval_max_steps,
        cfg.frame_skip, device)
    print(f"{tag}iter 0 eval (argmax, {cfg.eval_episodes} eps): "
          f"mean {mean_s:.0f}  max {max_s:.0f}  len {mean_len}", flush=True)

    # ---- DAGGER iterations ----
    for it in range(1, cfg.iterations + 1):
        print(f"{tag}== iter {it} ==  rolling student for {cfg.steps_per_iter} steps...",
              flush=True)
        reset_train_mode()
        pairs, _ = rollout_collect(
            env, policy_argmax, cfg.steps_per_iter,
            cfg.frame_skip, teacher, teacher_acts=False)
        aggregated.extend(pairs)
        # Action-dist on freshly collected pairs (the teacher labels)
        counts = np.bincount([p[1] for p in pairs], minlength=6)
        ce, acc = train_bc(policy, optim, aggregated,
                            cfg.epochs_per_iter, cfg.batch_size, device, rng)
        print(f"{tag}iter {it}: |D|={len(aggregated)}  "
              f"action dist (N/L/R/U/D/F) {counts.tolist()}  "
              f"bc ce={ce:.3f} acc={acc:.3f}", flush=True)
        reset_eval_mode()
        mean_s, max_s, mean_len = evaluate(
            env, policy, cfg.eval_episodes, cfg.eval_max_steps,
            cfg.frame_skip, device)
        elapsed = time.monotonic() - t0
        print(f"{tag}iter {it} eval (argmax, {cfg.eval_episodes} eps): "
              f"mean {mean_s:.0f}  max {max_s:.0f}  len {mean_len}  "
              f"wall {elapsed:.0f}s", flush=True)
        if cfg.save_every_iter:
            ckpt = ckpt_dir / f"dagger_iter{it:02d}.pt"
            torch.save({"policy": policy.state_dict(), "iter": it,
                        "config": cfg.__dict__}, ckpt)

    final = ckpt_dir / "dagger_final.pt"
    torch.save({"policy": policy.state_dict(),
                "iter": cfg.iterations,
                "config": cfg.__dict__}, final)
    print(f"{tag}done. final ckpt {final}  total wall {time.monotonic()-t0:.0f}s",
          flush=True)
    env.close()


if __name__ == "__main__":
    main()
