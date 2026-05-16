"""Train Dreamer world model offline on a recorded color trace.

Phase 1 of the Dreamer rollout: train the WorldModel (encoder+RSSM+decoder+
reward+continue) on existing replay data. No env interaction. No policy.
Lets us verify the world-model losses converge and rewards become predictable
from the latent state before we wire in the actor-critic + imagined rollouts.

Run:
    python train_dreamer.py data/trace1_color.npz --epochs 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dreamer import DreamerConfig, WorldModel

REPO = Path(__file__).parent.resolve()


def _resize_uint8(frames: np.ndarray, out_size: int) -> np.ndarray:
    """(N, H, W, 3) uint8 -> (N, 3, out_size, out_size) uint8 via torch area resize."""
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).float().mul_(1.0 / 255.0)
    t = F.interpolate(t, size=(out_size, out_size), mode="area")
    return (t * 255.0).clamp_(0, 255).byte().numpy()


def load_trace(path: Path, out_size: int = 64):
    d = np.load(path)
    if not bool(d.get("color", False)):
        raise ValueError(f"{path}: trace must be color (recorded with --color)")
    frames = d["frames"]                 # (N, 84, 84, 3) uint8
    actions = d["actions"].astype(np.int64)
    rewards = d["raw_rewards"].astype(np.float32)
    lives = d["lives"]
    resets = d["resets"].astype(bool) if "resets" in d.files \
        else np.zeros(len(actions), dtype=bool)

    if frames.shape[1] != out_size:
        frames = _resize_uint8(frames, out_size)
    else:
        frames = frames.transpose(0, 3, 1, 2)

    n = len(frames)
    real_done = np.zeros(n, dtype=bool)
    real_done[:-1] = ((lives[1:] == 0) & (lives[:-1] > 0)) | resets[1:]
    continues = (1.0 - real_done.astype(np.float32))
    return frames, actions, rewards, continues, real_done


def episode_segments(real_done: np.ndarray, min_len: int):
    """Return list of (start, end) half-open intervals where no episode ends.

    Drop any segment shorter than min_len, since we can't draw a sequence
    of length min_len from it.
    """
    n = len(real_done)
    segs, start = [], 0
    for t in range(n):
        if real_done[t]:
            if t + 1 - start >= min_len:
                segs.append((start, t + 1))
            start = t + 1
    if n - start >= min_len:
        segs.append((start, n))
    return segs


def sample_batch(frames, actions, rewards, continues, segs,
                 B: int, T: int, device: torch.device, rng) -> tuple:
    H = frames.shape[2]
    obs_b = np.empty((B, T, 3, H, H), dtype=np.uint8)
    act_b = np.empty((B, T), dtype=np.int64)
    rew_b = np.empty((B, T), dtype=np.float32)
    con_b = np.empty((B, T), dtype=np.float32)
    for i in range(B):
        seg = segs[rng.integers(len(segs))]
        s = int(rng.integers(seg[0], seg[1] - T + 1))
        obs_b[i] = frames[s:s + T]
        act_b[i] = actions[s:s + T]
        rew_b[i] = rewards[s:s + T]
        con_b[i] = continues[s:s + T]
    obs_t = torch.from_numpy(obs_b).to(device).float().mul_(1.0 / 255.0)
    return (obs_t,
            torch.from_numpy(act_b).to(device),
            torch.from_numpy(rew_b).to(device),
            torch.from_numpy(con_b).to(device))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-length", type=int, default=50)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--save-path", type=Path,
                   default=REPO / "data" / "checkpoints" / "dreamer_wm.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.force_cpu:
        device = torch.device("cpu")
    elif torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")
    print(f"device={device}  trace={args.trace}")

    frames, actions, rewards, continues, real_done = load_trace(args.trace)
    print(f"trace: {len(frames)} samples, {int(real_done.sum())} episode ends, "
          f"frames shape {frames.shape} dtype {frames.dtype}")
    segs = episode_segments(real_done, min_len=args.seq_length)
    print(f"usable segments (>= {args.seq_length} frames): {len(segs)}")
    if not segs:
        raise RuntimeError("no usable episode segments; reduce --seq-length")

    cfg = DreamerConfig(num_actions=6, obs_size=64)
    wm = WorldModel(cfg).to(device)
    n_params = sum(p.numel() for p in wm.parameters())
    print(f"WorldModel: {n_params:,} params, "
          f"batch={args.batch_size} seq={args.seq_length}")
    optim = torch.optim.Adam(wm.parameters(), lr=args.lr)

    t0 = time.monotonic()
    for epoch in range(args.epochs):
        sums = {}
        for step in range(args.steps_per_epoch):
            obs, act, rew, con = sample_batch(
                frames, actions, rewards, continues, segs,
                args.batch_size, args.seq_length, device, rng)
            losses = wm.loss(obs, act, rew, con)
            optim.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 1000.0)
            optim.step()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.item()
        avg = {k: v / args.steps_per_epoch for k, v in sums.items()}
        elapsed = time.monotonic() - t0
        sps = (epoch + 1) * args.steps_per_epoch / max(elapsed, 1e-6)
        print(f"epoch {epoch + 1:>2d}/{args.epochs}  "
              + "  ".join(f"{k}={v:.3f}" for k, v in avg.items())
              + f"  wall={elapsed:.1f}s  steps/s={sps:.1f}",
              flush=True)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": wm.state_dict(), "config": cfg.__dict__},
               args.save_path)
    print(f"saved {args.save_path}")


if __name__ == "__main__":
    main()
