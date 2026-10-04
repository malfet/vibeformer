"""Offline BC on search-teacher traces with a held-out split.

The same traces (tools/gen_search_trace.py) carry three views of every
decision-time state, so one script trains any of them against identical
labels:

  --input px84    84x84 RGB, 4-frame stack (the train_ppo pixel format)
  --input hires   native-aspect 100x160 RGB, 4-frame stack
  --input sym     symbolic 6x10x15 grid, single frame

Held-out accuracy is reported every epoch and the best epoch is kept.
That was missing from train_ppo's offline path, and it is the number that
exposed memorisation (74% train vs 42% held-out on px84).

Pixel checkpoints load with eval_symbolic --policy pixel and the
gen_search_trace --student path. Symbolic ones load with eval_symbolic
(--policy ckpt, frame_stack=1).

    python -m tools.train_bc_traces --input hires \\
        --traces data/traces/search_v2_{0..12}.npz \\
        --val data/traces/search_v2_13.npz --run-name bc_hires
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from train_ppo import CKPT_DIR, Agent, select_device

FIELD = {"px84": "frames", "hires": "frames_hires", "sym": "sym"}


def load(paths: list[str], field: str, stack: int):
    """Per-trace arrays plus a flat (trace, t) index that never stacks
    across an episode reset."""
    arrs, acts, index = [], [], []
    for i, p in enumerate(paths):
        d = np.load(p)
        x, a, r = d[field], d["actions"].astype(np.int64), d["resets"]
        arrs.append(x)
        acts.append(a)
        for t in range(stack - 1, len(a)):
            if not r[t - stack + 2:t + 1].any():
                index.append((i, t))
    return arrs, acts, np.array(index)


def batch(arrs, acts, index, ids, stack, pixel):
    xs, ys = [], []
    for i, t in index[ids]:
        if pixel:
            # (k, H, W, 3) -> (k*3, H, W): FrameStack's channel layout.
            w = arrs[i][t - stack + 1:t + 1]
            xs.append(w.transpose(0, 3, 1, 2).reshape(-1, *w.shape[1:3]))
        else:
            xs.append(arrs[i][t])
        ys.append(acts[i][t])
    return np.stack(xs), np.array(ys)


def random_shift(x: torch.Tensor, pad: int) -> torch.Tensor:
    """DrQ-style augmentation: replicate-pad and crop back at a random
    offset per sample. Sprites move by a few pixels; the policy should not
    care exactly where."""
    n, _, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    oy = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    rows = (oy[:, None] + torch.arange(h, device=x.device)[None])   # (n, h)
    cols = (ox[:, None] + torch.arange(w, device=x.device)[None])   # (n, w)
    idx_n = torch.arange(n, device=x.device)[:, None, None]
    return xp.permute(0, 2, 3, 1)[idx_n, rows[:, :, None],
                                  cols[:, None, :]].permute(0, 3, 1, 2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", choices=sorted(FIELD), required=True)
    p.add_argument("--traces", nargs="+", required=True)
    p.add_argument("--val", nargs="+", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--shift", type=int, default=0,
                   help="random-shift augmentation in pixels (pixel inputs)")
    p.add_argument("--width", type=int, default=1)
    p.add_argument("--init", type=Path, default=None,
                   help="start from this checkpoint (DAgger rounds)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--run-name", required=True)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = select_device(False)
    pixel = a.input != "sym"
    stack = 4 if pixel else 1
    field = FIELD[a.input]
    tr = load(a.traces, field, stack)
    va = load(a.val, field, stack)
    sample, _ = batch(*tr, np.arange(1), stack, pixel)
    in_ch, h, w = sample.shape[1:]
    print(f"[{a.run_name}] {a.input}: train {len(tr[2]):,}  val {len(va[2]):,}  "
          f"obs {sample.shape[1:]}", flush=True)

    if pixel:
        model = Agent(num_actions=6, in_channels=in_ch, width=a.width,
                      in_hw=(h, w))
        cfg = {"frame_stack": stack, "color": True, "obs_size": h,
               "obs_hw": [h, w], "frame_skip": 4, "encoder_width": a.width,
               "separate_critic": False, "episodic_life": True,
               "clip_reward": True}
        logits_of = lambda x: model.actor(model.encode(x))       # noqa: E731
    else:
        from train_ppo_symbolic import SymbolicAgent
        model = SymbolicAgent(in_channels=in_ch, num_actions=6, h=h, w=w)
        cfg = {"frame_stack": 1, "egocentric": False, "arch": "cnn"}
        logits_of = lambda x: model.actor(model.encode(x))       # noqa: E731
    if a.init is not None:
        model.load_state_dict(torch.load(a.init, weights_only=False,
                                         map_location="cpu")["agent"])
    model.to(device)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"[{a.run_name}] params {n_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, eps=1e-5,
                            weight_decay=a.weight_decay)
    out_dir = CKPT_DIR / a.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    def to_t(x):
        t = torch.from_numpy(x).to(device).float()
        return t.div_(255.0) if pixel else t

    def evaluate():
        model.eval()
        correct = 0
        with torch.no_grad():
            for s in range(0, len(va[2]), 1024):
                x, y = batch(*va, np.arange(s, min(s + 1024, len(va[2]))),
                             stack, pixel)
                correct += (logits_of(to_t(x)).argmax(-1).cpu().numpy()
                            == y).sum()
        model.train()
        return correct / len(va[2])

    best = -1.0
    for ep in range(a.epochs):
        t0 = time.monotonic()
        perm = rng.permutation(len(tr[2]))
        ce_sum = acc_sum = nb = 0
        for s in range(0, len(perm), a.batch_size):
            x, y = batch(*tr, perm[s:s + a.batch_size], stack, pixel)
            xt, yt = to_t(x), torch.from_numpy(y).to(device)
            if pixel and a.shift:
                xt = random_shift(xt, a.shift)
            logits = logits_of(xt)
            loss = F.cross_entropy(logits, yt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            ce_sum += loss.item()
            acc_sum += (logits.argmax(-1) == yt).float().mean().item()
            nb += 1
        val = evaluate()
        tag = ""
        if val > best:
            best = val
            torch.save({"agent": model.state_dict(), "config": cfg,
                        "epoch": ep + 1, "val_acc": val, "input": a.input},
                       out_dir / "best.pt")
            tag = "  *best"
        print(f"[{a.run_name}] epoch {ep + 1}/{a.epochs}  ce {ce_sum / nb:.3f}  "
              f"train acc {acc_sum / nb:.3f}  val acc {val:.3f}  "
              f"{time.monotonic() - t0:.0f}s{tag}", flush=True)
    print(f"[{a.run_name}] done. best val acc {best:.3f} -> {out_dir / 'best.pt'}",
          flush=True)


if __name__ == "__main__":
    main()
