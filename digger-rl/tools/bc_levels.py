"""Behavioural cloning on chosen levels, scored per level on held-out traces.

Answers two questions with the same machinery:

  - does training on more levels help on levels never trained on?
    (same arch, different `levels`)
  - which encoder head generalises across levels?
    (same levels, different `arch`)

  - does the network need RGB? (`input`: rgb, or one channel per frame)

Traces must carry `levels` (collect.py records it). Each run is
`name:arch:levels[:input]`, levels as "1-3" or "1,2,3,5", input one of
INPUTS; every run is cut to
the same number of training samples (`--samples`, default: the smallest
run's pool), so only the choice of levels or arch differs. The best
epoch is picked on validation accuracy over the run's own levels, never
the held-out ones. Each run's checkpoint is saved where evaluate.py can
load it, and a per-level accuracy table is printed at the end.

    python -m tools.bc_levels --traces data/traces/lv_{0..9}.npz \\
        --val data/traces/lv_val_{0,1}.npz \\
        --run early:impala:1-3 --run spread:impala:1-3,5-7,9-10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from digger_env import DiggerEnv
from train_pixel import (Agent, Config, _bc_minibatch, _build_bc_index,
                         _load_traces, random_shift)


# One channel per frame instead of RGB. Both are linear in RGB, so they
# commute with the area downscale and apply to stored 128x128 traces.
# Luma merges three of the 8 colours Digger uses (dark green, bright red,
# brown all land at ~0.30); "pal" is a linear map fitted to the palette
# that keeps all 8 apart (0, .09, .19, .28, .37, .72, .91, 1).
INPUTS = {"rgb": None,
          "luma": (0.299, 0.587, 0.114),
          "pal": (0.1846, 0.7231, 0.0923)}


def to_input(x: torch.Tensor, kind: str) -> torch.Tensor:
    w = INPUTS[kind]
    if w is None:
        return x
    n, c, h, wd = x.shape
    return torch.einsum("nkchw,c->nkhw", x.view(n, c // 3, 3, h, wd),
                        torch.tensor(w, device=x.device))


def parse_levels(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def load(paths: list[str], cfg: Config):
    traces = _load_traces(tuple(paths), cfg)
    data = _build_bc_index(traces, cfg)
    levels = [np.load(p)["levels"] for p in paths]
    lv = np.array([levels[i][t] for i, t in data[3]], np.int16)
    return data, lv


@torch.no_grad()
def per_level_acc(agent, data, lv, cfg, device,
                  kind: str = "rgb") -> dict[int, float]:
    frames, actions, _, index = data
    agent.eval()
    ok = np.zeros(len(index), bool)
    for s in range(0, len(index), 512):
        idxs = np.arange(s, min(s + 512, len(index)))
        obs, acts, _ = _bc_minibatch(frames, actions, None, index, idxs,
                                     cfg, device)
        obs = to_input(obs, kind)
        ok[idxs] = (agent.actor(agent.encode(obs)).argmax(-1)
                    == acts).cpu().numpy()
    agent.train()
    return {int(l): float(ok[lv == l].mean()) for l in np.unique(lv)}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--traces", nargs="+", required=True)
    p.add_argument("--val", nargs="+", required=True)
    p.add_argument("--run", action="append", required=True,
                   metavar="NAME:ARCH:LEVELS")
    p.add_argument("--samples", type=int, default=0)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--aug-shift", type=int, default=4)
    p.add_argument("--out-dir", type=Path, default=Path("data/checkpoints/bc_levels"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = Config(obs_size=128, color=True, frame_stack=4, arch="impala")
    train, tlv = load(args.traces, cfg)
    val, vlv = load(args.val, cfg)
    print("train samples per level:",
          dict(zip(*map(list, np.unique(tlv, return_counts=True)))))
    print("val samples per level:  ",
          dict(zip(*map(list, np.unique(vlv, return_counts=True)))), flush=True)

    runs = []
    for spec in args.run:
        name, arch, levels, *kind = spec.split(":")
        kind = kind[0] if kind else "rgb"
        assert kind in INPUTS, kind
        lvls = parse_levels(levels)
        pool = np.nonzero(np.isin(tlv, list(lvls)))[0]
        runs.append((name, arch, lvls, pool, kind))
    n = args.samples or min(len(r[3]) for r in runs)
    print(f"{n} training samples per run", flush=True)

    results = {}
    for name, arch, lvls, pool, kind in runs:
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(pool, size=min(n, len(pool)), replace=False)
        rcfg = Config(obs_size=128, color=True, frame_stack=4, arch=arch,
                      aug_shift=args.aug_shift, seed=args.seed,
                      run_name=f"bc_levels_{name}")
        torch.manual_seed(args.seed)
        agent = Agent(DiggerEnv.NUM_ACTIONS, 12 if kind == "rgb" else 4,
                      in_hw=(128, 128),
                      arch=arch).to(device)
        optim = Adam(agent.parameters(), lr=args.lr, eps=1e-5)
        nparam = sum(q.numel() for q in agent.parameters())
        best = (-1.0, None, None, -1)
        t0 = time.monotonic()
        for ep in range(args.epochs):
            perm = rng.permutation(pick)
            for s in range(0, len(perm), args.batch):
                obs, acts, _ = _bc_minibatch(train[0], train[1], None,
                                             train[3], perm[s:s + args.batch],
                                             rcfg, device)
                obs = to_input(obs, kind)
                if args.aug_shift:
                    obs = random_shift(obs, args.aug_shift)
                loss = F.cross_entropy(agent.actor(agent.encode(obs)), acts)
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optim.step()
            acc = per_level_acc(agent, val, vlv, rcfg, device, kind)
            w = {l: (vlv == l).sum() for l in acc}
            seen = (sum(acc[l] * w[l] for l in acc if l in lvls)
                    / max(1, sum(w[l] for l in acc if l in lvls)))
            held = {l: a for l, a in acc.items() if l not in lvls}
            print(f"[{name}] epoch {ep + 1}: seen {seen:.3f}  held-out "
                  + " ".join(f"L{l} {a:.3f}" for l, a in held.items())
                  + f"  ({time.monotonic() - t0:.0f}s)", flush=True)
            if seen > best[0]:
                best = (seen, {k: v.detach().clone()
                               for k, v in agent.state_dict().items()},
                        acc, ep + 1)
        out = args.out_dir / name
        out.mkdir(parents=True, exist_ok=True)
        # luma is what color=False preprocessing gives, so evaluate.py
        # can play it; a "pal" checkpoint is for the table only.
        ccfg = dict(rcfg.__dict__, color=kind == "rgb", input=kind)
        torch.save({"agent": best[1], "step": 0, "config": ccfg},
                   out / "ppo_digger_bc_only.pt")
        results[name] = dict(arch=arch, input=kind, levels=sorted(lvls),
                             params=nparam,
                             epoch=best[3], seen=best[0], per_level=best[2])
        del agent, optim

    levels = sorted({l for r in results.values() for l in r["per_level"]})
    print("\nval accuracy per level at each run's best epoch "
          "(* = not trained on)")
    print(f"{'run':<14}{'arch':<14}{'input':<6}{'params':>9} "
          + "".join(f"{'L' + str(l):>7}" for l in levels))
    for name, r in results.items():
        cells = "".join(
            f"{r['per_level'].get(l, float('nan')):6.3f}"
            + ("*" if l not in r["levels"] else " ") for l in levels)
        print(f"{name:<14}{r['arch']:<14}{r['input']:<6}{r['params']:>9} "
              f"{cells}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
