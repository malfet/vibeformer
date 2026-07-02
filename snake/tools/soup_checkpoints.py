"""Merge two or more agent checkpoints into a single 'souped' checkpoint.

Two modes:

  1. Uniform / weighted average (default):
         theta_out = sum_i w_i * theta_i

     The classic 'model soup'. Works when all inputs share an init and stayed
     in the same loss basin (i.e. they were fine-tuned from the same BC
     checkpoint). Fails silently if that isn't true — the average may not
     score above any individual specialist.

  2. Task arithmetic (--base):
         theta_out = theta_base + sum_i alpha_i * (theta_i - theta_base)

     Interprets each specialist as a 'task vector' delta from the base. Alpha
     controls how strongly each specialist's behavior is added on top. Set
     alphas < 1 to interpolate; > 1 to extrapolate; negative to subtract a
     behavior. Reference: Ilharco et al. 2022, 'Editing Models with Task
     Arithmetic'.

Usage:
    # uniform soup of three specialists
    python -m tools.soup_checkpoints \\
        --inputs checkpoints/ppo06a/ppo_tiny.pt \\
                 checkpoints/ppo06b/ppo_tiny.pt \\
                 checkpoints/ppo06c/ppo_tiny.pt \\
        --output checkpoints/soup1/ppo_tiny.pt

    # weighted soup (emphasize the survivor specialist)
    python -m tools.soup_checkpoints \\
        --inputs .../greedy.pt .../survivor.pt .../efficient.pt \\
        --weights 1 2 1 \\
        --output checkpoints/soup_weighted/ppo_tiny.pt

    # task arithmetic against a BC-init base
    python -m tools.soup_checkpoints \\
        --base checkpoints/ppo05/ppo_tiny.pt \\
        --inputs .../greedy.pt .../survivor.pt \\
        --alphas 0.5 0.5 \\
        --output checkpoints/task_arith/ppo_tiny.pt

Load the result with `tools/play_agent.py --ckpt <output>` or point
`train_ppo.py --load-bc <output>` at it to warm-start further training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _load(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _check_same_shapes(ref: dict, other: dict, other_path: str) -> None:
    if set(ref.keys()) != set(other.keys()):
        raise SystemExit(
            f"{other_path}: state_dict keys differ from reference "
            f"(missing {set(ref.keys()) - set(other.keys())}, "
            f"extra {set(other.keys()) - set(ref.keys())})")
    for k, t in ref.items():
        if other[k].shape != t.shape:
            raise SystemExit(
                f"{other_path}: shape mismatch on {k!r}: "
                f"{tuple(other[k].shape)} vs {tuple(t.shape)}")


def _norm_weights(raw: list[float] | None, n: int) -> list[float]:
    if raw is None:
        return [1.0 / n] * n
    if len(raw) != n:
        raise SystemExit(f"--weights has length {len(raw)}, need {n}")
    s = sum(raw)
    if s <= 0:
        raise SystemExit("--weights must sum to a positive value")
    return [w / s for w in raw]


def uniform_soup(ckpts: list[dict], weights: list[float]) -> dict:
    ref = ckpts[0]["agent"]
    for c, p in zip(ckpts[1:], range(1, len(ckpts))):
        _check_same_shapes(ref, c["agent"], f"input #{p}")
    out: dict[str, torch.Tensor] = {}
    for k, ref_t in ref.items():
        acc = torch.zeros_like(ref_t, dtype=torch.float32)
        for w, c in zip(weights, ckpts):
            acc.add_(c["agent"][k].to(torch.float32), alpha=w)
        out[k] = acc.to(ref_t.dtype)
    return out


def task_arithmetic(base: dict, ckpts: list[dict],
                    alphas: list[float]) -> dict:
    ref = base["agent"]
    for c, p in zip(ckpts, range(len(ckpts))):
        _check_same_shapes(ref, c["agent"], f"input #{p}")
    out: dict[str, torch.Tensor] = {}
    for k, base_t in ref.items():
        acc = base_t.to(torch.float32).clone()
        for a, c in zip(alphas, ckpts):
            delta = c["agent"][k].to(torch.float32) - base_t.to(torch.float32)
            acc.add_(delta, alpha=a)
        out[k] = acc.to(base_t.dtype)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True,
                   help="two or more checkpoints to merge")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--base", type=str, default=None,
                   help="if set, run task-arithmetic against this base "
                        "checkpoint instead of uniform soup")
    p.add_argument("--weights", type=float, nargs="*", default=None,
                   help="uniform-soup weights (default: uniform). Re-"
                        "normalized to sum to 1.")
    p.add_argument("--alphas", type=float, nargs="*", default=None,
                   help="task-arithmetic coefficients (default: 1/N each). "
                        "Only used with --base.")
    args = p.parse_args()

    if len(args.inputs) < 2 and args.base is None:
        raise SystemExit("uniform soup needs at least 2 inputs")

    ckpts = [_load(p) for p in args.inputs]

    if args.base is not None:
        base_ck = _load(args.base)
        if args.alphas is None:
            alphas = [1.0 / len(ckpts)] * len(ckpts)
        elif len(args.alphas) != len(ckpts):
            raise SystemExit(
                f"--alphas has length {len(args.alphas)}, need {len(ckpts)}")
        else:
            alphas = list(args.alphas)
        merged = task_arithmetic(base_ck, ckpts, alphas)
        cfg_source = base_ck
        meta = {"mode": "task_arithmetic",
                "base": args.base,
                "alphas": alphas}
    else:
        weights = _norm_weights(args.weights, len(ckpts))
        merged = uniform_soup(ckpts, weights)
        cfg_source = ckpts[0]
        meta = {"mode": "uniform_soup",
                "weights": weights}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "agent": merged,
        "config": cfg_source.get("config"),
        "souped_from": args.inputs,
        "merge": meta,
    }, out_path)
    print(f"wrote {out_path}  ({meta['mode']}, {len(ckpts)} inputs)")


if __name__ == "__main__":
    main()
