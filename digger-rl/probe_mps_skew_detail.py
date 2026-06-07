"""Rigorous MPS-vs-CPU sampling-bias measurement.

The first probe (probe_mps_sampling.py) measured 50k draws from one
6-class distribution and reported a max bin-deviation 0.42% on MPS vs
0.17% on CPU. That single-shot number doesn't separate:

  - Real bias (systematic over/under-sampling of specific actions)
  - Variance (just small-sample noise that scales like 1/sqrt(K))

This script does three things:

  1. Per-bin deviation across many independent trials, so we can
     report mean +/- std of the deviation for each action. If MPS's
     mean deviation per bin is centered on zero, it's noise. If it's
     biased away from zero, it's a real skew.

  2. Chi-square test against the target multinomial. A failing
     chi-square confirms statistically that MPS's draws are
     distinguishable from the target distribution.

  3. Sweep over distribution shapes -- nearly-uniform, mildly
     peaked, sharply peaked, deterministic-but-one. If the bias is
     a function of distribution entropy, that points to a specific
     algorithmic issue in the sampler.

Run:
    python probe_mps_skew_detail.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch


def empirical(logits: torch.Tensor, K: int, device: str,
              seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    if device == "mps":
        torch.mps.manual_seed(seed)
    big = logits.repeat(K, 1).to(device)
    s = torch.distributions.Categorical(logits=big).sample()
    counts = torch.bincount(s.cpu(), minlength=logits.numel()).float() / K
    return counts


def chi_square(observed: torch.Tensor, expected: torch.Tensor,
               K: int) -> float:
    """Pearson chi-square statistic for proportions vs expected (both
    normalised). Returns the raw statistic (not p-value).
    """
    obs = observed * K
    exp = expected * K
    return float(((obs - exp) ** 2 / exp).sum().item())


def sweep_one(name: str, logits: torch.Tensor, K: int, trials: int) -> None:
    target = torch.softmax(logits, dim=-1)
    n_classes = logits.numel()
    print(f"\n=== {name}: K={K:,} per trial, {trials} trials ===")
    print(f"  target: {[f'{x:.4f}' for x in target.tolist()]}")
    print(f"  entropy(target) = {(-target * target.log()).sum().item():.4f} "
          f"(max ln({n_classes}) = {np.log(n_classes):.4f})")
    cpu_devs = []
    mps_devs = []
    cpu_chi = []
    mps_chi = []
    for t in range(trials):
        seed = 1000 + t
        ecpu = empirical(logits, K, "cpu", seed)
        emps = empirical(logits, K, "mps", seed)
        cpu_devs.append((ecpu - target).numpy())
        mps_devs.append((emps - target).numpy())
        cpu_chi.append(chi_square(ecpu, target, K))
        mps_chi.append(chi_square(emps, target, K))

    cpu_devs = np.stack(cpu_devs, axis=0)   # (trials, n_classes)
    mps_devs = np.stack(mps_devs, axis=0)
    print(f"  per-bin mean dev (averaged over {trials} trials):")
    print(f"    CPU: {[f'{x:+.5f}' for x in cpu_devs.mean(0)]}")
    print(f"    MPS: {[f'{x:+.5f}' for x in mps_devs.mean(0)]}")
    print(f"  per-bin std dev (across {trials} trials):")
    print(f"    CPU: {[f'{x:.5f}' for x in cpu_devs.std(0)]}")
    print(f"    MPS: {[f'{x:.5f}' for x in mps_devs.std(0)]}")
    # Approx theoretical std for an unbiased multinomial bin:
    # std(p_hat) = sqrt(p * (1 - p) / K)
    th_std = np.sqrt(target.numpy() * (1 - target.numpy()) / K)
    print(f"    theoretical (unbiased): {[f'{x:.5f}' for x in th_std.tolist()]}")
    # Chi-square: critical value at p=0.001 with df=n-1 is roughly:
    # df=1 -> 10.83, df=2 -> 13.82, df=5 -> 20.52, df=10 -> 29.59
    print(f"  chi-square per trial (target=df-{n_classes-1}, "
          f"~p=0.001 critical -> {[10.83, 13.82, 16.27, 18.47, 20.52][min(n_classes-2, 4)]:.2f}):")
    print(f"    CPU mean {np.mean(cpu_chi):6.2f}  max {np.max(cpu_chi):6.2f}")
    print(f"    MPS mean {np.mean(mps_chi):6.2f}  max {np.max(mps_chi):6.2f}")


def main() -> None:
    if not torch.backends.mps.is_available():
        print("MPS not available on this machine; cannot run probe.")
        sys.exit(1)

    trials = 50
    K = 20_000

    # Sweep distribution shapes by entropy.
    sweeps: list[tuple[str, torch.Tensor]] = [
        ("uniform 6-class",
         torch.zeros(6)),
        ("mild softmax (range 0..1)",
         torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])),
        ("medium softmax (range 0..3)",
         torch.tensor([0.0, 0.6, 1.2, 1.8, 2.4, 3.0])),
        ("sharp softmax (range 0..6)",
         torch.tensor([0.0, 1.2, 2.4, 3.6, 4.8, 6.0])),
        ("one-hot-ish (logit 5 dominant)",
         torch.tensor([-2.0, -2.0, -2.0, -2.0, -2.0, 4.0])),
        ("12-class uniform",
         torch.zeros(12)),
        ("PPO-realistic 6-class",
         torch.tensor([0.1, 0.2, -0.1, 0.0, 0.5, -0.3])),
    ]

    for name, logits in sweeps:
        sweep_one(name, logits, K, trials)


if __name__ == "__main__":
    main()
