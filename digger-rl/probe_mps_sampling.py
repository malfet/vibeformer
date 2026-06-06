"""Did the forward/backward probe come back clean? Now check sampling.

The per-op forward/backward parity test (probe_mps_vs_cpu.py) showed
MPS and CPU computing bit-identical values for the full PPO forward,
loss, and parameter gradients. So the divergence at training time must
come from one of:

  1. torch.distributions.Categorical(logits=).sample()  -- the action
     drawn during rollout. If MPS's PRNG state isn't synced with CPU
     (or torch.multinomial draws differently on MPS), every rollout
     step gets a different action and the trajectory forks.
  2. torch.randint / torch.randperm used in the PPO update minibatch
     shuffle (we use np.random there, not torch, so probably safe).
  3. Weight init RNG -- but we explicitly copy weights from CPU to MPS
     in the probe and still see divergence in long-running training.

This script focuses on (1): does Categorical.sample() return the same
draws on MPS and CPU when both are seeded identically? If not, *that*
is the explanation: the rollouts forked from step 1.

Run:
    python probe_mps_sampling.py
"""

from __future__ import annotations

import sys

import torch


def main() -> None:
    if not torch.backends.mps.is_available():
        print("MPS not available; cannot run probe.")
        sys.exit(1)

    # Build fixed logits on both devices. Same values, no autograd.
    torch.manual_seed(0)
    logits = torch.tensor(
        [[0.1, 0.2, -0.1, 0.0, 0.5, -0.3]] * 16, dtype=torch.float32)
    logits_c = logits.clone()
    logits_m = logits.clone().to("mps")

    print("== Test A: Categorical.sample() under matched manual_seed ==")
    # Seed *both* devices identically.
    torch.manual_seed(42)
    if hasattr(torch, "mps"):
        torch.mps.manual_seed(42)
    dist_c = torch.distributions.Categorical(logits=logits_c)
    sample_c = dist_c.sample()

    torch.manual_seed(42)
    if hasattr(torch, "mps"):
        torch.mps.manual_seed(42)
    dist_m = torch.distributions.Categorical(logits=logits_m)
    sample_m = dist_m.sample()

    print(f"  CPU sample: {sample_c.tolist()}")
    print(f"  MPS sample: {sample_m.cpu().tolist()}")
    print(f"  match: {torch.equal(sample_c, sample_m.cpu())}")
    print()

    print("== Test B: torch.multinomial under matched manual_seed ==")
    probs = torch.softmax(logits, dim=-1)
    probs_c = probs.clone()
    probs_m = probs.to("mps")

    torch.manual_seed(123)
    if hasattr(torch, "mps"):
        torch.mps.manual_seed(123)
    mn_c = torch.multinomial(probs_c, num_samples=1).squeeze(-1)

    torch.manual_seed(123)
    if hasattr(torch, "mps"):
        torch.mps.manual_seed(123)
    mn_m = torch.multinomial(probs_m, num_samples=1).squeeze(-1)

    print(f"  CPU multinomial: {mn_c.tolist()}")
    print(f"  MPS multinomial: {mn_m.cpu().tolist()}")
    print(f"  match: {torch.equal(mn_c, mn_m.cpu())}")
    print()

    print("== Test C: action-distribution stats over 50k draws ==")
    # Even if individual draws diverge, the empirical distribution should
    # match log_softmax(logits) on both devices. If MPS systematically
    # over-samples some action, that's a real bug.
    K = 50_000
    big_logits = logits[0].repeat(K, 1)  # (K, 6)
    big_logits_c = big_logits.clone()
    big_logits_m = big_logits.to("mps")

    torch.manual_seed(7)
    if hasattr(torch, "mps"):
        torch.mps.manual_seed(7)
    s_c = torch.distributions.Categorical(logits=big_logits_c).sample()
    counts_c = torch.bincount(s_c, minlength=6).float() / K

    torch.manual_seed(7)
    if hasattr(torch, "mps"):
        torch.mps.manual_seed(7)
    s_m = torch.distributions.Categorical(logits=big_logits_m).sample()
    counts_m = torch.bincount(s_m.cpu(), minlength=6).float() / K

    target = torch.softmax(logits[0], dim=-1)
    print(f"  target softmax:  {[f'{x:.3f}' for x in target.tolist()]}")
    print(f"  CPU empirical:   {[f'{x:.3f}' for x in counts_c.tolist()]}")
    print(f"  MPS empirical:   {[f'{x:.3f}' for x in counts_m.tolist()]}")
    print(f"  CPU max-dev from target: {(counts_c - target).abs().max().item():.4f}")
    print(f"  MPS max-dev from target: {(counts_m - target).abs().max().item():.4f}")
    print()

    print("== Test D: torch.randn() seeded on each device ==")
    torch.manual_seed(99)
    if hasattr(torch, "mps"):
        torch.mps.manual_seed(99)
    r_c = torch.randn(8)

    torch.manual_seed(99)
    if hasattr(torch, "mps"):
        torch.mps.manual_seed(99)
    r_m = torch.randn(8, device="mps")

    print(f"  CPU randn(8): {r_c.tolist()}")
    print(f"  MPS randn(8): {r_m.cpu().tolist()}")
    print(f"  max-abs diff: {(r_c - r_m.cpu()).abs().max().item():.4e}")


if __name__ == "__main__":
    main()
