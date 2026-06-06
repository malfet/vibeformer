"""Localise the op that's causing MPS-vs-CPU divergence in our PPO loop.

After identical-seed 20k-step PPO runs showed CPU learning ~3x faster
than MPS (and the clip_frac frozen at 0.0 on MPS), we want to know
*which step in the forward/backward chain* is the source of the
discrepancy. Possible culprits:

  - Conv2d on small inputs (MPS has had numerical issues on
    (B, 24, 10, 15) shape that don't match CUDA-tuned kernels)
  - F.log_softmax / logsumexp (known historical drift)
  - torch.distributions.Categorical.log_prob (uses gather + log_softmax;
    has had silent-CPU-fallback bugs)
  - Adam optimizer step (less likely but plausible)
  - The .exp() in ratio = exp(new_logp - old_logp)

Method: build the same SymbolicAgent on CPU and MPS, copy weights so
both initial states are bit-identical, push a fixed batch through each
stage, and compute max-abs and max-relative error at each stage. Same
shapes & dtypes as the actual PPO loop.

Run:
    python probe_mps_vs_cpu.py
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from tools.game_state import MHEIGHT, MWIDTH
from tools.symbolic_env import BASE_OBS_CHANNELS
from train_ppo_symbolic import SymbolicAgent


def err(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Return (max_abs_diff, max_relative_diff) for two same-shape tensors."""
    d = (a.cpu().float() - b.cpu().float()).abs()
    scale = a.cpu().float().abs().max().item() + 1e-12
    return float(d.max().item()), float(d.max().item() / scale)


def main() -> None:
    if not torch.backends.mps.is_available():
        print("MPS not available on this machine; cannot run probe.")
        sys.exit(1)

    torch.manual_seed(0)
    in_ch = BASE_OBS_CHANNELS * 4
    N = 64
    A = 6

    # Build CPU agent first; copy weights to MPS so both starts are identical.
    agent_cpu = SymbolicAgent(in_channels=in_ch, num_actions=A).to("cpu")
    agent_mps = SymbolicAgent(in_channels=in_ch, num_actions=A).to("mps")
    agent_mps.load_state_dict(
        {k: v.to("mps") for k, v in agent_cpu.state_dict().items()})

    obs = torch.rand(N, in_ch, MHEIGHT, MWIDTH)
    acts = torch.randint(0, A, (N,))
    old_logp = torch.randn(N) * 0.1
    adv = torch.randn(N)

    obs_c, acts_c, old_c, adv_c = obs, acts, old_logp, adv
    obs_m = obs.to("mps")
    acts_m = acts.to("mps")
    old_m = old_logp.to("mps")
    adv_m = adv.to("mps")

    print(f"Batch shape: ({N}, {in_ch}, {MHEIGHT}, {MWIDTH})  actions={A}")
    print()
    print(f"{'stage':<32s} {'max-abs':>12s} {'max-rel':>10s}")
    print("-" * 56)

    # ---- 1. Encoder body ----
    z_c = agent_cpu.encode(obs_c)
    z_m = agent_mps.encode(obs_m)
    a, r = err(z_c, z_m)
    print(f"{'encoder body output':<32s} {a:12.2e} {r:9.2%}")

    # ---- 2. Conv layers individually ----
    # Walk the body Sequential and check each step.
    x_c, x_m = obs_c, obs_m
    for i, (lc, lm) in enumerate(zip(agent_cpu.body, agent_mps.body)):
        x_c = lc(x_c)
        x_m = lm(x_m)
        a, r = err(x_c, x_m)
        name = type(lc).__name__
        print(f"  body[{i}] {name:<22s} {a:12.2e} {r:9.2%}")

    # ---- 3. Actor logits ----
    logits_c = agent_cpu.actor(z_c)
    logits_m = agent_mps.actor(z_m)
    a, r = err(logits_c, logits_m)
    print(f"{'actor logits':<32s} {a:12.2e} {r:9.2%}")

    # ---- 4. F.log_softmax(logits) ----
    ls_c = F.log_softmax(logits_c, dim=-1)
    ls_m = F.log_softmax(logits_m, dim=-1)
    a, r = err(ls_c, ls_m)
    print(f"{'F.log_softmax(logits)':<32s} {a:12.2e} {r:9.2%}")

    # ---- 5. Categorical(logits).log_prob(acts) ----
    dist_c = torch.distributions.Categorical(logits=logits_c)
    dist_m = torch.distributions.Categorical(logits=logits_m)
    logp_c = dist_c.log_prob(acts_c)
    logp_m = dist_m.log_prob(acts_m)
    a, r = err(logp_c, logp_m)
    print(f"{'Categorical.log_prob':<32s} {a:12.2e} {r:9.2%}")

    # ---- 6. Categorical(logits).entropy() ----
    ent_c = dist_c.entropy()
    ent_m = dist_m.entropy()
    a, r = err(ent_c, ent_m)
    print(f"{'Categorical.entropy':<32s} {a:12.2e} {r:9.2%}")

    # ---- 7. ratio = exp(new_logp - old_logp) ----
    ratio_c = (logp_c - old_c).exp()
    ratio_m = (logp_m - old_m).exp()
    a, r = err(ratio_c, ratio_m)
    print(f"{'ratio = exp(d_logp)':<32s} {a:12.2e} {r:9.2%}")

    # ---- 8. Full PPO PG loss (clipped) ----
    clip = 0.1
    def pg(adv_t, ratio_t):
        return torch.max(-adv_t * ratio_t,
                          -adv_t * torch.clamp(ratio_t, 1 - clip, 1 + clip)).mean()
    pg_c = pg(adv_c, ratio_c)
    pg_m = pg(adv_m, ratio_m)
    a_, r_ = err(pg_c.unsqueeze(0), pg_m.unsqueeze(0))
    print(f"{'PG loss (clipped)':<32s} {a_:12.2e} {r_:9.2%}")
    print(f"  pg_cpu = {pg_c.item():+.6e}   pg_mps = {pg_m.item():+.6e}")

    # ---- 9. Backward: actor weight grad ----
    agent_cpu.zero_grad()
    agent_mps.zero_grad()
    # Re-do the forward to get a fresh autograd graph (since we already
    # consumed the previous graph by reading values out of it).
    z_c2 = agent_cpu.encode(obs_c)
    logits_c2 = agent_cpu.actor(z_c2)
    dist_c2 = torch.distributions.Categorical(logits=logits_c2)
    logp_c2 = dist_c2.log_prob(acts_c)
    ratio_c2 = (logp_c2 - old_c).exp()
    pg_c2 = pg(adv_c, ratio_c2)
    pg_c2.backward()

    z_m2 = agent_mps.encode(obs_m)
    logits_m2 = agent_mps.actor(z_m2)
    dist_m2 = torch.distributions.Categorical(logits=logits_m2)
    logp_m2 = dist_m2.log_prob(acts_m)
    ratio_m2 = (logp_m2 - old_m).exp()
    pg_m2 = pg(adv_m, ratio_m2)
    pg_m2.backward()

    print()
    print(f"{'parameter grad':<32s} {'max-abs':>12s} {'max-rel':>10s}")
    print("-" * 56)
    for (n_c, p_c), (n_m, p_m) in zip(agent_cpu.named_parameters(),
                                         agent_mps.named_parameters()):
        if p_c.grad is None:
            continue
        a, r = err(p_c.grad, p_m.grad)
        print(f"{n_c:<32s} {a:12.2e} {r:9.2%}")


if __name__ == "__main__":
    main()
