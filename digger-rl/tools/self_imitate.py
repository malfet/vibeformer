"""Filtered self-imitation: fine-tune a pixel policy on its own deathless games.

The traces come from tools/rewind_traces.py: every move was sampled by
the policy itself, and the timelines that led to deaths were cut away.
On such data the expected cross-entropy gradient equals that of
KL(policy_old || policy) plus the selection effect of keeping only the
surviving timelines, so no separate anchor is needed: a small learning
rate and a couple of epochs move the policy toward what survived.

Only the encoder and actor are trained; the critic head is kept as is.

    python -m tools.self_imitate checkpoints/digger_px128_pool8_grpoV2.pt \\
        --traces data/traces/rewind_V2_{0,1}.npz \\
        --out data/checkpoints/px128_selfimit1/ppo_digger_final.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from digger_env import stack_offsets
from tools.policies import load_pixel_agent


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("policy")
    p.add_argument("--traces", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device(a.device)
    agent, cfg = load_pixel_agent(a.policy, dev)
    ref, _ = load_pixel_agent(a.policy, dev)
    offs = np.array(stack_offsets(cfg["frame_stack"],
                                  cfg.get("frame_offsets", ())))
    frames, acts, idx = [], [], []
    base = 0
    for path in a.traces:
        z = np.load(path)
        f = z["frames"]                       # N, H, W, C uint8
        r = z["resets"].astype(bool)
        start = np.maximum.accumulate(np.where(r, np.arange(len(r)), 0))
        # a stack may not reach back past the start of its game/life
        t = np.arange(len(r))
        ok = t - offs.max() >= start
        frames.append(f)
        acts.append(z["actions"].astype(np.int64))
        idx.append(base + t[ok])
        base += len(f)
        print(f"{path}: {len(f)} steps, {ok.sum()} usable", flush=True)
    frames = torch.from_numpy(np.concatenate(frames))          # N,H,W,C
    acts = torch.from_numpy(np.concatenate(acts))
    idx = np.concatenate(idx)
    offs_t = torch.from_numpy(offs)

    def batch(j):
        j = torch.from_numpy(j)
        x = frames[j[:, None] - offs_t[None]]                  # B,S,H,W,C
        x = x.permute(0, 1, 4, 2, 3).reshape(len(j), -1, *x.shape[2:4])
        return x.to(dev).float().div_(255), acts[j].to(dev)

    params = [q for n, q in agent.named_parameters()
              if not n.startswith("critic")]
    opt = torch.optim.Adam(params, lr=a.lr)
    agent.train()
    for ep in range(a.epochs):
        perm = rng.permutation(idx)
        ce_s, kl_s, acc_s, n = 0.0, 0.0, 0.0, 0
        for s in range(0, len(perm), a.batch):
            x, y = batch(perm[s:s + a.batch])
            lp = F.log_softmax(agent.actor(agent.encode(x)), -1)
            loss = F.nll_loss(lp, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()
            with torch.no_grad():
                rl = F.log_softmax(ref.actor(ref.encode(x)), -1)
                kl = (rl.exp() * (rl - lp)).sum(-1).mean()
            ce_s += float(loss); kl_s += float(kl)
            acc_s += float((lp.argmax(-1) == y).float().mean()); n += 1
        print(f"epoch {ep + 1}: ce {ce_s / n:.4f} acc {acc_s / n:.3f} "
              f"KL(ref||pi) {kl_s / n:.4f}", flush=True)
    ck = torch.load(a.policy, map_location="cpu", weights_only=False)
    ck["agent"] = {k: v.cpu() for k, v in agent.state_dict().items()}
    ck["config"] = dict(ck["config"], self_imitate=dict(
        policy=a.policy, traces=a.traces, epochs=a.epochs, lr=a.lr))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
