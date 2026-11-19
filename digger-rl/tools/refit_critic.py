"""Refit a pixel policy's critic head to its own play.

GRPO trains only the actor, but the actor and critic share the encoder,
so after GRPO the critic head reads features it was never fit to. This
plays the policy (sampled) in parallel workers, records the frozen
256-d features of every live state and the discounted return that
followed it within that life,

    G_t = sum_k gamma^k x 0.02 x score gained,   0 at the death onset,

and refits only the linear critic head by ridge regression. The tail
of each worker's last life (no known end) is dropped. Reports held-out
R^2 for the old head and the new one and saves a copy of the checkpoint
with the new head.

    python -m tools.refit_critic checkpoints/digger_px128_pool8_grpoLt2.pt \\
        --out data/checkpoints/px128_grpoLt2_vfit/ppo_digger_final.pt
"""

from __future__ import annotations

import argparse
import collections
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from digger_env import preprocess_uint8, stack_offsets
from tools.policies import load_pixel_agent
from tools.symbolic_env import SymbolicDiggerEnv, env_step_skipped


def worker(idx: int, a: argparse.Namespace, q) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(a.seed * 100 + idx)
    agent, cfg = load_pixel_agent(a.policy, torch.device("cpu"))
    agent.eval()
    offs = stack_offsets(cfg["frame_stack"], cfg.get("frame_offsets", ()))
    maxlen = offs[0] + 1
    env = SymbolicDiggerEnv(max_steps=10**9, clip_reward=False,
                            episodic_life=False, start_states=a.start_states,
                            start_prob=a.start_prob)
    env.reset()
    st = collections.deque(maxlen=maxlen)
    fresh = True
    feats, rews, ends = [], [], []        # per live step
    score = env._env._last_score
    for _ in range(a.steps):
        f = preprocess_uint8(env._env._core.get_frame(), cfg["obs_size"],
                             cfg["color"]).transpose(2, 0, 1)
        if fresh:
            st.clear()
            st.extend([f] * maxlen)
        else:
            st.append(f)
        with torch.no_grad():
            x = torch.from_numpy(np.concatenate(
                [st[-1 - o] for o in offs], 0)).float().div_(255)[None]
            z = agent.encode(x)
            act = int(torch.distributions.Categorical(
                logits=agent.actor(z)[0]).sample())
        s = env._last_state
        live = s.digger is not None and not s.dying
        _, _, done, info = env_step_skipped(env, act, 4)
        new = int(info.get("score", score))
        if live:
            feats.append(z[0].numpy())
            rews.append(0.02 * (new - score))
            ends.append(bool(info.get("death_event", False) or done))
        score = new
        fresh = info.get("dying", False)
        if done:
            env.reset()
            fresh = True
            score = env._env._last_score
    env.close()
    # discounted returns within each life; drop the unfinished last life
    G = np.zeros(len(rews), np.float32)
    g, known = 0.0, False
    keep = np.zeros(len(rews), bool)
    for t in range(len(rews) - 1, -1, -1):
        if ends[t]:
            g, known = 0.0, True
        g = rews[t] + a.gamma * g
        G[t], keep[t] = g, known
    q.put((np.array(feats)[keep], G[keep]))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("policy")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--steps", type=int, default=4000, help="per worker")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--start-states", default="data/save/levels")
    p.add_argument("--start-prob", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(i, a, q), daemon=True)
             for i in range(a.workers)]
    for pr in procs:
        pr.start()
    parts = [q.get() for _ in procs]
    for pr in procs:
        pr.join(timeout=30)
    # hold out whole workers so test states are not neighbours of train ones
    test = parts[:max(1, a.workers // 5)]
    train = parts[len(test):]
    Xtr = np.concatenate([x for x, _ in train]).astype(np.float64)
    ytr = np.concatenate([y for _, y in train]).astype(np.float64)
    Xte = np.concatenate([x for x, _ in test]).astype(np.float64)
    yte = np.concatenate([y for _, y in test]).astype(np.float64)
    print(f"states: train {len(ytr)}, test {len(yte)}; "
          f"return mean {ytr.mean():.2f} sd {ytr.std():.2f}")

    agent, cfg = load_pixel_agent(a.policy, torch.device("cpu"))
    W = agent.critic.weight.detach().double().numpy()[0]
    b = float(agent.critic.bias.detach())

    def r2(pred, y):
        return 1 - ((pred - y) ** 2).mean() / y.var()

    print(f"old head: test R^2 {r2(Xte @ W + b, yte):.3f}, "
          f"mean prediction {(Xte @ W + b).mean():.2f} vs {yte.mean():.2f}")
    A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    reg = a.ridge * np.eye(A.shape[1])
    reg[-1, -1] = 0.0
    sol = np.linalg.solve(A.T @ A + reg, A.T @ ytr)
    Wn, bn = sol[:-1], sol[-1]
    print(f"new head: train R^2 {r2(Xtr @ Wn + bn, ytr):.3f}, "
          f"test R^2 {r2(Xte @ Wn + bn, yte):.3f}")

    ck = torch.load(a.policy, map_location="cpu", weights_only=False)
    sd = ck["agent"]
    sd["critic.weight"] = torch.tensor(Wn, dtype=sd["critic.weight"].dtype)[None]
    sd["critic.bias"] = torch.tensor([bn], dtype=sd["critic.bias"].dtype)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
