"""Search labels for near-death states, with a consistency check.

Plays a fixed pixel policy (sampled) in parallel workers. At each death it
takes states 2..12 steps back and tries every action there: hold it for
--hold steps, then let the policy continue (sampled) for the rest of
--horizon, --reps times per action. FIRE is only tried when loaded.

    value(a) = mean over reps of [0.02 x score gained - 10 x died]

A state is labelled with argmax value when the best action beats the
policy's greedy action by --margin. To see how much of that is noise the
reps are split in two halves and each half picks its own best action:
`agree` is how often both halves pick the same one, and `greedy_det`
records the label a single deterministic greedy continuation gives
(what train_grpo.py --labels used).

    python -m tools.death_labels checkpoints/digger_px128_pool8_grpo1.pt \\
        --states 400 --out data/traces/death_labels_grpo1.npz
"""

from __future__ import annotations

import argparse
import collections
import multiprocessing as mp
import time

import numpy as np
import torch

from digger_env import DiggerEnv, preprocess_uint8, stack_offsets
from tools.play_audit import ram
from tools.policies import load_pixel_agent
from tools.symbolic_env import SymbolicDiggerEnv, env_step_skipped


def worker(idx: int, a: argparse.Namespace, n_states: int, q) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(a.seed * 100 + idx)
    rng = np.random.default_rng(a.seed * 100 + idx)
    agent, cfg = load_pixel_agent(a.policy, torch.device("cpu"))
    agent.eval()
    offs = stack_offsets(cfg["frame_stack"], cfg.get("frame_offsets", ()))
    maxlen = offs[0] + 1
    env = SymbolicDiggerEnv(max_steps=10**9, clip_reward=False,
                            episodic_life=False, start_states=a.start_states,
                            start_prob=a.start_prob)

    def frame():
        f = preprocess_uint8(env._env._core.get_frame(), cfg["obs_size"],
                             cfg["color"])
        return f.transpose(2, 0, 1) if f.ndim == 3 else f[None]

    @torch.no_grad()
    def logits(st):
        x = torch.from_numpy(np.concatenate(
            [st[-1 - o] for o in offs], 0)).float().div_(255).unsqueeze(0)
        return agent.actor(agent.encode(x))[0]

    def act(st, greedy: bool) -> int:
        lg = logits(st)
        return int(lg.argmax() if greedy
                   else torch.distributions.Categorical(logits=lg).sample())

    def rollout(snap, stack, first: int, hold: int, greedy: bool) -> float:
        env.restore_snapshot(snap)
        st = collections.deque(stack, maxlen=maxlen)
        score0 = env._env._last_score
        score = score0
        for t in range(a.horizon):
            if t:
                st.append(frame())
            ac = first if t < hold else act(st, greedy)
            _, _, done, info = env_step_skipped(env, ac, 4)
            score = int(info.get("score", score))
            if info.get("dying", False) or info.get("death_event", False):
                return 0.02 * (score - score0) - 10.0
            if done or env._env.level_done():
                break
        return 0.02 * (score - score0)

    env.reset()
    st = collections.deque(maxlen=maxlen)
    hist = collections.deque(maxlen=12)
    fresh, done_states = True, 0
    while done_states < n_states:
        fr = frame()
        if fresh:
            st.clear()
            st.extend([fr] * maxlen)
        else:
            st.append(fr)
        s = env._last_state
        if s.digger is not None and not s.dying:
            hist.append((env.snapshot(), list(st)))
        _, _, done, info = env_step_skipped(env, act(st, False), 4)
        fresh = info.get("dying", False)
        if info.get("death_event", False):
            cands = list(hist)[:-1]
            cur = env.snapshot()
            pick = rng.choice(len(cands), size=min(a.per_death, len(cands)),
                              replace=False) if cands else []
            for i in pick:
                snap, stack = cands[i]
                back = len(cands) + 1 - i
                env.restore_snapshot(snap)
                d, _ = ram(env)
                loaded = bool(d[9] == 0 and d[23] != 0)
                g = int(logits(stack).argmax())
                V = np.full((DiggerEnv.NUM_ACTIONS, a.reps), np.nan, np.float32)
                det = np.full(DiggerEnv.NUM_ACTIONS, -np.inf, np.float32)
                for f in range(DiggerEnv.NUM_ACTIONS):
                    if f == DiggerEnv.FIRE and not loaded:
                        continue
                    h = 1 if f == DiggerEnv.FIRE else a.hold
                    for r in range(a.reps):
                        V[f, r] = rollout(snap, stack, f, h, False)
                    det[f] = rollout(snap, stack, f, h, True)
                q.put(dict(obs=np.concatenate([stack[-1 - o] for o in offs], 0),
                           V=V, det=det, greedy=g, loaded=loaded, back=back))
                done_states += 1
            env.restore_snapshot(cur)
            hist.clear()
        if done:
            env.reset()
            fresh = True
            hist.clear()
    q.put(None)
    env.close()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("policy")
    p.add_argument("--states", type=int, default=400)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--per-death", type=int, default=3)
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--hold", type=int, default=2)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--margin", type=float, default=2.0)
    p.add_argument("--start-states", default="data/save/levels")
    p.add_argument("--start-prob", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    per = -(-a.states // a.workers)
    procs = [ctx.Process(target=worker, args=(i, a, per, q), daemon=True)
             for i in range(a.workers)]
    for pr in procs:
        pr.start()
    recs, live, t0 = [], a.workers, time.monotonic()
    while live:
        r = q.get()
        if r is None:
            live -= 1
            continue
        recs.append(r)
        if len(recs) % 50 == 0:
            print(f"  {len(recs)} states ({time.monotonic() - t0:.0f}s)",
                  flush=True)
    for pr in procs:
        pr.join(timeout=30)

    h = a.reps // 2
    lab, agree, det_agree, gap = [], [], [], []
    for r in recs:
        V = r["V"]
        tried = ~np.isnan(V[:, 0])
        m = np.where(tried, np.nanmean(V, 1), -np.inf)
        m1 = np.where(tried, np.nanmean(V[:, :h], 1), -np.inf)
        m2 = np.where(tried, np.nanmean(V[:, h:], 1), -np.inf)
        best = int(m.argmax())
        gp = m[best] - m[r["greedy"]]
        r["best"], r["gap"] = best, float(gp)
        if gp >= a.margin:
            lab.append(r)
            agree.append(m1.argmax() == m2.argmax())
            det_agree.append(int(r["det"].argmax()) == best)
            gap.append(gp)
    n = len(recs)
    surv = np.array([np.nanmean(r["V"] > -5, 1) for r in recs])
    print(f"\n{n} near-death states; labelled (best beats greedy by "
          f">= {a.margin}): {len(lab)}")
    if lab:
        print(f"  halves pick the same best action: {np.mean(agree):.2f}")
        print(f"  deterministic greedy rollout picks it too: "
              f"{np.mean(det_agree):.2f}")
        print(f"  labels: {sorted(collections.Counter(r['best'] for r in lab).items())}")
        print(f"  mean gap {np.mean(gap):.2f}")
    g_surv = np.array([surv[i, r["greedy"]] for i, r in enumerate(recs)])
    b_surv = np.nanmax(surv, 1)
    print(f"  survival from these states: greedy action {g_surv.mean():.2f}, "
          f"best action {b_surv.mean():.2f}")
    if a.out:
        np.savez_compressed(
            a.out, obs=np.stack([r["obs"] for r in recs]),
            V=np.stack([r["V"] for r in recs]),
            det=np.stack([r["det"] for r in recs]),
            greedy=np.array([r["greedy"] for r in recs]),
            best=np.array([r["best"] for r in recs]),
            gap=np.array([r["gap"] for r in recs], np.float32),
            loaded=np.array([r["loaded"] for r in recs]),
            back=np.array([r["back"] for r in recs]))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
