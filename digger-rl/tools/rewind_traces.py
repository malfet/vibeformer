"""Deathless traces by rewinding: the policy's own play with its deaths undone.

The sampled pixel policy plays fresh games. When the digger dies, the
emulator rewinds d steps (16, 32, 64, 128, 256; --tries re-samples per
depth, deeper after each failed batch) and the policy plays on from
there with fresh samples. The timeline after the rewind point is
discarded, so what is left is one continuous game in which every move
was sampled by the policy itself and no life was lost. A death that no
rewind within 256 steps avoids is kept and the game goes on.

Only games that reach --to-level with no kept death are written, as a
train_pixel.py trace (frames, actions, scores, lives, resets), for
filtered self-imitation: no teacher, no scenarios, no hand-made labels.

    python -m tools.rewind_traces checkpoints/digger_px128_pool8_grpoV2.pt \\
        --games 50 --out data/traces/rewind_V2_0.npz
"""

from __future__ import annotations

import argparse
import collections
import multiprocessing as mp

import numpy as np
import torch

from digger_env import preprocess_uint8, stack_offsets
from tools.policies import load_pixel_agent
from tools.symbolic_env import SymbolicDiggerEnv, env_step_skipped

DEPTHS = (16, 32, 64, 128, 256)
KEEP = 8                       # snapshot every KEEP steps on the timeline


def play_game(env, agent, cfg, offs, a) -> dict:
    maxlen = offs[0] + 1

    def frame():
        return preprocess_uint8(env._env._core.get_frame(), cfg["obs_size"],
                                cfg["color"])          # HWC uint8

    def act(st):
        with torch.no_grad():
            x = torch.from_numpy(np.concatenate(
                [st[-1 - o].transpose(2, 0, 1) for o in offs], 0)
            ).float().div_(255)[None]
            return int(torch.distributions.Categorical(
                logits=agent.actor(agent.encode(x))[0]).sample())

    env.reset()
    f0 = frame()
    st = collections.deque([f0] * maxlen, maxlen=maxlen)
    frames, acts, scores, lives, resets = [], [], [], [], []
    snaps: list = []           # (timeline index, snapshot, stack)
    pending = None             # [death index, depth idx, tries left]
    kept, rewinds, steps = 0, 0, 0
    fresh = True
    while True:
        s = env._last_state
        if s.level >= a.to_level or steps > a.max_steps:
            break
        n = len(acts)
        if n % KEEP == 0 and s.digger is not None and not s.dying \
                and (not snaps or snaps[-1][0] != n):
            snaps.append((n, env.snapshot(), list(st)))
        x = act(st)
        frames.append(st[-1]); acts.append(x)
        scores.append(env._env._last_score); lives.append(s.lives)
        resets.append(fresh)
        fresh = False
        _, _, done, info = env_step_skipped(env, x, 4)
        steps += 1
        st.append(frame())
        if pending and len(acts) >= pending[0] + 32:
            pending = None                        # that death is avoided
        if not (info.get("dying") or info.get("death_event") or done):
            continue
        if pending is None:
            pending = [len(acts), 0, a.tries]
        else:
            pending[2] -= 1
            if pending[2] == 0:
                pending[1] += 1
                pending[2] = a.tries
        if pending[1] >= len(DEPTHS) or not snaps:
            # not avoidable: keep the death and play on
            kept += 1
            pending = None
            while not done and (env._last_state.dying
                                or env._last_state.digger is None):
                _, _, done, info = env_step_skipped(env, 0, 4)
            if done:
                break
            st = collections.deque([frame()] * maxlen, maxlen=maxlen)
            fresh = True
            snaps.clear()
            continue
        target = pending[0] - DEPTHS[pending[1]]
        cand = [z for z in snaps if z[0] <= target] or [snaps[0]]
        n0, sn, sk = cand[-1]
        env.restore_snapshot(sn)
        st = collections.deque(sk, maxlen=maxlen)
        del frames[n0:], acts[n0:], scores[n0:], lives[n0:], resets[n0:]
        while snaps and snaps[-1][0] > n0:
            snaps.pop()
        rewinds += 1
    return dict(frames=frames, acts=acts, scores=scores, lives=lives,
                resets=resets, kept=kept, rewinds=rewinds,
                level=env._last_state.level,
                score=env._env._last_score)


def worker(idx: int, a, q) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(a.seed * 1000 + idx)
    agent, cfg = load_pixel_agent(a.policy, torch.device("cpu"))
    agent.eval()
    offs = stack_offsets(cfg["frame_stack"], cfg.get("frame_offsets", ()))
    env = SymbolicDiggerEnv(max_steps=10**9, clip_reward=False,
                            episodic_life=False)
    for _ in range(a.games // a.workers):
        g = play_game(env, agent, cfg, offs, a)
        ok = g["kept"] == 0 and g["level"] >= a.to_level
        print(f"[w{idx}] level {g['level']} score {g['score']} steps "
              f"{len(g['acts'])} rewinds {g['rewinds']} kept deaths "
              f"{g['kept']} -> {'use' if ok else 'drop'}", flush=True)
        q.put(g if ok else None)
    q.put("done")
    env.close()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("policy")
    p.add_argument("--out", required=True)
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--tries", type=int, default=4)
    p.add_argument("--to-level", type=int, default=3,
                   help="stop each game when this level starts")
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(i, a, q), daemon=True)
             for i in range(a.workers)]
    for pr in procs:
        pr.start()
    games, live = [], a.workers
    while live:
        g = q.get()
        if g == "done":
            live -= 1
        elif g is not None:
            games.append(g)
    for pr in procs:
        pr.join(timeout=30)
    if not games:
        print("no usable games")
        return
    _, cfg = load_pixel_agent(a.policy, torch.device("cpu"))
    cat = lambda k, dt: np.concatenate([np.asarray(g[k], dt) for g in games])
    np.savez(a.out, frames=np.stack([f for g in games for f in g["frames"]]),
             actions=cat("acts", np.int64),
             raw_rewards=np.concatenate([np.diff(np.asarray(g["scores"] + [g["score"]], np.float32)) for g in games]),
             scores=cat("scores", np.int32), lives=cat("lives", np.int32),
             resets=cat("resets", bool), frame_skip=np.int32(4),
             obs_size=np.int32(cfg["obs_size"]), color=np.bool_(cfg["color"]),
             teacher=np.array(f"rewind:{a.policy}"))
    print(f"wrote {a.out}: {len(games)} games, "
          f"{sum(len(g['acts']) for g in games)} steps, "
          f"mean score {np.mean([g['score'] for g in games]):.0f}")


if __name__ == "__main__":
    main()
