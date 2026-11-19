"""GRPO fine-tuning of a pixel policy, using exact emulator restores.

GRPO (group relative policy optimisation) needs no critic: it plays a
group of G samples from the same starting point and uses each sample's
return relative to the group as its advantage,

    A_i = (R_i - mean(R)) / std(R),

applied to every action of sample i. In language models the group shares
a prompt; here it shares an emulator state. Save/restore is exact
(README Lesson 21) and the monsters' randomness is in the saved state,
so the G continuations differ only by the policy's own sampled actions.
The group mean is the baseline and the comparison is like for like.

Workers (CPU processes, one emulator each) play the current policy and
keep snapshots of what it sees. States from 6 to 16 steps before each of
its deaths go into a "hard" pool, and every 25th state into a general
pool. A group starts from a hard state with probability --hard-prob,
otherwise from a general one, and runs G rollouts of up to H steps:

    R = 0.02 x score gained - death_penalty x died

A confident policy often plays all G samples the same way, which gives
no signal. So with probability --branch-prob a group is a branch group
instead: one rollout per action, each forced for its first --hold
steps, and only that first choice is trained (see build_batch). This
is where FIRE and the rare escapes get compared on equal terms.

The learner (GPU) applies the PPO-clipped GRPO loss with a k3 KL penalty
toward the frozen starting policy, then sends the new weights back.
With --overlap (default) the workers already play the next batch while
the learner trains on this one. That data is one version stale; the
clipped ratio corrects for it, since it is taken against the logp the
worker recorded.

    python train_grpo.py data/checkpoints/px128_pool8_bagA/ppo_digger_final.pt \\
        --run-name px128_grpo1 --iters 200
"""

from __future__ import annotations

import argparse
import collections
import copy
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from digger_env import DiggerEnv, preprocess_uint8, stack_offsets
from tools.play_audit import line_of_fire, ram
from tools.policies import load_pixel_agent
from tools.symbolic_env import SymbolicDiggerEnv, env_step_skipped


def worker(idx: int, ckpt: str, conn, a: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(a.seed * 100 + idx)
    rng = np.random.default_rng(a.seed * 100 + idx)
    agent, cfg = load_pixel_agent(ckpt, torch.device("cpu"))
    offs = stack_offsets(cfg["frame_stack"], cfg.get("frame_offsets", ()))
    env = SymbolicDiggerEnv(max_steps=10**9, clip_reward=False,
                            episodic_life=False, start_states=a.start_states,
                            start_prob=a.start_prob)

    def frame():
        f = preprocess_uint8(env._env._core.get_frame(), cfg["obs_size"],
                             cfg["color"])
        return f.transpose(2, 0, 1) if f.ndim == 3 else f[None]

    @torch.no_grad()
    def sample(stack, force: int | None = None,
               greedy: bool = False) -> tuple[int, float]:
        """A sampled (or greedy) action, or `force`, and its log-prob."""
        x = torch.from_numpy(np.concatenate(
            [stack[-1 - o] for o in offs], 0)).float().div_(255).unsqueeze(0)
        logits = agent.actor(agent.encode(x))[0]
        d = torch.distributions.Categorical(logits=logits)
        act = (torch.tensor(force) if force is not None
               else logits.argmax() if greedy else d.sample())
        return int(act), float(d.log_prob(act))

    maxlen = offs[0] + 1
    stack: collections.deque = collections.deque(maxlen=maxlen)
    hard: collections.deque = collections.deque(maxlen=a.pool_size)
    pool: collections.deque = collections.deque(maxlen=a.pool_size)
    hist: collections.deque = collections.deque(maxlen=16)
    near: collections.deque = collections.deque(maxlen=a.pool_size)
    shoot: collections.deque = collections.deque(maxlen=a.pool_size)
    env.reset()
    fresh, played = True, 0

    def play(n: int) -> None:
        nonlocal fresh, played
        for _ in range(n):
            fr = frame()
            if fresh:
                stack.clear()
                stack.extend([fr] * maxlen)
            else:
                stack.append(fr)
            s = env._last_state
            if s.digger is not None and not s.dying:
                rec = (env.snapshot(), list(stack))
                hist.append(rec)
                played += 1
                if played % 25 == 0:
                    pool.append(rec)
            act, _ = sample(stack)
            if a.labels and s.digger is not None and not s.dying \
                    and s.digger.can_fire and (
                        act == DiggerEnv.FIRE or
                        line_of_fire(s, s.digger.dir)[0] == "aimed"):
                shoot.append(rec)
            _, _, done, info = env_step_skipped(env, act, 4)
            if info.get("death_event", False):
                hard.extend(list(hist)[:-5])        # 6..16 steps back
                near.extend(list(hist)[-12:-1])     # 2..12 steps back
                hist.clear()
            fresh = info.get("dying", False)
            if done:
                env.reset()
                fresh = True
                hist.clear()

    def rollout(snap, start_stack, force: int | None = None,
                greedy: bool = False, hold: int | None = None) -> dict:
        env.restore_snapshot(snap)
        st = collections.deque(start_stack, maxlen=maxlen)
        frames = list(start_stack)
        acts, logps = [], []
        score0 = env._env._last_score
        score, died = score0, False
        duds = misses = 0
        shot = None                      # None, or "did it kill" so far
        track = a.dud_cost or a.miss_cost
        for t in range(a.horizon):
            if t:
                fr = frame()
                st.append(fr)
                frames.append(fr)
            act, lp = sample(st, force if t < (a.hold if hold is None
                                                 else hold) else None,
                             greedy)
            acts.append(act)
            logps.append(lp)
            if track:
                d0, m0 = ram(env)
            _, _, done, info = env_step_skipped(env, act, 4)
            score = int(info.get("score", score))
            if track:
                # same RAM rules as tools/play_audit.py
                d1, m1 = ram(env)
                if act == DiggerEnv.FIRE:
                    if d0[9] == 0 and d1[9] > 0:
                        shot = False
                    elif d0[9] > 0 or d0[23] == 0:
                        duds += 1
                if shot is not None:
                    shot |= bool(((m0[:, 15] != 0) & (m1[:, 15] == 0)
                                  & (m0[:, 10] == 0)).any())
                    if d1[23] != 0:
                        misses += not shot
                        shot = None
            if info.get("dying", False) or info.get("death_event", False):
                died = True
                break
            if done or env._env.level_done():
                break
        V = 0.0
        if a.bootstrap and not died and not done:
            # what the rest of the life is worth, in the same units as R
            # (the critic was fit on 0.02 x score with gamma 0.99)
            st.append(frame())
            with torch.no_grad():
                x = torch.from_numpy(np.concatenate(
                    [st[-1 - o] for o in offs], 0)).float().div_(255)
                V = float(agent.value(x.unsqueeze(0))[0])
        return dict(frames=np.stack(frames), acts=np.array(acts, np.int64),
                    logp=np.array(logps, np.float32),
                    R=a.score_scale * (score - score0)
                    - died * (a.death_penalty + a.death_time_penalty
                              * (a.horizon - len(acts)) / a.horizon)
                    - a.dud_cost * duds - a.miss_cost * misses
                    + a.bootstrap * V,
                    V=V, duds=duds, misses=misses,
                    died=died)

    def label(src: str) -> dict | None:
        """Try every action from a stored state (hold it, then the greedy
        policy) and return a CE label when the best beats the policy's
        own greedy choice by --label-margin. FIRE is only tried when
        loaded (unloaded FIRE just stands still, like NOOP) and pays
        --fire-cost for the reload it starts."""
        q = near if src == "near" else shoot
        snap, st = q[rng.integers(len(q))]
        env.restore_snapshot(snap)
        d, _ = ram(env)
        loaded = d[9] == 0 and d[23] != 0
        g, _ = sample(collections.deque(st, maxlen=maxlen), greedy=True)
        R = np.full(DiggerEnv.NUM_ACTIONS, -np.inf, np.float32)
        for f in range(DiggerEnv.NUM_ACTIONS):
            if f == DiggerEnv.FIRE and not loaded:
                continue
            r = rollout(snap, st, f, greedy=True,
                        hold=1 if f == DiggerEnv.FIRE else a.label_hold)
            R[f] = r["R"] - (a.fire_cost if f == DiggerEnv.FIRE else 0.0)
        best = int(R.argmax())
        if R[best] - R[g] < a.label_margin:
            return None
        return dict(obs=np.concatenate([st[-1 - o] for o in offs], 0),
                    label=best, greedy=g, gap=float(R[best] - R[g]), src=src)

    while True:
        msg = conn.recv()
        if msg is None:
            break
        agent.load_state_dict(msg["weights"])
        groups = []
        while len(groups) < msg["groups"]:
            play(a.explore)
            if not pool:
                continue
            src = "hard" if hard and rng.random() < a.hard_prob else "pool"
            q = hard if src == "hard" else pool
            snap, start_stack = q[rng.integers(len(q))]
            cur = (env.snapshot(), list(stack))
            if rng.random() < a.branch_prob:
                groups.append(dict(src=src, branch=True, rollouts=[
                    rollout(snap, start_stack, f)
                    for f in range(DiggerEnv.NUM_ACTIONS)]))
            else:
                groups.append(dict(src=src, branch=False, rollouts=[
                    rollout(snap, start_stack) for _ in range(a.group)]))
            env.restore_snapshot(cur[0])
            stack.clear()
            stack.extend(cur[1])
        labels, tried = [], 0
        if a.labels:
            play(a.explore * 2)
        while tried < a.labels and (near or shoot):
            cur = (env.snapshot(), list(stack))
            src = ("near" if near and (not shoot or rng.random() < a.near_prob)
                   else "shoot")
            lab = label(src)
            tried += 1
            if lab is not None:
                labels.append(lab)
            env.restore_snapshot(cur[0])
            stack.clear()
            stack.extend(cur[1])
            if tried % 2 == 0:
                play(a.explore)
        conn.send(dict(groups=groups, labels=labels, tried=tried))
    env.close()


def build_batch(groups: list[dict], offs: tuple, maxlen: int):
    """Flatten groups into (frames, per-step frame index, acts, logp, adv,
    branch flag), skipping groups whose returns are all equal (no signal).

    A branch group contributes only its first step, the forced action:
    the later steps' returns mostly reflect that choice, not their own.
    With logp_old = the policy's probability of the forced action, the
    clipped surrogate's gradient at ratio 1 is A * grad log pi(a|s), the
    all-actions policy gradient without the pi(a) weight."""
    frames, idx, acts, logp, adv, branch = [], [], [], [], [], []
    n_frames, used = 0, 0
    for g in groups:
        R = np.array([r["R"] for r in g["rollouts"]], np.float32)
        if R.std() < 1e-6:
            continue
        used += 1
        A = (R - R.mean()) / (R.std() + 1e-6)
        for r, Ai in zip(g["rollouts"], A):
            T = 1 if g["branch"] else len(r["acts"])
            t = np.arange(T)[:, None]
            idx.append(n_frames + maxlen - 1 + t - np.array(offs)[None])
            fr = r["frames"][:maxlen] if g["branch"] else r["frames"]
            frames.append(fr)
            n_frames += len(fr)
            acts.append(r["acts"][:T])
            logp.append(r["logp"][:T])
            adv.append(np.full(T, Ai, np.float32))
            branch.append(np.full(T, g["branch"]))
    if not used:
        return None, 0
    return (np.concatenate(frames), np.concatenate(idx), np.concatenate(acts),
            np.concatenate(logp), np.concatenate(adv),
            np.concatenate(branch)), used


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("policy", help="pixel checkpoint to start from")
    p.add_argument("--run-name", required=True)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--groups", type=int, default=2,
                   help="groups per worker per iteration")
    p.add_argument("--group", type=int, default=8, help="G, rollouts per group")
    p.add_argument("--horizon", type=int, default=32, help="H, agent steps")
    p.add_argument("--explore", type=int, default=32,
                   help="steps of normal play between groups")
    p.add_argument("--hard-prob", type=float, default=0.5)
    p.add_argument("--pool-size", type=int, default=200)
    p.add_argument("--score-scale", type=float, default=0.02)
    p.add_argument("--death-penalty", type=float, default=10.0)
    p.add_argument("--bootstrap", type=float, default=0.0,
                   help="add this x the critic's V(s_end) to a rollout that "
                        "survives its horizon (1.0 = full value)")
    p.add_argument("--dud-cost", type=float, default=0.0,
                   help="per FIRE press while reloading (0.2 = 10 points)")
    p.add_argument("--miss-cost", type=float, default=0.0,
                   help="per shot that ends without a kill (1.0 = 50 points)")
    p.add_argument("--death-time-penalty", type=float, default=0.0,
                   help="extra death penalty scaled by the share of the "
                        "horizon not survived: dying at once costs "
                        "death_penalty + this, dying at the last step "
                        "death_penalty. Gives a signal when every rollout "
                        "of a group dies: the one that lasts longest wins.")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--minibatch", type=int, default=512)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=0.1,
                   help="KL penalty toward the starting policy")
    p.add_argument("--branch-prob", type=float, default=0.5,
                   help="P(a group forces each action in turn for its "
                        "first --hold steps; see build_batch)")
    p.add_argument("--hold", type=int, default=3)
    p.add_argument("--branch-coef", type=float, default=1.0,
                   help="weight of the branch steps' loss")
    p.add_argument("--labels", type=int, default=0,
                   help="states per worker per iteration to label by trying "
                        "every action (near-death and shooting states)")
    p.add_argument("--near-prob", type=float, default=0.6,
                   help="P(label a state 2..12 steps before a death) vs a "
                        "loaded aimed / firing state")
    p.add_argument("--label-hold", type=int, default=2)
    p.add_argument("--label-margin", type=float, default=2.0,
                   help="label only if best R - greedy's R >= this")
    p.add_argument("--fire-cost", type=float, default=1.0)
    p.add_argument("--ce-coef", type=float, default=0.5)
    p.add_argument("--kl-full", type=float, default=0.0,
                   help="weight of the full KL(ref || pi) on GRPO states")
    p.add_argument("--ce-batch", type=int, default=128)
    p.add_argument("--label-buffer", type=int, default=5000)
    p.add_argument("--no-overlap", dest="overlap", action="store_false")
    p.add_argument("--start-states", default="data/save/levels")
    p.add_argument("--start-prob", type=float, default=0.5)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    agent, cfg = load_pixel_agent(a.policy, device)
    agent.train()
    ref = copy.deepcopy(agent).eval().requires_grad_(False)
    offs = tuple(stack_offsets(cfg["frame_stack"], cfg.get("frame_offsets", ())))
    maxlen = offs[0] + 1
    optim = torch.optim.Adam(agent.parameters(), lr=a.lr, eps=1e-5)
    out = Path("data/checkpoints") / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    ccfg = dict(cfg, run_name=a.run_name, grpo=vars(a))

    def save(name: str, it: int) -> None:
        torch.save({"agent": agent.state_dict(), "step": it, "config": ccfg},
                   out / name)

    ctx = mp.get_context("spawn")
    conns, procs = [], []
    for i in range(a.workers):
        c1, c2 = ctx.Pipe()
        pr = ctx.Process(target=worker, args=(i, a.policy, c2, a), daemon=True)
        pr.start()
        conns.append(c1)
        procs.append(pr)

    def request() -> None:
        w = {k: v.detach().cpu() for k, v in agent.state_dict().items()}
        for c in conns:
            c.send(dict(weights=w, groups=a.groups))

    request()
    t0 = time.monotonic()
    steps_total = 0
    lab_obs: collections.deque = collections.deque(maxlen=a.label_buffer)
    lab_y: collections.deque = collections.deque(maxlen=a.label_buffer)
    lab_meta: list[tuple] = []

    def ce_loss():
        if len(lab_y) < 32:
            return None, None
        j = np.random.randint(len(lab_y), size=min(a.ce_batch, len(lab_y)))
        x = torch.from_numpy(np.stack([lab_obs[i] for i in j])).to(device)
        y = torch.from_numpy(np.array([lab_y[i] for i in j])).to(device)
        logits = agent.actor(agent.encode(x.float().div_(255)))
        return (F.cross_entropy(logits, y),
                (logits.argmax(-1) == y).float().mean())
    for it in range(1, a.iters + 1):
        tw = time.monotonic()
        msgs = [c.recv() for c in conns]
        groups = [g for m_ in msgs for g in m_["groups"]]
        new_labels = [l for m_ in msgs for l in m_["labels"]]
        tried = sum(m_["tried"] for m_ in msgs)
        for l in new_labels:
            lab_obs.append(l["obs"])
            lab_y.append(l["label"])
            lab_meta.append((it, l["src"], l["greedy"], l["label"], l["gap"]))
        t_wait = time.monotonic() - tw
        if a.overlap and it < a.iters:
            request()                    # next batch plays while we learn
        tl = time.monotonic()
        rolls = [r for g in groups for r in g["rollouts"]]
        steps = sum(len(r["acts"]) for r in rolls)
        steps_total += steps
        batch, used = build_batch(groups, offs, maxlen)
        stats = collections.defaultdict(list)
        if batch is not None:
            frames, idx, acts, logp_old, adv, branch = batch
            frames_t = torch.from_numpy(frames)
            n = len(acts)
            # Forced actions often have pi ~ 1e-5, where the worker's CPU
            # logp and the learner's differ by more than the clip range;
            # take their old logp from the learner itself.
            bi = np.nonzero(branch)[0]
            if len(bi):
                with torch.no_grad():
                    x = frames_t[torch.from_numpy(idx[bi])]
                    x = x.reshape(len(bi), -1, *x.shape[-2:]).to(device)
                    lp = F.log_softmax(agent.actor(agent.encode(
                        x.float().div_(255))), -1)
                    logp_old[bi] = lp.gather(1, torch.from_numpy(
                        acts[bi])[:, None].to(device))[:, 0].cpu().numpy()
            for _ in range(a.epochs):
                perm = np.random.permutation(n)
                for s in range(0, n, a.minibatch):
                    mb = perm[s:s + a.minibatch]
                    x = frames_t[torch.from_numpy(idx[mb])]
                    x = x.reshape(len(mb), -1, *x.shape[-2:]).to(device)
                    x = x.float().div_(255)
                    act = torch.from_numpy(acts[mb]).to(device)
                    lp = F.log_softmax(agent.actor(agent.encode(x)), -1)
                    lp_a = lp.gather(1, act[:, None])[:, 0]
                    with torch.no_grad():
                        ref_lp = F.log_softmax(ref.actor(ref.encode(x)), -1)
                        ref_a = ref_lp.gather(1, act[:, None])[:, 0]
                    A = torch.from_numpy(adv[mb]).to(device)
                    ratio = (lp_a - torch.from_numpy(logp_old[mb]).to(device)).exp()
                    pg = -torch.min(ratio * A, ratio.clamp(1 - a.clip,
                                                           1 + a.clip) * A)
                    # the few branch steps get their own mean, so the
                    # sampled steps don't drown them out
                    br = torch.from_numpy(branch[mb]).to(device)
                    loss = 0.0
                    if (~br).any():
                        loss = loss + pg[~br].mean()
                    if br.any():
                        loss = loss + a.branch_coef * pg[br].mean()
                    d = ref_a - lp_a
                    kl = d.exp() - d - 1
                    loss = loss + a.beta * kl.mean()
                    if a.kl_full:
                        # whole-distribution KL(ref || pi) on these
                        # ordinary states: what keeps the labels from
                        # rewriting the rest of the policy
                        klf = (ref_lp.exp() * (ref_lp - lp)).sum(-1).mean()
                        loss = loss + a.kl_full * klf
                        stats["klf"].append(float(klf.detach()))
                    ce, acc = ce_loss()
                    if ce is not None:
                        loss = loss + a.ce_coef * ce
                        stats["ce"].append(float(ce.detach()))
                        stats["acc"].append(float(acc))
                    optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    optim.step()
                    stats["kl"].append(float(kl.mean().detach()))
                    stats["clip"].append(float(((ratio - 1).abs()
                                                > a.clip).float().mean()))
                    stats["ent"].append(float(-(lp.exp() * lp).sum(-1)
                                              .mean().detach()))
        elif len(lab_y) >= 32:
            # no GRPO signal this iteration: label loss alone
            for _ in range(a.epochs * 2):
                ce, acc = ce_loss()
                loss = a.ce_coef * ce
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optim.step()
                stats["ce"].append(float(ce.detach()))
                stats["acc"].append(float(acc))
        if not a.overlap and it < a.iters:
            request()
        t_learn = time.monotonic() - tl
        R = np.array([r["R"] for r in rolls])
        samp = [r for g in groups if not g["branch"] for r in g["rollouts"]]
        died = np.mean([r["died"] for r in samp]) if samp else float("nan")
        nfire = sum(l["label"] == DiggerEnv.FIRE for l in new_labels)
        nshoot = sum(l["src"] == "shoot" for l in new_labels)
        bused = sum(1 for g in groups if g["branch"]
                    and np.std([r["R"] for r in g["rollouts"]]) > 1e-6)
        hard = np.mean([g["src"] == "hard" for g in groups])
        m = {k: np.mean(v) for k, v in stats.items()}
        print(f"[{a.run_name}] it {it:4d} steps {steps_total:8d} "
              f"groups {used}/{len(groups)} (branch {bused}) hard {hard:.2f} R {R.mean():+.2f} "
              f"died {died:.3f} duds {np.mean([r.get('duds', 0) for r in rolls]):.2f} "
              f"miss {np.mean([r.get('misses', 0) for r in rolls]):.2f} kl {m.get('kl', 0):.4f} "
              f"clip {m.get('clip', 0):.3f} ent {m.get('ent', 0):.3f} "
              f"labels {len(new_labels)}/{tried} (shoot {nshoot}, FIRE {nfire}) "
              f"klf {m.get('klf', 0):.3f} buf {len(lab_y)} ce {m.get('ce', 0):.3f} acc {m.get('acc', 0):.2f} "
              f"wait {t_wait:.1f}s learn {t_learn:.1f}s "
              f"({time.monotonic() - t0:.0f}s)", flush=True)
        if it % a.save_every == 0:
            save(f"grpo_it{it:04d}.pt", it)
    save("ppo_digger_final.pt", a.iters)
    if lab_meta:
        np.save(out / "labels_meta.npy", np.array(lab_meta, dtype=object))
    for c in conns:
        c.send(None)
    for pr in procs:
        pr.join(timeout=30)


if __name__ == "__main__":
    main()
