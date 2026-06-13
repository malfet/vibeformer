"""Behavioural-cloning trainer for snake — pixel obs, BFS teacher labels.

First iteration of the digger-rl recipe, ported to Nibbles. The lesson from
digger-rl Sessions: pure pixel PPO from scratch is hopeless; BC bootstrap
is the lever. So we start there.

Workflow:
    1. Warmup: drive the env with the BFS teacher for `--warmup-steps`,
       record (stacked-pixel-obs, teacher-action) pairs, train CE for
       `--warmup-epochs`.
    2. Optional DAGGER iterations: student drives, teacher labels every
       visited state, pairs appended to the aggregate set, CE re-trained.
       (Fixes pure BC's covariate shift: the student visits states the
       teacher never would.)
    3. Final stochastic eval over `--eval-eps` episodes.

The snake teacher works directly on `NibblesGame` state (no perception
step), so live labeling is cheaper here than in digger-rl. We can label
every visited state without sym-state extraction from pixels.

Typical first-run:
    python train_bc.py --warmup-steps 10000 --warmup-epochs 3 \\
        --dagger-iters 1 --dagger-collect-steps 10000 --dagger-epochs 3 \\
        --eval-eps 10
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from nibbles_env import (
    NibblesEnv, NibblesVecEnv,
    SymbolicVecEnv, SYM_NUM_TYPES,
    ARENA_ROWS, ARENA_COLS,
)
from tools.heuristic_agent import heuristic_action


CKPT_DIR = Path("checkpoints")


def select_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()
    return torch.device("cpu")


def layer_init(layer: nn.Module, std: float = np.sqrt(2),
               bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


def _nature_cnn(in_channels: int, obs_h: int, obs_w: int,
                c1: int, c2: int, c3: int, fc: int) -> nn.Sequential:
    """NatureCNN with conv-output shape computed at construction time.

    Handles rectangular inputs (e.g. the 50x80 symbolic grid) as well as
    the classic 84x84 / 168x168 pixel setups.
    """
    convs = nn.Sequential(
        layer_init(nn.Conv2d(in_channels, c1, 8, stride=4)), nn.ReLU(),
        layer_init(nn.Conv2d(c1, c2, 4, stride=2)), nn.ReLU(),
        layer_init(nn.Conv2d(c2, c3, 3, stride=1)), nn.ReLU(),
    )
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, obs_h, obs_w)
        flat = convs(dummy).flatten(1).shape[1]
    return nn.Sequential(
        convs,
        nn.Flatten(),
        layer_init(nn.Linear(flat, fc)), nn.ReLU(),
    )


class Agent(nn.Module):
    """NatureCNN trunk + actor / critic linear heads (shared encoder).

    Square obs by default (pixel mode). Pass (h, w) tuple for rectangular
    inputs (symbolic mode).
    """

    def __init__(self, num_actions: int, in_channels: int = 12,
                 obs_size: int | tuple[int, int] = 84, width: int = 1):
        super().__init__()
        c1, c2, c3, fc = 32 * width, 64 * width, 64 * width, 512 * width
        h, w = (obs_size, obs_size) if isinstance(obs_size, int) else obs_size
        self.encoder = _nature_cnn(in_channels, h, w, c1, c2, c3, fc)
        self.actor = layer_init(nn.Linear(fc, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(fc, 1), std=1.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def act(self, x: torch.Tensor, action: torch.Tensor | None = None):
        z = self.encoder(x)
        logits = self.actor(z)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        v = self.critic(z).squeeze(-1)
        return action, dist.log_prob(action), dist.entropy(), v


def _to_obs(obs_np: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(obs_np).to(device).float().mul_(1.0 / 255.0)


def _to_obs_symbolic(obs_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """(N, 50, 80) uint8 int codes -> (N, 5, 50, 80) float one-hot."""
    t = torch.from_numpy(obs_np).to(device).long()
    return F.one_hot(t, num_classes=SYM_NUM_TYPES).permute(0, 3, 1, 2).float()


def collect_with_teacher(vec, K: int,
                         obs_shape: tuple,
                         tag: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Teacher drives; record (obs, teacher_action) per step."""
    obs_store = np.zeros((K, *obs_shape), dtype=np.uint8)
    act_store = np.zeros((K,), dtype=np.int64)
    obs_np = vec.reset()
    t0 = time.monotonic()
    for k in range(K):
        a = heuristic_action(vec._env._game)
        obs_store[k] = obs_np[0]
        act_store[k] = a
        obs_np, _, _, _ = vec.step(np.array([a]))
        if (k + 1) % 2000 == 0:
            sps = (k + 1) / (time.monotonic() - t0)
            print(f"{tag}teacher-collect {k+1}/{K}  sps {sps:.0f}",
                  flush=True)
    hist = np.bincount(act_store, minlength=NibblesEnv.NUM_ACTIONS).tolist()
    print(f"{tag}teacher done. action hist (NOOP/UP/DN/L/R): {hist}",
          flush=True)
    return obs_store, act_store


def collect_with_student(agent: Agent, vec,
                         device: torch.device, K: int,
                         obs_shape: tuple, to_obs_fn,
                         tag: str = ""
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Student drives; teacher labels every state. DAGGER's covariate-shift fix."""
    obs_store = np.zeros((K, *obs_shape), dtype=np.uint8)
    act_store = np.zeros((K,), dtype=np.int64)
    obs_np = vec.reset()
    obs_t = to_obs_fn(obs_np, device)
    agree = 0
    t0 = time.monotonic()
    for k in range(K):
        teacher_a = heuristic_action(vec._env._game)
        with torch.no_grad():
            s_action, _, _, _ = agent.act(obs_t)
        s_a = int(s_action.cpu().item())
        obs_store[k] = obs_np[0]
        act_store[k] = teacher_a
        if s_a == teacher_a:
            agree += 1
        obs_np, _, _, _ = vec.step(np.array([s_a]))
        obs_t = to_obs_fn(obs_np, device)
        if (k + 1) % 2000 == 0:
            sps = (k + 1) / (time.monotonic() - t0)
            print(f"{tag}student-collect {k+1}/{K}  sps {sps:.0f}  "
                  f"agree {agree / (k+1):.1%}", flush=True)
    print(f"{tag}student done. agree {agree / K:.1%}", flush=True)
    return obs_store, act_store


def train_epochs(agent: Agent, optim: Adam,
                 obs_store: np.ndarray, act_store: np.ndarray,
                 epochs: int, bs: int, max_grad_norm: float,
                 device: torch.device, seed: int, to_obs_fn,
                 tag: str = "") -> None:
    K = len(act_store)
    if K == 0:
        return
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        perm = rng.permutation(K)
        ce_sum, acc_sum, nb = 0.0, 0.0, 0
        for start in range(0, K, bs):
            mb = perm[start:start + bs]
            mb_obs = to_obs_fn(obs_store[mb], device)
            mb_act = torch.from_numpy(act_store[mb]).to(device)
            logits = agent.actor(agent.encode(mb_obs))
            ce = F.cross_entropy(logits, mb_act)
            optim.zero_grad()
            ce.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optim.step()
            ce_sum += ce.item()
            with torch.no_grad():
                acc_sum += (logits.argmax(-1) == mb_act).float().mean().item()
            nb += 1
        print(f"{tag}epoch {epoch+1}/{epochs}  "
              f"ce {ce_sum / nb:.3f}  acc {acc_sum / nb:.3f}", flush=True)


def evaluate(agent: Agent, vec, n_eps: int,
             device: torch.device, to_obs_fn,
             greedy: bool = False,
             tag: str = "") -> list[int]:
    """Roll out `n_eps` episodes; return per-episode final scores.

    `greedy=False` (default) samples from Categorical(logits) — what PPO
    would do during rollouts. `greedy=True` takes argmax, which is the
    fair test of what the BC encoder actually learned.
    """
    scores: list[int] = []
    obs_np = vec.reset()
    obs_t = to_obs_fn(obs_np, device)
    ep_steps = 0
    while len(scores) < n_eps:
        with torch.no_grad():
            if greedy:
                logits = agent.actor(agent.encode(obs_t))
                action = logits.argmax(-1)
            else:
                action, _, _, _ = agent.act(obs_t)
        obs_np, _, dones, infos = vec.step(action.cpu().numpy())
        obs_t = to_obs_fn(obs_np, device)
        ep_steps += 1
        if dones[0]:
            score = int(infos[0].get("score", 0))
            trunc = bool(infos[0].get("truncated", False))
            scores.append(score)
            print(f"{tag}eval {len(scores)}/{n_eps}: score {score} "
                  f"steps {ep_steps}{' (truncated)' if trunc else ''}",
                  flush=True)
            ep_steps = 0
    return scores


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup-steps", type=int, default=10_000)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--dagger-iters", type=int, default=0)
    p.add_argument("--dagger-collect-steps", type=int, default=10_000)
    p.add_argument("--dagger-epochs", type=int, default=3)
    p.add_argument("--eval-eps", type=int, default=10)
    p.add_argument("--env-max-steps", type=int, default=10_000,
                   help="env max_steps cap; episodes truncate (done=True) here.")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--bc-batch-size", type=int, default=64)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--obs-size", type=int, default=84)
    p.add_argument("--encoder-width", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--obs-mode", choices=["pixel", "symbolic"],
                   default="pixel",
                   help="`pixel` (default) feeds 84/168-px RGB through a "
                        "NatureCNN; `symbolic` feeds a (5, 50, 80) one-hot "
                        "of cell types and skips frame-stack/preprocess.")
    p.add_argument("--greedy-eval", action="store_true",
                   help="argmax instead of Categorical sampling during eval")
    p.add_argument("--eval-only", type=str, default="",
                   metavar="CKPT_PATH",
                   help="load an existing checkpoint, run only the eval, "
                        "and exit (skip warmup, DAGGER, save)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.force_cpu)
    ckpt_dir = CKPT_DIR / args.run_name if args.run_name else CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = f"[{args.run_name}] " if args.run_name else ""

    print(f"{tag}device={device}", flush=True)
    print(f"{tag}args={vars(args)}", flush=True)

    env_kwargs = dict(max_steps=args.env_max_steps, rng_seed=args.seed)
    if args.obs_mode == "symbolic":
        vec = SymbolicVecEnv(env_kwargs=env_kwargs)
        obs_shape = (ARENA_ROWS, ARENA_COLS)
        in_ch = SYM_NUM_TYPES
        agent_obs_size = (ARENA_ROWS, ARENA_COLS)
        to_obs_fn = _to_obs_symbolic
    else:
        vec = NibblesVecEnv(num_envs=1, frame_skip=1,
                           frame_stack=args.frame_stack,
                           obs_size=args.obs_size, color=True,
                           env_kwargs=env_kwargs)
        in_ch = args.frame_stack * 3
        obs_shape = (in_ch, args.obs_size, args.obs_size)
        agent_obs_size = args.obs_size
        to_obs_fn = _to_obs

    agent = Agent(NibblesEnv.NUM_ACTIONS, in_channels=in_ch,
                  obs_size=agent_obs_size,
                  width=args.encoder_width).to(device)
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"{tag}agent: mode={args.obs_mode}  in_ch={in_ch}  "
          f"obs_shape={obs_shape}  width={args.encoder_width}  "
          f"params={n_params:,}", flush=True)

    if args.eval_only:
        ckpt = torch.load(args.eval_only, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["agent"])
        agent.eval()
        mode = "greedy" if args.greedy_eval else "stochastic"
        print(f"{tag}eval-only: loaded {args.eval_only}  mode={mode}",
              flush=True)
        scores = evaluate(agent, vec, args.eval_eps, device, to_obs_fn,
                          greedy=args.greedy_eval, tag=tag)
        arr = np.array(scores, dtype=np.int32)
        print(f"{tag}eval: mean {arr.mean():.0f}  "
              f"median {int(np.median(arr))}  "
              f"min/max {arr.min()}/{arr.max()}", flush=True)
        vec.close()
        return

    optim = Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    print(f"{tag}warmup: teacher-driven collect of {args.warmup_steps}",
          flush=True)
    obs_store, act_store = collect_with_teacher(
        vec, args.warmup_steps, obs_shape, tag=tag)
    print(f"{tag}warmup: training {args.warmup_epochs} epochs", flush=True)
    train_epochs(agent, optim, obs_store, act_store,
                 args.warmup_epochs, args.bc_batch_size,
                 args.max_grad_norm, device, args.seed, to_obs_fn, tag=tag)

    for it in range(args.dagger_iters):
        K = args.dagger_collect_steps
        print(f"{tag}DAGGER iter {it+1}/{args.dagger_iters}: "
              f"student-collect of {K}", flush=True)
        new_obs, new_act = collect_with_student(
            agent, vec, device, K, obs_shape, to_obs_fn, tag=tag)
        obs_store = np.concatenate([obs_store, new_obs], axis=0)
        act_store = np.concatenate([act_store, new_act], axis=0)
        del new_obs, new_act
        print(f"{tag}DAGGER iter {it+1}: dataset size {len(act_store):,}; "
              f"training {args.dagger_epochs} epochs", flush=True)
        train_epochs(agent, optim, obs_store, act_store,
                     args.dagger_epochs, args.bc_batch_size,
                     args.max_grad_norm, device, args.seed + it + 1,
                     to_obs_fn, tag=tag)
        mid = evaluate(agent, vec, 3, device, to_obs_fn,
                       greedy=args.greedy_eval, tag=tag + "mid-")
        arr_mid = np.array(mid, dtype=np.int32)
        print(f"{tag}DAGGER iter {it+1} mid-eval: mean {arr_mid.mean():.0f}  "
              f"max {arr_mid.max()}", flush=True)

    mode = "greedy" if args.greedy_eval else "stochastic"
    print(f"{tag}final eval ({args.eval_eps} episodes, {mode})", flush=True)
    scores = evaluate(agent, vec, args.eval_eps, device, to_obs_fn,
                      greedy=args.greedy_eval, tag=tag)
    arr = np.array(scores, dtype=np.int32)
    print(f"{tag}eval: mean {arr.mean():.0f}  "
          f"median {int(np.median(arr))}  "
          f"min/max {arr.min()}/{arr.max()}", flush=True)
    vec.close()

    out = ckpt_dir / "bc_nibbles.pt"
    torch.save({
        "agent": agent.state_dict(),
        "config": vars(args),
        "eval_scores": scores,
        "dataset_size": len(act_store),
    }, out)
    print(f"{tag}saved {out}", flush=True)


if __name__ == "__main__":
    main()
