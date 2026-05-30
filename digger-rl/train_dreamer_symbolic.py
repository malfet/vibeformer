"""Dreamer on top of the symbolic Digger observation.

Same Dreamer recipe as `train_dreamer_online.py` but the observation is the
6-channel (10x15) symbolic tile-grid from `SymbolicDiggerEnv`, not raw
pixels. The world model is a SymbolicEncoder/Decoder (MLP) over the mask
grid plus the existing RSSM + reward/continue heads.

Why this might work where the pixel Dreamer plateaued:
  - The recon loss is on tile-level masks, not pixels, so the world model
    can devote capacity to reward/dynamics instead of reconstructing
    nobbin walking-frames.
  - With the recently-fixed monster channel, channel 3 actually carries
    threat information -- imagined rollouts can include "what if the
    nobbin keeps walking left", which the pixel Dreamer never quite
    learned to predict.

Loop:
  1. Prefill replay with random-action steps.
  2. Forever:
     a. Run K real-env steps with the current actor (RSSM observes each
        frame so its hidden state stays grounded).
     b. Sample B subsequences from replay; WM update.
     c. Sample imagination starts from the latents seen during the WM
        forward pass; AC update via REINFORCE + critic baseline.

Run:
    python train_dreamer_symbolic.py --total-timesteps 100000 \\
        --run-name dreamer_sym_v1
"""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tools.dreamer import (Actor, Critic, DreamerConfig, ReplayBuffer,
                            WorldModel)
from tools.symbolic_env import BASE_OBS_CHANNELS, SymbolicDiggerEnv
from train_dagger import env_step_skipped
from train_dreamer import ac_loss, imagine_rollout
from train_ppo import select_device

REPO = Path(__file__).parent.resolve()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-timesteps", type=int, default=100_000,
                   help="agent steps (== frame_skip * emulator frames)")
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--buffer-capacity", type=int, default=20_000)
    p.add_argument("--prefill-steps", type=int, default=500)
    p.add_argument("--collect-per-iter", type=int, default=16)
    p.add_argument("--train-per-iter", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-length", type=int, default=50)
    p.add_argument("--ac-batch", type=int, default=64)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ac-lr", type=float, default=8e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=3e-3)
    p.add_argument("--log-every", type=int, default=20,
                   help="iterations between log lines")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--episodic-life", default=True,
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--run-name", type=str, default="dreamer_sym")
    p.add_argument("--init-from", type=Path, default=None,
                   help="optional checkpoint to warm-start WM/actor/critic")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = select_device(args.force_cpu)

    out_dir = REPO / "data" / "checkpoints" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"[{args.run_name}]"
    print(f"{tag} device={device}  out={out_dir}", flush=True)

    # SymbolicDiggerEnv doesn't accept frame_skip directly; we apply it via
    # env_step_skipped from train_dagger. frame_stack=1 is intentional --
    # the RSSM is responsible for temporal context.
    env = SymbolicDiggerEnv(max_steps=10**9,
                             episodic_life=args.episodic_life,
                             frame_stack=1)
    obs_shape = (BASE_OBS_CHANNELS, 10, 15)

    cfg = DreamerConfig(obs_type="symbolic",
                         sym_channels=BASE_OBS_CHANNELS,
                         sym_h=10, sym_w=15,
                         num_actions=env.NUM_ACTIONS)
    wm = WorldModel(cfg).to(device)
    actor = Actor(cfg).to(device)
    critic = Critic(cfg).to(device)
    if args.init_from is not None:
        ckpt = torch.load(args.init_from, map_location=device,
                           weights_only=False)
        wm.load_state_dict(ckpt["world_model"])
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        print(f"{tag} warm-started from {args.init_from}", flush=True)

    wm_params = sum(p.numel() for p in wm.parameters())
    ac_params = sum(p.numel() for p in actor.parameters()) \
              + sum(p.numel() for p in critic.parameters())
    print(f"{tag} WorldModel {wm_params:,}  Actor+Critic {ac_params:,}",
          flush=True)

    wm_optim = torch.optim.Adam(wm.parameters(), lr=args.lr)
    ac_optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=args.ac_lr)

    # ---- Replay buffer ------------------------------------------------------
    # Symbolic obs are already float32 in [0, 1]; no scaling at sample time.
    # num_envs=1 (single libretro instance).
    buf = ReplayBuffer(capacity=args.buffer_capacity, num_envs=1,
                        obs_shape=obs_shape,
                        obs_dtype=np.float32, obs_scale=1.0)

    # ---- Prefill with random actions ---------------------------------------
    print(f"{tag} prefilling buffer with {args.prefill_steps} random steps",
          flush=True)
    cur_obs = env.reset()                          # (C, H, W) float32
    ep_return = 0.0
    ep_returns = collections.deque(maxlen=20)
    for _ in range(args.prefill_steps):
        a = int(rng.integers(env.NUM_ACTIONS))
        next_obs, total_r, done, info = env_step_skipped(
            env, a, args.frame_skip)
        buf.add(cur_obs[None],                      # (1, C, H, W)
                np.array([a], dtype=np.int64),
                np.array([total_r], dtype=np.float32),
                np.array([done], dtype=bool))
        ep_return += total_r
        if done:
            ep_returns.append(ep_return)
            ep_return = 0.0
            cur_obs = env.reset()
        else:
            cur_obs = next_obs

    # ---- Online loop --------------------------------------------------------
    # Track the RSSM state across collect calls so the actor sees a
    # continuously-updated latent state. `last_action_oh` is what the RSSM
    # observe_step needs as input -- the *previous* action that produced
    # `cur_obs`. At reset there is no prior action, so we use zeros.
    h_actor, z_actor = wm.rssm.initial(1, device)
    last_action_oh = torch.zeros(1, env.NUM_ACTIONS, device=device)
    sums: dict[str, float] = {}
    n_logged_iters = 0
    total_steps = 0
    t0 = time.monotonic()
    last_save_step = 0
    iter_no = 0

    while total_steps < args.total_timesteps:
        iter_no += 1

        # ---- Collect K real-env steps under current actor ------------------
        for _ in range(args.collect_per_iter):
            with torch.no_grad():
                obs_t = torch.from_numpy(cur_obs[None]).to(device)  # (1,C,H,W)
                embed = wm.encoder(obs_t)
                h_actor, z_actor, _, _ = wm.rssm.observe_step(
                    h_actor, z_actor, last_action_oh, embed)
                feat = wm.rssm.feature(h_actor, z_actor)
                dist = actor(feat)
                a_t = dist.sample()
                a = int(a_t.item())
                last_action_oh = F.one_hot(
                    a_t, num_classes=env.NUM_ACTIONS).float()

            next_obs, total_r, done, info = env_step_skipped(
                env, a, args.frame_skip)
            buf.add(cur_obs[None],
                    np.array([a], dtype=np.int64),
                    np.array([total_r], dtype=np.float32),
                    np.array([done], dtype=bool))
            ep_return += total_r
            total_steps += 1

            if done:
                ep_returns.append(ep_return)
                ep_return = 0.0
                cur_obs = env.reset()
                h_actor, z_actor = wm.rssm.initial(1, device)
                last_action_oh = torch.zeros(
                    1, env.NUM_ACTIONS, device=device)
            else:
                cur_obs = next_obs

        # ---- WM + AC updates ----------------------------------------------
        for _ in range(args.train_per_iter):
            obs, act, rew, cont = buf.sample(
                args.batch_size, args.seq_length, device, rng)
            losses, h_seq, z_seq = wm.loss(
                obs, act, rew, cont, return_latents=True)
            wm_optim.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 1000.0)
            wm_optim.step()
            for k, v in losses.items():
                sums[f"wm_{k}"] = sums.get(f"wm_{k}", 0.0) + v.item()

            # Imagination starts from the WM forward pass's latents.
            B, T = h_seq.shape[:2]
            flat_h = h_seq.reshape(B * T, -1).detach()
            flat_z = z_seq.reshape(B * T, -1).detach()
            pick = torch.randint(B * T, (args.ac_batch,), device=device)
            start_h = flat_h[pick]
            start_z = flat_z[pick]
            rollout = imagine_rollout(
                wm, actor, start_h, start_z, args.horizon)
            ac_losses = ac_loss(
                rollout, critic, args.gamma, args.lam, args.ent_coef)
            ac_total = ac_losses["actor"] + ac_losses["critic"]
            ac_optim.zero_grad()
            ac_total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), 100.0)
            ac_optim.step()
            for k, v in ac_losses.items():
                sums[f"ac_{k}"] = sums.get(f"ac_{k}", 0.0) + v.item()
        n_logged_iters += 1

        # ---- Periodic logging ---------------------------------------------
        if iter_no % args.log_every == 0:
            avg = {k: v / max(1, n_logged_iters * args.train_per_iter)
                   for k, v in sums.items()}
            elapsed = time.monotonic() - t0
            sps = total_steps / max(elapsed, 1e-6)
            ret = (sum(ep_returns) / len(ep_returns)) if ep_returns else 0.0
            print(
                f"{tag} iter {iter_no:>5d}  step {total_steps:>8d}  "
                f"buf {len(buf):>5d}  avg_ret {ret:>7.1f}  "
                f"wm[recon {avg.get('wm_recon', 0):.3f} "
                f"rew {avg.get('wm_reward', 0):.2f} "
                f"cont {avg.get('wm_continue', 0):.3f} "
                f"kl {avg.get('wm_kl', 0):.2f}]  "
                f"ac[H {avg.get('ac_entropy', 0):.2f} "
                f"V {avg.get('ac_value_mean', 0):+.1f} "
                f"G {avg.get('ac_return_mean', 0):+.1f}]  "
                f"sps {sps:.0f}",
                flush=True)
            sums.clear()
            n_logged_iters = 0

        # ---- Periodic save ------------------------------------------------
        if total_steps - last_save_step >= args.save_every:
            ckpt_path = out_dir / "dreamer.pt"
            torch.save({"world_model": wm.state_dict(),
                         "actor": actor.state_dict(),
                         "critic": critic.state_dict(),
                         "config": cfg.__dict__,
                         "iter": iter_no,
                         "step": total_steps,
                         "args": vars(args)}, ckpt_path)
            last_save_step = total_steps

    final = out_dir / "dreamer_final.pt"
    torch.save({"world_model": wm.state_dict(),
                 "actor": actor.state_dict(),
                 "critic": critic.state_dict(),
                 "config": cfg.__dict__,
                 "iter": iter_no,
                 "step": total_steps,
                 "args": vars(args)}, final)
    print(f"{tag} saved final to {final}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
