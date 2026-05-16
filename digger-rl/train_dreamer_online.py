"""Phase 3 Dreamer: online training with replay buffer + real env.

Loop:
  1. Collect K real-env steps into a replay buffer using the current actor
     (with the RSSM observing each real frame to keep its state grounded).
  2. Sample B subsequences from the buffer; do a WM update and an AC
     update on imagined rollouts launched from the WM's latent states.
  3. Repeat.

Optionally initialise from an offline-trained dreamer checkpoint
(--init-from) to skip the cold start.

    python train_dreamer_online.py --total-timesteps 200000 --num-envs 4 \
        --init-from data/checkpoints/dreamer_wm.pt
"""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from digger_env import DiggerVecEnv
from dreamer import Actor, Critic, DreamerConfig, ReplayBuffer, WorldModel
from train_dreamer import ac_loss, imagine_rollout
from train_ppo import select_device

REPO = Path(__file__).parent.resolve()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-timesteps", type=int, default=200_000,
                   help="emulator frames (NOT policy steps)")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--obs-size", type=int, default=64)
    p.add_argument("--buffer-capacity", type=int, default=20_000,
                   help="per-env capacity; total samples = num_envs * capacity")
    p.add_argument("--prefill-steps", type=int, default=500,
                   help="random-action collection steps to seed the buffer")
    p.add_argument("--collect-per-iter", type=int, default=16,
                   help="real-env steps collected between training updates")
    p.add_argument("--train-per-iter", type=int, default=8,
                   help="WM+AC updates per outer iteration")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-length", type=int, default=50)
    p.add_argument("--ac-batch", type=int, default=64)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ac-lr", type=float, default=8e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=3e-3)
    p.add_argument("--init-from", type=Path, default=None,
                   help="warm-start from an offline-trained dreamer ckpt")
    p.add_argument("--save-path", type=Path,
                   default=REPO / "data" / "checkpoints" / "dreamer_online.pt")
    p.add_argument("--save-every", type=int, default=50,
                   help="outer iters between checkpoint saves")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--force-cpu", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = select_device(args.force_cpu)
    print(f"device={device}  num_envs={args.num_envs}  obs_size={args.obs_size}")

    # ---- Env (color, no frame-stack — RSSM provides temporal context) ----
    vec = DiggerVecEnv(
        num_envs=args.num_envs, frame_skip=args.frame_skip,
        frame_stack=1, obs_size=args.obs_size, color=True,
        env_kwargs=dict(max_steps=10**9, episodic_life=True))

    # ---- Models ----
    cfg = DreamerConfig(num_actions=6, obs_size=args.obs_size)
    wm = WorldModel(cfg).to(device)
    actor = Actor(cfg).to(device)
    critic = Critic(cfg).to(device)
    if args.init_from is not None and args.init_from.exists():
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        wm.load_state_dict(ckpt["world_model"]
                           if "world_model" in ckpt else ckpt["model"])
        if "actor" in ckpt:
            actor.load_state_dict(ckpt["actor"])
        if "critic" in ckpt:
            critic.load_state_dict(ckpt["critic"])
        print(f"warm-started from {args.init_from}")
    wm_optim = torch.optim.Adam(wm.parameters(), lr=args.lr)
    ac_optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=args.ac_lr)
    wm_n = sum(p.numel() for p in wm.parameters())
    ac_n = sum(p.numel() for p in actor.parameters()) \
         + sum(p.numel() for p in critic.parameters())
    print(f"WM: {wm_n:,} params  AC: {ac_n:,} params")

    # ---- Replay buffer ----
    buffer = ReplayBuffer(
        capacity=args.buffer_capacity, num_envs=args.num_envs,
        obs_shape=(3, args.obs_size, args.obs_size))

    # ---- Prefill with random actions to seed the buffer ----
    print(f"prefilling buffer with {args.prefill_steps} random steps...")
    obs = vec.reset()  # (N, 3, H, W) uint8
    for _ in range(args.prefill_steps):
        actions = rng.integers(0, 6, size=args.num_envs)
        next_obs, rewards, dones, _ = vec.step(actions)
        buffer.add(obs, actions, rewards, dones)
        obs = next_obs
    global_step = args.num_envs * args.prefill_steps * args.frame_skip
    print(f"  buffer now has {len(buffer)} per-env samples "
          f"({len(buffer) * args.num_envs} total)")

    # ---- Online loop ----
    h, z = wm.rssm.initial(args.num_envs, device)
    prev_a = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    ep_return = np.zeros(args.num_envs, dtype=np.float32)
    ep_length = np.zeros(args.num_envs, dtype=np.int64)
    ep_returns: collections.deque[float] = collections.deque(maxlen=30)

    iter_no = 0
    t0 = time.monotonic()

    while global_step < args.total_timesteps:
        iter_no += 1

        # ---- Collect K real-env steps under the current actor ----
        collect_sums = {"reward": 0.0}
        for _ in range(args.collect_per_iter):
            obs_t = torch.from_numpy(obs).to(device).float().mul_(1.0 / 255.0)
            with torch.no_grad():
                embed = wm.encoder(obs_t)
                prev_a_oh = F.one_hot(prev_a, num_classes=6).float()
                h, z, _, _ = wm.rssm.observe_step(h, z, prev_a_oh, embed)
                feat = wm.rssm.feature(h, z)
                actions = actor(feat).sample()
            actions_np = actions.cpu().numpy()
            next_obs, rewards, dones, infos = vec.step(actions_np)
            buffer.add(obs, actions_np, rewards, dones)
            ep_return += rewards
            ep_length += args.frame_skip
            for i, d in enumerate(dones):
                if d:
                    ep_returns.append(float(ep_return[i]))
                    ep_return[i] = 0.0
                    ep_length[i] = 0
                    # Reset RSSM state for that env on episode boundary
                    h = h.clone(); z = z.clone()
                    h[i].zero_(); z[i].zero_()
                    prev_a = prev_a.clone(); prev_a[i] = 0
            obs = next_obs
            prev_a = actions
            collect_sums["reward"] += float(rewards.sum())
            global_step += args.num_envs * args.frame_skip

        # ---- Train: WM + AC updates from replay ----
        if len(buffer) < args.seq_length:
            continue
        train_sums = {}
        for _ in range(args.train_per_iter):
            b_obs, b_act, b_rew, b_cont = buffer.sample(
                args.batch_size, args.seq_length, device, rng)
            losses, h_seq, z_seq = wm.loss(
                b_obs, b_act, b_rew, b_cont, return_latents=True)
            wm_optim.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 1000.0)
            wm_optim.step()
            for k, v in losses.items():
                train_sums[k] = train_sums.get(k, 0.0) + v.item()

            # AC update from imagined rollouts launched off WM latents
            B, T = h_seq.shape[:2]
            flat_h = h_seq.reshape(B * T, -1).detach()
            flat_z = z_seq.reshape(B * T, -1).detach()
            pick = torch.randint(B * T, (args.ac_batch,), device=device)
            rollout = imagine_rollout(
                wm, actor, flat_h[pick], flat_z[pick], args.horizon)
            ac_l = ac_loss(rollout, critic, args.gamma, args.lam, args.ent_coef)
            ac_total = ac_l["actor"] + ac_l["critic"]
            ac_optim.zero_grad()
            ac_total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), 100.0)
            ac_optim.step()
            for k, v in ac_l.items():
                train_sums[f"ac_{k}"] = train_sums.get(f"ac_{k}", 0.0) + v.item()

        avg_ret = (sum(ep_returns) / len(ep_returns)) if ep_returns else float("nan")
        elapsed = time.monotonic() - t0
        sps = global_step / max(elapsed, 1e-6)
        tr = {k: v / args.train_per_iter for k, v in train_sums.items()}
        print(f"iter {iter_no:>4d}  step {global_step:>8d}  "
              f"buf {len(buffer):>5d}  "
              f"avg_ret {avg_ret:6.1f}  "
              f"wm[recon {tr.get('recon',0):.3f} rew {tr.get('reward',0):.1f} "
              f"cont {tr.get('continue',0):.3f} kl {tr.get('kl',0):.2f}]  "
              f"ac[H {tr.get('ac_entropy',0):.2f} V {tr.get('ac_value_mean',0):+.1f} "
              f"G {tr.get('ac_return_mean',0):+.1f}]  "
              f"sps {sps:.0f}", flush=True)

        if args.save_every and iter_no % args.save_every == 0:
            args.save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"world_model": wm.state_dict(),
                        "actor": actor.state_dict(),
                        "critic": critic.state_dict(),
                        "config": cfg.__dict__,
                        "global_step": global_step}, args.save_path)
            print(f"  saved {args.save_path}", flush=True)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"world_model": wm.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "config": cfg.__dict__,
                "global_step": global_step}, args.save_path)
    print(f"done. final ckpt {args.save_path}")
    vec.close()


if __name__ == "__main__":
    main()
