"""Train Dreamer world model offline on a recorded color trace.

Phase 1 of the Dreamer rollout: train the WorldModel (encoder+RSSM+decoder+
reward+continue) on existing replay data. No env interaction. No policy.
Lets us verify the world-model losses converge and rewards become predictable
from the latent state before we wire in the actor-critic + imagined rollouts.

Run:
    python train_dreamer.py data/trace1_color.npz --epochs 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tools.dreamer import Actor, Critic, DreamerConfig, WorldModel

REPO = Path(__file__).parent.resolve()


def _resize_uint8(frames: np.ndarray, out_size: int) -> np.ndarray:
    """(N, H, W, 3) uint8 -> (N, 3, out_size, out_size) uint8 via torch area resize."""
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).float().mul_(1.0 / 255.0)
    t = F.interpolate(t, size=(out_size, out_size), mode="area")
    return (t * 255.0).clamp_(0, 255).byte().numpy()


def load_trace(path: Path, out_size: int = 64):
    d = np.load(path)
    if not bool(d.get("color", False)):
        raise ValueError(f"{path}: trace must be color (recorded with --color)")
    frames = d["frames"]                 # (N, 84, 84, 3) uint8
    actions = d["actions"].astype(np.int64)
    rewards = d["raw_rewards"].astype(np.float32)
    lives = d["lives"]
    resets = d["resets"].astype(bool) if "resets" in d.files \
        else np.zeros(len(actions), dtype=bool)

    if frames.shape[1] != out_size:
        frames = _resize_uint8(frames, out_size)
    else:
        frames = frames.transpose(0, 3, 1, 2)

    n = len(frames)
    real_done = np.zeros(n, dtype=bool)
    real_done[:-1] = ((lives[1:] == 0) & (lives[:-1] > 0)) | resets[1:]
    continues = (1.0 - real_done.astype(np.float32))
    return frames, actions, rewards, continues, real_done


def episode_segments(real_done: np.ndarray, min_len: int):
    """Return list of (start, end) half-open intervals where no episode ends.

    Drop any segment shorter than min_len, since we can't draw a sequence
    of length min_len from it.
    """
    n = len(real_done)
    segs, start = [], 0
    for t in range(n):
        if real_done[t]:
            if t + 1 - start >= min_len:
                segs.append((start, t + 1))
            start = t + 1
    if n - start >= min_len:
        segs.append((start, n))
    return segs


def imagine_rollout(wm: WorldModel, actor: Actor,
                    h0: torch.Tensor, z0: torch.Tensor, horizon: int):
    """Roll the world model forward `horizon` steps under the current policy.

    Gradient flow: only the actor receives gradients (through log_probs and
    entropies). The world model is evaluated under no_grad and its outputs
    (rewards, continues, next features) are constants from the AC update's
    perspective. The features used as actor inputs are detached so AC
    gradients never leak into the encoder/RSSM.

    Returns a dict of stacked tensors with leading dim = horizon (or
    horizon+1 for features, including the bootstrap state).
    """
    h, z = h0, z0
    features = [wm.rssm.feature(h, z)]
    rewards, continues = [], []
    log_probs, entropies, actions_out = [], [], []
    num_actions = wm.cfg.num_actions

    for _ in range(horizon):
        # Actor on detached feature -> gradient stays in actor only
        feat_in = features[-1].detach()
        dist = actor(feat_in)
        a = dist.sample()
        log_probs.append(dist.log_prob(a))
        entropies.append(dist.entropy())
        actions_out.append(a)

        with torch.no_grad():
            r = wm.reward_head(features[-1])
            c = wm.continue_head(features[-1])
            a_oh = F.one_hot(a, num_classes=num_actions).float()
            h, z, _ = wm.rssm.imagine_step(h, z, a_oh)
            features.append(wm.rssm.feature(h, z))
        rewards.append(r)
        continues.append(c)

    return {
        "features": torch.stack(features, dim=0),    # (H+1, N, feat)
        "actions":  torch.stack(actions_out, dim=0), # (H, N)
        "rewards":  torch.stack(rewards, dim=0),     # (H, N)
        "continues": torch.stack(continues, dim=0),  # (H, N)
        "log_probs": torch.stack(log_probs, dim=0),  # (H, N)
        "entropies": torch.stack(entropies, dim=0),  # (H, N)
    }


def lambda_returns(rewards, values, continues, gamma: float, lam: float):
    """Generalised-advantage lambda return.

    G_t = r_t + gamma * cont_t * [(1 - lam) * V_{t+1} + lam * G_{t+1}]

    rewards:   (H, N)
    values:    (H+1, N) — last is the bootstrap V at the imagined endpoint
    continues: (H, N)
    Returns:   (H, N)
    """
    H = rewards.shape[0]
    out = torch.zeros_like(rewards)
    last = values[-1]
    for t in reversed(range(H)):
        last = rewards[t] + gamma * continues[t] * (
            (1.0 - lam) * values[t + 1] + lam * last)
        out[t] = last
    return out


def ac_loss(rollout: dict, critic: Critic, gamma: float, lam: float,
            entropy_coef: float):
    """Actor-critic loss on imagined rollouts.

    Actor: REINFORCE with critic baseline + entropy bonus.
    Critic: 0.5 * (V - lambda-return)^2 against detached targets.
    """
    feats = rollout["features"]                    # (H+1, N, feat)
    # Critic on detached features so AC grads don't leak into WM.
    values = critic(feats.detach())                # (H+1, N)
    returns = lambda_returns(
        rollout["rewards"], values, rollout["continues"], gamma, lam)

    advantages = (returns - values[:-1]).detach()
    actor_loss = -(rollout["log_probs"] * advantages).mean() \
                 - entropy_coef * rollout["entropies"].mean()
    critic_loss = 0.5 * (values[:-1] - returns.detach()).pow(2).mean()

    return {
        "actor": actor_loss,
        "critic": critic_loss,
        "value_mean": values.detach().mean(),
        "return_mean": returns.detach().mean(),
        "entropy": rollout["entropies"].detach().mean(),
    }


def sample_batch(frames, actions, rewards, continues, segs,
                 B: int, T: int, device: torch.device, rng) -> tuple:
    H = frames.shape[2]
    obs_b = np.empty((B, T, 3, H, H), dtype=np.uint8)
    act_b = np.empty((B, T), dtype=np.int64)
    rew_b = np.empty((B, T), dtype=np.float32)
    con_b = np.empty((B, T), dtype=np.float32)
    for i in range(B):
        seg = segs[rng.integers(len(segs))]
        s = int(rng.integers(seg[0], seg[1] - T + 1))
        obs_b[i] = frames[s:s + T]
        act_b[i] = actions[s:s + T]
        rew_b[i] = rewards[s:s + T]
        con_b[i] = continues[s:s + T]
    obs_t = torch.from_numpy(obs_b).to(device).float().mul_(1.0 / 255.0)
    return (obs_t,
            torch.from_numpy(act_b).to(device),
            torch.from_numpy(rew_b).to(device),
            torch.from_numpy(con_b).to(device))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-length", type=int, default=50)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ac-lr", type=float, default=8e-5)
    p.add_argument("--horizon", type=int, default=15,
                   help="imagined-rollout horizon for actor-critic update")
    p.add_argument("--ac-batch", type=int, default=64,
                   help="number of imagined trajectories per AC update")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=3e-3)
    p.add_argument("--no-ac", action="store_true",
                   help="skip actor-critic updates (Phase 1 mode)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--save-path", type=Path,
                   default=REPO / "data" / "checkpoints" / "dreamer_wm.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.force_cpu:
        device = torch.device("cpu")
    elif torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")
    print(f"device={device}  trace={args.trace}")

    frames, actions, rewards, continues, real_done = load_trace(args.trace)
    print(f"trace: {len(frames)} samples, {int(real_done.sum())} episode ends, "
          f"frames shape {frames.shape} dtype {frames.dtype}")
    segs = episode_segments(real_done, min_len=args.seq_length)
    print(f"usable segments (>= {args.seq_length} frames): {len(segs)}")
    if not segs:
        raise RuntimeError("no usable episode segments; reduce --seq-length")

    cfg = DreamerConfig(num_actions=6, obs_size=64)
    wm = WorldModel(cfg).to(device)
    actor = Actor(cfg).to(device)
    critic = Critic(cfg).to(device)
    wm_n = sum(p.numel() for p in wm.parameters())
    ac_n = sum(p.numel() for p in actor.parameters()) + \
           sum(p.numel() for p in critic.parameters())
    print(f"WorldModel: {wm_n:,} params  Actor+Critic: {ac_n:,} params")
    print(f"batch={args.batch_size} seq={args.seq_length} "
          f"horizon={args.horizon} ac_batch={args.ac_batch}  "
          f"ac={'OFF' if args.no_ac else 'ON'}")

    wm_optim = torch.optim.Adam(wm.parameters(), lr=args.lr)
    ac_optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=args.ac_lr)

    t0 = time.monotonic()
    for epoch in range(args.epochs):
        sums = {}
        for step in range(args.steps_per_epoch):
            obs, act, rew, con = sample_batch(
                frames, actions, rewards, continues, segs,
                args.batch_size, args.seq_length, device, rng)
            losses, h_seq, z_seq = wm.loss(obs, act, rew, con,
                                           return_latents=True)
            wm_optim.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 1000.0)
            wm_optim.step()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.item()

            if not args.no_ac:
                # Use the WM forward pass's latent states as imagination
                # starts. Detach so AC gradients don't flow into WM.
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

        avg = {k: v / args.steps_per_epoch for k, v in sums.items()}
        elapsed = time.monotonic() - t0
        sps = (epoch + 1) * args.steps_per_epoch / max(elapsed, 1e-6)
        wm_summary = "  ".join(
            f"{k}={avg[k]:.3f}" for k in ("recon", "reward", "continue", "kl"))
        ac_summary = ""
        if not args.no_ac:
            ac_summary = "  ||  " + "  ".join(
                f"{k.removeprefix('ac_')}={avg[k]:+.3f}"
                for k in ("ac_actor", "ac_critic", "ac_value_mean",
                          "ac_return_mean", "ac_entropy"))
        print(f"epoch {epoch + 1:>2d}/{args.epochs}  {wm_summary}{ac_summary}  "
              f"wall={elapsed:.1f}s  steps/s={sps:.1f}", flush=True)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"world_model": wm.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "config": cfg.__dict__}, args.save_path)
    print(f"saved {args.save_path}")


if __name__ == "__main__":
    main()
