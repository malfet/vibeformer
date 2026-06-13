"""PPO on tiny snake, with optional BC checkpoint init + BC anchor loss.

Reuses the Agent / device / eval helpers from train_bc so we don't drift on
encoder details. Workflow:

  1. (Optional) load a BC checkpoint into the agent.
  2. Rollout `--num-steps` policy steps with stochastic Categorical actions,
     stash (obs, action, logp, value, reward, done) plus the teacher's label
     when --bc-anchor-coef > 0.
  3. Compute GAE advantages + Monte Carlo returns.
  4. PPO update with the standard clipped surrogate + entropy + value loss;
     interleave a BC anchor CE term so the policy stays close to teacher
     demonstrations while it learns from reward.
  5. Periodic stochastic eval; final eval at end.

Typical first-run (build on the run06 BC checkpoint):

    python train_ppo.py --load-bc checkpoints/run06/bc_nibbles.pt \\
        --total-timesteps 200000 --bc-anchor-coef 0.5 \\
        --eval-every 25 --run-name ppo01
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import tiny_snake
from train_bc import (
    Agent, select_device, _to_obs_symbolic, evaluate,
    CKPT_DIR,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--load-bc", type=str, default="",
                   help="path to a BC checkpoint to initialise the agent from")
    p.add_argument("--num-steps", type=int, default=128,
                   help="policy steps per rollout (= per update)")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--anneal-lr", action="store_true", default=True)
    p.add_argument("--no-anneal-lr", dest="anneal_lr", action="store_false")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.1)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--ent-coef-final", type=float, default=None,
                   help="if set, linearly anneal ent-coef from --ent-coef to "
                        "this value over total_timesteps.")
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--norm-adv", action="store_true", default=True)
    p.add_argument("--bc-anchor-coef", type=float, default=0.0,
                   help="weight on CE(actor(obs), teacher_action) added to "
                        "every PPO minibatch. Zero = pure PPO.")
    p.add_argument("--bc-anchor-final", type=float, default=None,
                   help="if set, linearly anneal --bc-anchor-coef toward "
                        "this value across training. Setting to 0 lets PPO "
                        "explore past the teacher's distribution once the "
                        "encoder has converged on it.")
    p.add_argument("--encoder-width", type=float, default=1.0)
    p.add_argument("--env-max-steps", type=int, default=500)
    p.add_argument("--eval-eps", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=25,
                   help="run eval every N updates (also at the end)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--run-name", type=str, default="")
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
    vec = tiny_snake.TinySnakeVecEnv(env_kwargs=env_kwargs)
    obs_shape = tiny_snake.TinySnakeVecEnv.OBS_SHAPE  # (12, 12)
    in_ch = tiny_snake.SYM_NUM_TYPES
    num_actions = tiny_snake.NUM_ACTIONS

    agent = Agent(num_actions, in_channels=in_ch,
                  obs_size=obs_shape,
                  width=args.encoder_width).to(device)
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"{tag}agent: in_ch={in_ch}  obs_shape={obs_shape}  "
          f"width={args.encoder_width}  params={n_params:,}", flush=True)
    if args.load_bc:
        ckpt = torch.load(args.load_bc, map_location=device,
                          weights_only=False)
        agent.load_state_dict(ckpt["agent"])
        print(f"{tag}loaded BC ckpt: {args.load_bc}", flush=True)
    optim = Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    # Rollout storage (num_envs=1, N policy steps per update).
    N = args.num_steps
    # Keep obs as uint8 on CPU to save device memory; we one-hot per-batch.
    obs_buf_np = np.zeros((N, *obs_shape), dtype=np.uint8)
    act_buf = torch.zeros(N, dtype=torch.long, device=device)
    logp_buf = torch.zeros(N, device=device)
    rew_buf = torch.zeros(N, device=device)
    done_buf = torch.zeros(N, device=device)
    val_buf = torch.zeros(N, device=device)
    teacher_buf = torch.full((N,), -1, dtype=torch.long, device=device)

    use_anchor = args.bc_anchor_coef > 0 or (
        args.bc_anchor_final is not None and args.bc_anchor_final > 0)

    obs_np = vec.reset()
    obs_t = _to_obs_symbolic(obs_np, device)
    done_t = torch.zeros(1, device=device)

    global_step = 0
    update = 0
    ep_returns: list[float] = []
    ep_return_running = 0.0
    t0 = time.monotonic()

    while global_step < args.total_timesteps:
        update += 1
        frac_remaining = max(0.0, 1.0 - global_step / args.total_timesteps)
        if args.anneal_lr:
            optim.param_groups[0]["lr"] = frac_remaining * args.lr
        if args.ent_coef_final is not None:
            current_ent_coef = (args.ent_coef_final
                + frac_remaining * (args.ent_coef - args.ent_coef_final))
        else:
            current_ent_coef = args.ent_coef
        if args.bc_anchor_final is not None:
            current_anchor = (args.bc_anchor_final
                + frac_remaining * (args.bc_anchor_coef - args.bc_anchor_final))
        else:
            current_anchor = args.bc_anchor_coef

        # ---- rollout --------------------------------------------------------
        for step in range(N):
            obs_buf_np[step] = obs_np[0]
            done_buf[step] = done_t.squeeze()
            if use_anchor:
                teacher_buf[step] = tiny_snake.heuristic_action(
                    vec._env._game)
            with torch.no_grad():
                action, logp, _, value = agent.act(obs_t)
            act_buf[step] = action.squeeze()
            logp_buf[step] = logp.squeeze()
            val_buf[step] = value.squeeze()
            obs_np, rewards, dones, infos = vec.step(action.cpu().numpy())
            global_step += 1
            rew_buf[step] = float(rewards[0])
            ep_return_running += float(rewards[0])
            if dones[0]:
                ep_returns.append(ep_return_running)
                ep_return_running = 0.0
            done_t = torch.tensor([float(dones[0])], device=device)
            obs_t = _to_obs_symbolic(obs_np, device)

        # ---- GAE ------------------------------------------------------------
        with torch.no_grad():
            _, _, _, next_value = agent.act(obs_t)
            advantages = torch.zeros_like(rew_buf)
            lastgae = torch.zeros(1, device=device)
            for t in reversed(range(N)):
                if t == N - 1:
                    next_nonterm = 1.0 - done_t.squeeze()
                    next_v = next_value.squeeze()
                else:
                    next_nonterm = 1.0 - done_buf[t + 1]
                    next_v = val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * next_v * next_nonterm \
                        - val_buf[t]
                lastgae = delta + args.gamma * args.gae_lambda * \
                          next_nonterm * lastgae
                advantages[t] = lastgae
            returns = advantages + val_buf

        # ---- PPO update over flattened batch -------------------------------
        flat_obs_t = _to_obs_symbolic(obs_buf_np, device)  # (N, 5, H, W)
        flat_act = act_buf
        flat_logp = logp_buf
        flat_val = val_buf
        flat_adv = advantages
        flat_ret = returns
        flat_teacher = teacher_buf

        total = N
        b_inds = np.arange(total)
        mb_size = max(1, total // args.num_minibatches)
        approx_kls = []
        clipfracs = []
        pg_losses = []
        v_losses = []
        anchor_losses = []
        ent_vals = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, total, mb_size):
                mb = b_inds[start:start + mb_size]
                _, new_logp, entropy, new_val = agent.act(
                    flat_obs_t[mb], flat_act[mb])
                ratio = (new_logp - flat_logp[mb]).exp()
                mb_adv = flat_adv[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / \
                             (mb_adv.std() + 1e-8)

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * (new_val - flat_ret[mb]).pow(2).mean()
                ent = entropy.mean()
                loss = pg_loss - current_ent_coef * ent + args.vf_coef * v_loss

                anchor_val = 0.0
                if use_anchor and current_anchor > 0:
                    bc_logits = agent.actor(agent.encode(flat_obs_t[mb]))
                    anchor_ce = F.cross_entropy(bc_logits,
                                                 flat_teacher[mb])
                    loss = loss + current_anchor * anchor_ce
                    anchor_val = float(anchor_ce.item())

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(),
                                         args.max_grad_norm)
                optim.step()

                with torch.no_grad():
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef)
                        .float().mean().item())
                    approx_kls.append(
                        ((ratio - 1) - (new_logp - flat_logp[mb]))
                        .mean().item())
                pg_losses.append(float(pg_loss.item()))
                v_losses.append(float(v_loss.item()))
                anchor_losses.append(anchor_val)
                ent_vals.append(float(ent.item()))

        # ---- logging --------------------------------------------------------
        recent_ret = float(np.mean(ep_returns[-20:])) if ep_returns else 0.0
        elapsed = time.monotonic() - t0
        sps = global_step / max(elapsed, 1e-6)
        if update % 5 == 0 or update == 1:
            print(f"{tag}upd {update:4d}  step {global_step:6d}  "
                  f"sps {sps:5.0f}  "
                  f"train_ret {recent_ret:5.2f}  "
                  f"pg {np.mean(pg_losses):+.3f}  "
                  f"v {np.mean(v_losses):.3f}  "
                  f"ent {np.mean(ent_vals):.3f}  "
                  f"anchor {np.mean(anchor_losses):.3f}  "
                  f"kl {np.mean(approx_kls):+.3f}  "
                  f"clip {np.mean(clipfracs):.2f}",
                  flush=True)

        if update % args.eval_every == 0:
            scores = evaluate(agent, vec, args.eval_eps, device,
                              _to_obs_symbolic, greedy=False, tag=tag)
            arr = np.array(scores, dtype=np.int32)
            print(f"{tag}EVAL @ upd {update}: mean {arr.mean():.1f}  "
                  f"max {arr.max()}  min {arr.min()}  "
                  f"median {int(np.median(arr))}", flush=True)

    # ---- final eval + save -------------------------------------------------
    print(f"{tag}final eval ({args.eval_eps} episodes)", flush=True)
    scores = evaluate(agent, vec, args.eval_eps, device,
                      _to_obs_symbolic, greedy=False, tag=tag)
    arr = np.array(scores, dtype=np.int32)
    print(f"{tag}FINAL: mean {arr.mean():.1f}  "
          f"median {int(np.median(arr))}  "
          f"min/max {arr.min()}/{arr.max()}", flush=True)
    out = ckpt_dir / "ppo_tiny.pt"
    torch.save({
        "agent": agent.state_dict(),
        "config": vars(args),
        "eval_scores": scores,
    }, out)
    print(f"{tag}saved {out}", flush=True)


if __name__ == "__main__":
    main()
