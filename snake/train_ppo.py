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
    Agent, select_device, _to_obs_symbolic, _to_obs_dist, evaluate,
    CKPT_DIR,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--load-bc", type=str, default="",
                   help="path to a BC checkpoint to initialise the agent from")
    p.add_argument("--num-steps", type=int, default=128,
                   help="policy steps per rollout (= per update), per env")
    p.add_argument("--num-envs", type=int, default=8,
                   help="parallel in-proc envs. Batch per update is "
                        "num_steps * num_envs (default 128*8 = 1024).")
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
    p.add_argument("--micro-cnn", action="store_true",
                   help="Use the micro CNN (from train_bc) instead of "
                        "NatureCNN. Required when --load-bc points at a "
                        "checkpoint trained with train_bc.py --micro-cnn "
                        "(e.g. run12).")
    p.add_argument("--dist-feature", action="store_true",
                   help="Augment obs with a BFS-distance potential-field "
                        "channel (matches train_bc's --dist-feature). Hands "
                        "the teacher's intermediate computation to PPO so it "
                        "doesn't have to derive long-horizon credit on its own.")
    p.add_argument("--extra-features", action="store_true",
                   help="Use the 11-channel obs (one-hot + distance + "
                        "body-age + heading planes; matches train_bc's "
                        "--extra-features). Supersedes --dist-feature.")
    p.add_argument("--teacher", choices=["bfs", "safe"], default="bfs",
                   help="which scripted teacher labels the BC anchor: "
                        "`bfs` = greedy shortest-path (mean ~27 @ 500 "
                        "steps), `safe` = tail-safe (mean ~39, never dies).")
    p.add_argument("--save-best", action="store_true",
                   help="at every eval, also run a greedy eval and save the "
                        "best-so-far checkpoint to ppo_tiny_best.pt.")
    p.add_argument("--env-max-steps", type=int, default=500)
    p.add_argument("--num-apples", type=int, default=1,
                   help="simultaneous apples on the board; 0 = survival-only "
                        "(no food, no growth).")
    p.add_argument("--canvas", type=int, default=12,
                   help="observation canvas size (square). Constant per "
                        "run — it fixes the CNN input shape.")
    p.add_argument("--field-min", type=int, default=0,
                   help="with --field-max, sample the playable field size "
                        "per episode from [min, max]; walls fill the rest "
                        "of the canvas. 0 = field fills the canvas.")
    p.add_argument("--field-max", type=int, default=0)
    p.add_argument("--start-len-min", type=int, default=0,
                   help="with --start-len-max, sample the starting snake "
                        "length per episode. 0 = fixed length 3.")
    p.add_argument("--start-len-max", type=int, default=0)
    p.add_argument("--reward-eat", type=float, default=1.0,
                   help="Reward per food eaten. Scaling up (e.g. 10) gives a "
                        "much stronger signal for the from-scratch case; "
                        "remember to scale --vf-coef DOWN inversely (value "
                        "loss grows quadratically with return magnitude).")
    p.add_argument("--reward-die", type=float, default=-1.0,
                   help="Reward on death. Default -1 balances against the "
                        "default +1 food. Drop toward 0 if you're scaling "
                        "--reward-eat up and don't want death to dominate.")
    p.add_argument("--reward-step", type=float, default=0.0,
                   help="Reward added on EVERY step (including eat/die). "
                        "Positive -> survivor specialist (living pays); "
                        "negative -> efficiency specialist (short paths pay). "
                        "Typical values: +0.001 or -0.01. Change one and "
                        "expect the policy to converge on a qualitatively "
                        "different behavior — souping across signs is the "
                        "interesting experiment.")
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

    env_kwargs = dict(max_steps=args.env_max_steps, rng_seed=args.seed,
                      reward_eat=args.reward_eat,
                      reward_die=args.reward_die,
                      reward_step=args.reward_step,
                      num_apples=args.num_apples,
                      canvas_rows=args.canvas, canvas_cols=args.canvas)
    if args.field_min > 0:
        env_kwargs["field_range"] = (args.field_min, args.field_max)
    if args.start_len_min > 0:
        env_kwargs["start_length_range"] = (args.start_len_min,
                                            args.start_len_max)
    vec = tiny_snake.TinySnakeVecEnv(
        env_kwargs=env_kwargs, add_distance=args.dist_feature,
        full_features=args.extra_features, num_envs=args.num_envs)
    # Separate single-env for eval so eval never disturbs rollout state.
    # Shaped rewards don't matter here — eval reads info["score"].
    eval_vec = tiny_snake.TinySnakeVecEnv(
        env_kwargs=dict(env_kwargs, rng_seed=args.seed + 777),
        add_distance=args.dist_feature,
        full_features=args.extra_features, num_envs=1)
    obs_grid = vec.OBS_SHAPE  # (canvas, canvas)
    if args.extra_features:
        in_ch = tiny_snake.FULL_OBS_CHANNELS
        obs_buf_shape = (in_ch, *obs_grid)
        obs_buf_dtype = np.float32
        to_obs_fn = _to_obs_dist
    elif args.dist_feature:
        in_ch = tiny_snake.SYM_NUM_TYPES + 1
        obs_buf_shape = (in_ch, *obs_grid)
        obs_buf_dtype = np.float32
        to_obs_fn = _to_obs_dist
    else:
        in_ch = tiny_snake.SYM_NUM_TYPES
        obs_buf_shape = obs_grid
        obs_buf_dtype = np.uint8
        to_obs_fn = _to_obs_symbolic
    obs_shape = obs_grid
    num_actions = tiny_snake.NUM_ACTIONS
    heuristic_fn = (tiny_snake.safe_heuristic_action
                    if args.teacher == "safe"
                    else tiny_snake.heuristic_action)

    agent = Agent(num_actions, in_channels=in_ch,
                  obs_size=obs_shape,
                  width=args.encoder_width,
                  micro=args.micro_cnn).to(device)
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"{tag}agent: in_ch={in_ch}  obs_shape={obs_shape}  "
          f"width={args.encoder_width}  params={n_params:,}", flush=True)
    if args.load_bc:
        ckpt = torch.load(args.load_bc, map_location=device,
                          weights_only=False)
        agent.load_state_dict(ckpt["agent"])
        print(f"{tag}loaded BC ckpt: {args.load_bc}", flush=True)
    optim = Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    # Rollout storage: (N policy steps, E envs) per update.
    N = args.num_steps
    E = args.num_envs
    # Symbolic mode: uint8 cell-type codes (one-hot at boundary). Dist/full
    # modes: already-stacked float32 (C, H, W) tensors.
    obs_buf_np = np.zeros((N, E, *obs_buf_shape), dtype=obs_buf_dtype)
    act_buf = torch.zeros(N, E, dtype=torch.long, device=device)
    logp_buf = torch.zeros(N, E, device=device)
    rew_buf = torch.zeros(N, E, device=device)
    done_buf = torch.zeros(N, E, device=device)
    # truncated_buf[t] = 1 iff step t-1 finished by --env-max-steps rather than
    # death — used to bootstrap V(terminal) instead of zeroing it.
    truncated_buf = torch.zeros(N, E, device=device)
    val_buf = torch.zeros(N, E, device=device)
    teacher_buf = torch.full((N, E), -1, dtype=torch.long, device=device)

    use_anchor = args.bc_anchor_coef > 0 or (
        args.bc_anchor_final is not None and args.bc_anchor_final > 0)

    obs_np = vec.reset()
    obs_t = to_obs_fn(obs_np, device)
    done_t = torch.zeros(E, device=device)
    truncated_t = torch.zeros(E, device=device)

    global_step = 0
    update = 0
    best_eval = -float("inf")
    ep_returns: list[float] = []
    ep_return_running = np.zeros(E, dtype=np.float64)
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
        # Map (step, env) -> terminal obs whenever that step truncated. Used
        # after the rollout to compute V(terminal_obs) for bootstrap.
        trunc_terminal_obs: dict[tuple[int, int], np.ndarray] = {}
        for step in range(N):
            obs_buf_np[step] = obs_np
            done_buf[step] = done_t
            truncated_buf[step] = truncated_t
            if use_anchor:
                for e, game in enumerate(vec.games):
                    teacher_buf[step, e] = heuristic_fn(game)
            with torch.no_grad():
                action, logp, _, value = agent.act(obs_t)
            act_buf[step] = action
            logp_buf[step] = logp
            val_buf[step] = value
            obs_np, rewards, dones, infos = vec.step(action.cpu().numpy())
            global_step += E
            rew_buf[step] = torch.from_numpy(rewards).to(device)
            trunc_flags = np.zeros(E, dtype=np.float32)
            for e in range(E):
                ep_return_running[e] += float(rewards[e])
                if dones[e]:
                    ep_returns.append(float(ep_return_running[e]))
                    ep_return_running[e] = 0.0
                    if infos[e].get("truncated", False):
                        trunc_flags[e] = 1.0
                        trunc_terminal_obs[(step, e)] = infos[e]["terminal_obs"]
            done_t = torch.from_numpy(dones.astype(np.float32)).to(device)
            truncated_t = torch.from_numpy(trunc_flags).to(device)
            obs_t = to_obs_fn(obs_np, device)

        # ---- GAE ------------------------------------------------------------
        with torch.no_grad():
            # Precompute V(terminal_obs) for every truncated (step, env) in
            # this rollout, batched for one forward pass.
            trunc_v_buf = torch.zeros(N, E, device=device)
            if trunc_terminal_obs:
                keys = sorted(trunc_terminal_obs.keys())
                obs_batch = np.stack([trunc_terminal_obs[k] for k in keys])
                _, _, _, vs = agent.act(to_obs_fn(obs_batch, device))
                for (t_k, e_k), v in zip(keys, vs):
                    trunc_v_buf[t_k, e_k] = v

            # Bootstrap value at end of rollout: for envs whose last step
            # truncated, V(terminal_obs) is the right estimate; otherwise
            # V(next obs) (either mid-episode or a fresh reset, both fine
            # since the death mask zeros the reset case).
            _, _, _, next_value = agent.act(obs_t)
            next_value = torch.where(truncated_t > 0.5,
                                     trunc_v_buf[N - 1], next_value)

            advantages = torch.zeros_like(rew_buf)
            lastgae = torch.zeros(E, device=device)
            for t in reversed(range(N)):
                if t == N - 1:
                    d_next = done_t
                    tr_next = truncated_t
                    v_next = next_value
                else:
                    d_next = done_buf[t + 1]
                    tr_next = truncated_buf[t + 1]
                    # If step t truncated, override bootstrap source with
                    # V(terminal_obs) rather than V(reset obs).
                    v_next = torch.where(tr_next > 0.5,
                                         trunc_v_buf[t], val_buf[t + 1])
                # bootstrap_mask is 0 only on real death (done & not truncated);
                # truncation keeps the bootstrap so long horizons aren't punished.
                real_term = d_next * (1.0 - tr_next)
                boot = 1.0 - real_term
                cont = 1.0 - d_next  # GAE λ-decay still stops at any episode end
                delta = rew_buf[t] + args.gamma * v_next * boot - val_buf[t]
                lastgae = delta + args.gamma * args.gae_lambda * cont * lastgae
                advantages[t] = lastgae
            returns = advantages + val_buf

        # ---- PPO update over flattened batch -------------------------------
        flat_obs_t = to_obs_fn(
            obs_buf_np.reshape(N * E, *obs_buf_shape), device)
        flat_act = act_buf.reshape(-1)
        flat_logp = logp_buf.reshape(-1)
        flat_adv = advantages.reshape(-1)
        flat_ret = returns.reshape(-1)
        flat_teacher = teacher_buf.reshape(-1)

        total = N * E
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
            scores = evaluate(agent, eval_vec, args.eval_eps, device,
                              to_obs_fn, greedy=False, tag=tag)
            arr = np.array(scores, dtype=np.int32)
            print(f"{tag}EVAL @ upd {update}: mean {arr.mean():.1f}  "
                  f"max {arr.max()}  min {arr.min()}  "
                  f"median {int(np.median(arr))}", flush=True)
            if args.save_best:
                g_scores = evaluate(agent, eval_vec, args.eval_eps, device,
                                    to_obs_fn, greedy=True,
                                    tag=tag + "greedy-")
                g_mean = float(np.mean(g_scores))
                print(f"{tag}GREEDY EVAL @ upd {update}: mean {g_mean:.1f}",
                      flush=True)
                if g_mean > best_eval:
                    best_eval = g_mean
                    best_out = ckpt_dir / "ppo_tiny_best.pt"
                    torch.save({
                        "agent": agent.state_dict(),
                        "config": vars(args),
                        "eval_scores": g_scores,
                        "update": update,
                        "global_step": global_step,
                    }, best_out)
                    print(f"{tag}new best {g_mean:.1f} -> {best_out}",
                          flush=True)

    # ---- final eval + save -------------------------------------------------
    print(f"{tag}final eval ({args.eval_eps} episodes)", flush=True)
    scores = evaluate(agent, eval_vec, args.eval_eps, device,
                      to_obs_fn, greedy=False, tag=tag)
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
