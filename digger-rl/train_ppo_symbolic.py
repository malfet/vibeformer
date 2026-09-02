"""PPO on the symbolic tile-grid observation.

Same PPO recipe as `train_ppo.py` but the agent sees the (C, 10, 15)
symbolic mask grid from SymbolicDiggerEnv instead of raw 84x84 RGB
frames. Two reasons:

  1. Smaller state space -> sample-efficient learning. The pixel
     PPO runs at exp1/exp4/exp7 plateaued at avg_ret ~50 because
     the policy collapsed onto one action while still in the
     "policy gradient = 0, clip_frac = 0" regime. The symbolic obs
     gives the encoder a one-hot starting point: monsters here,
     emeralds there, dirt there. No representation problem.

  2. SymbolicDiggerEnv already supports potential-based shaping
     via `shaping_coef`. The default reward signal (game score)
     gives +25/+100/etc. on emerald/bag/monster events; the shaping
     adds a small per-step term proportional to the Manhattan-
     distance reduction toward the nearest emerald, which gives PPO
     a non-zero gradient *every* step instead of waiting for sparse
     pickups.

Single env loop (DOSBox is one process per emulator instance).
frame_skip applied via env_step_skipped, same as DAGGER.

Run:
    python train_ppo_symbolic.py --total-timesteps 500000 \\
        --shaping-coef 0.5 --frame-stack 4 \\
        --run-name ppo_sym_v1
"""

from __future__ import annotations

import argparse
import collections
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from tools.game_state import MHEIGHT, MWIDTH
from tools.symbolic_env import SymbolicDiggerEnv
from train_dagger import env_step_skipped
from train_ppo import layer_init, select_device

REPO = Path(__file__).parent.resolve()
CKPT_DIR = REPO / "data" / "checkpoints"


@dataclass
class Config:
    total_timesteps: int = 500_000        # emulator frames (incl. skip)
    learning_rate: float = 2.5e-4
    num_steps: int = 256                  # policy steps per rollout
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    # See train_ppo.py for the rationale; clip_vloss at clip_coef=0.1
    # caps per-update value movement and is mostly neutral when ablated.
    clip_vloss: bool = False
    ent_coef: float = 0.01
    ent_coef_final: float | None = 0.005  # gentler anneal than pixel PPO
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    frame_skip: int = 4
    frame_stack: int = 4
    shaping_coef: float = 0.5
    time_penalty: float = 0.0
    death_penalty: float = 0.0
    # Survival-only objective: pay `survival_reward` per agent step alive
    # and set score_weight=0 so the game's score never reaches the agent.
    survival_reward: float = 0.0
    score_weight: float = 1.0
    egocentric: bool = False
    # Truncate an episode after this many agent steps (0 = never). Needed
    # once survival is the objective: a policy that finds a safe corner
    # would otherwise run one episode for the whole training job.
    max_episode_steps: int = 0
    episodic_life: bool = True
    force_cpu: bool = False
    resume_from: Path | None = None
    resume_prob: float = 1.0
    load_agent: Path | None = None
    save_best: bool = False
    best_metric: str = "score"            # score | length | return
    # Behavioural-cloning warmup + per-PPO-minibatch anchor. If bc_traces
    # is non-empty, pretrain the actor on (obs, action) pairs from those
    # .npz files for bc_epochs epochs *before* PPO starts. While PPO is
    # running, optionally add bc_anchor_coef * CE(pi(s_demo), a_demo)
    # to every minibatch loss so the policy can't drift far from the
    # teacher. Annealing bc_anchor_coef -> bc_anchor_final relaxes the
    # leash as PPO ramps up its own learning signal.
    bc_traces: tuple[str, ...] = ()
    bc_epochs: int = 10
    bc_batch_size: int = 256
    bc_anchor_coef: float = 0.0
    bc_anchor_final: float | None = None
    log_every: int = 1
    save_every: int = 100
    seed: int = 1
    arch: str = "cnn"                     # cnn | iter
    iter_channels: int = 32
    iter_steps: int = 10
    readout: int = 7
    run_name: str = "ppo_sym"


class SymbolicAgent(nn.Module):
    """Shared 3-conv body (same as PolicyNet in train_dagger), separate
    actor / critic heads. ~150k params at BASE_OBS_CHANNELS*frame_stack=24.

    Why share encoder: PPO needs both pi(a|s) and V(s); a shared trunk
    keeps value-error gradients from re-learning the same features.
    """

    def __init__(self, in_channels: int, num_actions: int,
                 h: int = MHEIGHT, w: int = MWIDTH):
        super().__init__()
        self.body = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 3, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, padding=1)), nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * h * w, 128)), nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(128, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self.encode(x)).squeeze(-1)

    def act(self, x: torch.Tensor, action: torch.Tensor | None = None):
        z = self.encode(x)
        logits = self.actor(z)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), \
            self.critic(z).squeeze(-1)


class IterAgent(SymbolicAgent):
    """Weight-tied conv iterator + head-local readout (snake finding #17).

    Two problems with the 3-conv trunk on this board. (a) Its receptive
    field is 7 tiles, but the grid is 15 wide, so nothing in the network
    can relate the digger to an emerald on the far side. Stacking more
    distinct conv layers buys range at a linear cost in parameters.
    (b) `Flatten -> Linear` assigns a private weight to every board
    position, which is exactly the memorisation failure the egocentric
    obs was introduced to remove — and on the 19x29 egocentric canvas
    that layer is 4.5M of the 4.6M parameters.

    Instead: one shared residual block applied `iter_steps` times (an
    unrolled propagation operator, so range grows with *iterations*, not
    parameters) and a readout that only looks at a `readout`x`readout`
    window around the canvas centre — where the digger always is under
    egocentric obs — plus a global mean-pool summary. Range and
    parameter count are decoupled.

    Requires egocentric obs: the readout crop is meaningless otherwise.
    """

    def __init__(self, in_channels: int, num_actions: int,
                 h: int, w: int, ch: int = 32, iters: int = 10,
                 readout: int = 7):
        nn.Module.__init__(self)
        self.iters = iters
        self.stem = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, ch, 3, padding=1)), nn.ReLU())
        # Applied `iters` times with the same weights. GroupNorm inside
        # the block and a small-gain init on its output conv are both
        # load-bearing: with the trunk's default orthogonal(sqrt(2)) init
        # on both convs, ten residual applications compound into a value
        # loss of 5e4 and instant entropy collapse (observed).
        self.block = nn.Sequential(
            layer_init(nn.Conv2d(ch, ch, 3, padding=1)),
            nn.GroupNorm(8, ch), nn.ReLU(),
            layer_init(nn.Conv2d(ch, ch, 3, padding=1), std=0.1))
        r = readout
        self.r0, self.c0 = h // 2 - r // 2, w // 2 - r // 2
        if self.r0 < 0 or self.c0 < 0 or self.r0 + r > h or self.c0 + r > w:
            raise SystemExit(f"--readout {r} does not fit in {h}x{w}")
        self.r = r
        self.proj = nn.Sequential(
            layer_init(nn.Linear(ch * r * r + ch, 128)), nn.ReLU())
        self.actor = layer_init(nn.Linear(128, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        for _ in range(self.iters):
            z = F.relu(z + self.block(z))
        local = z[:, :, self.r0:self.r0 + self.r,
                  self.c0:self.c0 + self.r].flatten(1)
        glob = z.mean(dim=(2, 3))
        return self.proj(torch.cat([local, glob], dim=1))


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-timesteps", type=int, default=Config.total_timesteps)
    p.add_argument("--lr", type=float, default=Config.learning_rate)
    p.add_argument("--num-steps", type=int, default=Config.num_steps)
    p.add_argument("--no-anneal-lr", action="store_true")
    p.add_argument("--gamma", type=float, default=Config.gamma)
    p.add_argument("--ent-coef", type=float, default=Config.ent_coef)
    p.add_argument("--ent-coef-final", type=float, default=Config.ent_coef_final)
    p.add_argument("--clip-coef", type=float, default=Config.clip_coef)
    p.add_argument("--frame-skip", type=int, default=Config.frame_skip)
    p.add_argument("--frame-stack", type=int, default=Config.frame_stack)
    p.add_argument("--shaping-coef", type=float, default=Config.shaping_coef,
                   help="potential-based shaping for emerald-direction. "
                        "Per step, adds shaping_coef * delta_manhattan to "
                        "the reward. 0 disables.")
    p.add_argument("--time-penalty", type=float, default=Config.time_penalty,
                   help="constant subtracted from reward every agent step. "
                        "0 disables. Try 0.01 for ~10 per 1000-step episode.")
    p.add_argument("--death-penalty", type=float, default=Config.death_penalty,
                   help="constant subtracted from reward on any life loss. "
                        "0 disables. Try 100-200.")
    p.add_argument("--survival-reward", type=float,
                   default=Config.survival_reward,
                   help="constant ADDED to reward for every agent step the "
                        "digger stays alive. With --score-weight 0 and "
                        "--shaping-coef 0 this is the pure 'maximise frames "
                        "survived' objective. Try 0.01.")
    p.add_argument("--score-weight", type=float, default=Config.score_weight,
                   help="multiplier on the game's own score delta. 0 makes "
                        "the agent blind to score.")
    p.add_argument("--egocentric", action="store_true",
                   help="re-centre the tile grid on the digger (19x29 canvas "
                        "+ off-board plane). Translation invariance by "
                        "construction; see snake finding #16.")
    p.add_argument("--max-episode-steps", type=int,
                   default=Config.max_episode_steps,
                   help="truncate an episode after this many agent steps "
                        "(0 = never). GAE bootstraps V(s') on truncation, so "
                        "this costs nothing but bounds episode length.")
    p.add_argument("--load-agent", type=Path, default=None,
                   help="initialise weights from a previous symbolic "
                        "checkpoint (BC or PPO) before training.")
    p.add_argument("--save-best", action="store_true",
                   help="checkpoint whenever the 20-episode rolling mean of "
                        "--best-metric hits a new high.")
    p.add_argument("--best-metric", choices=("score", "length", "return"),
                   default=Config.best_metric)
    p.add_argument("--arch", choices=("cnn", "iter"), default=Config.arch,
                   help="cnn = 3-conv trunk + flatten-FC (default); "
                        "iter = weight-tied residual conv iterator with a "
                        "head-local readout (requires --egocentric).")
    p.add_argument("--iter-channels", type=int, default=Config.iter_channels)
    p.add_argument("--iter-steps", type=int, default=Config.iter_steps,
                   help="how many times the tied block is applied. Each "
                        "application extends the propagation range by 2 "
                        "tiles; the board is 15 wide.")
    p.add_argument("--readout", type=int, default=Config.readout,
                   help="side of the digger-centred window the --arch iter "
                        "readout reads.")
    p.add_argument("--bc-traces", type=str, nargs="*", default=(),
                   help="one or more .npz files produced by "
                        "tools/gen_symbolic_trace.py. BC pretrains the actor "
                        "on these before PPO starts.")
    p.add_argument("--bc-epochs", type=int, default=Config.bc_epochs)
    p.add_argument("--bc-batch-size", type=int, default=Config.bc_batch_size)
    p.add_argument("--bc-anchor-coef", type=float, default=Config.bc_anchor_coef,
                   help="if >0, add bc_anchor_coef * CE(pi(s_demo), a_demo) "
                        "to every PPO minibatch loss. Keeps policy near "
                        "teacher during PPO. Try 0.1-1.0.")
    p.add_argument("--bc-anchor-final", type=float,
                   default=Config.bc_anchor_final,
                   help="if set, linearly anneal the BC anchor coef toward "
                        "this value across training.")
    p.add_argument("--episodic-life", default=Config.episodic_life,
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--resume-from", type=Path, default=None,
                   help="Pickle file produced by env.save_state() or by "
                        "run_digger.py --live S. If set, every env.reset() "
                        "is followed by env.load_state(<this file>) so the "
                        "agent always trains from this scenario rather "
                        "than the level-1 attract screen. Useful for "
                        "curriculum-from-checkpoint and for debugging "
                        "specific positions (e.g. 'stuck at right edge').")
    p.add_argument("--resume-prob", type=float, default=Config.resume_prob,
                   help="probability that a reset restores a --resume-from "
                        "scenario instead of starting a normal episode. "
                        "1.0 = always (default); 0.5 mixes ordinary play "
                        "back in to prevent forgetting.")
    p.add_argument("--save-every", type=int, default=Config.save_every)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--run-name", type=str, default=Config.run_name)
    a = p.parse_args()
    return Config(
        total_timesteps=a.total_timesteps,
        learning_rate=a.lr, num_steps=a.num_steps,
        anneal_lr=not a.no_anneal_lr, gamma=a.gamma,
        ent_coef=a.ent_coef, ent_coef_final=a.ent_coef_final,
        clip_coef=a.clip_coef,
        frame_skip=a.frame_skip, frame_stack=a.frame_stack,
        shaping_coef=a.shaping_coef,
        time_penalty=a.time_penalty,
        death_penalty=a.death_penalty,
        survival_reward=a.survival_reward,
        score_weight=a.score_weight,
        egocentric=a.egocentric,
        max_episode_steps=a.max_episode_steps,
        load_agent=a.load_agent,
        save_best=a.save_best,
        best_metric=a.best_metric,
        arch=a.arch, iter_channels=a.iter_channels,
        iter_steps=a.iter_steps, readout=a.readout,
        bc_traces=tuple(a.bc_traces),
        bc_epochs=a.bc_epochs,
        bc_batch_size=a.bc_batch_size,
        bc_anchor_coef=a.bc_anchor_coef,
        bc_anchor_final=a.bc_anchor_final,
        episodic_life=a.episodic_life,
        force_cpu=a.force_cpu,
        resume_from=a.resume_from, resume_prob=a.resume_prob,
        save_every=a.save_every, seed=a.seed,
        run_name=a.run_name,
    )


def main() -> None:
    cfg = parse_args()
    if cfg.resume_from is not None and not cfg.resume_from.exists():
        raise SystemExit(
            f"--resume-from {cfg.resume_from} does not exist")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = select_device(cfg.force_cpu)
    ckpt_dir = CKPT_DIR / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = f"[{cfg.run_name}]"
    print(f"{tag} device={device}  cfg={cfg}", flush=True)

    # SymbolicDiggerEnv.step applies time_penalty per *emulator* frame
    # (env.step == one frame). env_step_skipped below calls env.step
    # frame_skip times per agent step and sums rewards, so a naive
    # --time-penalty 0.01 would actually subtract 0.04 per agent step.
    # Divide here so the CLI value means "per agent step", matching the
    # pixel trainer's semantics.
    # The raw score signal is small relative to shaping_coef=0.5 deltas,
    # and the symbolic best recipe relies on raw +25/+100/+250 events
    # dominating shaping. Pass clip_reward=False explicitly so changing
    # DiggerEnv's default doesn't quietly squash the score signal here.
    env = SymbolicDiggerEnv(max_steps=10**9,
                             clip_reward=False,
                             episodic_life=cfg.episodic_life,
                             frame_stack=cfg.frame_stack,
                             egocentric=cfg.egocentric,
                             shaping_coef=cfg.shaping_coef,
                             time_penalty=cfg.time_penalty / max(cfg.frame_skip, 1),
                             survival_reward=cfg.survival_reward / max(cfg.frame_skip, 1),
                             score_weight=cfg.score_weight,
                             death_penalty=cfg.death_penalty)
    in_ch, obs_h, obs_w = env.obs_shape
    num_actions = env.NUM_ACTIONS

    # If --resume-from was given, load the pickle once. We re-apply it
    # after every env.reset() so each "episode" starts from that
    # scenario rather than the level-1 attract screen.
    resume_states: list[dict] | None = None
    resume_rng = np.random.default_rng(cfg.seed)
    if cfg.resume_from is not None:
        with cfg.resume_from.open("rb") as fh:
            loaded = pickle.load(fh)
        if isinstance(loaded, dict) and "scenarios" in loaded:
            # A scenario *set* from tools/collect_death_states.py: sample
            # one uniformly per reset so the policy sees the whole
            # distribution of dangerous states rather than memorising one.
            resume_states = list(loaded["scenarios"])
            print(f"{tag} --resume-from {cfg.resume_from}: "
                  f"{len(resume_states)} scenarios "
                  f"(lookback {loaded.get('lookback', '?')} steps, "
                  f"collected with {loaded.get('policy', '?')}); "
                  f"each reset() samples one", flush=True)
        else:
            # Tolerate either single-state format -- DiggerEnv.save_state()
            # returns a dict with "core"; run_digger.py --live S returns a
            # dict with "core" too but no python wrapper fields.
            # SymbolicDiggerEnv wraps it with an extra layer
            # ({"env": {...}, "prev_dist": ...}). Lift bare DiggerEnv dicts
            # so load_state finds the expected shape.
            if "core" in loaded and "env" not in loaded:
                loaded = {"env": loaded, "prev_dist": None}
            resume_states = [loaded]
            print(f"{tag} --resume-from {cfg.resume_from} loaded; "
                  f"every reset() will restore this scenario", flush=True)

    def reset_env():
        env.reset()
        # resume_prob < 1 mixes ordinary level-start episodes back in.
        # Training purely on near-death scenarios teaches escape at the
        # cost of forgetting how to play the rest of the game; the mix is
        # the cheap guard against that.
        if resume_states is not None and (
                cfg.resume_prob >= 1.0
                or resume_rng.random() < cfg.resume_prob):
            i = int(resume_rng.integers(len(resume_states)))
            return env.load_state(resume_states[i])
        return env.current_obs()

    if cfg.arch == "iter":
        if not cfg.egocentric:
            raise SystemExit("--arch iter needs --egocentric: its readout "
                             "crop assumes the digger is at the canvas "
                             "centre")
        agent = IterAgent(in_channels=in_ch, num_actions=num_actions,
                          h=obs_h, w=obs_w, ch=cfg.iter_channels,
                          iters=cfg.iter_steps, readout=cfg.readout).to(device)
    else:
        agent = SymbolicAgent(in_channels=in_ch, num_actions=num_actions,
                              h=obs_h, w=obs_w).to(device)
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"{tag} agent params={n_params:,}  arch={cfg.arch}  in_ch={in_ch}  "
          f"H={obs_h} W={obs_w}  egocentric={cfg.egocentric}", flush=True)
    if cfg.load_agent is not None:
        prev = torch.load(cfg.load_agent, weights_only=False,
                          map_location="cpu")
        agent.load_state_dict(prev["agent"])
        print(f"{tag} initialised weights from {cfg.load_agent} "
              f"(step {prev.get('step', '?')})", flush=True)
    optim = Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # ---- BC data + pretrain ------------------------------------------------
    bc_obs: torch.Tensor | None = None
    bc_acts: torch.Tensor | None = None
    if cfg.bc_traces:
        bc_obs_arrs = []
        bc_act_arrs = []
        for tp in cfg.bc_traces:
            d = np.load(tp)
            obs_arr = d["obs"]
            act_arr = d["actions"].astype(np.int64)
            if obs_arr.shape[1:] != (in_ch, obs_h, obs_w):
                raise SystemExit(
                    f"trace {tp} has obs shape {obs_arr.shape[1:]} but "
                    f"trainer expects {(in_ch, obs_h, obs_w)} "
                    f"(frame_stack / egocentric mismatch?). Re-record with "
                    f"--frame-stack {cfg.frame_stack}"
                    + (" --egocentric" if cfg.egocentric else "") + ".")
            bc_obs_arrs.append(obs_arr)
            bc_act_arrs.append(act_arr)
            print(f"{tag} loaded BC trace {tp}: {len(act_arr):,} samples",
                  flush=True)
        bc_obs = torch.from_numpy(np.concatenate(bc_obs_arrs, axis=0)).to(device)
        bc_acts = torch.from_numpy(np.concatenate(bc_act_arrs, axis=0)).to(device)
        M = bc_acts.numel()
        print(f"{tag} BC dataset: {M:,} (obs, action) pairs", flush=True)

        if cfg.bc_epochs > 0:
            rng_bc = np.random.default_rng(cfg.seed)
            for epoch in range(cfg.bc_epochs):
                perm = rng_bc.permutation(M)
                ce_sum = 0.0
                acc_sum = 0.0
                nb = 0
                for start in range(0, M, cfg.bc_batch_size):
                    mb = perm[start:start + cfg.bc_batch_size]
                    mb_t = torch.from_numpy(mb).to(device)
                    logits = agent.actor(agent.encode(bc_obs[mb_t]))
                    ce = F.cross_entropy(logits, bc_acts[mb_t])
                    optim.zero_grad()
                    ce.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(),
                                             cfg.max_grad_norm)
                    optim.step()
                    ce_sum += ce.item()
                    with torch.no_grad():
                        acc_sum += (logits.argmax(-1) == bc_acts[mb_t]
                                    ).float().mean().item()
                    nb += 1
                print(f"{tag}   bc epoch {epoch + 1}/{cfg.bc_epochs}  "
                      f"ce {ce_sum / nb:.3f}  acc {acc_sum / nb:.3f}",
                      flush=True)

    N = cfg.num_steps
    obs_buf = torch.zeros(N, in_ch, obs_h, obs_w, device=device)
    act_buf = torch.zeros(N, dtype=torch.long, device=device)
    logp_buf = torch.zeros(N, device=device)
    rew_buf = torch.zeros(N, device=device)
    done_buf = torch.zeros(N, device=device)
    val_buf = torch.zeros(N, device=device)

    obs_np = reset_env()
    obs_t = torch.from_numpy(obs_np).to(device)
    done_t = torch.zeros((), device=device)

    global_step = 0
    update = 0
    ep_return = 0.0
    ep_length = 0
    ep_returns: collections.deque[float] = collections.deque(maxlen=20)
    # Track the raw game score *at episode end* (info["score"] is cumulative
    # from libretro). Summing per-step info["score_reward"] would
    # under-count because env_step_skipped overwrites info each sub-step.
    ep_scores: collections.deque[float] = collections.deque(maxlen=20)
    ep_lengths: collections.deque[float] = collections.deque(maxlen=20)
    ep_agent_steps = 0          # agent steps in the current episode
    best_metric_value = float("-inf")
    t0 = time.monotonic()

    while global_step < cfg.total_timesteps:
        update += 1
        frac_remaining = max(0.0, 1.0 - global_step / cfg.total_timesteps)
        if cfg.anneal_lr:
            optim.param_groups[0]["lr"] = frac_remaining * cfg.learning_rate
        if cfg.ent_coef_final is not None:
            current_ent_coef = (cfg.ent_coef_final
                                + frac_remaining * (cfg.ent_coef - cfg.ent_coef_final))
        else:
            current_ent_coef = cfg.ent_coef
        if cfg.bc_anchor_final is not None:
            current_anchor = (cfg.bc_anchor_final
                              + frac_remaining * (cfg.bc_anchor_coef - cfg.bc_anchor_final))
        else:
            current_anchor = cfg.bc_anchor_coef
        anchor_active = bc_obs is not None and current_anchor > 0

        # ---- Rollout ----
        for step in range(N):
            obs_buf[step] = obs_t
            done_buf[step] = done_t
            with torch.no_grad():
                action, logp, _, value = agent.act(obs_t.unsqueeze(0))
            a = int(action.item())
            act_buf[step] = action.squeeze(0)
            logp_buf[step] = logp.squeeze(0)
            val_buf[step] = value.squeeze(0)

            next_obs, total_r, done, info = env_step_skipped(
                env, a, cfg.frame_skip)
            global_step += cfg.frame_skip
            ep_return += float(total_r)
            ep_length += cfg.frame_skip
            ep_agent_steps += 1

            truncated = (not done and cfg.max_episode_steps > 0
                         and ep_agent_steps >= cfg.max_episode_steps)
            if truncated:
                # Bootstrapped, not terminal: fold gamma*V(s') into the
                # reward so the cut-off doesn't read as "the world ended
                # here". Without this a survival-reward policy is taught
                # that surviving to the cap is as bad as dying there.
                with torch.no_grad():
                    boot = agent.value(
                        torch.from_numpy(next_obs).to(device).unsqueeze(0))
                total_r = float(total_r) + cfg.gamma * float(boot.item())
            rew_buf[step] = float(total_r)

            if done or truncated:
                final_score = float(info.get("score", 0))
                ep_returns.append(ep_return)
                ep_scores.append(final_score)
                ep_lengths.append(float(ep_length))
                print(f"{tag}   ep_end return={ep_return:.1f} "
                      f"score={int(final_score)} length={ep_length} "
                      f"lives={info.get('lives', 0)}"
                      + ("  [truncated]" if truncated else ""), flush=True)
                ep_return = 0.0
                ep_length = 0
                ep_agent_steps = 0
                next_obs = reset_env()
                done_t = torch.ones((), device=device)
            else:
                done_t = torch.zeros((), device=device)
            obs_t = torch.from_numpy(next_obs).to(device)

        # ---- Diagnostics on the rollout buffer ----
        flat_obs = obs_buf
        flat_act = act_buf
        with torch.no_grad():
            buf_logits = agent.actor(agent.encode(flat_obs))
            buf_log_probs = F.log_softmax(buf_logits, dim=-1)
            buf_probs = buf_log_probs.exp()
            buf_entropy = -(buf_probs * buf_log_probs).sum(-1)
            buf_spread = (buf_logits.max(-1).values - buf_logits.min(-1).values)
        ent_buf_mean = buf_entropy.mean().item()
        ent_buf_p10 = buf_entropy.quantile(0.1).item()
        spread_mean = buf_spread.mean().item()
        act_counts = torch.bincount(flat_act, minlength=num_actions).float()
        act_dist = act_counts / act_counts.sum()
        top_act_idx = int(act_dist.argmax().item())
        top_act_frac = act_dist.max().item()

        # ---- GAE ----
        with torch.no_grad():
            next_value = agent.value(obs_t.unsqueeze(0)).squeeze(0)
            advantages = torch.zeros_like(rew_buf)
            lastgae = torch.zeros((), device=device)
            for t in reversed(range(N)):
                if t == N - 1:
                    next_nonterminal = 1.0 - done_t
                    next_v = next_value
                else:
                    next_nonterminal = 1.0 - done_buf[t + 1]
                    next_v = val_buf[t + 1]
                delta = rew_buf[t] + cfg.gamma * next_v * next_nonterminal - val_buf[t]
                lastgae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * lastgae
                advantages[t] = lastgae
            returns = advantages + val_buf

        # ---- PPO update ----
        flat_logp = logp_buf
        flat_val = val_buf
        flat_adv = advantages
        flat_ret = returns
        b_inds = np.arange(N)
        clipfracs = []
        approx_kls = []
        for _ in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            mb_size = N // cfg.num_minibatches
            for start in range(0, N, mb_size):
                mb = b_inds[start:start + mb_size]
                _, new_logp, entropy, new_val = agent.act(
                    flat_obs[mb], flat_act[mb])
                ratio = (new_logp - flat_logp[mb]).exp()
                mb_adv = flat_adv[mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                if cfg.clip_vloss:
                    v_unclipped = (new_val - flat_ret[mb]) ** 2
                    v_clipped = flat_val[mb] + torch.clamp(
                        new_val - flat_val[mb], -cfg.clip_coef, cfg.clip_coef)
                    v_clipped_loss = (v_clipped - flat_ret[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped_loss).mean()
                else:
                    v_loss = 0.5 * ((new_val - flat_ret[mb]) ** 2).mean()
                ent = entropy.mean()
                loss = pg_loss - current_ent_coef * ent + cfg.vf_coef * v_loss
                if anchor_active:
                    bc_M = bc_acts.numel()
                    bc_idx = torch.randint(
                        bc_M, (min(cfg.bc_batch_size, bc_M),), device=device)
                    bc_logits = agent.actor(agent.encode(bc_obs[bc_idx]))
                    anchor_ce = F.cross_entropy(bc_logits, bc_acts[bc_idx])
                    loss = loss + current_anchor * anchor_ce
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optim.step()
                with torch.no_grad():
                    clipfracs.append(
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())
                    approx_kls.append(
                        ((ratio - 1) - (new_logp - flat_logp[mb])).mean().item())

        # ---- Log ----
        if update % cfg.log_every == 0:
            avg_ret = (sum(ep_returns) / len(ep_returns)) if ep_returns else float("nan")
            avg_score = (sum(ep_scores) / len(ep_scores)) if ep_scores else float("nan")
            avg_len = (sum(ep_lengths) / len(ep_lengths)) if ep_lengths else float("nan")
            elapsed = time.monotonic() - t0
            sps = global_step / max(elapsed, 1e-6)
            lr_now = optim.param_groups[0]["lr"]
            print(f"{tag} upd {update:>4d}  step {global_step:>8d}  "
                  f"pg {pg_loss.item():+.3f}  v {v_loss.item():.3f}  "
                  f"ent {ent.item():.3f}  clip {np.mean(clipfracs):.3f}  "
                  f"kl {np.mean(approx_kls):+.4f}  "
                  f"avg_ret {avg_ret:.1f}  score {avg_score:.0f}  "
                  f"len {avg_len:.0f}  "
                  f"lr {lr_now:.2e}  entc {current_ent_coef:.3f}  "
                  + (f"anc {current_anchor:.2f}  " if anchor_active else "")
                  + f"sps {sps:.0f}", flush=True)
            print(f"      buf: ent_mean {ent_buf_mean:.2f}  "
                  f"ent_p10 {ent_buf_p10:.2f}  spread {spread_mean:5.2f}  "
                  f"top_a {top_act_idx}@{top_act_frac:.0%}", flush=True)

        # ---- Best-so-far checkpoint ----
        # Rolling 20-episode mean, not a dedicated eval: an eval pass costs
        # another DOSBox run. Requires a full window so early luck on 2-3
        # episodes can't claim "best".
        if cfg.save_best and len(ep_lengths) == ep_lengths.maxlen:
            pool = {"score": ep_scores, "length": ep_lengths,
                    "return": ep_returns}[cfg.best_metric]
            current = sum(pool) / len(pool)
            if current > best_metric_value:
                best_metric_value = current
                torch.save({"agent": agent.state_dict(), "step": global_step,
                            "config": cfg.__dict__,
                            "best_metric": cfg.best_metric,
                            "best_value": current},
                           ckpt_dir / "ppo_sym_best.pt")
                print(f"{tag}   new best {cfg.best_metric} "
                      f"{current:.1f} -> ppo_sym_best.pt", flush=True)

        if cfg.save_every and update % cfg.save_every == 0:
            ckpt = ckpt_dir / f"ppo_sym_step{global_step:08d}.pt"
            torch.save({"agent": agent.state_dict(), "step": global_step,
                        "config": cfg.__dict__}, ckpt)
            print(f"{tag}   saved {ckpt}", flush=True)

    env.close()
    final = ckpt_dir / "ppo_sym_final.pt"
    torch.save({"agent": agent.state_dict(), "step": global_step,
                "config": cfg.__dict__}, final)
    print(f"{tag} done. final checkpoint {final}", flush=True)


if __name__ == "__main__":
    main()
