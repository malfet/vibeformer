"""PPO training for Digger, CleanRL-style: single file, no gymnasium.

Pipeline:
  - DiggerEnv (raw 640x400 RGBA frames) -- one env per process due to the
    LibretroCore singleton constraint
  - Per policy step: hold action for frame_skip=4 emulator frames, sum reward
  - Preprocess: drop alpha, grayscale, area-resize to 84x84, float32 in [0,1]
  - Frame-stack last 4 preprocessed frames -> (4, 84, 84) policy input
  - NatureCNN encoder + linear actor head (Discrete 6) + linear critic head
  - Standard PPO: GAE, clipped policy ratio, clipped value loss, entropy bonus,
    advantage normalization, orthogonal init, linear LR anneal

Defaults aim at a sane first run on Apple Silicon MPS in ~10-30 min wall clock.
Tune via --total-timesteps and friends.
"""

from __future__ import annotations

import argparse
import collections
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from digger_env import DiggerEnv, DiggerVecEnv

REPO = Path(__file__).parent.resolve()
CKPT_DIR = REPO / "data" / "checkpoints"


@dataclass
class Config:
    total_timesteps: int = 1_000_000  # emulator frames (NOT policy steps)
    learning_rate: float = 2.5e-4
    num_steps: int = 128              # policy steps per rollout
    num_envs: int = 1                 # parallel envs (subprocess if >1)
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    frame_skip: int = 4
    frame_stack: int = 4
    obs_size: int = 84
    encoder_width: int = 1            # NatureCNN width multiplier; 2 = ~4x params
    clip_reward: bool = False
    episodic_life: bool = False
    death_penalty: float = 0.0
    ent_coef_final: float | None = None  # if set, linearly anneal ent_coef -> this
    # Behavioural-cloning warmup. If bc_traces is non-empty, pretrain the
    # actor (cross-entropy on human-recorded actions) for bc_epochs *before*
    # PPO starts; with --bc-value, also fit the critic on MC returns from the
    # traces so the random V doesn't immediately wipe out the BC policy.
    bc_traces: tuple[str, ...] = ()
    bc_epochs: int = 5
    bc_batch_size: int = 64
    bc_value: bool = False
    # BC anchor: per PPO minibatch, add bc_anchor_coef * CE(pi(s_demo), a_demo)
    # to the loss. Keeps the policy close to demonstrations while PPO ramps up.
    # Optionally annealed linearly toward bc_anchor_final.
    bc_anchor_coef: float = 0.0
    bc_anchor_final: float | None = None
    log_every: int = 1                # updates
    save_every: int = 50              # updates
    device: str = "auto"              # resolved by select_device()
    seed: int = 1
    run_name: str = ""                # if set, checkpoints go to <CKPT>/run_name/


def select_device(force_cpu: bool = False) -> torch.device:
    """Pick the best available accelerator, with a CPU override.

    Delegates to torch.accelerator so we don't enumerate device names here --
    whatever backend (CUDA, MPS, XPU, ...) the torch build supports is what
    we use.
    """
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


class Agent(nn.Module):
    """NatureCNN trunk + linear actor / critic heads.

    `width` linearly scales conv channels (32->32w, 64->64w) and the FC
    bottleneck (512->512w). width=1 is the classic NatureCNN (~1.7M params);
    width=2 is ~6M params and substantially more representation capacity.
    """

    def __init__(self, num_actions: int, in_channels: int = 4, width: int = 1):
        super().__init__()
        c1, c2, c3, fc = 32 * width, 64 * width, 64 * width, 512 * width
        self.width = width
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, c1, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(c1, c2, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(c2, c3, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(c3 * 7 * 7, fc)), nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(fc, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(fc, 1), std=1.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self.encode(x)).squeeze(-1)

    def act(self, x: torch.Tensor, action: torch.Tensor | None = None):
        z = self.encode(x)
        logits = self.actor(z)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(z).squeeze(-1)


def _decimate_trace(d: dict, factor: int) -> dict:
    """Group every `factor` consecutive samples in a trace into one.

    Used to convert a low-skip trace (e.g. recorded at frame_skip=4) into a
    higher-skip representation (e.g. trainer running at frame_skip=8).
    Keeps the LAST frame/score/lives of each group (matching the env's
    skip-then-snapshot semantics), SUMS rewards across the group, and ORs
    resets so a reset that landed inside the group is preserved.
    """
    n = len(d["frames"])
    trimmed = n - (n % factor)
    frames = d["frames"][:trimmed].reshape(
        -1, factor, *d["frames"].shape[1:])[:, -1]
    actions = d["actions"][:trimmed].reshape(-1, factor)[:, -1]
    raw_rewards = d["raw_rewards"][:trimmed].reshape(-1, factor).sum(axis=1)
    scores = d["scores"][:trimmed].reshape(-1, factor)[:, -1]
    lives = d["lives"][:trimmed].reshape(-1, factor)[:, -1]
    resets = d["resets"][:trimmed].reshape(-1, factor).any(axis=1)
    return {
        "frames":      frames,
        "actions":     actions,
        "raw_rewards": raw_rewards,
        "scores":      scores,
        "lives":       lives,
        "resets":      resets,
        "frame_skip":  int(d["frame_skip"]) * factor,
        "obs_size":    int(d["obs_size"]),
    }


def _load_traces(paths: tuple[str, ...], cfg: Config) -> list[dict]:
    """Load .npz playtraces, decimating skip-mismatched ones at load time."""
    out: list[dict] = []
    for p in paths:
        raw = np.load(p)
        fs = int(raw["frame_skip"])
        os_ = int(raw["obs_size"])
        if os_ != cfg.obs_size:
            raise ValueError(
                f"{p}: trace obs_size={os_} but trainer uses {cfg.obs_size}")
        # Traces predating the R-reset feature won't carry a `resets` array;
        # default to all-False (no resets recorded).
        if "resets" in raw.files:
            resets = raw["resets"].astype(bool)
        else:
            resets = np.zeros(len(raw["actions"]), dtype=bool)
        d = {
            "frames":      raw["frames"],
            "actions":     raw["actions"],
            "raw_rewards": raw["raw_rewards"],
            "scores":      raw["scores"],
            "lives":       raw["lives"],
            "resets":      resets,
            "frame_skip":  fs,
            "obs_size":    os_,
        }
        if fs != cfg.frame_skip:
            if cfg.frame_skip % fs != 0 or cfg.frame_skip < fs:
                raise ValueError(
                    f"{p}: trace frame_skip={fs} not an integer divisor of "
                    f"trainer frame_skip={cfg.frame_skip}; re-record")
            factor = cfg.frame_skip // fs
            before = len(d["frames"])
            d = _decimate_trace(d, factor)
            print(f"  {p}: decimated x{factor} (frame_skip {fs}->{cfg.frame_skip}), "
                  f"{before}->{len(d['frames'])} samples", flush=True)
        d["path"] = p
        out.append(d)
    return out


def _trace_returns(rewards: np.ndarray, lives: np.ndarray,
                   resets: np.ndarray, gamma: float,
                   episodic_life: bool) -> np.ndarray:
    """Compute discounted returns G_t, resetting the bootstrap at episode ends.

    Reset bootstrap when (a) lives drops from >0 to 0 (true game-over),
    (b) `episodic_life` and any life loss, or (c) `resets[t+1]` flags a
    user-triggered emulator reset between t and t+1.
    """
    n = len(rewards)
    returns = np.zeros(n, dtype=np.float32)
    running = 0.0
    for t in range(n - 1, -1, -1):
        if t < n - 1:
            ended = (int(lives[t + 1]) == 0 and int(lives[t]) > 0) or (
                episodic_life and int(lives[t + 1]) < int(lives[t])) or (
                bool(resets[t + 1]))
            if ended:
                running = 0.0
        running = float(rewards[t]) + gamma * running
        returns[t] = running
    return returns


def _build_bc_index(traces: list[dict], cfg: Config
                    ) -> tuple[list[np.ndarray], list[np.ndarray],
                               list[np.ndarray], list[tuple[int, int]]]:
    """Prepare per-trace arrays and a flat (trace_idx, t) sample index.

    Stored in uint8 to keep memory modest; the minibatch loader divides by
    255 on the fly. The first (frame_stack - 1) timesteps of each trace are
    skipped because we can't build a full stack for them.
    """
    per_frames: list[np.ndarray] = []
    per_actions: list[np.ndarray] = []
    per_returns: list[np.ndarray] = []
    index: list[tuple[int, int]] = []

    for i, tr in enumerate(traces):
        frames = tr["frames"]
        actions = tr["actions"].astype(np.int64)
        rewards = tr["raw_rewards"].astype(np.float32).copy()
        lives = tr["lives"]
        resets = tr["resets"]

        if cfg.clip_reward:
            rewards = np.sign(rewards).astype(np.float32)
        if cfg.death_penalty != 0.0:
            died = np.zeros_like(rewards)
            died[1:] = (lives[1:] < lives[:-1]).astype(np.float32)
            rewards = rewards - cfg.death_penalty * died

        returns = _trace_returns(rewards, lives, resets, cfg.gamma,
                                 cfg.episodic_life)

        per_frames.append(frames)
        per_actions.append(actions)
        per_returns.append(returns)
        n = len(frames)
        # Skip stacks that straddle an R-reset: the stack at index t covers
        # frames [t-k+1..t], so a reset on any of (t-k+2)..t would mean
        # pre-reset and post-reset frames are mixed in the same input window.
        # resets[k] = True means a reset happened between sample k-1 and k.
        skipped = 0
        for t in range(cfg.frame_stack - 1, n):
            lo = max(0, t - cfg.frame_stack + 2)
            if bool(resets[lo:t + 1].any()):
                skipped += 1
                continue
            index.append((i, t))
        if skipped:
            print(f"  bc: skipped {skipped} cross-reset stacks in trace {i}",
                  flush=True)

    return per_frames, per_actions, per_returns, index


def _bc_minibatch(per_frames: list[np.ndarray], per_actions: list[np.ndarray],
                  per_returns: list[np.ndarray] | None,
                  index: list[tuple[int, int]], idxs: np.ndarray,
                  cfg: Config, device: torch.device,
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Materialise a minibatch of (stacked_obs, action, return) from a BC index."""
    bs = len(idxs)
    obs = np.empty((bs, cfg.frame_stack, cfg.obs_size, cfg.obs_size),
                   dtype=np.uint8)
    acts = np.empty(bs, dtype=np.int64)
    rets = np.empty(bs, dtype=np.float32) if per_returns is not None else None
    for j, ix in enumerate(idxs):
        ti, t = index[ix]
        obs[j] = per_frames[ti][t - cfg.frame_stack + 1: t + 1]
        acts[j] = per_actions[ti][t]
        if rets is not None:
            rets[j] = per_returns[ti][t]
    obs_t = torch.from_numpy(obs).to(device).float().mul_(1.0 / 255.0)
    acts_t = torch.from_numpy(acts).to(device)
    rets_t = torch.from_numpy(rets).to(device) if rets is not None else None
    return obs_t, acts_t, rets_t


def bc_pretrain(agent: "Agent", optim: Adam, bc_data: tuple,
                cfg: Config, device: torch.device) -> None:
    """Cross-entropy on (obs, action). Optionally MSE on critic vs MC return.

    Shares the PPO optimiser so Adam state carries over into the main loop.
    """
    per_frames, per_actions, per_returns, index = bc_data
    M = len(index)
    if M == 0:
        print("bc: no usable samples in traces; skipping", flush=True)
        return

    # Returns are in raw point units (~50..8000 in Digger). Regressing the
    # critic on them with the default vf_coef=0.5 produces a v_loss that
    # dwarfs the cross-entropy by ~4 orders of magnitude, so Adam learns the
    # critic and basically ignores the actor. Train V against standardised
    # targets, then absorb (mean, std) into the critic head so PPO sees a
    # raw-scale predictor.
    if cfg.bc_value:
        sampled_at = np.array([per_returns[i][t] for i, t in index],
                              dtype=np.float32)
        ret_mean = float(sampled_at.mean())
        ret_std = float(sampled_at.std()) + 1e-6
        norm_returns = [(r - ret_mean) / ret_std for r in per_returns]
    else:
        ret_mean = 0.0
        ret_std = 1.0
        norm_returns = per_returns

    extra = ""
    if cfg.bc_value:
        extra = f", +value head (returns N({ret_mean:.1f}, {ret_std:.1f}))"
    print(f"bc: {M} samples ({cfg.bc_epochs} epochs, "
          f"batch={cfg.bc_batch_size}{extra})", flush=True)

    rng = np.random.default_rng(cfg.seed)

    for epoch in range(cfg.bc_epochs):
        perm = rng.permutation(M)
        ce_sum = 0.0
        v_sum = 0.0
        acc_sum = 0.0
        nb = 0
        for start in range(0, M, cfg.bc_batch_size):
            mb = perm[start:start + cfg.bc_batch_size]
            obs_t, acts_t, rets_t = _bc_minibatch(
                per_frames, per_actions, norm_returns, index, mb, cfg, device)
            z = agent.encode(obs_t)
            logits = agent.actor(z)
            ce = F.cross_entropy(logits, acts_t)
            loss = ce
            v_val = 0.0
            if cfg.bc_value:
                v = agent.critic(z).squeeze(-1)
                v_loss = 0.5 * (v - rets_t).pow(2).mean()
                loss = loss + cfg.vf_coef * v_loss
                v_val = v_loss.item()
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optim.step()
            with torch.no_grad():
                acc_sum += (logits.argmax(-1) == acts_t).float().mean().item()
            ce_sum += ce.item()
            v_sum += v_val
            nb += 1
        tail = f"  v_mse_norm {v_sum / nb:.3f}" if cfg.bc_value else ""
        print(f"  bc epoch {epoch + 1}/{cfg.bc_epochs}  "
              f"ce {ce_sum / nb:.3f}  acc {acc_sum / nb:.3f}{tail}",
              flush=True)

    # Absorb the (mean, std) normalisation into the critic's last linear so
    # V outputs raw-scale returns from here on. V(s) = w·z + b learned to
    # predict (G - mean)/std, so de-normalised V is std*(w·z + b) + mean.
    if cfg.bc_value:
        with torch.no_grad():
            agent.critic.weight.mul_(ret_std)
            agent.critic.bias.mul_(ret_std)
            agent.critic.bias.add_(ret_mean)
        print(f"  bc: rescaled critic head to raw return units "
              f"(mul {ret_std:.2f}, add {ret_mean:.2f})", flush=True)


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=Config.total_timesteps)
    p.add_argument("--force-cpu", action="store_true",
                   help="force CPU even if a CUDA/MPS accelerator is available")
    p.add_argument("--lr", type=float, default=Config.learning_rate)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--num-steps", type=int, default=Config.num_steps)
    p.add_argument("--num-envs", type=int, default=Config.num_envs,
                   help="parallel envs (subprocess workers if >1)")
    p.add_argument("--frame-skip", type=int, default=Config.frame_skip)
    p.add_argument("--frame-stack", type=int, default=Config.frame_stack)
    p.add_argument("--save-every", type=int, default=Config.save_every)
    p.add_argument("--no-anneal-lr", action="store_true")
    p.add_argument("--clip-reward", action="store_true",
                   help="report sign(reward) to the agent instead of raw score delta")
    p.add_argument("--episodic-life", action="store_true",
                   help="emit done=True on every life loss, not just game over")
    p.add_argument("--ent-coef", type=float, default=Config.ent_coef,
                   help="entropy bonus weight (default 0.01; bump to 0.05-0.1 to fight collapse)")
    p.add_argument("--ent-coef-final", type=float, default=None,
                   help="linearly anneal entropy bonus toward this value over training")
    p.add_argument("--death-penalty", type=float, default=Config.death_penalty,
                   help="subtract this from reward on every life loss (default 0)")
    p.add_argument("--bc-traces", type=str, nargs="+", default=[],
                   help="one or more .npz playtraces from run_digger "
                        "--record-playtrace; if set, BC-pretrain the actor "
                        "before PPO starts")
    p.add_argument("--bc-epochs", type=int, default=Config.bc_epochs,
                   help="behavioural-cloning epochs over the traces (default 5)")
    p.add_argument("--bc-batch-size", type=int, default=Config.bc_batch_size)
    p.add_argument("--bc-value", action="store_true",
                   help="also fit the critic on MC returns from the traces "
                        "during BC pretrain (recommended; otherwise a random "
                        "V can wipe out the BC policy on the first PPO update)")
    p.add_argument("--bc-anchor-coef", type=float,
                   default=Config.bc_anchor_coef,
                   help="anchor weight for an extra CE(pi(s_demo), a_demo) "
                        "loss added to every PPO minibatch (default 0 = off). "
                        "Try 0.1-1.0 to keep PPO from drifting off BC.")
    p.add_argument("--bc-anchor-final", type=float, default=None,
                   help="if set, linearly anneal the BC anchor coef toward "
                        "this value across training")
    p.add_argument("--encoder-width", type=int, default=Config.encoder_width,
                   help="NatureCNN channel/FC width multiplier (default 1 = "
                        "~1.7M params; 2 = ~6M params for richer encoder)")
    p.add_argument("--run-name", type=str, default="",
                   help="subdir under data/checkpoints/ for this run; needed "
                        "to run several train_ppo invocations in parallel "
                        "without checkpoint filename collisions")
    a = p.parse_args()
    return Config(
        total_timesteps=a.total_timesteps,
        device=str(select_device(a.force_cpu)),
        learning_rate=a.lr, seed=a.seed,
        num_steps=a.num_steps, num_envs=a.num_envs,
        frame_skip=a.frame_skip, frame_stack=a.frame_stack,
        save_every=a.save_every, anneal_lr=not a.no_anneal_lr,
        clip_reward=a.clip_reward, episodic_life=a.episodic_life,
        ent_coef=a.ent_coef, ent_coef_final=a.ent_coef_final,
        death_penalty=a.death_penalty,
        bc_traces=tuple(a.bc_traces), bc_epochs=a.bc_epochs,
        bc_batch_size=a.bc_batch_size, bc_value=a.bc_value,
        bc_anchor_coef=a.bc_anchor_coef, bc_anchor_final=a.bc_anchor_final,
        encoder_width=a.encoder_width, run_name=a.run_name,
    )


def main() -> None:
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    ckpt_dir = CKPT_DIR / cfg.run_name if cfg.run_name else CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tag = f"[{cfg.run_name}] " if cfg.run_name else ""
    print(f"{tag}device={device} cfg={cfg}", flush=True)

    env_kwargs = dict(max_steps=10**9, clip_reward=cfg.clip_reward,
                      episodic_life=cfg.episodic_life,
                      death_penalty=cfg.death_penalty)
    vec = DiggerVecEnv(num_envs=cfg.num_envs, frame_skip=cfg.frame_skip,
                       frame_stack=cfg.frame_stack, obs_size=cfg.obs_size,
                       env_kwargs=env_kwargs)
    agent = Agent(num_actions=DiggerEnv.NUM_ACTIONS,
                  in_channels=cfg.frame_stack,
                  width=cfg.encoder_width).to(device)
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"{tag}agent: width={cfg.encoder_width}, params={n_params:,}  "
          f"num_envs={cfg.num_envs}", flush=True)
    optim = Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    bc_data = None
    if cfg.bc_traces:
        traces = _load_traces(cfg.bc_traces, cfg)
        bc_data = _build_bc_index(traces, cfg)
        if cfg.bc_epochs > 0:
            bc_pretrain(agent, optim, bc_data, cfg, device)
    anchor_active = bc_data is not None and cfg.bc_anchor_coef > 0
    anchor_rng = np.random.default_rng(cfg.seed + 17)

    # Rollout storage: (N, B, ...) so we can stack per-env trajectories,
    # then flatten to (N*B, ...) for minibatch sampling.
    N, B = cfg.num_steps, cfg.num_envs
    obs_buf = torch.zeros(N, B, cfg.frame_stack, cfg.obs_size, cfg.obs_size, device=device)
    act_buf = torch.zeros(N, B, dtype=torch.long, device=device)
    logp_buf = torch.zeros(N, B, device=device)
    rew_buf = torch.zeros(N, B, device=device)
    done_buf = torch.zeros(N, B, device=device)
    val_buf = torch.zeros(N, B, device=device)

    obs_np = vec.reset()  # (B, k, H, W) uint8
    obs_t = torch.from_numpy(obs_np).to(device).float().mul_(1.0 / 255.0)
    done_t = torch.zeros(B, device=device)

    global_step = 0
    update = 0
    ep_returns_per_env = np.zeros(B, dtype=np.float32)
    ep_lengths_per_env = np.zeros(B, dtype=np.int64)
    ep_returns: collections.deque[float] = collections.deque(maxlen=20)
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

        # ---- Rollout ----
        for step in range(N):
            obs_buf[step] = obs_t
            done_buf[step] = done_t
            with torch.no_grad():
                action, logp, _, value = agent.act(obs_t)
            act_buf[step] = action
            logp_buf[step] = logp
            val_buf[step] = value

            actions_np = action.cpu().numpy()
            obs_np, rewards, dones, infos = vec.step(actions_np)
            global_step += cfg.frame_skip * B
            rew_buf[step] = torch.from_numpy(rewards).to(device)
            ep_returns_per_env += rewards
            ep_lengths_per_env += cfg.frame_skip

            for i, d in enumerate(dones):
                if d:
                    ep_returns.append(float(ep_returns_per_env[i]))
                    print(f"{tag}  ep_end env={i} return={ep_returns_per_env[i]:.0f} "
                          f"length={ep_lengths_per_env[i]} "
                          f"score={infos[i].get('score', 0)} "
                          f"lives={infos[i].get('lives', 0)}", flush=True)
                    ep_returns_per_env[i] = 0.0
                    ep_lengths_per_env[i] = 0

            done_t = torch.from_numpy(dones.astype(np.float32)).to(device)
            obs_t = torch.from_numpy(obs_np).to(device).float().mul_(1.0 / 255.0)

        # ---- Pre-update diagnostics on the full rollout (N*B samples) ----
        flat_obs = obs_buf.reshape(-1, *obs_buf.shape[2:])
        flat_act = act_buf.reshape(-1)
        with torch.no_grad():
            buf_logits = agent.actor(agent.encode(flat_obs))
            buf_log_probs = F.log_softmax(buf_logits, dim=-1)
            buf_probs = buf_log_probs.exp()
            buf_entropy = -(buf_probs * buf_log_probs).sum(-1)
            buf_spread = (buf_logits.max(-1).values - buf_logits.min(-1).values)

        ent_buf_mean = buf_entropy.mean().item()
        ent_buf_p10 = buf_entropy.quantile(0.1).item()
        spread_mean = buf_spread.mean().item()

        act_counts = torch.bincount(flat_act, minlength=DiggerEnv.NUM_ACTIONS).float()
        act_dist = act_counts / act_counts.sum()
        top_act_idx = int(act_dist.argmax().item())
        top_act_frac = act_dist.max().item()

        collapse_signals = []
        if top_act_frac > 0.85:
            collapse_signals.append(f"action {top_act_idx} = {top_act_frac:.0%}")
        if ent_buf_p10 < 0.05:
            collapse_signals.append(f"ent_p10 = {ent_buf_p10:.3f}")
        if spread_mean > 15.0:
            collapse_signals.append(f"logit_spread = {spread_mean:.1f}")

        # ---- GAE per env ----
        with torch.no_grad():
            next_value = agent.value(obs_t)
            advantages = torch.zeros_like(rew_buf)
            lastgae = torch.zeros(B, device=device)
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

        # ---- PPO update over flattened (N*B, ...) samples ----
        flat_logp = logp_buf.reshape(-1)
        flat_val = val_buf.reshape(-1)
        flat_adv = advantages.reshape(-1)
        flat_ret = returns.reshape(-1)
        total = N * B
        b_inds = np.arange(total)
        clipfracs = []
        approx_kls = []
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            mb_size = total // cfg.num_minibatches
            for start in range(0, total, mb_size):
                mb = b_inds[start:start + mb_size]
                _, new_logp, entropy, new_val = agent.act(flat_obs[mb], flat_act[mb])
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

                if anchor_active and current_anchor > 0:
                    bc_M = len(bc_data[3])
                    bc_idxs = anchor_rng.choice(
                        bc_M, size=min(cfg.bc_batch_size, bc_M), replace=False)
                    bc_obs, bc_acts, _ = _bc_minibatch(
                        bc_data[0], bc_data[1], None, bc_data[3],
                        bc_idxs, cfg, device)
                    bc_logits = agent.actor(agent.encode(bc_obs))
                    anchor_ce = F.cross_entropy(bc_logits, bc_acts)
                    loss = loss + current_anchor * anchor_ce

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optim.step()

                with torch.no_grad():
                    clipfracs.append(
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())
                    approx_kls.append(((ratio - 1) - (new_logp - flat_logp[mb])).mean().item())

        # ---- Logging ----
        if update % cfg.log_every == 0:
            avg_ret = (sum(ep_returns) / len(ep_returns)) if ep_returns else float("nan")
            elapsed = time.monotonic() - t0
            sps = global_step / max(elapsed, 1e-6)
            lr_now = optim.param_groups[0]["lr"]
            anchor_tag = f"  anchor {current_anchor:.3f}" if anchor_active else ""
            print(f"{tag}upd {update:>4d}  step {global_step:>8d}  "
                  f"pg {pg_loss.item():+.3f}  v {v_loss.item():.3f}  "
                  f"ent {ent.item():.3f}  clip {np.mean(clipfracs):.3f}  "
                  f"avg_ret {avg_ret:.1f}  lr {lr_now:.2e}  "
                  f"entc {current_ent_coef:.3f}{anchor_tag}  sps {sps:.0f}",
                  flush=True)
            print(f"     buf: ent_mean {ent_buf_mean:.2f}  ent_p10 {ent_buf_p10:.2f}  "
                  f"spread {spread_mean:5.2f}  top_a {top_act_idx}@{top_act_frac:.0%}",
                  flush=True)
            if collapse_signals:
                print(f"     ** collapse-warning: {' | '.join(collapse_signals)}",
                      flush=True)

        if cfg.save_every and update % cfg.save_every == 0:
            ckpt = ckpt_dir / f"ppo_digger_step{global_step:08d}.pt"
            torch.save({"agent": agent.state_dict(), "step": global_step,
                        "config": cfg.__dict__}, ckpt)
            print(f"{tag}  saved {ckpt}", flush=True)

    vec.close()
    final = ckpt_dir / "ppo_digger_final.pt"
    torch.save({"agent": agent.state_dict(), "step": global_step,
                "config": cfg.__dict__}, final)
    print(f"{tag}done. final checkpoint {final}", flush=True)


if __name__ == "__main__":
    main()
