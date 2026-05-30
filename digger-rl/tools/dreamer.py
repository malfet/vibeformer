"""DreamerV3-flavored world model (continuous latents, simplified).

Components, following the Dreamer family:

  Encoder   pixels x_t -> embed e_t                      (CNN)
  RSSM      h_{t-1}, z_{t-1}, a_{t-1}, e_t -> h_t,
            prior  p(z_t | h_t),
            post   q(z_t | h_t, e_t),
            z_t ~ post during training (reparam)         (GRU + MLP heads)
  Decoder   (h_t, z_t) -> reconstructed x_t              (deconv mirror)
  Reward    (h_t, z_t) -> r_t                            (MLP)
  Continue  (h_t, z_t) -> P(episode continues)           (MLP)

Loss = recon_mse + reward_mse + bce(continue) + KL_balanced(post || prior).

Simplifications vs DreamerV3:
  - Continuous gaussian latents instead of discrete categoricals.
  - Plain MSE on rewards (no symlog / two-hot).
  - Plain KL with free bits, no KL balancing factor.
  Future iterations can promote each of these.

Spatial layout: 64x64 RGB input. Encoder/decoder stride-2 conv stacks give
clean (3,64,64) -> (256,4,4) -> (256*16=4096) latent embed, then deter+stoch
-> decoder -> (3,64,64).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal


@dataclass
class DreamerConfig:
    obs_size: int = 64
    num_actions: int = 6
    cnn_channels: tuple = (32, 64, 128, 256)  # encoder/decoder widths per stage
    stoch_dim: int = 32                       # gaussian latent dim
    deter_dim: int = 256                      # GRU hidden dim
    hidden_dim: int = 256                     # MLP width inside RSSM/heads
    embed_dim: int = 1024                     # post-CNN flattened width
    kl_free_nats: float = 1.0                 # free bits / dims
    reward_clip: float = 100.0                # gradient sanity
    # ---- Observation type ---------------------------------------------------
    # "pixel": 3x64x64 RGB, MSE recon, CNNEncoder/CNNDecoder.
    # "symbolic": (obs_channels x sym_h x sym_w) binary masks, BCE recon,
    #             SymbolicEncoder/SymbolicDecoder. The 10x15 Digger tile grid
    #             is too small to convolve meaningfully; we treat it as a
    #             flat feature vector.
    obs_type: str = "pixel"
    sym_channels: int = 6
    sym_h: int = 10
    sym_w: int = 15


# -------- Encoder / Decoder --------------------------------------------------

class CNNEncoder(nn.Module):
    """64x64 RGB -> embed_dim. Mirrors NatureCNN-ish but deeper & strided."""

    def __init__(self, cfg: DreamerConfig):
        super().__init__()
        c1, c2, c3, c4 = cfg.cnn_channels
        # 64 -> 32 -> 16 -> 8 -> 4
        self.net = nn.Sequential(
            nn.Conv2d(3, c1, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(c1, c2, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(c2, c3, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(c3, c4, 4, stride=2, padding=1), nn.SiLU(),
            nn.Flatten(),
        )
        flat = c4 * 4 * 4
        self.proj = nn.Linear(flat, cfg.embed_dim)
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 3, H, W) float in [0, 1] (any leading batch dims)
        lead = x.shape[:-3]
        x = x.reshape(-1, 3, self.cfg.obs_size, self.cfg.obs_size)
        e = self.proj(self.net(x))
        return e.reshape(*lead, -1)


class CNNDecoder(nn.Module):
    """(deter + stoch) -> reconstructed RGB 64x64."""

    def __init__(self, cfg: DreamerConfig):
        super().__init__()
        c1, c2, c3, c4 = cfg.cnn_channels
        feat = cfg.deter_dim + cfg.stoch_dim
        self.lin = nn.Linear(feat, c4 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(c4, c3, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(c1, 3, 4, stride=2, padding=1),
        )
        self.cfg = cfg

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        lead = feat.shape[:-1]
        c4 = self.cfg.cnn_channels[-1]
        h = self.lin(feat).reshape(-1, c4, 4, 4)
        x = self.deconv(h)
        return x.reshape(*lead, 3, self.cfg.obs_size, self.cfg.obs_size)


# -------- Symbolic encoder / decoder ----------------------------------------
# 10x15 tile grid is too small for repeated stride-2 conv. We flatten the
# (C, H, W) mask tensor and run an MLP -> embed_dim. The decoder mirrors
# this and outputs (C, H, W) logits (BCE recon downstream).

class SymbolicEncoder(nn.Module):
    def __init__(self, cfg: DreamerConfig):
        super().__init__()
        in_flat = cfg.sym_channels * cfg.sym_h * cfg.sym_w
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_flat, cfg.hidden_dim), nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim), nn.SiLU(),
        )
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-3]
        flat_in = (self.cfg.sym_channels * self.cfg.sym_h * self.cfg.sym_w)
        x = x.reshape(-1, flat_in)
        e = self.net(x)
        return e.reshape(*lead, -1)


class SymbolicDecoder(nn.Module):
    """(deter + stoch) -> per-tile per-channel logits (no sigmoid here)."""

    def __init__(self, cfg: DreamerConfig):
        super().__init__()
        feat = cfg.deter_dim + cfg.stoch_dim
        out_flat = cfg.sym_channels * cfg.sym_h * cfg.sym_w
        self.net = nn.Sequential(
            nn.Linear(feat, cfg.hidden_dim), nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.SiLU(),
            nn.Linear(cfg.hidden_dim, out_flat),
        )
        self.cfg = cfg

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        lead = feat.shape[:-1]
        logits = self.net(feat)
        return logits.reshape(*lead, self.cfg.sym_channels,
                              self.cfg.sym_h, self.cfg.sym_w)


# -------- RSSM ---------------------------------------------------------------

def _gaussian(mean: torch.Tensor, std_pre: torch.Tensor, min_std: float = 0.1
              ) -> Independent:
    """Build a Normal from raw (mean, pre-softplus std) head outputs."""
    std = F.softplus(std_pre) + min_std
    return Independent(Normal(mean, std), 1)


class RSSM(nn.Module):
    """Recurrent state-space model: deterministic h_t (GRU) + stochastic z_t.

    Posterior q(z_t | h_t, e_t) is used during training (reparam sample).
    Prior     p(z_t | h_t)        is what we'd predict without observing.
    KL(q || p) is the dynamics-prediction loss.
    """

    def __init__(self, cfg: DreamerConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.stoch_dim + cfg.num_actions
        # MLP that prepares the GRU input: prev z + prev a -> hidden
        self.proj_in = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim), nn.SiLU())
        self.gru = nn.GRUCell(cfg.hidden_dim, cfg.deter_dim)
        # Prior head from h
        self.prior = nn.Sequential(
            nn.Linear(cfg.deter_dim, cfg.hidden_dim), nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 2 * cfg.stoch_dim))
        # Posterior head from (h, embed)
        self.post = nn.Sequential(
            nn.Linear(cfg.deter_dim + cfg.embed_dim, cfg.hidden_dim), nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 2 * cfg.stoch_dim))

    def initial(self, batch_size: int, device: torch.device) -> tuple:
        h = torch.zeros(batch_size, self.cfg.deter_dim, device=device)
        z = torch.zeros(batch_size, self.cfg.stoch_dim, device=device)
        return h, z

    def observe_step(self, h, z, action_onehot, embed):
        """One step of the world model conditioned on a real observation.

        Returns: new_h, new_z (sampled from posterior), prior_dist, post_dist.
        """
        x = self.proj_in(torch.cat([z, action_onehot], dim=-1))
        new_h = self.gru(x, h)
        prior_params = self.prior(new_h)
        p_mean, p_std = prior_params.chunk(2, dim=-1)
        prior = _gaussian(p_mean, p_std)
        post_params = self.post(torch.cat([new_h, embed], dim=-1))
        q_mean, q_std = post_params.chunk(2, dim=-1)
        post = _gaussian(q_mean, q_std)
        new_z = post.rsample()
        return new_h, new_z, prior, post

    def imagine_step(self, h, z, action_onehot):
        """Step the model without observing (uses prior to sample z)."""
        x = self.proj_in(torch.cat([z, action_onehot], dim=-1))
        new_h = self.gru(x, h)
        prior_params = self.prior(new_h)
        p_mean, p_std = prior_params.chunk(2, dim=-1)
        prior = _gaussian(p_mean, p_std)
        new_z = prior.rsample()
        return new_h, new_z, prior

    def feature(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, z], dim=-1)


# -------- Reward / continue heads -------------------------------------------

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int = 1,
                 activation: str | None = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim))
        self.act = activation

    def forward(self, x):
        y = self.net(x).squeeze(-1)
        if self.act == "sigmoid":
            y = torch.sigmoid(y)
        return y


# -------- Actor / Critic (operate on RSSM feature) --------------------------

class Actor(nn.Module):
    """Categorical policy over discrete actions.

    Feature is the concatenation of deterministic h and stochastic z. We use
    a small MLP head -> logits -> Categorical. Discrete actions = REINFORCE
    in the imagined-rollout loss (with critic baseline).
    """

    def __init__(self, cfg: DreamerConfig):
        super().__init__()
        feat_dim = cfg.deter_dim + cfg.stoch_dim
        self.net = nn.Sequential(
            nn.Linear(feat_dim, cfg.hidden_dim), nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.num_actions),
        )

    def forward(self, feature: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.net(feature))


class Critic(nn.Module):
    """V(feature) -> scalar lambda-return target."""

    def __init__(self, cfg: DreamerConfig):
        super().__init__()
        feat_dim = cfg.deter_dim + cfg.stoch_dim
        self.net = nn.Sequential(
            nn.Linear(feat_dim, cfg.hidden_dim), nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature).squeeze(-1)


# -------- Full world model ---------------------------------------------------

class WorldModel(nn.Module):
    def __init__(self, cfg: DreamerConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.obs_type == "symbolic":
            self.encoder = SymbolicEncoder(cfg)
            self.decoder = SymbolicDecoder(cfg)
        elif cfg.obs_type == "pixel":
            self.encoder = CNNEncoder(cfg)
            self.decoder = CNNDecoder(cfg)
        else:
            raise ValueError(f"unknown obs_type: {cfg.obs_type!r}")
        self.rssm = RSSM(cfg)
        feat_dim = cfg.deter_dim + cfg.stoch_dim
        self.reward_head = MLPHead(feat_dim, cfg.hidden_dim)
        self.continue_head = MLPHead(feat_dim, cfg.hidden_dim,
                                     activation="sigmoid")

    def observe(self, obs, actions):
        """Run encoder + RSSM observe steps; return (h_seq, z_seq, prior, post).

        h_seq, z_seq: (B, T, ...) latent trajectories (gradient-tracked).
        prior_dists, post_dists: lists of length T of Independent Normals;
        used by the caller to compute KL.
        """
        B, T = obs.shape[:2]
        device = obs.device
        cfg = self.cfg
        embed = self.encoder(obs)
        actions_oh = F.one_hot(actions, num_classes=cfg.num_actions).float()

        h, z = self.rssm.initial(B, device)
        h_seq, z_seq, priors, posts = [], [], [], []
        for t in range(T):
            h, z, prior, post = self.rssm.observe_step(
                h, z, actions_oh[:, t], embed[:, t])
            h_seq.append(h)
            z_seq.append(z)
            priors.append(prior)
            posts.append(post)
        h_seq = torch.stack(h_seq, dim=1)
        z_seq = torch.stack(z_seq, dim=1)
        return h_seq, z_seq, priors, posts

    def loss(self, obs, actions, rewards, continues, return_latents: bool = False):
        """Compute world model loss on a batch of trajectories.

        obs:        (B, T, 3, H, W) float in [0, 1]
        actions:    (B, T) int64
        rewards:    (B, T) float
        continues:  (B, T) float in [0, 1]   (1.0 = episode continues)

        Returns: dict of scalar losses; total in `total`. If return_latents,
        also returns (h_seq, z_seq) so caller can use them as imagination
        start states for the actor-critic phase.
        """
        cfg = self.cfg
        h_seq, z_seq, priors, posts = self.observe(obs, actions)
        feats = torch.cat([h_seq, z_seq], dim=-1)

        recon = self.decoder(feats)
        if cfg.obs_type == "symbolic":
            # `recon` is logits; `obs` is in {0, 1}. BCE-with-logits handles
            # the per-tile binary nature of each channel.
            recon_loss = F.binary_cross_entropy_with_logits(recon, obs)
        else:
            recon_loss = F.mse_loss(recon, obs)

        pred_reward = self.reward_head(feats)
        pred_continue = self.continue_head(feats)
        reward_loss = F.mse_loss(
            pred_reward, rewards.clamp(-cfg.reward_clip, cfg.reward_clip))
        continue_loss = F.binary_cross_entropy(
            pred_continue.clamp(1e-6, 1.0 - 1e-6), continues)

        # KL with free bits (per-dim threshold).
        kls = []
        for prior, post in zip(priors, posts):
            kl = torch.distributions.kl_divergence(post, prior)
            kls.append(kl.clamp(min=cfg.kl_free_nats))
        kl_loss = torch.stack(kls, dim=1).mean()

        total = recon_loss + reward_loss + continue_loss + kl_loss
        losses = {
            "total": total,
            "recon": recon_loss,
            "reward": reward_loss,
            "continue": continue_loss,
            "kl": kl_loss,
        }
        if return_latents:
            return losses, h_seq, z_seq
        return losses


# -------- Replay buffer ------------------------------------------------------

class ReplayBuffer:
    """Per-env circular buffer of (obs, action, reward, done) transitions.

    Layout is (num_envs, capacity, ...). We sample length-T subsequences from
    a single env at random start positions; subsequences are allowed to cross
    done flags (the trainer learns the boundary from `continues=0` there).

    obs are stored as uint8 to keep memory in check. With capacity=20_000
    and num_envs=4 at (3, 64, 64): ~1 GB. Halve capacity if needed.
    """

    def __init__(self, capacity: int, num_envs: int,
                 obs_shape: tuple[int, ...],
                 obs_dtype: np.dtype = np.uint8,
                 obs_scale: float = 1.0 / 255.0):
        """`obs_dtype` is what's stored on disk-equivalent (uint8 for pixels,
        float32 for symbolic mask grids). `obs_scale` is the multiplier
        applied at sample time when promoting to float (1/255 for pixels,
        1.0 for already-in-[0,1] symbolic obs).
        """
        self.capacity = capacity
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.obs_scale = obs_scale
        self.obs = np.zeros((num_envs, capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros((num_envs, capacity), dtype=np.int64)
        self.rewards = np.zeros((num_envs, capacity), dtype=np.float32)
        self.dones = np.zeros((num_envs, capacity), dtype=bool)
        self.idx = 0       # next write index (per env)
        self.full = False  # has the buffer wrapped at least once

    def add(self, obs: np.ndarray, actions: np.ndarray,
            rewards: np.ndarray, dones: np.ndarray) -> None:
        i = self.idx
        self.obs[:, i] = obs
        self.actions[:, i] = actions
        self.rewards[:, i] = rewards
        self.dones[:, i] = dones
        self.idx = (i + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int, seq_length: int,
               device: torch.device, rng) -> tuple:
        """Return (obs, actions, rewards, continues) sequences ready for the WM.

        Avoids sampling a sub-sequence that crosses the write head when the
        buffer is full (otherwise we'd mix the oldest and newest data
        non-causally). When the buffer isn't yet full, sample only from the
        contiguous [0, idx) region.
        """
        n = len(self)
        max_start = n - seq_length
        if max_start <= 0:
            raise ValueError(
                f"buffer has {n} samples, need >= {seq_length}")
        envs = rng.integers(self.num_envs, size=batch_size)
        starts = rng.integers(0, max_start + 1, size=batch_size)

        obs = np.empty((batch_size, seq_length, *self.obs_shape),
                       dtype=self.obs_dtype)
        act = np.empty((batch_size, seq_length), dtype=np.int64)
        rew = np.empty((batch_size, seq_length), dtype=np.float32)
        don = np.empty((batch_size, seq_length), dtype=bool)
        for i in range(batch_size):
            e, s = int(envs[i]), int(starts[i])
            obs[i] = self.obs[e, s:s + seq_length]
            act[i] = self.actions[e, s:s + seq_length]
            rew[i] = self.rewards[e, s:s + seq_length]
            don[i] = self.dones[e, s:s + seq_length]

        cont = 1.0 - don.astype(np.float32)
        obs_t = torch.from_numpy(obs).to(device).float()
        if self.obs_scale != 1.0:
            obs_t.mul_(self.obs_scale)
        return (obs_t,
                torch.from_numpy(act).to(device),
                torch.from_numpy(rew).to(device),
                torch.from_numpy(cont).to(device))
