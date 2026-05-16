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


# -------- Full world model ---------------------------------------------------

class WorldModel(nn.Module):
    def __init__(self, cfg: DreamerConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = CNNEncoder(cfg)
        self.decoder = CNNDecoder(cfg)
        self.rssm = RSSM(cfg)
        feat_dim = cfg.deter_dim + cfg.stoch_dim
        self.reward_head = MLPHead(feat_dim, cfg.hidden_dim)
        self.continue_head = MLPHead(feat_dim, cfg.hidden_dim,
                                     activation="sigmoid")

    def loss(self, obs, actions, rewards, continues):
        """Compute world model loss on a batch of trajectories.

        obs:        (B, T, 3, H, W) float in [0, 1]
        actions:    (B, T) int64
        rewards:    (B, T) float
        continues:  (B, T) float in [0, 1]   (1.0 = episode continues)

        Returns: dict of scalar losses; total in `total`.
        """
        B, T = obs.shape[:2]
        device = obs.device
        cfg = self.cfg

        embed = self.encoder(obs)                          # (B, T, embed_dim)
        actions_oh = F.one_hot(actions, num_classes=cfg.num_actions).float()

        h, z = self.rssm.initial(B, device)
        feats, kls = [], []
        for t in range(T):
            h, z, prior, post = self.rssm.observe_step(
                h, z, actions_oh[:, t], embed[:, t])
            feats.append(self.rssm.feature(h, z))
            # KL with free bits (per-dim threshold so trivial dims aren't
            # punished into a tight prior match).
            kl = torch.distributions.kl_divergence(post, prior)
            kl = kl.clamp(min=cfg.kl_free_nats)
            kls.append(kl)
        feats = torch.stack(feats, dim=1)                  # (B, T, feat_dim)
        kls = torch.stack(kls, dim=1)

        # Reconstruction loss (per-pixel MSE)
        recon = self.decoder(feats)                        # (B, T, 3, H, W)
        recon_loss = F.mse_loss(recon, obs)

        # Reward and continue prediction
        pred_reward = self.reward_head(feats)              # (B, T)
        pred_continue = self.continue_head(feats)          # (B, T)
        reward_loss = F.mse_loss(pred_reward,
                                 rewards.clamp(-cfg.reward_clip, cfg.reward_clip))
        continue_loss = F.binary_cross_entropy(
            pred_continue.clamp(1e-6, 1.0 - 1e-6), continues)

        kl_loss = kls.mean()
        total = recon_loss + reward_loss + continue_loss + kl_loss
        return {
            "total": total,
            "recon": recon_loss,
            "reward": reward_loss,
            "continue": continue_loss,
            "kl": kl_loss,
        }
