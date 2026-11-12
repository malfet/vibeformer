"""Everything that can play Digger, behind one interface.

    spec := teacher | search | random | path/to/checkpoint.pt

Checkpoints from both trainers load here: train_symbolic.py (symbolic grid
obs) and train_pixel.py (84x84 frames); which one is read from the
checkpoint's config. evaluate.py and watch.py both use this, so anything
one can run the other can too.

    env = SymbolicDiggerEnv(**env_kwargs(spec), episodic_life=False)
    policy = make_policy(spec, env, device)
    obs = env.reset(); policy.reset()
    a = policy(obs)                         # before each agent step
    obs, r, done, info = env_step_skipped(env, a, 4)
    policy.observe(info)                    # after it
"""

from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import torch

from digger_env import DiggerEnv, preprocess_uint8
from tools.heuristic_agent import SmartHeuristic
from tools.symbolic_env import SymbolicDiggerEnv

BUILTIN = ("teacher", "search", "random")


def _config(spec: str) -> dict:
    return torch.load(spec, weights_only=False, map_location="cpu")["config"]


def is_pixel(cfg: dict) -> bool:
    return "encoder_width" in cfg          # only train_pixel.py writes it


def env_kwargs(spec: str) -> dict:
    """SymbolicDiggerEnv kwargs a policy needs (frame stack, egocentric)."""
    if spec in BUILTIN:
        return {}
    cfg = _config(spec)
    if is_pixel(cfg):
        return {}
    return dict(frame_stack=cfg["frame_stack"],
                egocentric=cfg.get("egocentric", False))


def load_symbolic_agent(path, device) -> tuple[torch.nn.Module, dict]:
    from train_symbolic import IterAgent, SymbolicAgent
    cfg = _config(path)
    in_ch, h, w = SymbolicDiggerEnv(**env_kwargs(path)).obs_shape
    if cfg.get("arch", "cnn") == "iter":
        agent = IterAgent(in_channels=in_ch, num_actions=DiggerEnv.NUM_ACTIONS,
                          h=h, w=w, ch=cfg["iter_channels"],
                          iters=cfg["iter_steps"], readout=cfg["readout"])
    else:
        agent = SymbolicAgent(in_channels=in_ch,
                              num_actions=DiggerEnv.NUM_ACTIONS, h=h, w=w)
    agent.load_state_dict(
        torch.load(path, weights_only=False, map_location="cpu")["agent"])
    return agent.to(device).eval(), cfg


def load_pixel_agent(path, device) -> tuple[torch.nn.Module, dict]:
    from train_pixel import Agent
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    cfg = ckpt["config"]
    agent = Agent(num_actions=DiggerEnv.NUM_ACTIONS,
                  in_channels=cfg["frame_stack"] * (3 if cfg["color"] else 1),
                  width=cfg["encoder_width"],
                  separate_critic=cfg.get("separate_critic", False),
                  in_hw=(cfg["obs_size"], cfg["obs_size"]))
    agent.load_state_dict(ckpt["agent"])
    return agent.to(device).eval(), cfg


class _Policy:
    def reset(self) -> None:
        pass

    def observe(self, info: dict) -> None:
        pass


class _Heuristic(_Policy):
    def __init__(self, env: SymbolicDiggerEnv, teacher):
        self.env, self.teacher = env, teacher

    def reset(self) -> None:
        self.teacher.reset()

    def __call__(self, obs) -> int:
        return int(self.teacher(self.env._last_state))


class _Random(_Policy):
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs) -> int:
        return int(self.rng.integers(DiggerEnv.NUM_ACTIONS))


class _Net(_Policy):
    def __init__(self, agent, device, argmax: bool):
        self.agent, self.device, self.argmax = agent, device, argmax

    @torch.no_grad()
    def _act(self, x: torch.Tensor) -> int:
        logits = self.agent.actor(self.agent.encode(x))
        if self.argmax:
            return int(logits.argmax(-1).item())
        return int(torch.distributions.Categorical(
            logits=logits).sample().item())


class _Symbolic(_Net):
    def __call__(self, obs) -> int:
        return self._act(torch.from_numpy(np.asarray(obs, np.float32))
                         .to(self.device).unsqueeze(0))


class _Pixel(_Net):
    """Keeps its own frame stack of the emulator's frames, refilled at a
    reset and after each death, as DiggerVecEnv does in training."""

    def __init__(self, env, agent, cfg, device, argmax):
        super().__init__(agent, device, argmax)
        self.env, self.cfg = env, cfg
        self.stack = collections.deque(maxlen=cfg["frame_stack"])
        self.refill = True

    def reset(self) -> None:
        self.refill = True

    def observe(self, info: dict) -> None:
        if info.get("dying", False):
            self.refill = True

    def __call__(self, obs) -> int:
        fr = preprocess_uint8(self.env._env._core.get_frame(),
                              self.cfg["obs_size"], self.cfg["color"])
        if self.refill:
            self.stack.clear()
            self.stack.extend([fr] * self.stack.maxlen)
            self.refill = False
        else:
            self.stack.append(fr)
        planes = [f.transpose(2, 0, 1) if f.ndim == 3 else f[None]
                  for f in self.stack]
        x = torch.from_numpy(np.concatenate(planes, 0)).to(self.device)
        return self._act(x.float().div_(255.0).unsqueeze(0))


def make_policy(spec: str, env: SymbolicDiggerEnv, device,
                argmax: bool = False, seed: int = 0, **search_kwargs):
    """A callable obs -> action for `spec` (see module docstring).
    `env` must be built with env_kwargs(spec)."""
    if spec == "teacher":
        return _Heuristic(env, SmartHeuristic())
    if spec == "search":
        from tools.search_teacher import SearchTeacher
        return _Heuristic(env, SearchTeacher(env, **search_kwargs))
    if spec == "random":
        return _Random(seed)
    if not Path(spec).exists():
        raise SystemExit(f"no such checkpoint {spec!r} "
                         f"(or use one of {', '.join(BUILTIN)})")
    if is_pixel(_config(spec)):
        agent, cfg = load_pixel_agent(spec, device)
        return _Pixel(env, agent, cfg, device, argmax)
    agent, _ = load_symbolic_agent(spec, device)
    return _Symbolic(agent, device, argmax)
