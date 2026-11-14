"""What a pixel policy sees and thinks, drawn next to the game (watch.py).

`Hud.update(policy, action)` reads the input stack and logits the policy
just acted on (`_Net.last_x` / `last_logits`) and draws a 320x400 panel:

  - the network's input: the newest of the stacked 128x128 frames,
    enlarged;
  - motion across the stack as an onion skin: the frames overlaid, with
    the pixels that differ from the newest frame in one false colour per
    age (t-3 blue, t-2 violet, t-1 magenta) and the newest's changed
    pixels in their true colour over a dimmed static background;
  - each action's probability, the chosen one
    highlighted;
  - the critic's value and the policy entropy, with the value's recent
    history as a sparkline.

With `saliency=True` it also computes |d log pi(a|x) / dx|: the input
pixels whose change would most change the probability of the chosen
action (Simonyan et al. 2013), painted onto the panel's input view.
"""

from __future__ import annotations

import collections

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from tools.heuristic_agent import ACTION_NAMES

W, H = 320, 400
VIEW = 150                        # input / motion views, px on the panel
AGES = [(60, 90, 255), (170, 70, 255), (255, 60, 190)]   # t-3, t-2, t-1
BG = (18, 18, 24)
FG = (220, 220, 220)
DIM = (120, 120, 130)
ACCENT = (255, 196, 0)
BAR = (70, 130, 220)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                      # Pillow without FreeType sizes
        return ImageFont.load_default()


class Hud:
    def __init__(self, saliency: bool = False, history: int = 300):
        self.saliency = saliency
        self.values: collections.deque[float] = collections.deque(
            maxlen=history)
        self.sal: np.ndarray | None = None          # (128, 128) in [0, 1]
        self.panel = np.zeros((H, W, 3), np.uint8)
        self.font, self.small = _font(13), _font(11)

    def reset(self) -> None:
        self.values.clear()
        self.sal = None

    def update(self, policy, action: int) -> None:
        x = policy.last_x
        agent = policy.agent
        with torch.no_grad():
            logp = F.log_softmax(policy.last_logits.float(), -1).cpu()
            value = float(agent.value(x)[0])
        self.values.append(value)
        if self.saliency:
            xg = x.detach().clone().requires_grad_(True)
            lp = F.log_softmax(agent.actor(agent.encode(xg)), -1)[0, action]
            (g,) = torch.autograd.grad(lp, xg)
            s = g[0].abs().amax(0, keepdim=True)[None]       # (1,1,H,W)
            s = F.avg_pool2d(s, 5, stride=1, padding=2)[0, 0]
            top = torch.quantile(s.flatten(), 0.995).clamp_min(1e-12)
            self.sal = (s / top).clamp(0, 1).cpu().numpy()
        self.panel = self._draw(x[0].cpu().numpy(), logp, action, value)

    # ---- motion -------------------------------------------------------------

    @staticmethod
    def onion(frames: list[np.ndarray]) -> np.ndarray:
        """The stacked frames (oldest first, HxWx3 uint8) overlaid: what
        is unchanged across the stack dimmed grey; where an older frame
        differs from the newest and had something lit there, its
        brightness in that age's colour; on top, the newest frame's own
        changed pixels in true colour."""
        new = frames[-1].astype(np.float32)
        out = new.mean(2, keepdims=True).repeat(3, 2) * 0.3
        changed = np.zeros(new.shape[:2], bool)
        for f, tint in zip(frames[:-1], AGES[-(len(frames) - 1):]):
            f = f.astype(np.float32)
            diff = np.abs(f - new).max(2) > 24
            changed |= diff
            luma = f.mean(2) / 255
            m = diff & (luma > 0.1)
            out[m] = np.array(tint, np.float32) * (0.4 + 0.6 * luma[m, None])
        m = changed & (new.mean(2) > 25)
        out[m] = new[m]
        return out.clip(0, 255).astype(np.uint8)

    # ---- drawing ------------------------------------------------------------

    def _frames(self, x: np.ndarray) -> list[np.ndarray]:
        """(C, h, w) floats in [0, 1] -> stacked frames, oldest first."""
        per = 3 if x.shape[0] % 3 == 0 else 1
        out = []
        for i in range(0, x.shape[0], per):
            f = x[i:i + per].transpose(1, 2, 0)
            out.append((np.repeat(f, 3 // per, 2) * 255).astype(np.uint8))
        return out

    def _heat(self, rgb: np.ndarray, sal: np.ndarray) -> np.ndarray:
        # Cyan, not red: Digger's dirt is red. Weak gradients stay clear.
        a = np.clip((sal - 0.15) / 0.85, 0, 1)[..., None] * 0.8
        hot = np.array([0, 255, 255], np.float32)
        return (rgb * (1 - a) + hot * a).astype(np.uint8)

    def _draw(self, x: np.ndarray, logp: torch.Tensor, action: int,
              value: float) -> np.ndarray:
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        frames = self._frames(x)
        newest = frames[-1]
        kind = "RGB" if x.shape[0] == 3 * len(frames) else "gray"
        d.text((8, 2), f"input {len(frames)} x {kind} {x.shape[1]}x"
               f"{x.shape[2]} = {x.shape[0]} ch", font=self.small, fill=DIM)
        view = self._heat(newest, self.sal) if self.sal is not None else newest
        img.paste(Image.fromarray(view).resize((VIEW, VIEW), Image.NEAREST),
                  (8, 18))
        if self.sal is not None:
            d.text((8, 18 + VIEW + 2), "saliency", font=self.small, fill=DIM)

        mx = W - 8 - VIEW
        img.paste(Image.fromarray(self.onion(frames)).resize(
            (VIEW, VIEW), Image.NEAREST), (mx, 18))
        lx = mx
        for k, tint in enumerate(AGES[-(len(frames) - 1):]):
            lab = f"t-{len(frames) - 1 - k}"
            d.text((lx, 18 + VIEW + 2), lab, font=self.small, fill=tint)
            lx += d.textlength(lab, font=self.small) + 6
        d.text((lx, 18 + VIEW + 2), "t", font=self.small, fill=FG)

        y0 = 220
        probs = logp.exp()
        for i, name in enumerate(ACTION_NAMES[:len(probs)]):
            y = y0 + i * 20
            col = ACCENT if i == action else FG
            d.text((8, y + 2), name, font=self.font, fill=col)
            d.rectangle((62, y + 3, 62 + 200, y + 15), outline=(50, 50, 60))
            d.rectangle((62, y + 3, 62 + int(200 * float(probs[i])), y + 15),
                        fill=ACCENT if i == action else BAR)
            d.text((310, y + 3), f"{float(probs[i]):.2f}", font=self.small,
                   fill=col, anchor="ra")
        d.text((310, y0 - 14), "p", font=self.small, fill=DIM, anchor="ra")

        ent = float(-(probs * logp).sum())
        d.text((8, 344), f"value {value:6.2f}    entropy {ent:4.2f}",
               font=self.font, fill=FG)
        vs = np.array(self.values, np.float32)
        if len(vs) > 1:
            lo, hi = float(vs.min()), float(vs.max())
            span = max(hi - lo, 1e-6)
            pts = [(8 + i * 304 / (self.values.maxlen - 1),
                    396 - (v - lo) / span * 32) for i, v in enumerate(vs)]
            d.line(pts, fill=BAR, width=1)
            d.text((270, 362), f"{hi:5.1f}", font=self.small, fill=DIM)
        return np.asarray(img)
