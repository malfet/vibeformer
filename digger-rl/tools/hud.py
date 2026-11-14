"""What a pixel policy sees and thinks, drawn next to the game (watch.py).

`Hud.update(policy, action)` reads the input stack and logits the policy
just acted on (`_Net.last_x` / `last_logits`) and draws a 320x400 panel:

  - the network's input: the newest of the stacked 128x128 frames,
    enlarged;
  - motion across the stack: where each moving 8x8 block of the newest
    frame was in the oldest one, found by block matching as a video
    encoder would, drawn as arrows over the dimmed frame;
  - each action's probability and log-probability, the chosen one
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
BLOCK, REACH = 8, 6               # block matching: block size, search radius
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
    def motion(old: np.ndarray, new: np.ndarray) -> list[tuple]:
        """Block matching from `new` back to `old` (HxWx3 uint8): for each
        BLOCK x BLOCK tile of `new` that changed, the shift (dy, dx) within
        +-REACH whose patch of `old` matches it best. Neighbouring tiles
        that moved alike (one sprite straddling tiles) are merged. Returns
        (y, x, dy, dx) per moving thing: centre now, and the displacement
        since `old`."""
        a = old.astype(np.int16).sum(2)
        b = new.astype(np.int16).sum(2)
        h, w = b.shape
        nb = (h // BLOCK, BLOCK, w // BLOCK, BLOCK)
        pad = np.pad(a, REACH, mode="edge")
        shifts = [(dy, dx) for dy in range(-REACH, REACH + 1)
                  for dx in range(-REACH, REACH + 1)]
        sad = np.stack([
            np.abs(b - pad[REACH - dy:REACH - dy + h, REACH - dx:REACH - dx + w])
            .reshape(nb).sum((1, 3)) for dy, dx in shifts])   # (S, h/B, w/B)
        still = sad[shifts.index((0, 0))]
        # Prefer short shifts on ties; a move has to beat standing still.
        cost = sad + 4 * np.array([abs(dy) + abs(dx) for dy, dx in shifts]
                                  )[:, None, None]
        best = cost.argmin(0)
        moved = {}
        for i, j in zip(*np.nonzero(still > 40 * BLOCK)):
            k = best[i, j]
            if sad[k, i, j] * 2 < still[i, j] and shifts[k] != (0, 0):
                moved[i, j] = shifts[k]
        out = []
        while moved:                     # flood-fill groups of alike tiles
            group, todo = [], [next(iter(moved))]
            while todo:
                t = todo.pop()
                if t not in moved:
                    continue
                v = moved.pop(t)
                group.append((*t, *v))
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        n = (t[0] + di, t[1] + dj)
                        if n in moved and max(abs(moved[n][0] - v[0]),
                                              abs(moved[n][1] - v[1])) <= 2:
                            todo.append(n)
            i, j, dy, dx = np.mean(group, 0)
            out.append((i * BLOCK + BLOCK / 2, j * BLOCK + BLOCK / 2, dy, dx))
        return out

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

        # Motion view: the newest frame dimmed, arrows from where each
        # moving block was in the oldest frame to where it is now.
        mx = W - 8 - VIEW
        img.paste(Image.fromarray(newest // 3).resize((VIEW, VIEW),
                                                      Image.NEAREST), (mx, 18))
        k = VIEW / newest.shape[0]
        for cy, cx, dy, dx in self.motion(frames[0], newest):
            x1, y1 = mx + cx * k, 18 + cy * k
            x0, y0 = x1 - dx * k * 2, y1 - dy * k * 2   # arrows drawn 2x long
            d.line((x0, y0, x1, y1), fill=ACCENT, width=2)
            n = np.hypot(dx, dy)
            ux, uy = dx / n, dy / n
            d.polygon([(x1 + 3 * ux, y1 + 3 * uy),
                       (x1 - 4 * ux - 4 * uy, y1 - 4 * uy + 4 * ux),
                       (x1 - 4 * ux + 4 * uy, y1 - 4 * uy - 4 * ux)],
                      fill=ACCENT)
        d.text((mx, 18 + VIEW + 2), f"motion t-{len(frames) - 1} -> t",
               font=self.small, fill=DIM)

        y0 = 220
        probs = logp.exp()
        for i, name in enumerate(ACTION_NAMES[:len(probs)]):
            y = y0 + i * 20
            col = ACCENT if i == action else FG
            d.text((8, y + 2), name, font=self.font, fill=col)
            d.rectangle((62, y + 3, 62 + 140, y + 15), outline=(50, 50, 60))
            d.rectangle((62, y + 3, 62 + int(140 * float(probs[i])), y + 15),
                        fill=ACCENT if i == action else BAR)
            d.text((246, y + 3), f"{float(probs[i]):.2f}", font=self.small,
                   fill=col, anchor="ra")
            d.text((310, y + 3), f"{float(logp[i]):.2f}", font=self.small,
                   fill=col, anchor="ra")
        d.text((246, y0 - 14), "p", font=self.small, fill=DIM, anchor="ra")
        d.text((310, y0 - 14), "log p", font=self.small, fill=DIM, anchor="ra")

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
