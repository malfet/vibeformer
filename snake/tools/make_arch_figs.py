"""Generate the obs/architecture illustration PNGs referenced in README.md.

All boards are real `tiny_snake` states pushed through the real transforms
(`egocentric` shift, canonical rotation), so the figures cannot drift from
the code. Run from snake/:

    python -m tools.make_arch_figs        # writes docs/*.png
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import tiny_snake as ts

DOCS = Path(__file__).resolve().parent.parent / "docs"

# empty, wall, body, head, food
CMAP = colors.ListedColormap(
    ["#ffffff", "#3b4d61", "#e8a13c", "#d1495b", "#4caf50"])


def make_state(off_r: int, off_c: int, canvas: int = 21,
               field: int = 9) -> ts.TinySnake:
    """A snake in a `field`x`field` room placed at (off_r, off_c) on the
    canvas — identical local situation regardless of placement."""
    s = ts.TinySnake(canvas_rows=canvas, canvas_cols=canvas, rng_seed=0)
    s.reset()
    s.field_rows = s.field_cols = field
    s.off_r, s.off_c = off_r, off_c
    s._build_walls()
    hr, hc = off_r + field // 2 + 1, off_c + field // 2
    # L-shaped body, head facing RIGHT.
    s.body = deque([(hr + 2, hc - 1), (hr + 1, hc - 1),
                    (hr, hc - 1), (hr, hc)])
    s.direction = ts.RIGHT
    s.foods = [(hr - 2, hc + 2)]
    return s


def ego_grid(s: ts.TinySnake) -> np.ndarray:
    """Symbolic grid shifted so the head sits at the canvas center
    (int-grid version of `egocentric_obs`; out-of-canvas fill = wall)."""
    g = s.obs()
    H, W = g.shape
    out = np.full_like(g, ts.SYM_WALL)
    hr, hc = s.head
    dr, dc = H // 2 - hr, W // 2 - hc
    sr0, sr1 = max(0, -dr), min(H, H - dr)
    sc0, sc1 = max(0, -dc), min(W, W - dc)
    out[sr0 + dr:sr1 + dr, sc0 + dc:sc1 + dc] = g[sr0:sr1, sc0:sc1]
    return out


def canon_grid(s: ts.TinySnake) -> np.ndarray:
    """Egocentric grid rotated so the heading points up (real rot table)."""
    return np.rot90(ego_grid(s), k=ts._ROT_K[s.direction])


def draw_board(ax, grid: np.ndarray, title: str = "",
               center_marker: bool = False) -> None:
    ax.imshow(grid, cmap=CMAP, vmin=0, vmax=4, interpolation="nearest")
    H, W = grid.shape
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="#dddddd", linewidth=0.4)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    if title:
        ax.set_title(title, fontsize=10)
    if center_marker:
        ax.add_patch(Rectangle((W // 2 - 0.5, H // 2 - 0.5), 1, 1,
                               fill=False, edgecolor="black",
                               linewidth=2.0, linestyle="--"))


def box(ax, x, y, w, h, text, fc="#eef2f7", fontsize=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02",
                                fc=fc, ec="#3b4d61", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize)


def arrow(ax, x0, y0, x1, y1, **kw):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color="#3b4d61", lw=1.4, **kw))


def fig_allocentric() -> None:
    fig = plt.figure(figsize=(9, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.15], hspace=0.32)
    a = make_state(1, 1)
    b = make_state(10, 10)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    draw_board(ax0, a.obs(), "placement A")
    draw_board(ax1, b.obs(), "placement B — same situation")
    axs = fig.add_subplot(gs[1, :])
    axs.set_xlim(0, 10)
    axs.set_ylim(0, 2)
    axs.axis("off")
    box(axs, 0.3, 0.5, 1.8, 1.0, "conv trunk\n(equivariant)")
    arrow(axs, 2.15, 1.0, 2.9, 1.0)
    box(axs, 2.95, 0.5, 1.6, 1.0, "flatten")
    arrow(axs, 4.6, 1.0, 5.35, 1.0)
    box(axs, 5.4, 0.5, 2.2, 1.0,
        "FC head:\nprivate weight per (x, y)", fc="#fbe4e4")
    arrow(axs, 7.65, 1.0, 8.4, 1.0)
    axs.text(8.5, 1.0, "3 logits", va="center", fontsize=9)
    fig.subplots_adjust(top=0.80)
    fig.suptitle(
        "Allocentric CNN — the same situation at two placements is two "
        "unrelated inputs to the FC head:\nit can only memorize every "
        "placement (holdout CE diverged; score 7 vs teacher 44)",
        fontsize=10, y=0.99)
    fig.savefig(DOCS / "allocentric_cnn.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)


def fig_egocentric() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
    a = make_state(1, 1)
    b = make_state(10, 10)
    draw_board(axes[0], a.obs(), "placement A")
    draw_board(axes[1], b.obs(), "placement B")
    assert np.array_equal(ego_grid(a), ego_grid(b))
    draw_board(axes[2], ego_grid(a),
               "egocentric view of A AND B\n(head pinned at center)",
               center_marker=True)
    arrow(axes[2], -3.0, 10.0, -0.5, 10.0, clip_on=False)
    fig.subplots_adjust(top=0.76)
    fig.suptitle(
        "Egocentric CNN — translate the window so the head is always at "
        "the center: both placements become the identical input.\n"
        "Translation invariance by construction (score 7 → 23; unseen "
        "sizes for free)", fontsize=10, y=0.99)
    fig.savefig(DOCS / "egocentric_cnn.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)


def fig_canonical_iterator() -> None:
    fig = plt.figure(figsize=(9.6, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.1, 1.5], hspace=0.35)
    s = make_state(6, 3)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    draw_board(ax0, s.obs(), "raw board (facing right)")
    draw_board(ax1, ego_grid(s), "egocentric (head centered)",
               center_marker=True)
    draw_board(ax2, canon_grid(s),
               "canonical: rotated to face UP\nSTR/L/R = fixed cells",
               center_marker=True)
    # Highlight the three candidate cells on the canonical board.
    H = canon_grid(s).shape[0]
    c = H // 2
    for (rr, cc, lab) in ((c - 1, c, "S"), (c, c - 1, "L"), (c, c + 1, "R")):
        ax2.add_patch(Rectangle((cc - 0.5, rr - 0.5), 1, 1, fill=False,
                                edgecolor="#1565c0", linewidth=2.0))
        ax2.text(cc, rr, lab, ha="center", va="center", fontsize=8,
                 color="#1565c0", fontweight="bold")

    axs = fig.add_subplot(gs[1, :])
    axs.set_xlim(0, 12)
    axs.set_ylim(0, 3)
    axs.axis("off")
    box(axs, 0.2, 1.0, 1.9, 1.0, "canonical obs\n(6-7 ch)")
    arrow(axs, 2.15, 1.5, 2.75, 1.5)
    box(axs, 2.8, 1.0, 1.7, 1.0, "stem\n2× conv3×3")
    arrow(axs, 4.55, 1.5, 5.15, 1.5)
    box(axs, 5.2, 1.0, 2.3, 1.0,
        "ONE tied conv block\napplied K times", fc="#e7f2e7")
    # weight-tying loop arrow
    axs.add_patch(FancyArrowPatch((6.9, 2.05), (5.7, 2.05),
                                  connectionstyle="arc3,rad=0.9",
                                  arrowstyle="-|>", mutation_scale=12,
                                  color="#2e7d32", lw=1.4))
    axs.text(6.3, 2.75, "×K (propagates 1 cell / step;\nK is an "
             "inference-time dial)", ha="center", fontsize=8,
             color="#2e7d32")
    arrow(axs, 7.55, 1.5, 8.15, 1.5)
    box(axs, 8.2, 1.6, 3.4, 0.85,
        "policy: shared MLP on the S/L/R cells → 3 logits", fontsize=8)
    box(axs, 8.2, 0.55, 3.4, 0.85,
        "value: center ⊕ global-avg features → V", fontsize=8)
    fig.subplots_adjust(top=0.84)
    fig.suptitle(
        "Canonical iterator — egocentric + rotated-to-face-up obs, "
        "weight-tied conv iteration (learned BFS), head-local readout.\n"
        "240k params, size-agnostic; BC 39 vs teacher 44 with NO "
        "engineered distance channel", fontsize=10, y=0.99)
    fig.savefig(DOCS / "canonical_iterator.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    fig_allocentric()
    fig_egocentric()
    fig_canonical_iterator()
    print(f"wrote {DOCS}/allocentric_cnn.png, egocentric_cnn.png, "
          f"canonical_iterator.png")
