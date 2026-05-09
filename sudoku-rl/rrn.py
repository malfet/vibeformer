"""Recurrent Relational Network for Sudoku (Palm, Paquet, Winther 2018).

Each of the 81 cells is a node. Two cells are connected by a directed edge
in both directions if they share a row, column, or 3x3 box (20 neighbors each).
At each iteration step k:

    m_ij  = MSG(h_i, h_j)              for every edge i -> j
    a_j   = sum of m_ij over incoming edges
    h_j   = GRU(a_j, h_j)
    y_j   = HEAD(h_j)                  (9-way logit per cell)

Loss is cross-entropy of y_j against the full solution, summed over every
iteration step — pressure on the model to make progress at every step. At
inference you can run more steps than you trained with; the paper's signature
result is that accuracy keeps improving with more iterations.

Auto-selects MPS on Apple Silicon, then CUDA, then CPU.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---- graph ---------------------------------------------------------------

def build_edges() -> tuple[torch.Tensor, torch.Tensor]:
    """Directed edges (src -> dst) for every pair of cells sharing row/col/box."""
    src, dst = [], []
    for j in range(81):  # destination cell
        rj, cj = j // 9, j % 9
        bj = (rj // 3) * 3 + (cj // 3)
        for i in range(81):
            if i == j:
                continue
            ri, ci = i // 9, i % 9
            bi = (ri // 3) * 3 + (ci // 3)
            if ri == rj or ci == cj or bi == bj:
                src.append(i)
                dst.append(j)
    return torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)


# ---- model ---------------------------------------------------------------

class RRN(nn.Module):
    def __init__(self, hidden: int = 96, emb: int = 16, steps: int = 32):
        super().__init__()
        self.hidden = hidden
        self.steps = steps

        self.digit_emb = nn.Embedding(10, emb)  # 0=blank, 1..9=clue/value
        self.row_emb = nn.Embedding(9, emb)
        self.col_emb = nn.Embedding(9, emb)
        self.box_emb = nn.Embedding(9, emb)

        self.input_mlp = nn.Sequential(
            nn.Linear(4 * emb, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.gru = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, 9)

        idx = torch.arange(81)
        rows = idx // 9
        cols = idx % 9
        boxes = (rows // 3) * 3 + (cols // 3)
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)
        self.register_buffer("boxes", boxes, persistent=False)

        src, dst = build_edges()
        self.register_buffer("src", src, persistent=False)
        self.register_buffer("dst", dst, persistent=False)

    def forward(self, puzzle: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        """puzzle: (B, 81) long, values 0..9 (0 = blank).

        Returns logits of shape (B, steps, 81, 9). Apply softmax/argmax along the
        last dim. Use steps=K to run more iterations at inference than at train.
        """
        B = puzzle.shape[0]
        N = 81
        H = self.hidden
        K = steps if steps is not None else self.steps

        d = self.digit_emb(puzzle)
        r = self.row_emb(self.rows).unsqueeze(0).expand(B, -1, -1)
        c = self.col_emb(self.cols).unsqueeze(0).expand(B, -1, -1)
        bx = self.box_emb(self.boxes).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([d, r, c, bx], dim=-1)
        h = self.input_mlp(x)  # (B, 81, H)

        outs = []
        for _ in range(K):
            h_src = h[:, self.src]                                 # (B, E, H)
            h_dst = h[:, self.dst]
            msg = self.msg_mlp(torch.cat([h_src, h_dst], dim=-1))  # (B, E, H)

            agg = torch.zeros(B, N, H, device=h.device, dtype=h.dtype)
            agg.index_add_(1, self.dst, msg)

            h = self.gru(agg.reshape(B * N, H), h.reshape(B * N, H)).reshape(B, N, H)
            outs.append(self.head(h))                              # (B, 81, 9)
        return torch.stack(outs, dim=1)                            # (B, K, 81, 9)


def step_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """logits: (B, K, 81, 9). target: (B, 81) long in 0..8 (= solution value - 1)."""
    B, K, N, C = logits.shape
    return F.cross_entropy(
        logits.reshape(B * K * N, C),
        target.unsqueeze(1).expand(-1, K, -1).reshape(-1),
    )


# ---- data ----------------------------------------------------------------

def load_npz(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    z = np.load(path)
    puzzles = torch.from_numpy(z["puzzles"]).long().reshape(-1, 81)        # 0..9
    solutions = torch.from_numpy(z["solutions"]).long().reshape(-1, 81) - 1  # 0..8
    return puzzles, solutions


def split(puzzles: torch.Tensor, solutions: torch.Tensor, val_frac: float = 0.1):
    n = puzzles.shape[0]
    n_val = max(1, int(n * val_frac))
    return (puzzles[:-n_val], solutions[:-n_val]), (puzzles[-n_val:], solutions[-n_val:])


# ---- evaluation ----------------------------------------------------------

@torch.no_grad()
def evaluate(model: RRN, loader: DataLoader, device: torch.device, steps: int):
    model.eval()
    cell_correct = 0
    cell_total = 0
    blank_correct = 0
    blank_total = 0
    full_correct = 0
    full_total = 0
    for puzzle, sol in loader:
        puzzle = puzzle.to(device)
        sol = sol.to(device)
        logits = model(puzzle, steps=steps)
        pred = logits[:, -1].argmax(dim=-1)  # (B, 81), 0..8
        match = pred == sol
        cell_correct += match.sum().item()
        cell_total += match.numel()

        blank = puzzle == 0
        blank_correct += (match & blank).sum().item()
        blank_total += blank.sum().item()

        full = match.all(dim=1)
        full_correct += full.sum().item()
        full_total += full.shape[0]
    return {
        "cell_acc": cell_correct / cell_total,
        "blank_acc": blank_correct / blank_total if blank_total else 0.0,
        "full_acc": full_correct / full_total,
    }


# ---- training ------------------------------------------------------------

def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(args: argparse.Namespace) -> None:
    device = pick_device()
    print(f"device: {device}")

    puzzles, solutions = load_npz(args.data)
    (tr_p, tr_s), (va_p, va_s) = split(puzzles, solutions, args.val_frac)
    print(f"train: {tr_p.shape[0]}  val: {va_p.shape[0]}")

    train_loader = DataLoader(
        TensorDataset(tr_p, tr_s), batch_size=args.batch_size,
        shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(va_p, va_s), batch_size=args.batch_size, shuffle=False,
    )

    model = RRN(hidden=args.hidden, steps=args.steps).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}  hidden={args.hidden}  steps={args.steps}")

    Path("checkpoints").mkdir(exist_ok=True)
    best_full = -1.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        losses = []
        for i, (puzzle, sol) in enumerate(train_loader):
            puzzle = puzzle.to(device)
            sol = sol.to(device)
            logits = model(puzzle)
            loss = step_loss(logits, sol)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if (i + 1) % args.log_every == 0:
                avg = sum(losses[-args.log_every:]) / args.log_every
                print(f"  epoch {epoch} step {i+1}/{len(train_loader)}  "
                      f"loss={avg:.4f}  elapsed={time.time()-t0:.1f}s")

        eval_steps = args.eval_steps or args.steps
        m = evaluate(model, val_loader, device, steps=eval_steps)
        print(f"epoch {epoch}  train_loss={np.mean(losses):.4f}  "
              f"val_cell={m['cell_acc']:.3f}  "
              f"val_blank={m['blank_acc']:.3f}  "
              f"val_full={m['full_acc']:.3f}  "
              f"(K={eval_steps})")

        if m["full_acc"] > best_full:
            best_full = m["full_acc"]
            torch.save(
                {"model": model.state_dict(), "args": vars(args)},
                f"checkpoints/rrn_best.pt",
            )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/sudoku_10k.npz")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--steps", type=int, default=32, help="iterations per training forward")
    p.add_argument("--eval-steps", type=int, default=0, help="iterations at eval (0=same as train)")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=20)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
