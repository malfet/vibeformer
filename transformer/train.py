"""Training loop for the decoder-only transformer."""

import math
import os
import torch
from torch.utils.data import DataLoader

from dataset import get_datasets
from model import Transformer

# Hyperparameters
BLOCK_SIZE = 128
BATCH_SIZE = 64
MAX_ITERS = 20000
EVAL_INTERVAL = 500
EVAL_ITERS = 200
LEARNING_RATE = 3e-4
WARMUP_ITERS = 400
LR_DECAY_ITERS = 20000
MIN_LR = 1e-5
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
D_FF = 512
DROPOUT = 0.1
CHECKPOINT_PATH = "best_model.pt"
CHECKPOINT_DIR = "checkpoints"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_lr(step: int) -> float:
    """Learning rate schedule with warmup and cosine decay (per the paper)."""
    if step < WARMUP_ITERS:
        return LEARNING_RATE * step / WARMUP_ITERS
    if step > LR_DECAY_ITERS:
        return MIN_LR
    decay_ratio = (step - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, device):
    model.eval()
    out = {}
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        losses = []
        loader_iter = iter(loader)
        for _ in range(EVAL_ITERS):
            try:
                xb, yb = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                xb, yb = next(loader_iter)
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def main():
    device = get_device()
    print(f"Using device: {device}")

    train_dataset, val_dataset, tokenizer = get_datasets(BLOCK_SIZE)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Train size: {len(train_dataset):,} | Val size: {len(val_dataset):,}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98))

    best_val_loss = float("inf")
    step = 0

    # Resume from checkpoint if available
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "step" in ckpt:
            step = ckpt["step"] + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from step {step} (best val loss {best_val_loss:.4f})")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    train_iter = iter(train_loader)

    while step < MAX_ITERS:
        # Get batch
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)

        xb, yb = xb.to(device), yb.to(device)

        # Update learning rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward + backward
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Evaluate
        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
            losses = estimate_loss(model, train_loader, val_loader, device)
            print(f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | lr {lr:.2e}", flush=True)
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                        "best_val_loss": best_val_loss,
                        "vocab_size": tokenizer.vocab_size,
                        "stoi": tokenizer.stoi,
                        "itos": tokenizer.itos,
                    },
                    CHECKPOINT_PATH,
                )
                print(f"  -> saved best checkpoint (val loss {best_val_loss:.4f})", flush=True)

            # Periodic checkpoint
            periodic_path = os.path.join(CHECKPOINT_DIR, f"step_{step:06d}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "vocab_size": tokenizer.vocab_size,
                    "stoi": tokenizer.stoi,
                    "itos": tokenizer.itos,
                },
                periodic_path,
            )

        step += 1

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
