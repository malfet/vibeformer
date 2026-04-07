"""Compare fp32 vs bf16 autocast vs full bf16 training for 200 steps."""

import torch
from torch.utils.data import DataLoader

from dataset import get_datasets
from model import Transformer

SEED = 42
BLOCK_SIZE = 128
BATCH_SIZE = 64
STEPS = 200
EVAL_EVERY = 50
EVAL_ITERS = 100
LR = 3e-4
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
D_FF = 512


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def eval_loss(model, val_loader, device, mode="fp32"):
    model.eval()
    losses = []
    it = iter(val_loader)
    autocast_dev = "cuda" if device.type == "cuda" else device.type
    for _ in range(EVAL_ITERS):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(val_loader)
            xb, yb = next(it)
        xb, yb = xb.to(device), yb.to(device)
        if mode == "autocast":
            with torch.autocast(autocast_dev, dtype=torch.bfloat16):
                _, loss = model(xb, yb)
        else:
            # fp32 or full_bf16 — model dtype handles it
            _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_run(label, device, train_loader, val_loader, tokenizer, mode="fp32"):
    """mode: 'fp32', 'autocast', or 'full_bf16'"""
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed(SEED)

    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=D_FF, block_size=BLOCK_SIZE, dropout=0.0,
    ).to(device)

    if mode == "full_bf16":
        model = model.bfloat16()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.98))
    autocast_dev = "cuda" if device.type == "cuda" else device.type

    train_it = iter(train_loader)
    history = []

    for step in range(STEPS):
        try:
            xb, yb = next(train_it)
        except StopIteration:
            train_it = iter(train_loader)
            xb, yb = next(train_it)
        xb, yb = xb.to(device), yb.to(device)

        if mode == "autocast":
            with torch.autocast(autocast_dev, dtype=torch.bfloat16):
                _, loss = model(xb, yb)
        else:
            _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            val = eval_loss(model, val_loader, device, mode)
            history.append((step, loss.item(), val))
            print(f"  [{label:>12s}] step {step:4d} | train loss {loss.item():.4f} | val loss {val:.4f}")

    return history


def main():
    device = get_device()
    print(f"Device: {device}\n")

    train_ds, val_ds, tokenizer = get_datasets(BLOCK_SIZE)

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    print("=== fp32 ===")
    g.manual_seed(SEED)
    fp32_hist = train_run("fp32", device, train_loader, val_loader, tokenizer, mode="fp32")

    print("\n=== bf16 autocast ===")
    g.manual_seed(SEED)
    autocast_hist = train_run("bf16 autocast", device, train_loader, val_loader, tokenizer, mode="autocast")

    print("\n=== full bf16 ===")
    g.manual_seed(SEED)
    full_hist = train_run("full bf16", device, train_loader, val_loader, tokenizer, mode="full_bf16")

    print("\n=== Comparison ===")
    print(f"{'step':>6s}  {'fp32 val':>10s}  {'autocast':>10s}  {'full bf16':>10s}  {'ac diff':>10s}  {'full diff':>10s}")
    for (s, _, v1), (_, _, v2), (_, _, v3) in zip(fp32_hist, autocast_hist, full_hist):
        print(f"{s:6d}  {v1:10.4f}  {v2:10.4f}  {v3:10.4f}  {v2 - v1:+10.4f}  {v3 - v1:+10.4f}")

    print(f"\nFinal val loss — fp32: {fp32_hist[-1][2]:.4f}, autocast: {autocast_hist[-1][2]:.4f}, full bf16: {full_hist[-1][2]:.4f}")


if __name__ == "__main__":
    main()
