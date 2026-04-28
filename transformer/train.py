"""Training loop for the decoder-only transformer."""

import argparse
import json
import math
import os
import time
import torch
from torch.utils.data import DataLoader
from safetensors.torch import save_file, load_file
from safetensors import safe_open

from dataset import get_datasets
from model import Transformer

# Hyperparameters
BLOCK_SIZE = 128
BATCH_SIZE = 64
MAX_ITERS = 100_000
EVAL_INTERVAL = 500
EVAL_ITERS = 200
LEARNING_RATE = 3e-4
WARMUP_ITERS = 400
LR_DECAY_ITERS = 100_000
MIN_LR = 1e-5
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
D_FF = 512
DROPOUT = 0.1
DATA_PATH = "data/tiny_shakespeare.txt"


def _flatten_optim_state(optimizer):
    """Flatten optimizer state_dict into tensors + JSON metadata for safetensors."""
    tensors = {}
    state = optimizer.state_dict()
    # Store param_groups as JSON (they contain scalars, bools, Nones, tuples)
    meta = {"optim_param_groups": json.dumps(state["param_groups"])}
    for param_id, param_state in state["state"].items():
        for key, val in param_state.items():
            if isinstance(val, torch.Tensor):
                tensors[f"optim.state.{param_id}.{key}"] = val
            else:
                tensors[f"optim.state.{param_id}.{key}"] = torch.tensor(val)
    return tensors, meta


def _unflatten_optim_state(tensors, metadata, optimizer):
    """Restore optimizer state_dict from flattened safetensors."""
    state_dict = optimizer.state_dict()
    if "optim_param_groups" in metadata:
        loaded_groups = json.loads(metadata["optim_param_groups"])
        for i, group in enumerate(loaded_groups):
            # Restore all keys except 'params' (which must match current model)
            for key, val in group.items():
                if key == "params":
                    continue
                # Convert lists back to tuples where needed (e.g. betas)
                if isinstance(val, list):
                    val = tuple(val)
                state_dict["param_groups"][i][key] = val
    for key, tensor in tensors.items():
        if not key.startswith("optim.state."):
            continue
        parts = key.split(".")
        param_id, param_name = int(parts[2]), ".".join(parts[3:])
        if param_id not in state_dict["state"]:
            state_dict["state"][param_id] = {}
        val = tensor if tensor.dim() > 0 else tensor.item()
        state_dict["state"][param_id][param_name] = val
    optimizer.load_state_dict(state_dict)


def save_checkpoint(path, model, optimizer, step, best_val_loss, tokenizer):
    """Save checkpoint as safetensors."""
    # Skip tied weights (head.weight == token_emb.weight)
    tensors = {}
    seen_data_ptrs = {}
    for k, v in model.state_dict().items():
        ptr = v.data_ptr()
        if ptr in seen_data_ptrs:
            continue
        seen_data_ptrs[ptr] = k
        tensors[f"model.{k}"] = v.contiguous().cpu()
    optim_tensors, optim_meta = _flatten_optim_state(optimizer)
    tensors.update({k: v.contiguous().cpu() for k, v in optim_tensors.items()})
    metadata = {
        "step": str(step),
        "best_val_loss": str(best_val_loss),
        "vocab_size": str(tokenizer.vocab_size),
        "stoi": json.dumps(tokenizer.stoi),
        "itos": json.dumps({str(k): v for k, v in tokenizer.itos.items()}),
        **optim_meta,
    }
    save_file(tensors, path, metadata=metadata)


def load_checkpoint(path, model, optimizer, device):
    """Load checkpoint from safetensors. Returns (step, best_val_loss)."""
    with safe_open(path, framework="pt") as f:
        meta = f.metadata()
    tensors = load_file(path, device=str(device))
    model_state = {k[len("model."):]: v for k, v in tensors.items() if k.startswith("model.")}
    model.load_state_dict(model_state, strict=False)
    optim_tensors = {k: v for k, v in tensors.items() if k.startswith("optim.")}
    if optim_tensors:
        _unflatten_optim_state(optim_tensors, meta, optimizer)
    step = int(meta.get("step", "-1"))
    best_val_loss = float(meta.get("best_val_loss", "inf"))
    return step, best_val_loss


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
    parser = argparse.ArgumentParser(description="Train decoder-only transformer")
    parser.add_argument("--data", default=DATA_PATH, help="Path to training text file")
    args = parser.parse_args()

    # Derive checkpoint prefix from dataset filename (e.g. "tiny_shakespeare")
    data_name = os.path.splitext(os.path.basename(args.data))[0]
    checkpoint_path = f"{data_name}_best.safetensors"
    checkpoint_dir = f"checkpoints/{data_name}"

    device = get_device()
    print(f"Using device: {device}")
    print(f"Dataset: {args.data} (prefix: {data_name})")

    train_dataset, val_dataset, tokenizer = get_datasets(BLOCK_SIZE, data_path=args.data)
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
    ).to(device=device, dtype=torch.bfloat16)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    tokens_per_step = BATCH_SIZE * BLOCK_SIZE
    # Approximate FLOPs per token: 6 * N_params (2x forward, 4x backward)
    flops_per_step = 6 * param_count * tokens_per_step

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98))

    best_val_loss = float("inf")
    step = 0

    # Resume from checkpoint if available
    if os.path.exists(checkpoint_path):
        loaded_step, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer, device)
        if loaded_step >= 0:
            step = loaded_step + 1
        print(f"Resumed from step {step} (best val loss {best_val_loss:.4f})")
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_iter = iter(train_loader)
    step_times = []

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

        # Forward + backward (timed)
        t0 = time.perf_counter()
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        step_times.append(dt)

        # Evaluate
        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
            avg_dt = sum(step_times) / len(step_times)
            tok_per_sec = tokens_per_step / avg_dt
            gflops = flops_per_step / avg_dt / 1e9
            losses = estimate_loss(model, train_loader, val_loader, device)
            print(
                f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}"
                f" | lr {lr:.2e} | {tok_per_sec:,.0f} tok/s | {gflops:.1f} GFLOP/s",
                flush=True,
            )
            step_times = []
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(checkpoint_path, model, optimizer, step, best_val_loss, tokenizer)
                print(f"  -> saved best checkpoint (val loss {best_val_loss:.4f})", flush=True)

            # Periodic checkpoint
            periodic_path = os.path.join(checkpoint_dir, f"step_{step:06d}.safetensors")
            save_checkpoint(periodic_path, model, optimizer, step, best_val_loss, tokenizer)

        step += 1

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
