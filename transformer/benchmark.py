"""Benchmark various parts of the training loop."""

import time
import torch
from dataset import get_datasets
from model import Transformer

BLOCK_SIZE = 128
BATCH_SIZE = 64
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
D_FF = 512
WARMUP_STEPS = 5
BENCH_STEPS = 20


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def bench(name: str, fn, device: torch.device):
    # Warmup
    for _ in range(WARMUP_STEPS):
        fn()
    sync(device)

    times = []
    for _ in range(BENCH_STEPS):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    print(f"  {name:30s}: {avg*1000:8.2f} ms  (std {std*1000:.2f} ms)")
    return avg


def main():
    device = get_device()
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}, Block size: {BLOCK_SIZE}")
    print(f"Model: d={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}, d_ff={D_FF}")
    print()

    _, _, tokenizer = get_datasets(BLOCK_SIZE)

    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        block_size=BLOCK_SIZE,
        dropout=0.0,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Random batch
    xb = torch.randint(0, tokenizer.vocab_size, (BATCH_SIZE, BLOCK_SIZE), device=device)
    yb = torch.randint(0, tokenizer.vocab_size, (BATCH_SIZE, BLOCK_SIZE), device=device)

    # --- Forward only ---
    print("\n--- Forward pass ---")
    def fwd():
        with torch.no_grad():
            model(xb, yb)
    bench("forward (no grad)", fwd, device)

    model.train()
    def fwd_grad():
        model(xb, yb)
    bench("forward (with grad)", fwd_grad, device)

    # --- Backward ---
    print("\n--- Backward pass ---")
    def fwd_bwd():
        _, loss = model(xb, yb)
        loss.backward()
    bench("forward + backward", fwd_bwd, device)

    # --- Full step ---
    print("\n--- Full training step ---")
    def full_step():
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    t_full = bench("full step (fwd+bwd+optim)", full_step, device)

    # --- Throughput (fp32) ---
    tokens_per_step = BATCH_SIZE * BLOCK_SIZE
    tokens_per_sec = tokens_per_step / t_full
    print(f"\n  Throughput (fp32): {tokens_per_sec:,.0f} tokens/sec")
    print(f"  Tokens per step: {tokens_per_step:,}")

    # --- bf16 autocast ---
    autocast_dtype = torch.bfloat16
    autocast_device = "cuda" if device.type == "cuda" else device.type
    print(f"\n=== bf16 autocast ({autocast_device}) ===")

    print("\n--- Forward pass (bf16) ---")
    def fwd_bf16():
        with torch.no_grad(), torch.autocast(autocast_device, dtype=autocast_dtype):
            model(xb, yb)
    bench("forward (no grad, bf16)", fwd_bf16, device)

    model.train()
    def fwd_grad_bf16():
        with torch.autocast(autocast_device, dtype=autocast_dtype):
            model(xb, yb)
    bench("forward (with grad, bf16)", fwd_grad_bf16, device)

    print("\n--- Full training step (bf16) ---")
    def full_step_bf16():
        with torch.autocast(autocast_device, dtype=autocast_dtype):
            _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    t_bf16 = bench("full step (bf16)", full_step_bf16, device)
    print(f"\n  Throughput (bf16): {tokens_per_step / t_bf16:,.0f} tokens/sec")

    # --- torch.compile ---
    print("\n=== torch.compile ===")
    compiled_model = torch.compile(model)

    print("\n--- Forward pass (compiled, fp32) ---")
    def fwd_compiled():
        with torch.no_grad():
            compiled_model(xb, yb)
    bench("forward (compiled, fp32)", fwd_compiled, device)

    print("\n--- Full training step (compiled, fp32) ---")
    def full_step_compiled():
        _, loss = compiled_model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    t_compiled = bench("full step (compiled)", full_step_compiled, device)
    print(f"\n  Throughput (compiled): {tokens_per_step / t_compiled:,.0f} tokens/sec")

    print("\n--- Full training step (compiled + bf16) ---")
    def full_step_compiled_bf16():
        with torch.autocast(autocast_device, dtype=autocast_dtype):
            _, loss = compiled_model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    t_both = bench("full step (compiled+bf16)", full_step_compiled_bf16, device)
    print(f"\n  Throughput (compiled+bf16): {tokens_per_step / t_both:,.0f} tokens/sec")


if __name__ == "__main__":
    main()
