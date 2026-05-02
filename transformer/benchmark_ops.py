"""Per-operator forward/backward benchmark for the Transformer in model.py.

Each operator is exercised in isolation at the same shapes and dtypes the real
model uses, with forward and backward measured separately. Backward is timed on
a freshly-built autograd graph each iteration so we measure the backward kernel,
not graph-reuse artifacts.
"""

import math
import time
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import get_datasets


# ---- Config (matches benchmark.py defaults) ----
BLOCK_SIZE = 128
BATCH_SIZE = 64
D_MODEL = 128
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS
D_FF = 512
DROPOUT = 0.1

WARMUP = 10
BENCH = 50


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


def _stats(times):
    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    return avg, std


def time_fwd(fn: Callable, device) -> tuple[float, float]:
    for _ in range(WARMUP):
        fn()
    sync(device)
    times = []
    for _ in range(BENCH):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - t0)
    return _stats(times)


def time_bwd(make_graph: Callable, device) -> tuple[float, float]:
    """make_graph() returns (out, grad_out); only `out.backward(grad_out)` is timed."""
    for _ in range(WARMUP):
        out, grad_out = make_graph()
        out.backward(grad_out)
    sync(device)
    times = []
    for _ in range(BENCH):
        out, grad_out = make_graph()
        sync(device)
        t0 = time.perf_counter()
        out.backward(grad_out)
        sync(device)
        times.append(time.perf_counter() - t0)
    return _stats(times)


def _shape(t: torch.Tensor) -> str:
    return "x".join(str(d) for d in t.shape)


def _dtype(t: torch.Tensor) -> str:
    return str(t.dtype).replace("torch.", "")


def _desc(t: torch.Tensor) -> str:
    return f"{_shape(t)} {_dtype(t)}"


def _zero_grads(*tensors):
    for t in tensors:
        if t.grad is not None:
            t.grad = None


def _zero_module_grads(*mods):
    for m in mods:
        m.zero_grad(set_to_none=True)


def _print_header():
    print(
        f"  {'op':30s}  {'fwd (ms)':>17s}  {'bwd (ms)':>17s}  "
        f"{'input':<30s}  {'output':<30s}"
    )
    print("  " + "-" * 124)


def _print_row(name, fwd, bwd, in_desc, out_desc):
    fa, fs = fwd
    f_str = f"{fa*1000:8.3f} ± {fs*1000:6.3f}"
    if bwd is not None:
        ba, bs = bwd
        b_str = f"{ba*1000:8.3f} ± {bs*1000:6.3f}"
    else:
        b_str = f"{'-':>17s}"
    print(f"  {name:30s}  {f_str}  {b_str}  {in_desc:<30s}  {out_desc:<30s}")


def run_benchmarks(device, dtype, vocab_size):
    print(f"\n========== dtype={str(dtype).replace('torch.', '')} ==========")
    _print_header()

    B, T, D, H, HD, FF, V = (
        BATCH_SIZE, BLOCK_SIZE, D_MODEL, N_HEADS, HEAD_DIM, D_FF, vocab_size,
    )

    # ---------- Token embedding ----------
    emb = nn.Embedding(V, D).to(device=device, dtype=dtype)
    idx = torch.randint(0, V, (B, T), device=device)
    g_emb = torch.randn(B, T, D, device=device, dtype=dtype)
    fwd = time_fwd(lambda: emb(idx), device)
    def mg():
        _zero_module_grads(emb)
        return emb(idx), g_emb
    bwd = time_bwd(mg, device)
    _print_row("Embedding", fwd, bwd, f"({B},{T}) int64", _desc(g_emb))

    # ---------- Positional add (x + pe) ----------
    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    pe = torch.randn(1, T, D, device=device, dtype=dtype)
    g = torch.randn_like(x)
    fwd = time_fwd(lambda: x + pe, device)
    def mg():
        _zero_grads(x)
        return x + pe, g
    bwd = time_bwd(mg, device)
    _print_row("PositionalAdd", fwd, bwd, _desc(x), _desc(g))

    # ---------- Dropout (input embedding) ----------
    drop = nn.Dropout(DROPOUT)
    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn_like(x)
    fwd = time_fwd(lambda: drop(x), device)
    def mg():
        _zero_grads(x)
        return drop(x), g
    bwd = time_bwd(mg, device)
    _print_row("Dropout (B,T,D)", fwd, bwd, _desc(x), _desc(g))

    # ---------- LayerNorm ----------
    ln = nn.LayerNorm(D).to(device=device, dtype=dtype)
    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn_like(x)
    fwd = time_fwd(lambda: ln(x), device)
    def mg():
        _zero_grads(x); _zero_module_grads(ln)
        return ln(x), g
    bwd = time_bwd(mg, device)
    _print_row("LayerNorm", fwd, bwd, _desc(x), _desc(g))

    # ---------- QKV Linear ----------
    qkv_proj = nn.Linear(D, 3 * D).to(device=device, dtype=dtype)
    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn(B, T, 3 * D, device=device, dtype=dtype)
    fwd = time_fwd(lambda: qkv_proj(x), device)
    def mg():
        _zero_grads(x); _zero_module_grads(qkv_proj)
        return qkv_proj(x), g
    bwd = time_bwd(mg, device)
    _print_row("Linear qkv_proj (D->3D)", fwd, bwd, _desc(x), f"{B}x{T}x{3*D} {_dtype(g)}")

    # ---------- QKV split + view + transpose -> q (per-head) ----------
    qkv = torch.randn(B, T, 3 * D, device=device, dtype=dtype, requires_grad=True)
    g_q = torch.randn(B, H, T, HD, device=device, dtype=dtype)
    def fwd_reshape():
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, T, H, HD).transpose(1, 2)
        return q  # exercise one branch; cost is the same per branch
    fwd = time_fwd(fwd_reshape, device)
    def mg():
        _zero_grads(qkv)
        return fwd_reshape(), g_q
    bwd = time_bwd(mg, device)
    _print_row("split+view+transpose (1/3)", fwd, bwd, _desc(qkv), _desc(g_q))

    # ---------- attn scores: (q @ k^T) * scale ----------
    q = torch.randn(B, H, T, HD, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, T, HD, device=device, dtype=dtype, requires_grad=True)
    g_attn = torch.randn(B, H, T, T, device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(HD)
    fwd = time_fwd(lambda: (q @ k.transpose(-2, -1)) * scale, device)
    def mg():
        _zero_grads(q, k)
        return (q @ k.transpose(-2, -1)) * scale, g_attn
    bwd = time_bwd(mg, device)
    _print_row("matmul q@k^T * scale", fwd, bwd, f"{B}x{H}x{T}x{HD}", _desc(g_attn))

    # ---------- masked_fill + softmax ----------
    a = torch.randn(B, H, T, T, device=device, dtype=dtype, requires_grad=True)
    mask = torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)
    g = torch.randn_like(a)
    def fwd_sm():
        z = a.masked_fill(mask == 0, float("-inf"))
        return F.softmax(z, dim=-1)
    fwd = time_fwd(fwd_sm, device)
    def mg():
        _zero_grads(a)
        return fwd_sm(), g
    bwd = time_bwd(mg, device)
    _print_row("masked_fill + softmax", fwd, bwd, _desc(a), _desc(g))

    # ---------- attn dropout ----------
    a = torch.randn(B, H, T, T, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn_like(a)
    fwd = time_fwd(lambda: drop(a), device)
    def mg():
        _zero_grads(a)
        return drop(a), g
    bwd = time_bwd(mg, device)
    _print_row("Dropout (B,H,T,T)", fwd, bwd, _desc(a), _desc(g))

    # ---------- attn @ v ----------
    a = torch.randn(B, H, T, T, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, T, HD, device=device, dtype=dtype, requires_grad=True)
    g_av = torch.randn(B, H, T, HD, device=device, dtype=dtype)
    fwd = time_fwd(lambda: a @ v, device)
    def mg():
        _zero_grads(a, v)
        return a @ v, g_av
    bwd = time_bwd(mg, device)
    _print_row("matmul attn@v", fwd, bwd, f"{B}x{H}x{T}x{T}, {B}x{H}x{T}x{HD}", _desc(g_av))

    # ---------- attn output reshape (transpose+contiguous+view) ----------
    h_out = torch.randn(B, H, T, HD, device=device, dtype=dtype, requires_grad=True)
    g_proj = torch.randn(B, T, D, device=device, dtype=dtype)
    def fwd_oreshape():
        return h_out.transpose(1, 2).contiguous().view(B, T, D)
    fwd = time_fwd(fwd_oreshape, device)
    def mg():
        _zero_grads(h_out)
        return fwd_oreshape(), g_proj
    bwd = time_bwd(mg, device)
    _print_row("transpose+contig+view", fwd, bwd, _desc(h_out), _desc(g_proj))

    # ---------- out_proj Linear (D -> D) ----------
    out_proj = nn.Linear(D, D).to(device=device, dtype=dtype)
    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn_like(x)
    fwd = time_fwd(lambda: out_proj(x), device)
    def mg():
        _zero_grads(x); _zero_module_grads(out_proj)
        return out_proj(x), g
    bwd = time_bwd(mg, device)
    _print_row("Linear out_proj (D->D)", fwd, bwd, _desc(x), _desc(g))

    # ---------- residual add ----------
    a = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn_like(a)
    fwd = time_fwd(lambda: a + b, device)
    def mg():
        _zero_grads(a, b)
        return a + b, g
    bwd = time_bwd(mg, device)
    _print_row("Residual add", fwd, bwd, _desc(a), _desc(g))

    # ---------- FFN linear 1 (D -> D_FF) ----------
    ffn1 = nn.Linear(D, FF).to(device=device, dtype=dtype)
    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    g_ff = torch.randn(B, T, FF, device=device, dtype=dtype)
    fwd = time_fwd(lambda: ffn1(x), device)
    def mg():
        _zero_grads(x); _zero_module_grads(ffn1)
        return ffn1(x), g_ff
    bwd = time_bwd(mg, device)
    _print_row("Linear ffn1 (D->FF)", fwd, bwd, _desc(x), _desc(g_ff))

    # ---------- ReLU ----------
    x = torch.randn(B, T, FF, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn_like(x)
    fwd = time_fwd(lambda: F.relu(x), device)
    def mg():
        _zero_grads(x)
        return F.relu(x), g
    bwd = time_bwd(mg, device)
    _print_row("ReLU", fwd, bwd, _desc(x), _desc(g))

    # ---------- FFN linear 2 (D_FF -> D) ----------
    ffn2 = nn.Linear(FF, D).to(device=device, dtype=dtype)
    x = torch.randn(B, T, FF, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn(B, T, D, device=device, dtype=dtype)
    fwd = time_fwd(lambda: ffn2(x), device)
    def mg():
        _zero_grads(x); _zero_module_grads(ffn2)
        return ffn2(x), g
    bwd = time_bwd(mg, device)
    _print_row("Linear ffn2 (FF->D)", fwd, bwd, _desc(x), _desc(g))

    # ---------- LM head (D -> V), no bias, weight-tied in real model ----------
    head = nn.Linear(D, V, bias=False).to(device=device, dtype=dtype)
    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    g_logits = torch.randn(B, T, V, device=device, dtype=dtype)
    fwd = time_fwd(lambda: head(x), device)
    def mg():
        _zero_grads(x); _zero_module_grads(head)
        return head(x), g_logits
    bwd = time_bwd(mg, device)
    _print_row("Linear lm_head (D->V)", fwd, bwd, _desc(x), _desc(g_logits))

    # ---------- cross_entropy ----------
    logits = torch.randn(B * T, V, device=device, dtype=dtype, requires_grad=True)
    targets = torch.randint(0, V, (B * T,), device=device)
    g_loss = torch.tensor(1.0, device=device, dtype=dtype)
    fwd = time_fwd(lambda: F.cross_entropy(logits, targets), device)
    def mg():
        _zero_grads(logits)
        return F.cross_entropy(logits, targets), g_loss
    bwd = time_bwd(mg, device)
    _print_row("cross_entropy", fwd, bwd, _desc(logits), "scalar f32")

    # ---------- Reference: scaled_dot_product_attention (fused) ----------
    q = torch.randn(B, H, T, HD, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, T, HD, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, T, HD, device=device, dtype=dtype, requires_grad=True)
    g_sdpa = torch.randn_like(q)
    fwd = time_fwd(
        lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), device
    )
    def mg():
        _zero_grads(q, k, v)
        return F.scaled_dot_product_attention(q, k, v, is_causal=True), g_sdpa
    bwd = time_bwd(mg, device)
    _print_row(
        "[ref] SDPA (fused, causal)", fwd, bwd, f"{B}x{H}x{T}x{HD}", _desc(g_sdpa)
    )


def main():
    device = get_device()
    print(f"Device: {device}  |  torch {torch.__version__}")
    print(
        f"B={BATCH_SIZE}  T={BLOCK_SIZE}  D={D_MODEL}  H={N_HEADS}  "
        f"HD={HEAD_DIM}  FF={D_FF}  dropout={DROPOUT}"
    )
    print(f"Warmup={WARMUP}  Bench iters={BENCH}")

    _, _, tokenizer = get_datasets(BLOCK_SIZE)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    run_benchmarks(device, torch.float32, vocab_size)
    if device.type in ("cuda", "mps", "cpu"):
        run_benchmarks(device, torch.bfloat16, vocab_size)


if __name__ == "__main__":
    main()
