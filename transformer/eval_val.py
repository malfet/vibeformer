"""Deterministic full-pass validation loss for a checkpoint on Balmont.

train.py's in-loop number samples 200 random batches, so it's noisy and the
two runs aren't strictly comparable. This sweeps the entire Balmont val split
(last 10%, same split train.py uses) once and reports mean nats/char + bits/char
-- a fair apples-to-apples metric across checkpoints regardless of vocab size.
"""

import argparse
import json
import math
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from dataset import load_text
from model import Transformer

BLOCK_SIZE = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--data", default="data/tiny_balmont.txt")
    args = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    with safe_open(args.checkpoint, framework="pt") as f:
        meta = f.metadata()
    stoi = json.loads(meta["stoi"])
    vocab_size = int(meta["vocab_size"])
    unk = stoi.get("�")

    text = load_text(args.data)
    val_text = text[int(0.9 * len(text)):]  # matches get_datasets' 90/10 split
    ids = [stoi.get(c, unk) if unk is not None else stoi[c] for c in val_text]
    data = torch.tensor(ids, dtype=torch.long)

    model = Transformer(vocab_size=vocab_size).to(dev).bfloat16().eval()
    tensors = load_file(args.checkpoint, device=str(dev))
    model.load_state_dict({k[6:]: v for k, v in tensors.items()
                           if k.startswith("model.")}, strict=False)

    # Non-overlapping windows over the whole val split; sum exact token NLL.
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(data) - 1, BLOCK_SIZE):
            x = data[i:i + BLOCK_SIZE].unsqueeze(0).to(dev)
            y = data[i + 1:i + 1 + BLOCK_SIZE].to(dev)
            x = x[:, :len(y)]
            logits, _ = model(x)
            logp = torch.log_softmax(logits[0].float(), dim=-1)
            total_nll += -logp[torch.arange(len(y)), y].sum().item()
            total_tok += len(y)

    nats = total_nll / total_tok
    print(f"{args.checkpoint}")
    print(f"  vocab={vocab_size}  val chars={total_tok:,}")
    print(f"  nats/char = {nats:.4f}   bits/char = {nats / math.log(2):.4f}")


if __name__ == "__main__":
    main()
