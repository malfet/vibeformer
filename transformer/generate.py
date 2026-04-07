"""Generate text from a trained transformer."""

import argparse
import json
import torch
from safetensors.torch import load_file
from safetensors import safe_open

from model import Transformer


def generate(
    model: Transformer,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Autoregressive generation with temperature sampling."""
    model.eval()
    for _ in range(max_new_tokens):
        # Crop to block_size
        idx_cond = idx[:, -model.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=1)
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best_model.safetensors")
    parser.add_argument("--prompt", default="\n", help="Starting text")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    with safe_open(args.checkpoint, framework="pt") as f:
        meta = f.metadata()
    tensors = load_file(args.checkpoint, device="cpu")

    stoi = json.loads(meta["stoi"])
    itos = {int(k): v for k, v in json.loads(meta["itos"]).items()}
    vocab_size = int(meta["vocab_size"])

    model = Transformer(vocab_size=vocab_size)
    model_state = {k[len("model."):]: v for k, v in tensors.items() if k.startswith("model.")}
    model.load_state_dict(model_state, strict=False)

    # Encode prompt
    idx = torch.tensor([[stoi[c] for c in args.prompt]], dtype=torch.long)

    with torch.no_grad():
        out = generate(model, idx, args.max_tokens, args.temperature)

    text = "".join(itos[i] for i in out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
