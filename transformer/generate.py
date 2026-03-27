"""Generate text from a trained transformer."""

import argparse
import torch
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
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--prompt", default="\n", help="Starting text")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    vocab_size = checkpoint["vocab_size"]

    model = Transformer(vocab_size=vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Encode prompt
    idx = torch.tensor([[stoi[c] for c in args.prompt]], dtype=torch.long)

    with torch.no_grad():
        out = generate(model, idx, args.max_tokens, args.temperature)

    text = "".join(itos[i] for i in out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
