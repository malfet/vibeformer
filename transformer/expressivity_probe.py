"""Measure generated-text expressivity, not just Balmont-mimicry loss.

The question: did pretraining on other Silver-Age poets give the model a wider
vocabulary (and better rhyme) than a Balmont-only model -- and did fine-tuning
keep it? We generate a batch of poems from a checkpoint and classify the words
it produces against two reference vocabularies:

  * BAL  = words appearing in tiny_balmont.txt   (a Balmont-only model's world)
  * SIL  = words appearing in russian_silver_age.txt (the wider era)

Key metrics per model:
  real-word rate     fraction of generated words that exist in BAL ∪ SIL
                     (a fluency proxy; char models also invent plausible junk)
  era-only words     distinct generated words in SIL but NOT in BAL
                     -- vocabulary the baseline could not have learned
  rhyme rate         fraction of adjacent line pairs whose endings rhyme
"""

import argparse
import json
import re
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from model import Transformer
from generate import generate


def load(checkpoint, dev):
    with safe_open(checkpoint, framework="pt") as f:
        meta = f.metadata()
    stoi = json.loads(meta["stoi"])
    itos = {int(k): v for k, v in json.loads(meta["itos"]).items()}
    model = Transformer(vocab_size=int(meta["vocab_size"])).to(dev).bfloat16().eval()
    t = load_file(checkpoint, device=str(dev))
    model.load_state_dict({k[6:]: v for k, v in t.items() if k.startswith("model.")},
                          strict=False)
    return model, stoi, itos


def word_set(path):
    return set(w for w in re.findall(r"[а-яё]+", open(path, encoding="utf-8").read().lower())
               if len(w) >= 3)


def rhyme_rate(text):
    """Crude: fraction of adjacent non-blank line pairs whose last 3 letters match."""
    lines = [re.sub(r"[^а-яё]", "", l.lower()) for l in text.splitlines()]
    lines = [l for l in lines if len(l) >= 3]
    if len(lines) < 2:
        return 0.0
    hits = sum(lines[i][-3:] == lines[i + 1][-3:] for i in range(len(lines) - 1))
    return hits / (len(lines) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--tokens", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)
    model, stoi, itos = load(args.checkpoint, dev)

    prompt = "✦\n"
    idx = torch.tensor([[stoi[c] for c in prompt]] * args.batch, dtype=torch.long, device=dev)
    with torch.no_grad():
        out = generate(model, idx, args.tokens, args.temperature)
    text = "\n".join("".join(itos[i] for i in row) for row in out.tolist())

    BAL, SIL = word_set("data/tiny_balmont.txt"), word_set("data/russian_silver_age.txt")
    gen = [w for w in re.findall(r"[а-яё]+", text.lower()) if len(w) >= 3]
    uniq = set(gen)
    real = uniq & (BAL | SIL)
    era_only = (uniq & SIL) - BAL
    pure_oov = uniq - BAL - SIL

    print(f"{args.checkpoint}  (vocab {len(itos)})")
    print(f"  generated words: {len(gen):,} total, {len(uniq):,} distinct")
    print(f"  real-word rate : {len(uniq & (BAL|SIL))/len(uniq):.1%} "
          f"(distinct real {len(real):,}, invented {len(pure_oov):,})")
    print(f"  era-only words : {len(era_only):,} distinct (in other poets, not in Balmont)")
    print(f"  rhyme rate     : {rhyme_rate(text):.1%}")
    print(f"  era-only examples: {', '.join(sorted(era_only)[:25])}")


if __name__ == "__main__":
    main()
