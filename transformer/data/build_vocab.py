"""Build the shared character vocabulary for pretrain + fine-tune.

Both stages must use one identical stoi/itos, or the pretrained embeddings are
meaningless after fine-tuning. The vocab is every character Balmont uses (the
fine-tune target must be fully representable) plus every Silver-Age character
that appears often enough to be worth a row; rare OCR/Greek/archaic leftovers
fold into a single <unk> token at encode time.
"""

import collections
import json
import os

HERE = os.path.dirname(__file__)
PRETRAIN = os.path.join(HERE, "russian_silver_age.txt")
FINETUNE = os.path.join(HERE, "tiny_balmont.txt")
OUT = os.path.join(HERE, "russian_silver_age_vocab.json")

UNK = "�"  # replacement char; stands in for any out-of-vocab character
MIN_COUNT = 10  # keep a Silver-Age-only char only if it appears this often


def main():
    pre = open(PRETRAIN, encoding="utf-8").read()
    fine = open(FINETUNE, encoding="utf-8").read()
    pre_counts = collections.Counter(pre)

    chars = set(fine)  # every Balmont char is mandatory
    chars |= {c for c, n in pre_counts.items() if n >= MIN_COUNT}
    chars.add(UNK)
    vocab = sorted(chars)

    dropped = sorted(set(pre) - set(vocab))
    json.dump({"vocab": vocab, "unk": UNK}, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=0)
    print(f"vocab size: {len(vocab)}")
    print(f"balmont chars covered: {set(fine) <= set(vocab)}")
    print(f"silver-age chars folded into <unk>: {len(dropped)}")
    print(f"  {''.join(c for c in dropped if c.isprintable())!r}")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
