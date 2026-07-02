"""Character-level dataset for Tiny Shakespeare."""

import json
import os
import torch
from torch.utils.data import Dataset


def load_text(path: str = "data/tiny_shakespeare.txt") -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    """Char-level tokenizer.

    Pass `text` to derive the vocab from a corpus (default behaviour), or a
    fixed `vocab` list for a shared vocabulary across pretrain/fine-tune. With a
    fixed vocab, characters outside it map to `unk` instead of raising.
    """

    def __init__(self, text: str | None = None, vocab: list[str] | None = None,
                 unk: str | None = None):
        if vocab is not None:
            chars = list(vocab)
        else:
            chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.unk_id = self.stoi.get(unk) if unk is not None else None

    @classmethod
    def from_vocab_file(cls, path: str) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            spec = json.load(f)
        return cls(vocab=spec["vocab"], unk=spec.get("unk"))

    def encode(self, s: str) -> list[int]:
        if self.unk_id is not None:
            return [self.stoi.get(c, self.unk_id) for c in s]
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def metadata(self) -> dict:
        return {
            "tokenizer": "char",
            "vocab_size": str(self.vocab_size),
            "stoi": json.dumps(self.stoi),
            "itos": json.dumps({str(k): v for k, v in self.itos.items()}),
        }


class BPETokenizer:
    """SentencePiece subword tokenizer (see data/build_bpe.py).

    Newlines are meaningful in verse, so we swap \\n <-> the atomic <nl> token
    around SentencePiece, which otherwise treats input line-by-line.
    """

    NL = "<nl>"

    def __init__(self, model_path: str):
        import sentencepiece as spm
        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, s: str) -> list[int]:
        return self.sp.encode(s.replace("\n", self.NL))

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids).replace(self.NL, "\n")

    def metadata(self) -> dict:
        return {"tokenizer": "bpe", "vocab_size": str(self.vocab_size),
                "bpe_model": self.model_path}


def load_tokenizer(meta: dict):
    """Rebuild a tokenizer from checkpoint metadata (used by generate/eval)."""
    if meta.get("tokenizer") == "bpe":
        return BPETokenizer(meta["bpe_model"])
    stoi = json.loads(meta["stoi"])
    tok = CharTokenizer(vocab=list(stoi.keys()))
    return tok


class ShakespeareDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def get_datasets(block_size: int = 128, data_path: str = "data/tiny_shakespeare.txt",
                 vocab_path: str | None = None, bpe_path: str | None = None):
    text = load_text(data_path)
    if bpe_path is not None:
        tokenizer = BPETokenizer(bpe_path)
    elif vocab_path is not None:
        tokenizer = CharTokenizer.from_vocab_file(vocab_path)
    else:
        tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    train_dataset = ShakespeareDataset(train_data, block_size)
    val_dataset = ShakespeareDataset(val_data, block_size)
    return train_dataset, val_dataset, tokenizer


def build_dataset(data_path: str, block_size: int, tokenizer: "CharTokenizer"):
    """Encode a whole file with an existing tokenizer (for replay/mix data)."""
    data = torch.tensor(tokenizer.encode(load_text(data_path)), dtype=torch.long)
    return ShakespeareDataset(data, block_size)
