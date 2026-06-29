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
                 vocab_path: str | None = None):
    text = load_text(data_path)
    if vocab_path is not None:
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
