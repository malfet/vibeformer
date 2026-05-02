"""Character-level dataset for Tiny Shakespeare."""

import os
import torch
from torch.utils.data import Dataset


def load_text(path: str = "data/tiny_shakespeare.txt") -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, s: str) -> list[int]:
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


def get_datasets(block_size: int = 128, data_path: str = "data/tiny_shakespeare.txt"):
    text = load_text(data_path)
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    train_dataset = ShakespeareDataset(train_data, block_size)
    val_dataset = ShakespeareDataset(val_data, block_size)
    return train_dataset, val_dataset, tokenizer
