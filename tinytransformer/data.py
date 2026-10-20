"""Character vocabulary + batching for the movie-title corpus."""
import torch
from pathlib import Path


class CharVocab:
    """Maps characters <-> ids. Id 0 is '\\n', our title separator and EOS."""
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        try:
            return [self.stoi[ch] for ch in s]
        except KeyError as e:
            raise ValueError(f"character {e.args[0]!r} is not in the vocabulary") from e

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path):
        Path(path).write_text("".join(self.itos[i] for i in range(self.size)))

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        chars = Path(path).read_text()
        obj.stoi = {ch: i for i, ch in enumerate(chars)}
        obj.itos = {i: ch for ch, i in obj.stoi.items()}
        return obj


def load_corpus(path="data/movie_titles.txt"):
    text = Path(path).read_text()
    vocab = CharVocab(text)
    data = torch.tensor(vocab.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return vocab, data[:n], data[n:]


def get_batch(data: torch.Tensor, block_size: int, batch_size: int):
    """Random windows: input is data[i:i+T], target is the same shifted by 1."""
    n_positions = len(data) - block_size - 1
    if n_positions <= 0:
        raise ValueError(
            f"corpus too small ({len(data)} tokens) for block_size={block_size}")
    ix = torch.randint(n_positions, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y
