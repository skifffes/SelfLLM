"""Comparison baselines -- NOT your core code.

These three already satisfy the `Tokenizer` protocol from core_bpe.py, so every metric,
figure and CSV in this project can be produced before you write a single line of BPE.
Your ToyBPE then joins the same table and is compared against identical numbers.

  char      : one token per character (the lower bound on granularity)
  word      : whitespace words with a train-only vocabulary, OOV -> <unk>
  hf_bpe_N  : HuggingFace `tokenizers` BPE, byte-level pre-tokenization (reference upper
              bound -- the week-1 spec explicitly allows using the library)

Methodological rule enforced here: every tokenizer is fitted on the TRAIN split only.
Fitting on train+test would leak test statistics into the vocabulary and inflate the
comparison.  The audit pipeline checks the same thing for documents.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

UNK = "<unk>"


class CharTokenizer:
    name = "char"

    def __init__(self, vocab: set[str]):
        self._vocab = vocab

    @classmethod
    def fit(cls, texts: Iterable[str]) -> "CharTokenizer":
        vocab: set[str] = set()
        for t in texts:
            vocab.update(t)
        return cls(vocab)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def encode(self, text: str) -> list[str]:
        return [c if c in self._vocab else UNK for c in text]

    def decode(self, tokens: Sequence[str]) -> str:
        return "".join("" if t == UNK else t for t in tokens)


class WordTokenizer:
    """Whitespace word level. decode() is lossy: original whitespace is not recoverable."""

    name = "word"

    def __init__(self, vocab: set[str]):
        self._vocab = vocab

    @classmethod
    def fit(cls, texts: Iterable[str], min_freq: int = 1) -> "WordTokenizer":
        counts: Counter[str] = Counter()
        for t in texts:
            counts.update(t.split())
        return cls({w for w, c in counts.items() if c >= min_freq})

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def encode(self, text: str) -> list[str]:
        return [w if w in self._vocab else UNK for w in text.split()]

    def decode(self, tokens: Sequence[str]) -> str:
        return " ".join(tokens)


class HFBpeTokenizer:
    name = "hf_bpe"

    def __init__(self, tok, label: str):
        self._tok = tok
        self.name = label

    @classmethod
    def fit(
        cls,
        texts: Iterable[str],
        vocab_size: int = 2000,
        min_frequency: int = 2,
        label: str | None = None,
    ) -> "HFBpeTokenizer":
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tok = Tokenizer(models.BPE(unk_token=UNK))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=[UNK],
            show_progress=False,
        )
        tok.train_from_iterator(texts, trainer=trainer)
        return cls(tok, label or f"hf_bpe_{vocab_size}")

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> list[str]:
        return self._tok.encode(text).tokens

    def decode(self, tokens: Sequence[str]) -> str:
        ids = [self._tok.token_to_id(t) for t in tokens]
        return self._tok.decode([i for i in ids if i is not None])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: str | Path, label: str | None = None) -> "HFBpeTokenizer":
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(str(path)), label or Path(path).stem)


def build_baselines(
    train_texts: Sequence[str], hf_vocab_sizes: Sequence[int] = (2000,)
) -> list:
    """char + word + one HF BPE per requested vocab size, all fitted on train only."""
    out: list = [CharTokenizer.fit(train_texts), WordTokenizer.fit(train_texts)]
    for n in hf_vocab_sizes:
        out.append(HFBpeTokenizer.fit(train_texts, vocab_size=n))
    return out