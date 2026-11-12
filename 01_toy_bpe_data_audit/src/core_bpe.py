"""YOUR CORE CODE #1 -- Toy BPE.

Paper: Sennrich, Haddow & Birch (2016), *Neural Machine Translation of Rare Words with
Subword Units*, arXiv:1508.07909.  Algorithm 1 in that paper is exactly `get_stats` +
`merge_vocab` + the training loop.

Everything below is a stub that raises NotImplementedError.  The rest of the project
(baselines, metrics, plots, notebook, audit) talks to your code only through the
`Tokenizer` protocol at the bottom of this file, so you can implement these in any order
and the pipeline will pick them up.

Suggested order (each step is independently testable):
    1. get_stats      -> tests/test_bpe_core.py::test_get_stats_hand_computed
    2. merge_vocab    -> tests/test_bpe_core.py::test_merge_vocab_hand_computed
    3. train_bpe      -> tests/test_bpe_core.py::test_first_three_merges
    4. encode         -> tests/test_bpe_core.py::test_roundtrip
    5. decode         -> same roundtrip test
    6. save / load    -> tests/test_bpe_core.py::test_save_load

Design decisions that are YOURS to make (they are the interesting part of the experiment,
so no code is provided for them):
  * Pre-tokenization.  The paper splits on whitespace and appends the end-of-word marker
    "</w>" so that "low" and "lower" stay distinguishable.  What happens to Chinese, which
    has no whitespace?  Record whatever you choose in notes/failure_analysis.md -- the
    char/word/subword length comparison in the notebook is where it shows up.
  * Tie-breaking when several pairs share the max frequency.  REQUIRED: break ties
    lexicographically on the pair tuple, so runs are reproducible and the hand-computed
    fixtures pass.
  * Unknown characters at encode time.  REQUIRED: emit the single token "<unk>" for any
    character not in the base vocabulary; metrics.py measures the UNK rate.
  * Whether to keep "</w>" inside the returned tokens.  REQUIRED: keep the same symbol
    strings you trained on, so decode() can invert encode() exactly.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Protocol, Sequence, runtime_checkable

Symbol = str
Pair = tuple[Symbol, Symbol]
Word = tuple[Symbol, ...]
Vocab = dict[Word, int]

END_OF_WORD = "</w>"
UNK = "<unk>"


def get_stats(vocab: Vocab) -> dict[Pair, int]:
    """Count adjacent symbol pairs over a frequency-weighted vocabulary.

    Args:
        vocab: mapping from a word represented as a tuple of symbols to its corpus count.
            Example: {("l", "o", "w", "</w>"): 5}

    Returns:
        Mapping from adjacent pair -> total count, summed over all words weighted by freq.
        For the example above: {("l","o"): 5, ("o","w"): 5, ("w","</w>"): 5}.
        A pair occurring twice inside one word counts twice.

    Paper: the "statistics" step of Algorithm 1.
    """
    stats: dict[Pair, int] = {}
    for word, freq in vocab.items():
        for a, b in zip(word, word[1:]):
            pair = (a, b)
            stats[pair] = stats.get(pair, 0) + freq
    return stats
    


def merge_vocab(pair: Pair, vocab: Vocab) -> Vocab:
    """Replace every occurrence of `pair` in every word of `vocab` with a single symbol.

    Args:
        pair: the pair to merge, e.g. ("e", "s").
        vocab: current frequency-weighted vocabulary; not mutated in place.

    Returns:
        A NEW vocab where each word has all non-overlapping left-to-right occurrences of
        the pair replaced by the concatenation "es".  Frequencies are preserved.
        Words that do not contain the pair are carried over unchanged.

    Watch out: merging ("e","s") in ("n","e","w","e","s","t","</w>") must not also merge
    the newly created symbol again in the same pass.
    """
    a, b = pair
    merged = a + b
    out: Vocab = {}
    for word, freq in vocab.items():
        new_word: list[Symbol] = []
        i = 0
        while i < len(word):
            if i + 1 < len(word) and word[i] == a and word[i + 1] == b:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        key = tuple(new_word)
        out[key] = out.get(key, 0) + freq
    return out


_BLOCK_RE = re.compile(r"\s+|\S+")


def _split_blocks(text: str) -> list[str]:
    """Split text into non-whitespace blocks and whitespace blocks, keeping every char.

    Why not text.split(): it swallows the original whitespace (double spaces, newlines),
    so decode() could never restore the input and round-trip would stay below 1.0.
    """
    return _BLOCK_RE.findall(text)


def _block_symbols(block: str, scheme: str = "paper") -> Word:
    """One block -> symbol sequence.

    Whitespace blocks stay atomic (a single symbol, so get_stats never sees a pair in
    them); word blocks become characters plus a trailing END_OF_WORD marker.
    """
    if not block.strip():
        return (block,)
    if scheme == "char":
        return tuple(block)
    return tuple(block) + (END_OF_WORD,)


def _build_vocab(texts: Iterable[str], scheme: str = "paper") -> Vocab:
    """Count word-block frequencies once: each distinct word is stored a single time."""
    counter: Counter[Word] = Counter()
    for text in texts:
        for block in _split_blocks(text):
            counter[_block_symbols(block, scheme)] += 1
    return dict(counter)


def train_bpe(
    texts: Iterable[str],
    num_merges: int,
    min_freq: int = 2,
    pretokenize: str = "paper",
) -> "ToyBPE":
    """Learn `num_merges` BPE merge rules from raw text.

    Args:
        texts: raw documents (the corpus is a few MB; do not hold more than needed).
        num_merges: exact number of merge rules to learn.  Stop early if no pair reaches
            `min_freq`.
        min_freq: skip pairs whose total count is below this.
        pretokenize: name of your pre-tokenization scheme; "paper" = whitespace + "</w>".
            Keep it a parameter so the notebook can ablate it.

    Returns:
        A fitted ToyBPE.  REQUIRED attributes: `.merges` (ordered list of Pair),
        `.vocab` (set or dict of all symbols: base characters + every merged symbol),
        `.name`, `.vocab_size`.

    Notes:
        * Build the initial vocab from characters of pre-tokenized words, weighted by
          word frequency -- counting words once and reusing the counts is what makes this
          fast enough for 5MB on CPU.
        * Re-counting the whole corpus from scratch on every merge is the naive O(n*m)
          version.  It is fine for a first correct implementation; if it is too slow,
          incremental pair-count updates are the optimization (measure before/after).
    """
    vocab = _build_vocab(texts, pretokenize)

    tok = ToyBPE()
    tok.vocab = {s for word in vocab for s in word}
    tok.config = {
        "num_merges": num_merges,
        "min_freq": min_freq,
        "pretokenize": pretokenize,
        "n_merges_learned": 0,
    }

    for _ in range(num_merges):
        stats = get_stats(vocab)
        if not stats:
            break
        # Highest frequency wins; ties are broken lexicographically on the pair tuple.
        pair, freq = min(stats.items(), key=lambda kv: (-kv[1], kv[0]))
        if freq < min_freq:
            break

        tok.merges.append(pair)
        tok.vocab.add(pair[0] + pair[1])
        vocab = merge_vocab(pair, vocab)

    tok.config["n_merges_learned"] = len(tok.merges)
    return tok


class ToyBPE:
    """A fitted toy BPE tokenizer.  Fill in the methods; keep the signatures."""

    name: str = "toy_bpe"

    def __init__(self) -> None:
        self.merges: list[Pair] = []
        self.vocab: set[Symbol] = set()
        self.config: dict = {}
        self._ranks: dict[Pair, int] | None = None

    @property
    def vocab_size(self) -> int:
        """Number of distinct symbols: base characters + one per learned merge."""
        return len(self.vocab)

    @property
    def ranks(self) -> dict[Pair, int]:
        """Map each merge pair to its training order; lower rank = higher priority."""
        if self._ranks is None or len(self._ranks) != len(self.merges):
            self._ranks = {pair: i for i, pair in enumerate(self.merges)}
        return self._ranks

    def _merge_block(self, symbols: list[Symbol]) -> list[Symbol]:
        """Merge the lowest-rank adjacent pair repeatedly until nothing merges."""
        if len(symbols) < 2:
            return symbols
        ranks = self.ranks
        while True:
            best_i, best_r = -1, None
            for i in range(len(symbols) - 1):
                a, b = symbols[i], symbols[i + 1]
                if a == UNK or b == UNK:
                    continue
                r = ranks.get((a, b))
                if r is not None and (best_r is None or r < best_r):
                    best_i, best_r = i, r
            if best_i < 0:
                return symbols
            a, b = symbols[best_i], symbols[best_i + 1]
            symbols = symbols[:best_i] + [a + b] + symbols[best_i + 2:]

    def encode(self, text: str) -> list[Symbol]:
        """Tokenize raw text into subword symbol strings.

        REQUIRED behavior:
          * Pre-tokenize the same way as during training.
          * Apply learned merges in training order, highest priority first (rank 0 wins).
          * Characters absent from the base vocabulary become the single token "<unk>".
          * Deterministic: same input -> same output, across processes.
        """
        scheme = self.config.get("pretokenize", "paper")
        out: list[Symbol] = []
        for block in _split_blocks(text):
            symbols = [s if s in self.vocab else UNK
                       for s in _block_symbols(block, scheme)]
            out.extend(self._merge_block(symbols))
        return out

    def decode(self, tokens: Sequence[Symbol]) -> str:
        """Invert encode(): remove "</w>" markers, re-insert word boundaries, join.

        REQUIRED: decode(encode(text)) == text for text made of known characters,
        including whitespace and Chinese.  Round-trip is asserted in the tests.
        """
        text = "".join(tokens)
        return text.replace(END_OF_WORD, "")

    def save(self, path: str | Path) -> None:
        """Persist to JSON so experiments are reproducible without retraining.

        REQUIRED schema (report.py and the notebook read this back):
            {"name": str, "config": {...},
             "merges": [["e", "s"], ["es", "t"], ...],
             "vocab": ["a", "b", ..., "es", ...]}
        """
        payload = {
            "name": self.name,
            "config": self.config,
            "merges": [list(pair) for pair in self.merges],
            # sorted: the file must be byte-identical across processes (set order is not).
            "vocab": sorted(self.vocab),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ToyBPE":
        """Rebuild a fitted tokenizer from the JSON written by save()."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))

        tok = cls()
        tok.name = payload.get("name") or cls.name
        tok.config = payload.get("config") or {}
        # JSON has no tuples: turn the 2-element lists back into Pair tuples so that
        # self.ranks keys match the tuples produced by _merge_block.
        tok.merges = [tuple(pair) for pair in payload.get("merges", [])]
        tok.vocab = set(payload.get("vocab", []))

        if not tok.vocab:
            # Tolerate a hand-written file without "vocab": rebuild it from the merges.
            for a, b in tok.merges:
                tok.vocab.update((a, b, a + b))
            tok.vocab.add(END_OF_WORD)

        tok._ranks = None  # rebuilt lazily from merges, in case load() is called twice
        return tok


@runtime_checkable
class Tokenizer(Protocol):
    """The only interface the rest of the project depends on.

    baselines.py (char / whitespace-word / HuggingFace BPE) already satisfies it, so you
    can run every experiment and produce every figure before your BPE exists -- then drop
    yours in and compare against the same numbers.
    """

    name: str

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> list[str]: ...

    def decode(self, tokens: Sequence[str]) -> str: ...