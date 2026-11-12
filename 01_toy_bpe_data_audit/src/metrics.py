"""Length distributions and ratio metrics -- the measurement side of the acceptance criteria.

"准确计算不同切分策略的长度分布" lives here.  Nothing in this file tokenizes anything
itself: it takes an object satisfying the Tokenizer protocol and counts.  That keeps the
measurement independent of the implementation, so a bug in your BPE shows up as a weird
number instead of silently cancelling out.

Word counts: English uses whitespace.  Chinese has no whitespace, so tokens/word needs a
segmenter -- jieba is used when installed.  Without it the Chinese tokens/word column is
NaN rather than a fake number (documented in the data card).
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .baselines import UNK

_SEGMENTER = None
_SEGMENTER_TRIED = False


def _get_segmenter():
    global _SEGMENTER, _SEGMENTER_TRIED
    if not _SEGMENTER_TRIED:
        _SEGMENTER_TRIED = True
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import jieba

            jieba.setLogLevel(60)
            jieba.initialize()
            _SEGMENTER = jieba
        except ImportError:
            _SEGMENTER = None
    return _SEGMENTER


def count_words(text: str, lang: str) -> int | None:
    if lang != "zh":
        return len(text.split())
    seg = _get_segmenter()
    if seg is None:
        return None
    return len([w for w in seg.lcut(text) if w.strip()])


def length_stats(values: Sequence[int]) -> dict[str, float]:
    if len(values) == 0:
        return {k: math.nan for k in ("n", "mean", "std", "p50", "p90", "p95", "p99", "min", "max")}
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def doc_stats(docs: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Precompute per-document char/word counts once, reused across all tokenizers."""
    out: dict[str, dict[str, Any]] = {}
    for d in docs:
        out[d["doc_id"]] = {
            "lang": d["lang"],
            "n_chars": len(d["text"]),
            "n_words": count_words(d["text"], d["lang"]),
        }
    return out


def evaluate_tokenizer(
    tok: Any,
    docs: Sequence[dict[str, Any]],
    stats: dict[str, dict[str, Any]],
    lang: str = "all",
    max_docs: int | None = None,
) -> tuple[dict[str, Any], list[int]]:
    """Tokenize every doc of one language and summarize the resulting length distribution."""
    selected = [d for d in docs if lang == "all" or d["lang"] == lang]
    if max_docs is not None and len(selected) > max_docs:
        step = len(selected) / max_docs
        selected = [selected[int(i * step)] for i in range(max_docs)]

    lengths: list[int] = []
    n_tokens = n_chars = n_unk = 0
    n_words = 0
    words_known = True
    for d in selected:
        toks = tok.encode(d["text"])
        lengths.append(len(toks))
        n_tokens += len(toks)
        n_unk += sum(1 for t in toks if t == UNK)
        st = stats[d["doc_id"]]
        n_chars += st["n_chars"]
        if st["n_words"] is None:
            words_known = False
        else:
            n_words += st["n_words"]

    ls = length_stats(lengths)
    row = {
        "tokenizer": tok.name,
        "lang": lang,
        "vocab_size": int(tok.vocab_size),
        "n_docs": len(selected),
        "n_chars": n_chars,
        "n_words": n_words if words_known else None,
        "n_tokens": n_tokens,
        "mean_len": ls["mean"],
        "std_len": ls["std"],
        "p50_len": ls["p50"],
        "p90_len": ls["p90"],
        "p95_len": ls["p95"],
        "p99_len": ls["p99"],
        "max_len": ls["max"],
        "tokens_per_char": n_tokens / n_chars if n_chars else math.nan,
        "chars_per_token": n_chars / n_tokens if n_tokens else math.nan,
        "tokens_per_word": (n_tokens / n_words) if (words_known and n_words) else math.nan,
        "unk_rate": n_unk / n_tokens if n_tokens else 0.0,
    }
    return row, lengths


def roundtrip_rate(tok: Any, texts: Sequence[str], limit: int = 200) -> float:
    """Fraction of docs where decode(encode(text)) == text.

    The word baseline is intentionally lossy (whitespace is dropped), so a low score there
    is expected, not a bug.  For your ToyBPE this must be 1.0 on known characters.
    """
    ok = 0
    sample = texts[:limit]
    for t in sample:
        if tok.decode(tok.encode(t)) == t:
            ok += 1
    return ok / len(sample) if sample else math.nan


def build_comparison(
    tokenizers: Sequence[Any],
    docs: Sequence[dict[str, Any]],
    langs: Sequence[str] = ("en", "zh", "all"),
    max_docs: int | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str], list[int]]]:
    """Run every tokenizer over every language slice -> tidy metrics table + raw lengths."""
    stats = doc_stats(docs)
    rows: list[dict[str, Any]] = []
    lengths: dict[tuple[str, str], list[int]] = {}
    for tok in tokenizers:
        for lang in langs:
            row, lens = evaluate_tokenizer(tok, docs, stats, lang=lang, max_docs=max_docs)
            rows.append(row)
            lengths[(tok.name, lang)] = lens
    df = pd.DataFrame(rows)
    for lang in langs:
        base = df[(df.tokenizer == "char") & (df.lang == lang)]["n_tokens"]
        if len(base) and base.iloc[0]:
            mask = df.lang == lang
            df.loc[mask, "compression_vs_char"] = base.iloc[0] / df.loc[mask, "n_tokens"]
    return df, lengths