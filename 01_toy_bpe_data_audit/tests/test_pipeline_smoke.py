"""Invariants of the scaffolding itself (baselines / metrics / audit).

These must pass regardless of whether your core code exists -- they protect the
measurement layer, so a weird number later means your tokenizer, not the ruler.
"""

from __future__ import annotations

import math

import pytest

from src.baselines import CharTokenizer, WordTokenizer, build_baselines
from src.core_bpe import Tokenizer
from src.metrics import build_comparison, length_stats


@pytest.fixture(scope="module")
def small_docs():
    from src.common import load_corpus

    return load_corpus("train")[:120] + load_corpus("test")[:40]


def test_baselines_satisfy_protocol(small_docs):
    for tok in build_baselines([d["text"] for d in small_docs], hf_vocab_sizes=(500,)):
        assert isinstance(tok, Tokenizer), tok.name
        assert tok.vocab_size > 0
        toks = tok.encode(small_docs[0]["text"])
        assert isinstance(toks, list) and all(isinstance(t, str) for t in toks)


def test_char_baseline_is_the_ruler(small_docs):
    df, lengths = build_comparison([CharTokenizer.fit([d["text"] for d in small_docs])], small_docs)
    en = df[(df.tokenizer == "char") & (df.lang == "en")].iloc[0]
    assert en.tokens_per_char == pytest.approx(1.0)
    assert en.compression_vs_char == pytest.approx(1.0)
    assert sum(lengths[("char", "en")]) == en.n_tokens


def test_word_baseline_tokens_per_word_is_one_on_english(small_docs):
    df, _ = build_comparison([WordTokenizer.fit([d["text"] for d in small_docs])], small_docs)
    en = df[(df.tokenizer == "word") & (df.lang == "en")].iloc[0]
    assert en.tokens_per_word == pytest.approx(1.0)


def test_percentiles_ordered(small_docs):
    df, _ = build_comparison(build_baselines([d["text"] for d in small_docs], (500,)), small_docs)
    for r in df.itertuples():
        assert r.p50_len <= r.p90_len <= r.p95_len <= r.p99_len <= r.max_len
        assert r.mean_len > 0


def test_length_stats_empty():
    s = length_stats([])
    assert math.isnan(s["mean"])


def test_chinese_word_baseline_is_broken_on_purpose(small_docs):
    """Documents the known failure mode so a future 'fix' has to be deliberate."""
    train_texts = [d["text"] for d in small_docs if d["split"] == "train"]
    df, _ = build_comparison([WordTokenizer.fit(train_texts)], small_docs)
    zh = df[(df.tokenizer == "word") & (df.lang == "zh")].iloc[0]
    assert zh.tokens_per_word < 0.2
    assert zh.unk_rate > 0.05