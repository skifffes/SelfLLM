"""Hand-computed BPE fixtures.  These are the acceptance tests for YOUR core code.

They skip (not fail) while src/core_bpe.py is still a stub, so `pytest` is green on day one
and turns into real assertions the moment you implement each function.

The vocabulary below is the classic example from the BPE paper / karpathy's minbpe:
    low x5, lower x2, newest x6, widest x3
Pair counts: es=9, st=9, t</w>=9, we=8, lo=7, ow=7, ...
With lexicographic tie-breaking the first three merges are es, es+t, est+</w>.
"""

from __future__ import annotations

import pytest

from src.core_bpe import UNK, get_stats, merge_vocab, train_bpe

V = {
    ("l", "o", "w", "</w>"): 5,
    ("l", "o", "w", "e", "r", "</w>"): 2,
    ("n", "e", "w", "e", "s", "t", "</w>"): 6,
    ("w", "i", "d", "e", "s", "t", "</w>"): 3,
}
TEXT = "low low low low low lower lower newest newest newest newest newest newest widest widest widest"


def call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} is still a stub in src/core_bpe.py")


def test_get_stats_hand_computed():
    stats = call(get_stats, V)
    assert stats[("e", "s")] == 9
    assert stats[("s", "t")] == 9
    assert stats[("t", "</w>")] == 9
    assert stats[("w", "e")] == 8
    assert stats[("l", "o")] == 7
    assert stats[("o", "w")] == 7
    assert stats[("w", "</w>")] == 5
    assert sum(stats.values()) == sum(len(w) - 1 for w, c in V.items() for _ in range(c))


def test_merge_vocab_hand_computed():
    out = call(merge_vocab, ("e", "s"), V)
    assert ("n", "e", "w", "es", "t", "</w>") in out
    assert out[("n", "e", "w", "es", "t", "</w>")] == 6
    assert ("w", "i", "d", "es", "t", "</w>") in out
    assert out[("l", "o", "w", "</w>")] == 5
    assert ("n", "e", "w", "e", "s", "t", "</w>") not in out
    assert V[("n", "e", "w", "e", "s", "t", "</w>")] == 6, "merge_vocab must not mutate its input"


def test_first_three_merges():
    tok = call(train_bpe, [TEXT], num_merges=3)
    assert list(tok.merges) == [("e", "s"), ("es", "t"), ("est", "</w>")]


def test_merge_budget_is_exact():
    tok = call(train_bpe, [TEXT], num_merges=2)
    assert len(tok.merges) == 2
    assert tok.vocab_size >= len({c for w in V for c in w}) + 2


def test_roundtrip():
    tok = call(train_bpe, [TEXT, "hello world hello", "中文 测试 中文"], num_merges=40)
    for text in (TEXT, "hello world hello", "中文 测试 中文", "low lower newest widest"):
        assert tok.decode(tok.encode(text)) == text


def test_unknown_char_becomes_unk():
    tok = call(train_bpe, [TEXT], num_merges=10)
    assert UNK in tok.encode("low zzz")


def test_save_load(tmp_path):
    tok = call(train_bpe, [TEXT], num_merges=10)
    path = tmp_path / "toy.json"
    tok.save(path)
    from src.core_bpe import ToyBPE

    back = ToyBPE.load(path)
    assert back.encode(TEXT) == tok.encode(TEXT)
    assert list(back.merges) == list(tok.merges)