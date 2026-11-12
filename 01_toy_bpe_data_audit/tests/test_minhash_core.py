"""Acceptance tests for YOUR MinHash/LSH code.  Skip while the stub is unimplemented.

The statistical test (signature tracks Jaccard) is the one that catches the classic bugs:
built-in hash() salting, wrong modulus, off-by-one in the band split.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from src import core_minhash as cm
from src.common import ROOT


def call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} is still a stub in src/core_minhash.py")


def test_shingles_char_and_word():
    assert call(cm.shingles, "abcdef", k=5, mode="char") == {"abcde", "bcdef"}
    assert call(cm.shingles, "a b c d e f", k=2, mode="word") == {"a b", "b c", "c d", "d e", "e f"}
    assert call(cm.shingles, "abc", k=5, mode="char") == set()


def test_jaccard_known_sets():
    a, b = {"x", "y", "z"}, {"y", "z", "w"}
    assert call(cm.jaccard, a, b) == pytest.approx(2 / 4)
    assert call(cm.jaccard, set(), set()) == 1.0


def test_signature_shape_and_determinism():
    s = {"abcde", "bcdef", "cdefg"}
    sig = call(cm.minhash_signature, s, num_perm=64, seed=0)
    assert len(sig) == 64
    assert list(sig) == list(call(cm.minhash_signature, s, num_perm=64, seed=0))


def test_signature_deterministic_across_processes():
    s = {"abcde", "bcdef", "cdefg"}
    sig = list(call(cm.minhash_signature, s, num_perm=32, seed=0))
    code = (
        "from src.core_minhash import minhash_signature;"
        "print(list(minhash_signature({'abcde','bcdef','cdefg'}, num_perm=32, seed=0)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == str(sig), "built-in hash() is salted per process -- use hashlib"


def test_signature_tracks_jaccard():
    universe = [f"s{i:04d}" for i in range(200)]
    a = set(universe[:100])
    b = set(universe[40:140])  # |a&b|=60, |a|b|=140 -> Jaccard 3/7
    true = 60 / 140
    sa = call(cm.minhash_signature, a, num_perm=512, seed=1)
    sb = call(cm.minhash_signature, b, num_perm=512, seed=1)
    est = sum(x == y for x, y in zip(sa, sb)) / 512
    assert abs(est - true) < 0.08, f"MinHash estimate {est:.3f} far from Jaccard {true:.3f}"


def test_banding_arithmetic_and_candidates():
    sig = [i % 7 for i in range(128)]
    sigs = {"a": sig, "b": list(sig), "c": [(i + 1) % 7 for i in range(128)]}
    with pytest.raises(ValueError):
        call(cm.lsh_candidate_pairs, sigs, bands=30, rows=4)
    pairs = call(cm.lsh_candidate_pairs, sigs, bands=32, rows=4)
    assert ("a", "b") in pairs
    assert ("a", "a") not in pairs
    assert all(x < y for x, y in pairs)


def test_verify_respects_threshold():
    base = "the quick brown fox jumps over the lazy dog " * 8
    docs = {
        "a": base,
        "b": base.replace("fox", "cat"),
        "c": "completely unrelated text about databases and indexes " * 8,
    }
    sigs = call(cm.build_signatures, docs, num_perm=64, k=5, mode="char")
    cand = call(cm.lsh_candidate_pairs, sigs, bands=16, rows=4)
    kept = call(cm.verify_pairs, docs, cand, threshold=0.8, k=5, mode="char")
    ids = {tuple(sorted((a, b))) for a, b, _ in kept}
    assert ("a", "b") in ids
    assert ("a", "c") not in ids and ("b", "c") not in ids
    assert all(s >= 0.8 for _, _, s in kept)