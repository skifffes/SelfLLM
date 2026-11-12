"""Ground-truth oracle -- used to LABEL fixtures and SCORE detectors, never to detect.

Deliberately separate from src/core_minhash.py: if the detector and its own evaluation
shared code, a bug in that code would silently cancel out and every test would pass.
These are brute-force reference definitions (exact set Jaccard, no hashing, no LSH).

You are still expected to write your own shingles/jaccard in core_minhash.py; comparing
the two on random inputs is a useful sanity check.
"""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def shingles_oracle(text: str, k: int = 5, mode: str = "char") -> set[str]:
    if mode == "word":
        toks = text.split()
        return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def jaccard_oracle(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def similarity_oracle(text_a: str, text_b: str, k: int = 5, mode: str = "char") -> float:
    return jaccard_oracle(shingles_oracle(text_a, k, mode), shingles_oracle(text_b, k, mode))


def normalize_for_exact(text: str) -> str:
    """Whitespace/case normalization for the *exact* dedup fingerprint.

    Two docs identical up to case and whitespace are the same document for dedup purposes.
    Whether to normalize this aggressively is an experiment: see the boilerplate trap.
    """
    return _WS.sub(" ", text).strip().lower()