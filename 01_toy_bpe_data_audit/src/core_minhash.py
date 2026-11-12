"""YOUR CORE CODE #2 -- MinHash + LSH near-duplicate detection.

Paper: Lee et al. (2021), *Deduplicating Training Data Makes Language Models Better*,
arXiv:2107.06499.  Section 3 ("Near-duplicate detection") is the pipeline you are
reimplementing: shingle -> MinHash signature -> LSH banding -> verify with exact Jaccard.

Everything below is a stub.  src/audit_pipeline.py calls these four stages in order and
never looks at data/fixtures/labels.json -- the ground truth is only used afterwards, by
tests and report.py, to score your detector.  So you cannot accidentally pass by reading
the answers, and you can see exactly which stage is losing recall.

Suggested order:
    1. shingles            -> tests/test_minhash_core.py::test_shingles
    2. jaccard             -> tests/test_minhash_core.py::test_jaccard
    3. minhash_signature   -> tests/test_minhash_core.py::test_signature_bounds_and_determinism
                              tests/test_minhash_core.py::test_signature_tracks_jaccard
    4. lsh_candidate_pairs -> tests/test_minhash_core.py::test_banding_arithmetic
    5. verify_pairs        -> tests/test_minhash_core.py::test_verify_respects_threshold
    6. end to end          -> tests/test_acceptance.py  (recall / precision on injected fixtures)

Hard requirements (these are correctness constraints, not style):
  * DETERMINISM ACROSS PROCESSES.  Python's built-in hash() is randomly salted per process
    for str/bytes, so a signature built on hash() changes every run and your recall numbers
    become irreproducible.  Use hashlib (e.g. blake2b/sha1 of the shingle, truncated to an
    int) or a fixed-seed PRNG.  This is the single most common bug in toy MinHash.
  * Empty shingle sets must not crash: return a well-defined signature (e.g. all maxima)
    and let the caller decide.  Empty/near-empty docs exist in real corpora.
  * `bands * rows == num_perm` must hold; raise ValueError otherwise.
  * Candidate pairs are unordered and canonical: always (a, b) with a < b, no self-pairs,
    no duplicates.

Design decisions that are YOURS (each one is a knob the notebook sweeps):
  * char k-shingles vs word k-shingles, and the value of k.  The paper uses word 5-grams
    for NearDup; char 5-grams behave very differently on Chinese.  Try both, plot both.
  * Normalization before shingling: lowercase? collapse whitespace? strip punctuation?
    Boilerplate-heavy text (headers, licenses) creates false positives -- there is a
    deliberately planted "boilerplate trap" group in the fixtures to expose this.
  * How to turn hashes into a signature: the (a*h + b) mod p universal-hash family, or
    num_perm independent permutations.  Both are valid; state which you used.
  * Threshold.  With b bands and r rows, the probability that a pair with Jaccard s becomes
    a candidate is 1 - (1 - s**r)**b.  Derive the S-curve, pick (b, r) for your target
    threshold, and compare the predicted curve with the measured one -- that comparison is
    one of the required figures.

Performance budget (this corpus: ~4k docs x ~1200 chars, CPU only, <2GB):
  Hashing every shingle num_perm times with hashlib is ~600M hash calls -> many minutes.
  The cheap route is one base hash per shingle, then num_perm integer ops of the
  (a*h + b) mod p family: ~5M hashlib calls + ~600M int ops, tens of seconds.  numpy
  vectorization over the permutation axis helps further.  Measure it; the notebook has a
  cell for timing.  For fast iteration use `--max-docs 400 --num-perm 64` on the audit CLI
  and only run the full corpus once your tests pass.
"""

from __future__ import annotations

from typing import Iterable, Sequence

DocMap = dict[str, str]
Signature = Sequence[int]
DocPair = tuple[str, str]


def shingles(text: str, k: int = 5, mode: str = "char") -> set[str]:
    """Turn a document into a set of k-shingles.

    Args:
        text: raw document text.
        k: shingle width.
        mode: "char" -> sliding window of k characters;
              "word" -> sliding window of k whitespace-separated tokens.

    Returns:
        A set (duplicates collapse -- that is what makes Jaccard meaningful).
        Documents shorter than k should yield an empty set, not an exception.

    Normalization is your choice; keep it inside this function so that signatures and
    verification always agree on it.
    """
    raise NotImplementedError("TODO(you): k-shingles, char or word mode")


def jaccard(a: set[str], b: set[str]) -> float:
    """|a ∩ b| / |a ∪ b|.  Two empty sets -> define as 1.0."""
    raise NotImplementedError("TODO(you): exact Jaccard similarity")


def minhash_signature(shingle_set: set[str], num_perm: int = 128, seed: int = 0) -> list[int]:
    """MinHash signature: for each of num_perm hash functions, the minimum hash value
    over the shingle set.

    REQUIRED:
      * len(result) == num_perm
      * same (shingle_set, num_perm, seed) -> identical result in any process/run
      * P(signature_i(A) == signature_i(B)) == Jaccard(A, B) -- this is the property the
        whole method rests on, and tests/test_minhash_core.py checks it empirically.
    """
    raise NotImplementedError("TODO(you): num_perm independent min-hashes")


def build_signatures(
    docs: DocMap, num_perm: int = 128, k: int = 5, mode: str = "char", seed: int = 0
) -> dict[str, list[int]]:
    """Shingle + sign every document.  Convenience wrapper used by the audit pipeline.

    Args:
        docs: doc_id -> text.

    Returns:
        doc_id -> signature.  Keep memory in mind: ~5k docs x 128 perms is a few MB, fine;
        do not materialize all shingle sets of the whole corpus at once if you can avoid it.
    """
    raise NotImplementedError("TODO(you): loop shingles + minhash_signature")


def lsh_candidate_pairs(
    signatures: dict[str, list[int]], bands: int = 32, rows: int = 4
) -> set[DocPair]:
    """Locality-sensitive hashing: split each signature into `bands` bands of `rows`
    values; two docs are candidates if ANY band is identical.

    REQUIRED:
      * bands * rows == num_perm (raise ValueError otherwise)
      * return canonical unordered pairs (a, b), a < b, no self-pairs
      * a doc with an empty signature must not collide with every other empty doc unless
        that is genuinely what banding implies -- decide and document.

    This is the recall/precision knob: more bands -> higher recall, more false positives.
    """
    raise NotImplementedError("TODO(you): band the signatures and bucket them")


def verify_pairs(
    docs: DocMap,
    pairs: Iterable[DocPair],
    threshold: float = 0.8,
    k: int = 5,
    mode: str = "char",
) -> list[tuple[str, str, float]]:
    """Compute exact Jaccard for each LSH candidate and keep those >= threshold.

    Returns:
        List of (doc_a, doc_b, jaccard), sorted by descending jaccard then by pair, so
        reports are stable.  LSH is a filter; this step is what makes the answer exact.
    """
    raise NotImplementedError("TODO(you): exact verification of candidates")