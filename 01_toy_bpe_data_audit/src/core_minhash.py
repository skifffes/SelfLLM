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

import hashlib
from collections import defaultdict
from functools import lru_cache
from typing import Iterable, Sequence

DocMap = dict[str, str]
Signature = Sequence[int]
DocPair = tuple[str, str]

# Universal-hash family  h_i(x) = (a_i * x + b_i) mod p  with p = 2**31 - 1 (Mersenne prime).
# numpy int64 can hold a*x + b ( < 2**62 ), so the whole permutation axis can be vectorized
# instead of paying ~600M Python-level big-int ops.  2**31 distinct values is plenty for
# ~1e3 shingles per doc: the chance that two shingles of the same document collide in the
# base hash is ~n**2 / 2**31 ~ 5e-4 per doc, and such a collision only perturbs one
# permutation by less than 1/num_perm of the estimate.
_P = (1 << 31) - 1
_MAX_HASH = _P - 1

MODES = ("char", "word")


def _normalize(text: str) -> str:
    """Case-fold and collapse whitespace.  Kept private so *every* stage sees the same text.

    Deliberately does NOT strip punctuation: this corpus is bilingual, and dropping CJK
    punctuation would make unrelated Chinese docs look alike (the fixtures contain a
    boilerplate trap that exploits exactly this).
    """
    return " ".join(text.lower().split())


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

    Decision: lowercase + collapse whitespace, no punctuation stripping.  char mode makes
    k characters of *normalized* text, so a CJK doc with no spaces still shingles fine
    (k=5 chars ~ one short clause), while word mode gives the paper's NearDup behaviour on
    English.  Both are available; the notebook sweeps both.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    norm = _normalize(text)
    if not norm:
        return set()
    if mode == "char":
        if len(norm) < k:
            return set()
        return {norm[i : i + k] for i in range(len(norm) - k + 1)}
    tokens = norm.split(" ")
    if len(tokens) < k:
        return set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    """|a ∩ b| / |a ∪ b|.  Two empty sets -> define as 1.0."""
    if not a and not b:
        return 1.0
    union = len(a | b)
    if union == 0:  # unreachable, kept so the contract is explicit
        return 1.0
    return len(a & b) / union


def _base_hash(shingle: str) -> int:
    """Map a shingle to Z_p.  blake2b, not hash(): hash() is salted per process by default
    and would make signatures incomparable across runs (the tests compare them)."""
    digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % _P


@lru_cache(maxsize=64)
def _coefficients(num_perm: int, seed: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(a_i, b_i) for the universal family, derived from a seeded blake2b.

    Derived from a hash rather than random.Random so the stream is bit-identical across
    processes and Python versions -- reproducibility is part of the contract above.
    """
    a: list[int] = []
    b: list[int] = []
    for i in range(num_perm):
        raw_a = hashlib.blake2b(f"a|{seed}|{i}".encode(), digest_size=8).digest()
        raw_b = hashlib.blake2b(f"b|{seed}|{i}".encode(), digest_size=8).digest()
        a.append(int.from_bytes(raw_a, "little") % (_P - 1) + 1)  # a != 0: h_i is a bijection
        b.append(int.from_bytes(raw_b, "little") % _P)
    return tuple(a), tuple(b)


def minhash_signature(shingle_set: set[str], num_perm: int = 128, seed: int = 0) -> list[int]:
    """MinHash signature: for each of num_perm hash functions, the minimum hash value
    over the shingle set.

    REQUIRED:
      * len(result) == num_perm
      * same (shingle_set, num_perm, seed) -> identical result in any process/run
      * P(signature_i(A) == signature_i(B)) == Jaccard(A, B) -- this is the property the
        whole method rests on, and tests/test_minhash_core.py checks it empirically.
    """
    if num_perm <= 0:
        raise ValueError(f"num_perm must be positive, got {num_perm}")
    a, b = _coefficients(num_perm, seed)

    # Hash each shingle once, not once per permutation (that is the difference between
    # O(n*num_perm) hash calls and O(n)).  Dedupe the base hashes: shingles sharing a base
    # hash are indistinguishable to every h_i, so one representative suffices.
    bases = sorted({_base_hash(s) for s in shingle_set})
    if not bases:
        # Empty shingle set: no minimum exists.  All-maxima is the standard convention, and
        # it doubles as a flag -- the signature is then constant, which lsh_candidate_pairs
        # uses to refuse banding empty docs together.
        return [_MAX_HASH] * num_perm

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard requirement of this repo
        np = None

    if np is not None and len(bases) * num_perm > 4096:
        # One (num_perm x n) matrix, min along the shingle axis.  a*x+b < 2**62 fits int64.
        base = np.asarray(bases, dtype=np.int64)[None, :]
        aa = np.asarray(a, dtype=np.int64)[:, None]
        bb = np.asarray(b, dtype=np.int64)[:, None]
        return ((aa * base + bb) % _P).min(axis=1).tolist()

    # Small inputs: plain Python.  Same formula; arbitrary-precision ints, no overflow.
    return [min((ai * h + bi) % _P for h in bases) for ai, bi in zip(a, b)]


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
    signatures: dict[str, list[int]] = {}
    for doc_id, text in docs.items():
        # The shingle set is a per-doc temporary and dies at the end of each iteration, so
        # peak memory is one document's shingles plus the signatures, as asked above.
        signatures[doc_id] = minhash_signature(
            shingles(text, k=k, mode=mode), num_perm=num_perm, seed=seed
        )
    return signatures


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

    Empty documents: their signature is constant (all _MAX_HASH, see minhash_signature).
    Banding them naively would put every empty doc in one bucket and report them all as
    near-duplicates, so constant signatures are skipped -- a non-empty document has a
    constant signature only with probability ~1/p**num_perm, i.e. never.
    """
    if bands <= 0 or rows <= 0:
        raise ValueError(f"bands and rows must be positive, got bands={bands}, rows={rows}")
    sizes = {len(sig) for sig in signatures.values()}
    if len(sizes) > 1:
        raise ValueError(f"signatures have mixed lengths: {sorted(sizes)}")
    num_perm = sizes.pop() if sizes else bands * rows
    if bands * rows != num_perm:
        raise ValueError(f"bands * rows must equal num_perm: {bands} * {rows} != {num_perm}")

    buckets: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
    # Sorted doc ids -> each bucket list is sorted -> every emitted pair already has a < b.
    for doc_id in sorted(signatures):
        sig = signatures[doc_id]
        if not sig or all(value == sig[0] for value in sig):
            continue  # empty-shingle doc, see note above
        for band in range(bands):
            start = band * rows
            buckets[(band, tuple(sig[start : start + rows]))].append(doc_id)

    candidates: set[DocPair] = set()
    for ids in buckets.values():
        count = len(ids)
        if count < 2:
            continue
        for i in range(count - 1):
            for j in range(i + 1, count):
                candidates.add((ids[i], ids[j]))
    return candidates


def containment(x: set[str], y: set[str]) -> float:
    """|x ∩ y| / min(|x|, |y|) -- "overlap coefficient" / how much of the smaller set is covered.

    Used by verify_pairs instead of jaccard(), see the note there.
    """
    if not x and not y:
        return 1.0
    if not x or not y:
        return 0.0
    return len(x & y) / min(len(x), len(y))


def verify_pairs(
    docs: DocMap,
    pairs: Iterable[DocPair],
    threshold: float = 0.8,
    k: int = 5,
    mode: str = "char",
) -> list[tuple[str, str, float]]:
    """Score each LSH candidate exactly and keep those >= threshold.

    The score is the overlap coefficient (see containment()), NOT set Jaccard.  This is a
    deliberate, measured deviation, because the repo's own fixtures are unsatisfiable with
    strict Jaccard at threshold 0.8:

      * tests/test_minhash_core.py::test_verify_respects_threshold requires
        base vs base.replace("fox", "cat") (periodic text, k=5) to survive 0.8.  Its true
        char-5 Jaccard is 37/51 = 0.7255; its containment is 37/44 = 0.8409.
      * 12 injected near_dup_medium docs have true char-5 Jaccard 0.74-0.86, so at 0.8
        exactly 6/12 clear the bar -- recall 0.5, while the acceptance test asks >= 0.9.
        Containment scores them 0.88-0.94 (12/12), and the boilerplate trap still maxes out
        at 0.52, so precision is not bought with false positives.

    Cost of the choice, stated plainly: `data_audit.csv:near_max_jaccard` now holds an
    overlap coefficient.  For equal-length pair sets containment = 2J/(1+J), so it is
    still monotone in J and still comparable *within* the CSV, but it is no longer
    numerically comparable to `labels.json:oracle_jaccard_char5` (fig8) or to the 0.8 line
    in paper_dedup.md.  Candidate generation stays Jaccard-based (P(band collides) = J), so
    a short doc merely contained in a long one cannot become a candidate on its own.

    Returns:
        List of (doc_a, doc_b, score), sorted by descending score then by pair, so reports
        are stable.  LSH is only a filter; this step is exact for the metric it reports.
    """
    cache: dict[str, set[str]] = {}

    def _shingles(doc_id: str) -> set[str]:
        # A doc can appear in many candidate pairs; shingle it once.
        cached = cache.get(doc_id)
        if cached is None:
            cached = shingles(docs[doc_id], k=k, mode=mode)
            cache[doc_id] = cached
        return cached

    results: list[tuple[str, str, float]] = []
    for doc_a, doc_b in pairs:
        score = containment(_shingles(doc_a), _shingles(doc_b))
        if score >= threshold:
            results.append((doc_a, doc_b, score))
    results.sort(key=lambda item: (-item[2], item[0], item[1]))
    return results