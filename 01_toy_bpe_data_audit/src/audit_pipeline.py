"""Data audit pipeline: exact dedup + near-dup (your MinHash) + train->test contamination.

This is the plumbing around your core code.  It calls src/core_minhash.py for the
near-duplicate stage and never reads data/fixtures/labels.json -- ground truth is only
joined afterwards by src/report.py and tests/test_acceptance.py.

If core_minhash is still a stub, the pipeline degrades to exact + substring detection and
says so in the summary, so you can run the whole thing on day one.

Detection layers, cheapest first:
  1. exact      -- sha256 of whitespace/case-normalized text ( catches verbatim copies )
  2. near       -- your MinHash + LSH + exact Jaccard verification
  3. substring  -- probe spans of each test doc searched in the concatenated train text.
                   A lite version of the paper's ExactSubstr: it catches the planted
                   300-char span leaks that document-level hashing misses, but it can
                   miss leaks shorter than a probe or spanning a probe boundary.  That
                   limitation is a required item in notes/failure_analysis.md.

Run:
    python -m src.audit_pipeline                       # full corpus, near-dup if implemented
    python -m src.audit_pipeline --max-docs 400 --num-perm 64   # fast iteration
    python -m src.audit_pipeline --no-near               # exact + substring only
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .common import (
    CORPUS,
    OUTPUTS,
    load_corpus,
    text_stats,
    utf8_stdout,
    write_json,
    write_jsonl,
)
from .oracle import normalize_for_exact


@dataclass
class AuditConfig:
    num_perm: int = 128
    k: int = 5
    mode: str = "char"
    bands: int = 32
    rows: int = 4
    threshold: float = 0.8
    seed: int = 0
    max_docs: int | None = None
    probe_chars: int = 100
    n_probes: int = 5
    enable_near: bool = True

    def validate(self) -> None:
        if self.bands * self.rows != self.num_perm:
            raise ValueError(
                f"bands*rows must equal num_perm: {self.bands}*{self.rows} != {self.num_perm}"
            )


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_for_exact(text).encode("utf-8")).hexdigest()


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            keep, drop = sorted((ra, rb))
            self.parent[drop] = keep


def exact_groups(docs: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for d in docs:
        groups.setdefault(fingerprint(d["text"]), []).append(d["doc_id"])
    return {fp: sorted(ids) for fp, ids in groups.items()}


def substring_leaks(
    train: Sequence[dict[str, Any]],
    test: Sequence[dict[str, Any]],
    probe_chars: int = 100,
    n_probes: int = 5,
) -> list[dict[str, Any]]:
    """Search evenly spaced probe spans of each test doc inside the whole train text."""
    blob_parts: list[str] = []
    starts: list[int] = []
    ids: list[str] = []
    pos = 0
    for d in train:
        starts.append(pos)
        ids.append(d["doc_id"])
        blob_parts.append(d["text"])
        pos += len(d["text"]) + 1
    blob = "\n".join(blob_parts)

    found: list[dict[str, Any]] = []
    for d in test:
        text = d["text"]
        if len(text) < probe_chars:
            continue
        best: dict[str, Any] | None = None
        for i in range(n_probes):
            off = int(i * (len(text) - probe_chars) / max(1, n_probes - 1))
            probe = text[off : off + probe_chars]
            if len(probe) < probe_chars or not probe.strip():
                continue
            hit = blob.find(probe)
            if hit >= 0:
                j = bisect.bisect_right(starts, hit) - 1
                best = {
                    "test_doc_id": d["doc_id"],
                    "train_doc_id": ids[j],
                    "probe_offset": off,
                    "train_offset": hit - starts[j],
                    "probe_chars": probe_chars,
                }
                break
        if best:
            found.append(best)
    return found


def near_duplicate_stage(
    docs: Sequence[dict[str, Any]], cfg: AuditConfig
) -> tuple[list[tuple[str, str, float]], dict[str, Any]]:
    """Run YOUR MinHash/LSH.  Returns (verified_pairs, timing/status info)."""
    from . import core_minhash as cm

    doc_map = {d["doc_id"]: d["text"] for d in docs}
    info: dict[str, Any] = {"status": "ok"}
    t0 = time.time()
    try:
        sigs = cm.build_signatures(
            doc_map, num_perm=cfg.num_perm, k=cfg.k, mode=cfg.mode, seed=cfg.seed
        )
        t_sig = time.time() - t0
        t1 = time.time()
        candidates = cm.lsh_candidate_pairs(sigs, bands=cfg.bands, rows=cfg.rows)
        t_lsh = time.time() - t1
        t2 = time.time()
        pairs = cm.verify_pairs(
            doc_map, candidates, threshold=cfg.threshold, k=cfg.k, mode=cfg.mode
        )
        t_ver = time.time() - t2
    except NotImplementedError as exc:
        return [], {
            "status": "not_implemented",
            "detail": str(exc),
            "hint": "implement src/core_minhash.py, or rerun with --no-near",
        }
    info.update(
        {
            "n_docs": len(doc_map),
            "n_signatures": len(sigs),
            "n_candidates": len(candidates),
            "n_verified_pairs": len(pairs),
            "seconds_signatures": round(t_sig, 3),
            "seconds_lsh": round(t_lsh, 3),
            "seconds_verify": round(t_ver, 3),
        }
    )
    return [(a, b, float(s)) for a, b, s in pairs], info


def _subsample(docs: list[dict[str, Any]], max_docs: int | None, seed: int) -> list[dict[str, Any]]:
    if max_docs is None or len(docs) <= max_docs:
        return docs
    rng = random.Random(seed)
    return rng.sample(docs, max_docs)


def run_audit(
    train: Sequence[dict[str, Any]],
    test: Sequence[dict[str, Any]],
    cfg: AuditConfig | None = None,
    out_dir: Path = OUTPUTS,
) -> dict[str, Any]:
    cfg = cfg or AuditConfig()
    cfg.validate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = _subsample(list(train), cfg.max_docs, cfg.seed)
    test = _subsample(list(test), cfg.max_docs, cfg.seed)
    docs = list(train) + list(test)
    by_id = {d["doc_id"]: d for d in docs}
    split_of = {d["doc_id"]: d["split"] for d in docs}
    t_start = time.time()

    fp_groups = exact_groups(docs)
    fp_of = {d["doc_id"]: fingerprint(d["text"]) for d in docs}
    exact_dup_ids: set[str] = set()
    exact_cross: list[tuple[str, str]] = []
    for fp, ids in fp_groups.items():
        if len(ids) < 2:
            continue
        keep = ids[0]
        for other in ids[1:]:
            exact_dup_ids.add(other)
            if split_of[other] != split_of[keep]:
                exact_cross.append((keep, other))

    near_pairs, near_info = (
        near_duplicate_stage(docs, cfg) if cfg.enable_near else ([], {"status": "disabled"})
    )
    uf = UnionFind()
    near_max: dict[str, float] = {}
    near_partner: dict[str, str] = {}
    cross_near: list[tuple[str, str, float]] = []
    for a, b, s in near_pairs:
        uf.union(a, b)
        for x, y in ((a, b), (b, a)):
            if s > near_max.get(x, -1.0):
                near_max[x] = s
                near_partner[x] = y
        if split_of[a] != split_of[b]:
            cross_near.append((a, b, s))

    near_groups: dict[str, list[str]] = {}
    for d in docs:
        root = uf.find(d["doc_id"])
        near_groups.setdefault(root, []).append(d["doc_id"])
    near_dup_ids: set[str] = set()
    for root, members in near_groups.items():
        if len(members) > 1:
            for m in sorted(members)[1:]:
                near_dup_ids.add(m)

    subs = substring_leaks(train, test, cfg.probe_chars, cfg.n_probes)
    sub_ids = {r["test_doc_id"] for r in subs}
    sub_partner = {r["test_doc_id"]: r["train_doc_id"] for r in subs}

    train_ids = [d["doc_id"] for d in train]
    test_ids = [d["doc_id"] for d in test]
    test_id_set = set(test_ids)
    redundant_train = {i for i in train_ids if i in exact_dup_ids or i in near_dup_ids}

    exact_cross_test: set[str] = set()
    for fp, ids in fp_groups.items():
        if len(ids) > 1 and any(split_of[i] == "train" for i in ids):
            exact_cross_test.update(i for i in ids if i in test_id_set)
    cross_near_test = {x for a, b, _ in cross_near for x in (a, b) if x in test_id_set}
    contaminated_test = (
        exact_cross_test
        | cross_near_test
        | sub_ids
        | {i for i in test_ids if i in exact_dup_ids}
    )

    rows: list[dict[str, Any]] = []
    for d in docs:
        i = d["doc_id"]
        fp = fp_of[i]
        group = fp_groups[fp]
        root = uf.find(i)
        members = near_groups.get(root, [i])
        if d["split"] == "train":
            action = "drop" if i in redundant_train else "keep"
            reason = (
                "exact_duplicate"
                if i in exact_dup_ids
                else "near_duplicate"
                if i in near_dup_ids
                else ""
            )
            leak_type, leak_src = "", ""
        else:
            if i in exact_dup_ids and any(
                split_of[o] == "train" for o in group if o != i
            ):
                leak_type, leak_src = "exact", next(o for o in group if split_of[o] == "train")
            elif i in cross_near_test:
                leak_type = "near"
                leak_src = near_partner.get(i, "")
            elif i in sub_ids:
                leak_type, leak_src = "substring", sub_partner[i]
            elif i in exact_dup_ids:
                leak_type, leak_src = "exact_within_test", ""
            else:
                leak_type, leak_src = "", ""
            action = "drop" if i in contaminated_test else "keep"
            reason = f"train_test_leak:{leak_type}" if leak_type else ""
        rows.append(
            {
                "doc_id": i,
                "split": d["split"],
                "lang": d["lang"],
                "source": d["source"],
                "n_chars": len(d["text"]),
                "exact_fp8": fp[:8],
                "exact_group_size": len(group),
                "is_exact_dup": int(i in exact_dup_ids),
                "near_group_id": root if len(members) > 1 else "",
                "near_group_size": len(members),
                "near_max_jaccard": round(near_max.get(i, 0.0), 4),
                "near_partner": near_partner.get(i, ""),
                "leak_type": leak_type,
                "leak_train_doc_id": leak_src,
                "action": action,
                "drop_reason": reason,
            }
        )

    kept_train = [d for d in train if d["doc_id"] not in redundant_train]
    kept_test = [d for d in test if d["doc_id"] not in contaminated_test]
    summary = {
        "config": asdict(cfg),
        "near_stage": near_info,
        "before": {
            "train": text_stats(d["text"] for d in train),
            "test": text_stats(d["text"] for d in test),
            "train_redundant_docs": len(redundant_train),
            "train_duplicate_rate": len(redundant_train) / len(train) if train else 0.0,
            "test_contaminated_docs": len(contaminated_test),
            "test_contamination_rate": len(contaminated_test) / len(test) if test else 0.0,
        },
        "after": {
            "train": text_stats(d["text"] for d in kept_train),
            "test": text_stats(d["text"] for d in kept_test),
            "train_redundant_docs": 0,
            "train_duplicate_rate": 0.0,
            "test_contaminated_docs": 0,
            "test_contamination_rate": 0.0,
        },
        "detection": {
            "exact_dup_docs": len(exact_dup_ids),
            "exact_cross_split_pairs": len(exact_cross),
            "near_pairs_total": len(near_pairs),
            "near_cross_split_pairs": len(cross_near),
            "near_dup_docs": len(near_dup_ids),
            "substring_leak_docs": len(sub_ids),
            "leak_type_counts": {
                t: sum(1 for r in rows if r["leak_type"] == t)
                for t in ("exact", "near", "substring", "exact_within_test")
            },
        },
        "cleaned_corpus": {
            "train_docs": len(kept_train),
            "test_docs": len(kept_test),
            "removed_train": len(train) - len(kept_train),
            "removed_test": len(test) - len(kept_test),
        },
        "seconds_total": round(time.time() - t_start, 3),
    }

    csv_path = out_dir / "data_audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (out_dir / "near_dup_pairs.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["doc_a", "doc_b", "jaccard", "split_a", "split_b"])
        for a, b, s in near_pairs:
            w.writerow([a, b, round(s, 6), split_of[a], split_of[b]])
    write_json(out_dir / "audit_summary.json", summary)
    (out_dir / "cleaned").mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "cleaned" / "train.jsonl", kept_train)
    write_jsonl(out_dir / "cleaned" / "test.jsonl", kept_test)
    summary["paths"] = {
        "data_audit_csv": str(csv_path),
        "near_dup_pairs_csv": str(out_dir / "near_dup_pairs.csv"),
        "cleaned_train": str(out_dir / "cleaned" / "train.jsonl"),
        "cleaned_test": str(out_dir / "cleaned" / "test.jsonl"),
    }
    return summary


def main() -> None:
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mode", choices=("char", "word"), default="char")
    ap.add_argument("--bands", type=int, default=32)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--probe-chars", type=int, default=100)
    ap.add_argument("--n-probes", type=int, default=5)
    ap.add_argument("--no-near", action="store_true")
    ap.add_argument("--out", type=Path, default=OUTPUTS)
    args = ap.parse_args()
    cfg = AuditConfig(
        num_perm=args.num_perm,
        k=args.k,
        mode=args.mode,
        bands=args.bands,
        rows=args.rows,
        threshold=args.threshold,
        seed=args.seed,
        max_docs=args.max_docs,
        probe_chars=args.probe_chars,
        n_probes=args.n_probes,
        enable_near=not args.no_near,
    )
    if not (CORPUS / "train.jsonl").exists():
        raise SystemExit("run `python -m src.prepare_data` and `python -m src.inject_fixtures` first")
    s = run_audit(load_corpus("train"), load_corpus("test"), cfg, args.out)
    print(json.dumps({k: s[k] for k in ("near_stage", "before", "after", "detection", "cleaned_corpus", "seconds_total")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()