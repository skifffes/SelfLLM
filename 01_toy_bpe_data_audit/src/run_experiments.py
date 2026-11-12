"""One-shot reproduction of every experiment, figure, CSV and the data card.

    python -m src.run_experiments                 # full run
    python -m src.run_experiments --quick         # small subsamples, fast iteration
    python -m src.run_experiments --no-near       # skip the MinHash stage (still a stub)

The notebook calls these same functions, so notebook results and CLI results are identical.
Your core code is optional everywhere: if src/core_bpe.py or src/core_minhash.py still
raises NotImplementedError, the run completes with the library baselines and says which
parts are pending.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from . import plotting
from .audit_pipeline import AuditConfig, run_audit
from .baselines import HFBpeTokenizer, build_baselines
from .common import (
    CORPUS,
    FIGURES,
    FIXTURES,
    OUTPUTS,
    load_corpus,
    peak_memory_mb,
    read_json,
    utf8_stdout,
)
from .metrics import build_comparison, doc_stats, evaluate_tokenizer, roundtrip_rate
from .oracle import similarity_oracle
from .report import score_audit, write_data_card

NEAR_TRUTH_KINDS = ("exact_dup_train", "near_dup_train", "leak_exact", "leak_near")


def try_toy_bpe(train_texts: Sequence[str], num_merges: int) -> Any | None:
    from .core_bpe import train_bpe

    try:
        tok = train_bpe(train_texts, num_merges=num_merges)
    except NotImplementedError:
        return None
    tok.name = f"toy_bpe_{num_merges}"
    return tok


def merge_sweep(
    train_texts: Sequence[str],
    docs: Sequence[dict[str, Any]],
    grid: Sequence[int],
    max_docs: int | None,
) -> pd.DataFrame:
    """Compression vs merge budget.  Uses your BPE when available, else HF BPE as proxy.

    For byte-level HF BPE, vocab = 256 bytes + 1 special + n_merges, so n_merges is
    recovered as vocab_size - 257.  The `backend` column records which one was used.
    """
    stats = doc_stats(docs)
    rows = []
    for n in grid:
        tok = try_toy_bpe(train_texts, n)
        backend = "toy_bpe"
        if tok is None:
            tok = HFBpeTokenizer.fit(train_texts, vocab_size=n + 257, label=f"hf_bpe_{n + 257}")
            backend = "hf_bpe_proxy"
        for lang in ("en", "zh"):
            row, _ = evaluate_tokenizer(tok, docs, stats, lang=lang, max_docs=max_docs)
            rows.append(
                {
                    "backend": backend,
                    "num_merges": n,
                    "vocab_size": row["vocab_size"],
                    "lang": lang,
                    "tokens_per_char": row["tokens_per_char"],
                    "tokens_per_word": row["tokens_per_word"],
                    "mean_len": row["mean_len"],
                    "p99_len": row["p99_len"],
                    "unk_rate": row["unk_rate"],
                }
            )
        print(f"  sweep num_merges={n:<6} backend={backend:<14} vocab={tok.vocab_size}")
    return pd.DataFrame(rows)


def lsh_threshold_sweep(
    docs: Sequence[dict[str, Any]],
    labels: dict[str, Any],
    cfg: AuditConfig,
    thresholds: Sequence[float],
    sweep_docs: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Measured recall/precision of YOUR detector vs the theoretical S-curve.

    The doc set always contains every injected doc plus its origin, so no ground-truth
    pair is lost to subsampling.  leak_span is excluded from near-dup truth: its Jaccard
    is ~0.28, it belongs to the substring layer.
    """
    from . import core_minhash as cm

    injected = {i: v for i, v in labels.items() if v.get("kind") in NEAR_TRUTH_KINDS}
    needed = set(injected) | {v["origin_doc_id"] for v in injected.values() if v.get("origin_doc_id")}
    by_id = {d["doc_id"]: d for d in docs}
    chosen = [by_id[i] for i in needed if i in by_id]
    rest = [d for d in docs if d["doc_id"] not in needed]
    rng = random.Random(seed)
    if len(chosen) < sweep_docs:
        chosen += rng.sample(rest, min(len(rest), sweep_docs - len(chosen)))
    doc_map = {d["doc_id"]: d["text"] for d in chosen}

    truth = {
        tuple(sorted((i, v["origin_doc_id"])))
        for i, v in injected.items()
        if v.get("origin_doc_id") in doc_map and i in doc_map
    }
    try:
        sigs = cm.build_signatures(doc_map, num_perm=cfg.num_perm, k=cfg.k, mode=cfg.mode, seed=cfg.seed)
        candidates = cm.lsh_candidate_pairs(sigs, bands=cfg.bands, rows=cfg.rows)
    except NotImplementedError:
        return pd.DataFrame()

    rows = []
    for thr in thresholds:
        try:
            pairs = cm.verify_pairs(doc_map, candidates, threshold=thr, k=cfg.k, mode=cfg.mode)
        except NotImplementedError:
            return pd.DataFrame()
        pred = {tuple(sorted((a, b))) for a, b, _ in pairs}
        tp = len(pred & truth)
        rows.append(
            {
                "threshold": thr,
                "recall": tp / len(truth) if truth else 0.0,
                "precision": tp / len(pred) if pred else 1.0,
                "n_predicted": len(pred),
                "n_truth": len(truth),
                "n_candidates": len(candidates),
                "n_docs": len(doc_map),
            }
        )
    return pd.DataFrame(rows)


def similarity_groups(
    docs: Sequence[dict[str, Any]], labels: dict[str, Any], n_negative: int = 400, seed: int = 0
) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = {}
    for v in labels.values():
        sim = v.get("oracle_jaccard_char5")
        if sim is None:
            continue
        kind = v.get("kind", "?")
        key = f"{kind}:{v['level']}" if kind == "near_dup_train" else kind
        groups.setdefault(key, []).append(sim)
    rng = random.Random(seed)
    negatives: list[float] = []
    for _ in range(n_negative):
        a, b = rng.sample(docs, 2)
        negatives.append(similarity_oracle(a["text"], b["text"], 5, "char"))
    groups["random_negative_pairs"] = negatives
    return groups


def main() -> None:
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUTPUTS)
    ap.add_argument("--max-docs", type=int, default=None, help="subsample per language for metrics")
    ap.add_argument("--sweep-docs", type=int, default=1500)
    ap.add_argument("--merges-grid", default="500,1000,2000,4000")
    ap.add_argument("--hf-vocabs", default="1000,4000")
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--bands", type=int, default=32)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mode", choices=("char", "word"), default="char")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--no-near", action="store_true")
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.max_docs = args.max_docs or 400
        args.sweep_docs = 600
        args.merges_grid = "500,2000"
        args.hf_vocabs = "1000"

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    figs = out / "figures"
    t0 = time.time()

    train, test = load_corpus("train"), load_corpus("test")
    docs = train + test
    train_texts = [d["text"] for d in train]
    print(f"corpus: train {len(train)} docs, test {len(test)} docs")

    print("\n[1/6] tokenizer comparison")
    tokenizers = build_baselines(train_texts, [int(x) for x in args.hf_vocabs.split(",")])
    grid = [int(x) for x in args.merges_grid.split(",")]
    toy_missing = False
    for n in grid:
        tok = try_toy_bpe(train_texts, n)
        if tok is None:
            toy_missing = True
        else:
            tokenizers.append(tok)
    if toy_missing:
        print("  NOTE: src/core_bpe.py is still a stub -> toy BPE rows omitted (baselines only)")
    metrics_df, lengths = build_comparison(tokenizers, docs, max_docs=args.max_docs)
    rt = {t.name: roundtrip_rate(t, train_texts[:100]) for t in tokenizers}
    metrics_df["roundtrip"] = metrics_df.tokenizer.map(rt)
    metrics_df.to_csv(out / "metrics.csv", index=False, encoding="utf-8")
    print(metrics_df[["tokenizer", "lang", "vocab_size", "tokens_per_char", "tokens_per_word", "mean_len", "unk_rate"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n[2/6] merge-budget sweep")
    sweep = merge_sweep(train_texts, docs, grid, args.max_docs)
    sweep.to_csv(out / "merges_sweep.csv", index=False, encoding="utf-8")

    print("\n[3/6] figures: lengths / percentiles / ratios / sweep")
    names = [t.name for t in tokenizers]
    plotting.plot_length_distributions(lengths, names, out_dir=figs)
    plotting.plot_length_percentiles(metrics_df, out_dir=figs)
    plotting.plot_ratios(metrics_df, out_dir=figs)
    plotting.plot_merges_sweep(sweep, out_dir=figs)

    summary: dict[str, Any] = {}
    scores: dict[str, Any] = {}
    if not args.skip_audit:
        print("\n[4/6] data audit")
        cfg = AuditConfig(
            num_perm=args.num_perm, k=args.k, mode=args.mode, bands=args.bands, rows=args.rows,
            threshold=args.threshold, max_docs=None, enable_near=not args.no_near,
        )
        summary = run_audit(train, test, cfg, out)
        print(f"  near stage: {summary['near_stage']['status']}")
        print(f"  train dup rate {summary['before']['train_duplicate_rate']*100:.2f}% -> {summary['after']['train_duplicate_rate']*100:.2f}%")
        print(f"  test contamination {summary['before']['test_contamination_rate']*100:.2f}% -> {summary['after']['test_contamination_rate']*100:.2f}%")
        plotting.plot_dedup_before_after(summary, out_dir=figs)
        scores = score_audit(out / "data_audit.csv", FIXTURES / "labels.json")
        pd.DataFrame(scores["per_kind"]).T.to_csv(out / "acceptance_scores.csv", encoding="utf-8")
    else:
        print("\n[4/6] audit skipped")

    print("\n[5/6] LSH S-curve + similarity distribution")
    labels_payload = read_json(FIXTURES / "labels.json") if (FIXTURES / "labels.json").exists() else {"labels": {}}
    labels = labels_payload.get("labels", {})
    cfg = AuditConfig(num_perm=args.num_perm, k=args.k, mode=args.mode, bands=args.bands, rows=args.rows)
    measured = (
        lsh_threshold_sweep(docs, labels, cfg, (0.5, 0.6, 0.7, 0.8, 0.9), args.sweep_docs)
        if not args.no_near
        else pd.DataFrame()
    )
    if len(measured):
        measured.to_csv(out / "lsh_sweep.csv", index=False, encoding="utf-8")
        print(measured.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        print("  NOTE: src/core_minhash.py is still a stub -> theory-only S-curve")
    b = args.bands
    configs = [(b, args.rows)] + [(x, args.num_perm // x) for x in (128, 64, 16, 8) if args.num_perm % x == 0 and x != b]
    plotting.plot_lsh_scurve(configs, measured if len(measured) else None, out_dir=figs)
    groups = similarity_groups(docs, labels)
    plotting.plot_similarity_distribution(groups, out_dir=figs)

    print("\n[6/6] data card")
    card = write_data_card(metrics_df, summary or None, scores or None, out / "data_card.md")

    print(f"\nwrote: {out}")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out)}  ({p.stat().st_size/1024:.1f} KB)")
    print(f"\nelapsed {time.time()-t0:.1f}s   peak RSS {peak_memory_mb():.0f} MB (budget <2048 MB)")
    if toy_missing or not len(measured):
        print("PENDING: implement src/core_bpe.py and src/core_minhash.py, then rerun for the full deliverable.")


if __name__ == "__main__":
    main()