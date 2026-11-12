"""Plant controlled duplicates and train->test leakage into the corpus, with ground truth.

This is the "构造少量重复样本" step of the week-1 project.  It is scaffolding, not the
detector: it writes labels that ONLY tests/test_acceptance.py and src/report.py read.
src/audit_pipeline.py never opens data/fixtures/, so your detector cannot cheat.

Injected docs get neutral ids drawn from the same numbering space as real docs, and the
final doc order is shuffled -- you cannot spot an injected doc by its id or position.

Run AFTER prepare_data (prepare_data rebuilds the corpus and wipes injections):
    python -m src.prepare_data
    python -m src.inject_fixtures
    python -m src.inject_fixtures --scale 2     # double every injection count
"""

from __future__ import annotations

import argparse
import csv
import re
from typing import Any

from .common import (
    CORPUS,
    FIXTURES,
    ensure_dirs,
    load_corpus,
    seeded,
    text_stats,
    utf8_stdout,
    write_json,
    write_jsonl,
)
from .oracle import similarity_oracle

BOILERPLATE = (
    "NOTICE: This document was retrieved from an automated archive mirror. "
    "Redistribution requires that this header remain attached to the file. "
    "The archive makes no representation about the accuracy of the transcription. "
    "Any copyright notice in the original volume applies to the underlying work only. "
    "Please report transcription errors to the maintainers before citing this text. "
    "This header is identical across every file in the batch and carries no information "
    "about the document body that follows it. "
) * 2

BASE_COUNTS = {
    "exact_dup_train": 40,
    "near_dup_light": 12,
    "near_dup_medium": 12,
    "near_dup_heavy": 12,
    "leak_exact": 20,
    "leak_near": 15,
    "leak_span": 15,
    "boilerplate_trap": 12,
}
EDIT_RATE = {"near_dup_light": 0.03, "near_dup_medium": 0.10, "near_dup_heavy": 0.25}
SPAN_CHARS = 300
_WS_SPLIT = re.compile(r"(\s+)")


def build_pool(docs: list[dict[str, Any]], lang: str, limit: int = 3000) -> list[str]:
    freq: dict[str, int] = {}
    for d in docs:
        if d["lang"] != lang:
            continue
        units = d["text"].split() if lang == "en" else list(d["text"])
        for u in units:
            if u.strip():
                freq[u] = freq.get(u, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [u for u, _ in ranked[:limit]]


def perturb(text: str, lang: str, rate: float, rng: Any, pool: list[str]) -> str:
    """Apply replace / delete / insert edits to a fraction `rate` of words (en) or chars (zh)."""
    parts = _WS_SPLIT.split(text)
    out: list[str] = []
    for part in parts:
        if not part or part.isspace():
            out.append(part)
            continue
        units = part.split(" ") if lang == "en" else list(part)
        edited: list[str] = []
        for u in units:
            if not u:
                continue
            if rng.random() < rate:
                op = rng.choice(("replace", "delete", "insert"))
                if op == "replace":
                    edited.append(rng.choice(pool))
                elif op == "delete":
                    continue
                else:
                    edited.append(rng.choice(pool))
                    edited.append(u)
            else:
                edited.append(u)
        out.append((" " if lang == "en" else "").join(edited))
    return "".join(out)


def next_id_factory(docs: list[dict[str, Any]]) -> Any:
    """Hand out ids that continue each source's existing numbering, so injected docs look native."""
    high: dict[str, int] = {}
    for d in docs:
        m = re.match(r"(.+)_(\d+)$", d["doc_id"])
        if m:
            high[m.group(1)] = max(high.get(m.group(1), -1), int(m.group(2)))

    def make(source: str) -> str:
        high[source] = high.get(source, -1) + 1
        return f"{source}_{high[source]:05d}"

    return make


def inject(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dirs()
    rng = seeded(args.seed)
    train = load_corpus("train")
    test = load_corpus("test")
    all_docs = train + test
    make_id = next_id_factory(all_docs)
    pools = {lang: build_pool(all_docs, lang) for lang in ("en", "zh")}
    counts = {k: max(1, int(v * args.scale)) for k, v in BASE_COUNTS.items()}

    labels: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    new_train: list[dict[str, Any]] = []
    new_test: list[dict[str, Any]] = []

    def record(
        doc: dict[str, Any],
        kind: str,
        origin: dict[str, Any] | None,
        target: str,
        level: str = "",
    ) -> None:
        sim_c = sim_w = None
        if origin is not None:
            sim_c = round(similarity_oracle(origin["text"], doc["text"], 5, "char"), 4)
            sim_w = round(similarity_oracle(origin["text"], doc["text"], 5, "word"), 4)
        labels[doc["doc_id"]] = {
            "kind": kind,
            "origin_doc_id": origin["doc_id"] if origin else None,
            "origin_split": origin["split"] if origin else None,
            "split": target,
            "lang": doc["lang"],
            "source": doc["source"],
            "level": level,
            "oracle_jaccard_char5": sim_c,
            "oracle_jaccard_word5": sim_w,
            "n_chars": len(doc["text"]),
        }
        rows.append(
            {
                "doc_id": doc["doc_id"],
                "kind": kind,
                "level": level,
                "origin_doc_id": origin["doc_id"] if origin else "",
                "split": target,
                "lang": doc["lang"],
                "n_chars": len(doc["text"]),
                "oracle_jaccard_char5": sim_c if sim_c is not None else "",
                "oracle_jaccard_word5": sim_w if sim_w is not None else "",
            }
        )

    def clone(origin: dict[str, Any], text: str, target: str) -> dict[str, Any]:
        return {
            "doc_id": make_id(origin["source"]),
            "source": origin["source"],
            "lang": origin["lang"],
            "split": target,
            "text": text,
        }

    eligible_train = [d for d in train if len(d["text"]) >= 400]
    picks = rng.sample(eligible_train, min(len(eligible_train), sum(counts.values()) + 40))
    cursor = 0

    def take(n: int) -> list[dict[str, Any]]:
        nonlocal cursor
        out = picks[cursor : cursor + n]
        cursor += n
        return out

    for origin in take(counts["exact_dup_train"]):
        d = clone(origin, origin["text"], "train")
        new_train.append(d)
        record(d, "exact_dup_train", origin, "train")

    for kind, rate in EDIT_RATE.items():
        for origin in take(counts[kind]):
            text = perturb(origin["text"], origin["lang"], rate, rng, pools[origin["lang"]])
            d = clone(origin, text, "train")
            new_train.append(d)
            record(d, "near_dup_train", origin, "train", level=kind.split("_")[-1])

    for origin in take(counts["leak_exact"]):
        d = clone(origin, origin["text"], "test")
        new_test.append(d)
        record(d, "leak_exact", origin, "test")

    for origin in take(counts["leak_near"]):
        text = perturb(origin["text"], origin["lang"], 0.05, rng, pools[origin["lang"]])
        d = clone(origin, text, "test")
        new_test.append(d)
        record(d, "leak_near", origin, "test")

    for origin in take(counts["leak_span"]):
        body = origin["text"]
        start = rng.randrange(0, max(1, len(body) - SPAN_CHARS))
        span = body[start : start + SPAN_CHARS]
        d = clone(origin, span, "test")
        new_test.append(d)
        record(d, "leak_span", origin, "test")
        labels[d["doc_id"]]["span_offset"] = start
        labels[d["doc_id"]]["span_chars"] = len(span)

    trap_origins = take(counts["boilerplate_trap"])
    trap_ids = []
    for origin in trap_origins:
        body = origin["text"][:600]
        d = clone(origin, BOILERPLATE + "\n\n" + body, "train")
        new_train.append(d)
        record(d, "boilerplate_trap", origin, "train")
        trap_ids.append(d["doc_id"])

    control_ids = [d["doc_id"] for d in rng.sample(eligible_train, 30)]
    for cid in control_ids:
        labels[cid] = {"kind": "control", "origin_doc_id": None, "split": "train"}

    train_out = train + new_train
    test_out = test + new_test
    rng.shuffle(train_out)
    rng.shuffle(test_out)
    write_jsonl(CORPUS / "train.jsonl", train_out)
    write_jsonl(CORPUS / "test.jsonl", test_out)

    trap_sims = [
        round(similarity_oracle(
            next(d["text"] for d in new_train if d["doc_id"] == a),
            next(d["text"] for d in new_train if d["doc_id"] == b), 5, "char"), 4)
        for i, a in enumerate(trap_ids) for b in trap_ids[i + 1 :][:3]
    ]
    payload = {
        "seed": args.seed,
        "scale": args.scale,
        "counts": counts,
        "n_injected_train": len(new_train),
        "n_injected_test": len(new_test),
        "boilerplate_trap_ids": trap_ids,
        "boilerplate_trap_pairwise_char5": trap_sims,
        "control_ids": control_ids,
        "labels": labels,
        "train": text_stats(d["text"] for d in train_out),
        "test": text_stats(d["text"] for d in test_out),
    }
    write_json(FIXTURES / "labels.json", payload)
    with (FIXTURES / "manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"injected: train +{len(new_train)}, test +{len(new_test)}")
    for kind in ("exact_dup_train", "near_dup_train", "leak_exact", "leak_near", "leak_span", "boilerplate_trap"):
        got = [v for v in labels.values() if v.get("kind") == kind]
        sims = [v["oracle_jaccard_char5"] for v in got if v.get("oracle_jaccard_char5") is not None]
        rng_txt = f" char5 Jaccard {min(sims):.2f}-{max(sims):.2f}" if sims else ""
        print(f"  {kind:<20} n={len(got):>3}{rng_txt}")
    print(f"  boilerplate trap pairwise char5: {trap_sims[:6]}")
    print(f"labels -> {FIXTURES / 'labels.json'}")
    return payload


def main() -> None:
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    if not (CORPUS / "meta.json").exists():
        raise SystemExit("run `python -m src.prepare_data` first")
    inject(args)


if __name__ == "__main__":
    main()