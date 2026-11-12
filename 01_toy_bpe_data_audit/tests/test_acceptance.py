"""End-to-end acceptance on the planted fixtures: "对给定重复/泄漏样本能检测出来".

Runs the real audit pipeline over the full corpus and scores it against ground truth.
Exact / substring assertions pass today; near-duplicate assertions activate once
src/core_minhash.py is implemented.
"""

from __future__ import annotations

import csv

import pytest

from src.audit_pipeline import AuditConfig, run_audit
from src.common import FIXTURES, load_corpus, read_json

REQUIRED_COLUMNS = {
    "doc_id", "split", "lang", "source", "n_chars", "exact_fp8", "exact_group_size",
    "is_exact_dup", "near_group_id", "near_group_size", "near_max_jaccard", "near_partner",
    "leak_type", "leak_train_doc_id", "action", "drop_reason",
}


@pytest.fixture(scope="module")
def audit(tmp_path_factory):
    out = tmp_path_factory.mktemp("audit")
    cfg = AuditConfig(num_perm=64, bands=16, rows=4, threshold=0.8)
    summary = run_audit(load_corpus("train"), load_corpus("test"), cfg, out)
    with (out / "data_audit.csv").open(encoding="utf-8", newline="") as fh:
        rows = {r["doc_id"]: r for r in csv.DictReader(fh)}
    labels = read_json(FIXTURES / "labels.json")
    return summary, rows, labels


def flagged(rows, doc_id):
    r = rows[doc_id]
    return r["action"] == "drop" or r["leak_type"] != "" or int(r["near_group_size"]) > 1


def recall(rows, labels, kind, level=None):
    ids = [
        i for i, v in labels["labels"].items()
        if v.get("kind") == kind and (level is None or v.get("level") == level)
    ]
    assert ids, f"no fixtures of kind {kind}"
    return sum(flagged(rows, i) for i in ids) / len(ids)


def test_csv_schema(audit):
    _, rows, _ = audit
    assert REQUIRED_COLUMNS <= set(next(iter(rows.values())).keys())


def test_exact_duplicates_fully_detected(audit):
    _, rows, labels = audit
    assert recall(rows, labels, "exact_dup_train") == 1.0


def test_exact_leakage_fully_detected(audit):
    _, rows, labels = audit
    assert recall(rows, labels, "leak_exact") == 1.0


def test_span_leakage_fully_detected(audit):
    _, rows, labels = audit
    assert recall(rows, labels, "leak_span") == 1.0


def test_near_leakage_mostly_detected(audit):
    _, rows, labels = audit
    assert recall(rows, labels, "leak_near") >= 0.9


def test_near_duplicates_light_and_medium(audit):
    summary, rows, labels = audit
    if summary["near_stage"]["status"] != "ok":
        pytest.skip("src/core_minhash.py still a stub")
    assert recall(rows, labels, "near_dup_train", "light") >= 0.9
    assert recall(rows, labels, "near_dup_train", "medium") >= 0.9


def test_boilerplate_trap_not_flagged(audit):
    _, rows, labels = audit
    assert not any(flagged(rows, i) for i in labels["boilerplate_trap_ids"])


def test_controls_not_flagged(audit):
    _, rows, labels = audit
    assert not any(flagged(rows, i) for i in labels["control_ids"])


def test_cleaning_removes_contamination(audit):
    summary, _, _ = audit
    assert summary["before"]["test_contamination_rate"] > 0
    assert summary["after"]["test_contamination_rate"] == 0.0
    assert summary["after"]["train"]["docs"] < summary["before"]["train"]["docs"]