"""Result figures.  English labels on purpose: CJK fonts are not guaranteed on every
machine, and tofu boxes in a submitted figure are worse than English labels.

Every function takes already-computed data and only draws -- no experiment logic here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import FIGURES

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})


def lsh_probability(s: np.ndarray | float, bands: int, rows: int) -> np.ndarray | float:
    """P(a pair with Jaccard s becomes an LSH candidate) = 1 - (1 - s**rows)**bands."""
    return 1 - (1 - np.asarray(s) ** rows) ** bands


def _save(fig, name: str, out_dir: Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else FIGURES
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_length_distributions(
    lengths: dict[tuple[str, str], list[int]],
    names: Sequence[str],
    langs: Sequence[str] = ("en", "zh"),
    out_dir: Path | None = None,
    bins: int = 60,
) -> Path:
    fig, axes = plt.subplots(1, len(langs), figsize=(6.2 * len(langs), 4.0))
    axes = np.atleast_1d(axes)
    for ax, lang in zip(axes, langs):
        for name in names:
            vals = lengths.get((name, lang))
            if not vals:
                continue
            ax.hist(vals, bins=bins, histtype="step", linewidth=1.8, label=name, density=True)
        ax.set_title(f"document token count distribution ({lang})")
        ax.set_xlabel("tokens per document")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    return _save(fig, "fig1_length_distribution.png", out_dir)


def plot_length_percentiles(df: pd.DataFrame, out_dir: Path | None = None) -> Path:
    sub = df[df.lang.isin(["en", "zh"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, lang in zip(axes, ["en", "zh"]):
        d = sub[sub.lang == lang]
        x = np.arange(len(d))
        width = 0.25
        for off, col in zip((-width, 0, width), ["p50_len", "p90_len", "p99_len"]):
            ax.bar(x + off, d[col], width, label=col.replace("_len", ""))
        ax.set_xticks(x, d.tokenizer, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{lang}: length percentiles")
        ax.set_ylabel("tokens per document")
        ax.legend(fontsize=8)
    return _save(fig, "fig2_length_percentiles.png", out_dir)


def plot_merges_sweep(sweep: pd.DataFrame, out_dir: Path | None = None) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for lang, ax in zip(["en", "zh"], axes):
        d = sweep[sweep.lang == lang]
        for name, g in d.groupby("backend"):
            ax.plot(g.num_merges, g.tokens_per_char, marker="o", label=name)
        ax.set_xlabel("number of BPE merges")
        ax.set_ylabel("tokens per character")
        ax.set_title(f"{lang}: compression vs merge budget")
        ax.legend(fontsize=8)
    ax2 = axes[1].twinx()
    d = sweep[sweep.lang == "en"].drop_duplicates("num_merges")
    ax2.plot(d.num_merges, d.vocab_size, "k--", alpha=0.5, label="vocab size")
    ax2.set_ylabel("vocab size")
    ax2.grid(False)
    ax2.legend(fontsize=8, loc="center right")
    return _save(fig, "fig3_merges_sweep.png", out_dir)


def plot_ratios(df: pd.DataFrame, out_dir: Path | None = None) -> Path:
    sub = df[df.lang.isin(["en", "zh"])]
    names = list(dict.fromkeys(sub.tokenizer))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(names))
    for ax, col, title in zip(
        axes, ["tokens_per_char", "tokens_per_word"], ["tokens / character", "tokens / word"]
    ):
        for off, lang in zip((-0.2, 0.2), ["en", "zh"]):
            vals = [
                sub[(sub.tokenizer == n) & (sub.lang == lang)][col].mean() for n in names
            ]
            ax.bar(x + off, vals, 0.38, label=lang)
        ax.axhline(1.0, color="red", linestyle=":", linewidth=1)
        ax.set_xticks(x, names, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("red dotted line = 1.0 (above it, tokenization EXPANDS the text)", fontsize=9)
    return _save(fig, "fig4_ratio_by_lang.png", out_dir)


def plot_dedup_before_after(summary: dict[str, Any], out_dir: Path | None = None) -> Path:
    b, a = summary["before"], summary["after"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    metrics = [
        ("train_duplicate_rate", "train duplicate rate"),
        ("test_contamination_rate", "test contamination rate"),
    ]
    for ax, (key, title) in zip(axes, metrics):
        vals = [b[key] * 100, a[key] * 100]
        bars = ax.bar(["before cleaning", "after cleaning"], vals, color=["#c44", "#4a7"])
        ax.bar_label(bars, fmt="%.2f%%")
        ax.set_title(title)
        ax.set_ylabel("percent of documents")
        ax.set_ylim(0, max(vals) * 1.35 + 0.1)
    det = summary["detection"]
    labels = ["exact dup", "near dup", "substring leak", "cross-split near"]
    vals = [
        det["exact_dup_docs"],
        det["near_dup_docs"],
        det["substring_leak_docs"],
        det["near_cross_split_pairs"],
    ]
    fig2, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.barh(labels, vals, color="#4878a8")
    ax.bar_label(bars, padding=3)
    ax.set_xlabel("documents / pairs flagged")
    ax.set_title("detections by layer (layers overlap)")
    _save(fig2, "fig6_detection_layers.png", out_dir)
    return _save(fig, "fig5_dedup_before_after.png", out_dir)


def plot_lsh_scurve(
    configs: Sequence[tuple[int, int]],
    measured: pd.DataFrame | None = None,
    out_dir: Path | None = None,
) -> Path:
    s = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for bands, rows in configs:
        ax.plot(s, lsh_probability(s, bands, rows), label=f"b={bands}, r={rows}")
        thr = (1 / bands) ** (1 / rows)
        ax.axvline(thr, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    if measured is not None and len(measured):
        ax.scatter(
            measured.threshold, measured.recall, marker="o", color="black", zorder=5, label="measured recall"
        )
        if "precision" in measured:
            ax.scatter(
                measured.threshold, measured.precision, marker="x", color="crimson", zorder=5, label="measured precision"
            )
    ax.set_xlabel("Jaccard similarity s")
    ax.set_ylabel("P(candidate pair)")
    ax.set_title("LSH S-curve: theory vs measured recall")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.02, 1.02)
    return _save(fig, "fig7_lsh_scurve.png", out_dir)


def plot_similarity_distribution(
    groups: dict[str, Sequence[float]], out_dir: Path | None = None
) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label, vals in groups.items():
        if len(vals):
            ax.hist(vals, bins=30, range=(0, 1), histtype="step", linewidth=1.8, label=f"{label} (n={len(vals)})")
    ax.axvline(0.8, color="red", linestyle=":", label="threshold 0.8")
    ax.set_xlabel("exact Jaccard (char 5-shingles, oracle)")
    ax.set_ylabel("number of pairs")
    ax.set_title("injected duplicate groups vs random negative pairs")
    ax.legend(fontsize=8)
    return _save(fig, "fig8_similarity_distribution.png", out_dir)