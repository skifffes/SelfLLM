"""Scoring against ground truth + the data card deliverable.

This is the ONLY place (with tests/) that joins detector output to data/fixtures/labels.json.

Important caveat, stated in the data card too: precision measured against injected labels
is a LOWER BOUND.  Real corpora contain genuine duplicates that nobody labelled (repeated
chapter headings, the same passage quoted twice), so "flagged but not injected" is not the
same as "false positive".  The clean false-positive signal is the planted boilerplate trap
and the control group, which are known-distinct by construction.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .common import CORPUS, FIXTURES, OUTPUTS, read_json, utf8_stdout

TRAIN_KINDS = ("exact_dup_train", "near_dup_train", "boilerplate_trap")
LEAK_KINDS = ("leak_exact", "leak_near", "leak_span")


def score_audit(
    audit_csv: Path = OUTPUTS / "data_audit.csv",
    labels_json: Path = FIXTURES / "labels.json",
) -> dict[str, Any]:
    payload = read_json(labels_json)
    labels: dict[str, Any] = payload["labels"]
    with Path(audit_csv).open("r", encoding="utf-8", newline="") as fh:
        rows = {r["doc_id"]: r for r in csv.DictReader(fh)}

    def flagged(doc_id: str) -> bool:
        r = rows.get(doc_id)
        if r is None:
            return False
        return r["action"] == "drop" or r["leak_type"] != "" or int(r["near_group_size"]) > 1

    per_kind: dict[str, Any] = {}
    for kind in TRAIN_KINDS + LEAK_KINDS:
        ids = [i for i, v in labels.items() if v.get("kind") == kind]
        if not ids:
            continue
        hit = [i for i in ids if flagged(i)]
        entry: dict[str, Any] = {
            "expected": len(ids),
            "detected": len(hit),
            "recall": len(hit) / len(ids),
            "missed_ids": sorted(set(ids) - set(hit))[:20],
        }
        if kind == "near_dup_train":
            entry["by_level"] = {}
            for level in ("light", "medium", "heavy"):
                sub = [i for i in ids if labels[i].get("level") == level]
                if sub:
                    got = [i for i in sub if flagged(i)]
                    entry["by_level"][level] = {
                        "expected": len(sub),
                        "detected": len(got),
                        "recall": len(got) / len(sub),
                    }
        if kind == "boilerplate_trap":
            entry["false_positives"] = len(hit)
            entry.pop("recall")
            entry.pop("missed_ids")
            entry["note"] = "known-distinct docs sharing a long identical header; any hit is a FP"
        per_kind[kind] = entry

    controls = payload.get("control_ids", [])
    control_fp = [i for i in controls if flagged(i)]
    injected_ids = {i for i, v in labels.items() if v.get("kind") not in (None, "control")}
    unlabeled = [i for i, r in rows.items() if r["action"] == "drop" and i not in injected_ids]
    return {
        "per_kind": per_kind,
        "control_false_positives": {"expected": len(controls), "flagged": len(control_fp), "ids": control_fp[:20]},
        "flagged_but_not_injected": {
            "count": len(unlabeled),
            "sample_ids": sorted(unlabeled)[:20],
            "note": "lower-bound precision only: may be genuine unlabelled corpus duplicates",
        },
        "near_stage_status": payload.get("near_stage_status", "unknown"),
    }


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(out)


def write_data_card(
    metrics_df: pd.DataFrame | None = None,
    summary: dict[str, Any] | None = None,
    scores: dict[str, Any] | None = None,
    out_path: Path = OUTPUTS / "data_card.md",
) -> Path:
    meta = read_json(CORPUS / "meta.json")
    fix = read_json(FIXTURES / "labels.json") if (FIXTURES / "labels.json").exists() else {}
    summary = summary or (read_json(OUTPUTS / "audit_summary.json") if (OUTPUTS / "audit_summary.json").exists() else {})
    scores = scores or {}
    total = meta["total"]
    b = summary.get("before", {})
    a = summary.get("after", {})
    det = summary.get("detection", {})
    cfg = summary.get("config", {})

    src_rows = [
        [s["title"], s["lang"], s["n_docs"], f"{s['clean_chars']:,}", s["license"].split(";")[0]]
        for s in meta["sources"]
    ]
    parts = [
        "# 数据卡 · Toy BPE + Data Audit（LLM Atlas V3 · 第 1 周）",
        "",
        f"- 生成日期：{date.today().isoformat()}",
        f"- 用途：Tokenizer 对照实验与数据去重/污染检测的教学语料，**不用于训练发布模型**",
        f"- 随机种子：语料 {meta['seed']}／夹具 {fix.get('seed', 'n/a')}／审计 {cfg.get('seed', 'n/a')}",
        f"- 复现：`python -m src.prepare_data && python -m src.inject_fixtures && python -m src.run_experiments`",
        "",
        "## 1. 规模",
        "",
        f"- 总量：**{total['utf8_bytes']/1e6:.2f} MB**（UTF-8），{total['chars']:,} 字符，{total['docs']} 篇文档",
        f"- 划分：train {meta['n_train_docs']} 篇 / test {meta['n_test_docs']} 篇（{meta['split_policy']}）",
        f"- 注入夹具后：train {b.get('train', {}).get('docs', 'n/a')} 篇 / test {b.get('test', {}).get('docs', 'n/a')} 篇",
        "",
        _md_table(
            ["语言", "文档数", "字符数", "UTF-8 字节"],
            [[k, v["docs"], f"{v['chars']:,}", f"{v['utf8_bytes']:,}"] for k, v in meta["by_lang"].items()],
        ),
        "",
        "## 2. 来源与许可",
        "",
        _md_table(["来源", "语言", "文档数", "清洗后字符", "许可"], src_rows),
        "",
        "- 英文：Project Gutenberg，美国公有领域；已剥离 Gutenberg 商标头尾。",
        "- 中文：维基文库，原文公有领域（红楼梦/三国演义/鲁迅/论语），排版版本 CC BY-SA 4.0，需署名。",
        "- 中文为**繁体**（维基文库源文本未做繁简转换）；`--nfkc` 消融会改变这一点，见 meta.json。",
        "",
        "## 3. 预处理",
        "",
        f"- 文档切分：按段落聚合到约 {meta['target_chars']} 字符，上限 {meta['max_chars']}，**块间不重叠**",
        f"- Unicode NFKC：{meta['nfkc']}",
        "- 中文页面抽取：MediaWiki API `action=parse` + `<p>` 过滤，剔除导航/版权/姊妹计划等 boilerplate",
        "- 精确去重指纹：sha256(小写化 + 空白折叠)",
        "",
        "## 4. 注入的重复与泄漏夹具（ground truth）",
        "",
        _md_table(
            ["类型", "数量", "说明"],
            [[k, v, d] for k, v, d in [
                ("exact_dup_train", fix.get("counts", {}).get("exact_dup_train"), "train 内逐字复制"),
                ("near_dup_light/medium/heavy", fix.get("counts", {}).get("near_dup_light"), "词/字级 replace-delete-insert，编辑率 3%/10%/25%"),
                ("leak_exact", fix.get("counts", {}).get("leak_exact"), "train 文档逐字进入 test"),
                ("leak_near", fix.get("counts", {}).get("leak_near"), "train 文档轻度扰动后进入 test"),
                ("leak_span", fix.get("counts", {}).get("leak_span"), "train 文档中 300 字符连续片段进入 test"),
                ("boilerplate_trap", fix.get("counts", {}).get("boilerplate_trap"), "共享长头但正文不同，**已知不重复**，用于测误报"),
            ]],
        ),
        "",
        f"- 陷阱组两两 char-5 Jaccard：{fix.get('boilerplate_trap_pairwise_char5', [])[:6]}",
        "- 注入文档使用与真实文档同一编号空间的 id，并打乱顺序；检测器不读标签文件。",
        "",
        "## 5. 去重与污染检测",
        "",
        f"- 近重复：MinHash num_perm={cfg.get('num_perm')}, k={cfg.get('k')}, mode={cfg.get('mode')}, "
        f"bands={cfg.get('bands')}, rows={cfg.get('rows')}, threshold={cfg.get('threshold')}",
        f"- 近重复阶段状态：**{summary.get('near_stage', {}).get('status', 'n/a')}**",
        f"- 子串泄漏：每篇 test 取 {cfg.get('n_probes')} 个 {cfg.get('probe_chars')} 字符探针，在 train 全文中查找",
        f"- 检出（各层有重叠）：exact {det.get('exact_dup_docs')} 篇，near {det.get('near_dup_docs')} 篇，"
        f"substring {det.get('substring_leak_docs')} 篇，跨划分 near 对 {det.get('near_cross_split_pairs')}",
        "",
        "## 6. 清洗前后",
        "",
        _md_table(
            ["指标", "清洗前", "清洗后"],
            [
                ["train 文档数", b.get("train", {}).get("docs"), a.get("train", {}).get("docs")],
                ["test 文档数", b.get("test", {}).get("docs"), a.get("test", {}).get("docs")],
                ["train 重复率", f"{b.get('train_duplicate_rate', 0)*100:.2f}%", f"{a.get('train_duplicate_rate', 0)*100:.2f}%"],
                ["test 污染率", f"{b.get('test_contamination_rate', 0)*100:.2f}%", f"{a.get('test_contamination_rate', 0)*100:.2f}%"],
                ["train 字节", b.get("train", {}).get("utf8_bytes"), a.get("train", {}).get("utf8_bytes")],
            ],
        ),
        "",
        "## 7. 验收得分（对照 ground truth）",
        "",
    ]
    per_kind = scores.get("per_kind", {})
    if per_kind:
        rows = []
        for kind, v in per_kind.items():
            if "recall" in v:
                rows.append([kind, v["expected"], v["detected"], f"{v['recall']*100:.1f}%"])
            else:
                rows.append([kind, v["expected"], v.get("false_positives", 0), "误报数（越低越好）"])
        parts.append(_md_table(["夹具类型", "应有", "检出", "召回 / 误报"], rows))
        for kind, v in per_kind.items():
            if "by_level" in v:
                parts += ["", "近重复按扰动强度："]
                parts.append(
                    _md_table(
                        ["强度", "应有", "检出", "召回"],
                        [[k, x["expected"], x["detected"], f"{x['recall']*100:.1f}%"] for k, x in v["by_level"].items()],
                    )
                )
        cf = scores.get("control_false_positives", {})
        parts += ["", f"- 对照组误报：{cf.get('flagged', 0)} / {cf.get('expected', 0)}"]
        ub = scores.get("flagged_but_not_injected", {})
        parts += [f"- 被标记但非注入：{ub.get('count', 0)} 篇（**精确率下界**，可能是语料本身的真实重复）"]
    else:
        parts.append("_尚未运行评分：`python -m src.run_experiments`_")

    parts += ["", "## 8. Tokenizer 对照", ""]
    if metrics_df is not None and len(metrics_df):
        d = metrics_df[metrics_df.lang.isin(["en", "zh"])]
        parts.append(
            _md_table(
                ["tokenizer", "lang", "vocab", "tokens/char", "tokens/word", "mean len", "p99 len", "UNK率"],
                [
                    [r.tokenizer, r.lang, r.vocab_size, f"{r.tokens_per_char:.3f}",
                     "n/a" if pd.isna(r.tokens_per_word) else f"{r.tokens_per_word:.3f}",
                     f"{r.mean_len:.1f}", f"{r.p99_len:.0f}", f"{r.unk_rate*100:.2f}%"]
                    for r in d.itertuples()
                ],
            )
        )
    else:
        parts.append("_尚未运行 tokenizer 对照_")

    parts += [
        "",
        "## 9. 已知局限与失败模式",
        "",
        "1. **词级基线在中文上失效**：中文无空格，`text.split()` 把整段当成一个词，UNK 率极高。",
        "2. **小词表 byte-level BPE 对中文是膨胀而非压缩**：一个汉字 3 字节，merges 不足时 tokens/char > 1。",
        "3. **文档级近重复检测漏掉片段泄漏**：leak_span 的 Jaccard 只有 0.26–0.30，低于阈值，必须靠子串层。",
        "4. **探针式子串检测不完备**：短于探针长度或跨探针边界的泄漏会漏；论文用后缀数组做全子串去重。",
        "5. **精确率只是下界**：语料本身存在未标注的真实重复。",
        "6. **体裁与语言不平衡**：英文以 19 世纪小说为主，中文以古典小说为主，中文仅占约 10% 字节、4% 文档。",
        "7. **boilerplate 陷阱**：共享长头会把 Jaccard 抬到 0.3 左右，阈值设得过低就会误杀。",
        "8. 中文 tokens/word 依赖 jieba 分词；未安装时该列为空而非假数字。",
        "",
        "## 10. 伦理与使用约束",
        "",
        "- 全部来源为公有领域或 CC BY-SA，可自由用于教学；转发中文部分需署名维基文库。",
        "- 语料含 19 世纪文学中的历史偏见与刻板表述，未做内容过滤，不适合直接做对齐数据。",
        "- 本数据卡描述的清洗只针对重复与划分污染，**不代表内容质量过滤**。",
        "",
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def main() -> None:
    utf8_stdout()
    scores = score_audit()
    print(pd.DataFrame(scores["per_kind"]).T.to_string())
    p = write_data_card(scores=scores)
    print(f"data card -> {p}")


if __name__ == "__main__":
    main()