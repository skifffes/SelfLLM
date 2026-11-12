"""Download a ~5MB permissively-licensed corpus, clean it, chunk it into docs, split train/test.

Sources: Project Gutenberg (public domain English) + zh.wikisource (public domain Chinese).
Nothing here is BPE or MinHash logic -- this is the data plumbing you can trust and skip past.

Run:
    python -m src.prepare_data                 # uses cached downloads in data/raw
    python -m src.prepare_data --refresh       # re-download
    python -m src.prepare_data --nfkc          # ablation: Unicode NFKC normalization
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from .common import (
    CORPUS,
    RAW,
    ensure_dirs,
    seeded,
    sha256_file,
    text_stats,
    utf8_stdout,
    write_json,
    write_jsonl,
)

UA = {"User-Agent": "LLM-Atlas-Week1/1.0 (personal study project; small corpus audit)"}

EN_BOOKS: list[tuple[int, str, str]] = [
    (1342, "Pride and Prejudice", "Jane Austen"),
    (2701, "Moby Dick; Or, The Whale", "Herman Melville"),
    (1661, "The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
    (98, "A Tale of Two Cities", "Charles Dickens"),
    (84, "Frankenstein; Or, The Modern Prometheus", "Mary Wollstonecraft Shelley"),
    (74, "The Adventures of Tom Sawyer", "Mark Twain"),
    (11, "Alice's Adventures in Wonderland", "Lewis Carroll"),
]

ZH_PAGES: list[tuple[str, str]] = (
    [(f"紅樓夢/第{i:03d}回", "红楼梦") for i in range(1, 21)]
    + [(f"三國演義/第{i:03d}回", "三国演义") for i in range(1, 5)]
    + [("狂人日記", "呐喊"), ("孔乙己", "呐喊"), ("故鄉", "呐喊")]
    + [("論語/學而第一", "论语"), ("論語/為政第二", "论语")]
)

GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG", re.I)
GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", re.I)

ZH_NOISE = re.compile(
    r"回目录|回目錄|下一回|上一回|目錄|目录|姊妹计划|姊妹計劃|维基文库|維基文庫|检索自|檢索自"
    r"|本作品收錄|本作品收录|另參見|另参见|版权|版權|许可|許可|协议|協議|Creative Commons"
    r"|Public domain|public domain|GNU|维基百科|維基百科|←|→|◀|▶|https?://"
)
EN_SENT = re.compile(r"(?<=[.!?])\s+")
ZH_SENT = re.compile(r"(?<=[。！？；])")


def fetch(url: str, dest: Path, refresh: bool = False, params: dict | None = None) -> str:
    if dest.exists() and not refresh:
        return dest.read_text(encoding="utf-8")
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for wait in (5, 20, 60):
        try:
            r = requests.get(url, headers=UA, params=params, timeout=90)
            if r.status_code == 429:
                retry_after = int(r.headers.get("Retry-After", wait) or wait)
                print(f"    rate limited, sleeping {max(retry_after, wait)}s ...")
                time.sleep(max(retry_after, wait))
                continue
            r.raise_for_status()
            r.encoding = "utf-8"
            dest.write_text(r.text, encoding="utf-8")
            return r.text
        except Exception as exc:  # network boundary: retry with backoff
            last = exc
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch {url}: {last}")


def fetch_wikisource(page: str, refresh: bool) -> tuple[str, Path, bool]:
    slug = re.sub(r"[^\w]+", "_", page)
    dest = RAW / f"wikisource_{slug}.json"
    cached = dest.exists() and not refresh
    payload = fetch(
        "https://zh.wikisource.org/w/api.php",
        dest,
        refresh,
        params={
            "action": "parse",
            "page": page,
            "prop": "text",
            "format": "json",
            "redirects": 1,
        },
    )
    parsed = json.loads(payload)
    if "error" in parsed:
        raise RuntimeError(f"wikisource page missing: {page} ({parsed['error'].get('code')})")
    return parsed["parse"]["text"]["*"], dest, cached


def clean_gutenberg(raw: str) -> str:
    start = GUTENBERG_START.search(raw)
    end = GUTENBERG_END.search(raw)
    body = raw[start.end() : end.start()] if start and end else raw
    body = body.split("\n", 1)[-1] if start else body
    lines = []
    for line in body.replace("\r\n", "\n").split("\n"):
        s = line.strip()
        if re.fullmatch(r"\[Illustration[^\]]*\]", s, re.I):
            continue
        lines.append(s)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_wikisource(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    out = soup.select_one("div.mw-parser-output")
    if out is None:
        return ""
    paras = []
    for p in out.select("p"):
        t = p.get_text("", strip=True)
        if not t or ZH_NOISE.search(t):
            continue
        if len(t) < 12 and not re.search(r"[。！？；]", t):
            continue
        paras.append(t)
    return "\n\n".join(paras).strip()


def chunk_text(text: str, target_chars: int, max_chars: int, lang: str) -> list[str]:
    """Group whole paragraphs into ~target_chars docs; never overlap, never split mid-paragraph
    unless a single paragraph exceeds max_chars."""
    splitter = ZH_SENT if lang == "zh" else EN_SENT
    chunks: list[str] = []
    buf: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal buf, size
        if buf:
            chunks.append("\n\n".join(buf).strip())
            buf, size = [], 0

    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            flush()
            piece = ""
            for sent in splitter.split(para):
                if len(piece) + len(sent) > max_chars and piece:
                    chunks.append(piece.strip())
                    piece = ""
                piece += sent + (" " if lang == "en" else "")
            if piece.strip():
                buf, size = [piece.strip()], len(piece)
            continue
        if size + len(para) > target_chars and buf:
            flush()
        buf.append(para)
        size += len(para) + 2
    flush()
    return [c for c in chunks if len(c) >= 80]


def build(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dirs()
    rng = seeded(args.seed)
    docs: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []

    for gid, title, author in EN_BOOKS:
        url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
        dest = RAW / f"gutenberg_{gid}.txt"
        raw = fetch(url, dest, args.refresh)
        text = clean_gutenberg(raw)
        if args.nfkc:
            text = unicodedata.normalize("NFKC", text)
        pieces = chunk_text(text, args.target_chars, args.max_chars, "en")
        sid = f"pg{gid}"
        for i, piece in enumerate(pieces):
            docs.append({"doc_id": f"{sid}_{i:05d}", "source": sid, "lang": "en", "text": piece})
        sources.append(
            {
                "source_id": sid,
                "title": title,
                "author": author,
                "lang": "en",
                "url": url,
                "license": "Public domain (US); Project Gutenberg trademark header/footer removed",
                "raw_sha256": sha256_file(dest),
                "raw_bytes": dest.stat().st_size,
                "clean_chars": len(text),
                "n_docs": len(pieces),
            }
        )
        print(f"[en] {sid:<8} {title[:38]:<40} chars={len(text):>8} docs={len(pieces):>5}")

    for page, work in ZH_PAGES:
        slug = re.sub(r"[^\w]+", "_", page)
        html, dest, cached = fetch_wikisource(page, args.refresh)
        text = clean_wikisource(html)
        if args.nfkc:
            text = unicodedata.normalize("NFKC", text)
        pieces = chunk_text(text, args.target_chars, args.max_chars, "zh")
        sid = "zh_" + slug
        for i, piece in enumerate(pieces):
            docs.append({"doc_id": f"{sid}_{i:05d}", "source": sid, "lang": "zh", "text": piece})
        sources.append(
            {
                "source_id": sid,
                "title": page,
                "author": work,
                "lang": "zh",
                "url": f"https://zh.wikisource.org/wiki/{page}",
                "license": "Underlying text public domain; Wikisource edition CC BY-SA 4.0",
                "raw_sha256": sha256_file(dest),
                "raw_bytes": dest.stat().st_size,
                "clean_chars": len(text),
                "n_docs": len(pieces),
            }
        )
        print(f"[zh] {page:<22} chars={len(text):>7} docs={len(pieces):>4}")
        if not cached:
            time.sleep(2.0)

    by_source: dict[str, list[dict[str, Any]]] = {}
    for d in docs:
        by_source.setdefault(d["source"], []).append(d)
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for sid, group in by_source.items():
        rng.shuffle(group)
        cut = max(1, int(round(len(group) * (1 - args.test_frac)))) if len(group) > 1 else 1
        for d in group[:cut]:
            d["split"] = "train"
            train.append(d)
        for d in group[cut:]:
            d["split"] = "test"
            test.append(d)
    rng.shuffle(train)
    rng.shuffle(test)

    n_train = write_jsonl(CORPUS / "train.jsonl", train)
    n_test = write_jsonl(CORPUS / "test.jsonl", test)
    meta = {
        "seed": args.seed,
        "nfkc": args.nfkc,
        "target_chars": args.target_chars,
        "max_chars": args.max_chars,
        "test_frac": args.test_frac,
        "split_policy": "doc-level random split, stratified by source; chunks never overlap",
        "sources": sources,
        "train": text_stats(d["text"] for d in train),
        "test": text_stats(d["text"] for d in test),
        "total": text_stats(d["text"] for d in docs),
        "by_lang": {
            lang: text_stats(d["text"] for d in docs if d["lang"] == lang) for lang in ("en", "zh")
        },
        "n_train_docs": n_train,
        "n_test_docs": n_test,
    }
    write_json(CORPUS / "meta.json", meta)
    print(
        f"\ncorpus: {meta['total']['utf8_bytes']/1e6:.2f} MB, "
        f"{meta['total']['chars']:,} chars, {meta['total']['docs']} docs "
        f"(train {n_train} / test {n_test})"
    )
    print(f"by lang: {meta['by_lang']}")
    return meta


def main() -> None:
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-chars", type=int, default=1200)
    ap.add_argument("--max-chars", type=int, default=2400)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--nfkc", action="store_true", help="ablation: apply Unicode NFKC")
    ap.add_argument("--refresh", action="store_true", help="ignore cached downloads")
    build(ap.parse_args())


if __name__ == "__main__":
    main()