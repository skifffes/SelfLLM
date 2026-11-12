# Week 1 · Toy BPE + Data Audit（LLM Atlas V3）

5MB 小语料上的 tokenizer 对照实验 + 数据去重/污染审计。
**核心代码（BPE、MinHash/LSH）留给你写**；其余管线已全部实现并实跑验证。

- 精读：arXiv:1508.07909（BPE）、arXiv:2107.06499（Deduplicating Training Data）
- 交付：`notebooks/tokenizer_demo.ipynb` + `outputs/data_audit.csv` + `outputs/data_card.md`
- 验收：① 不同切分策略的长度分布准确；② 给定重复/泄漏样本能检出
- 资源：CPU，峰值内存 < 2GB（基线全量实测 **330 MB / 40.6s**，见 `run_experiments` 结尾打印）

## 目录

```
src/
  core_bpe.py        ← 你写：get_stats / merge_vocab / train_bpe / encode / decode / save / load
  core_minhash.py    ← 你写：shingles / jaccard / minhash_signature / build_signatures /
                          lsh_candidate_pairs / verify_pairs
  prepare_data.py      已实现：下载 Gutenberg+维基文库、清洗、切 doc、train/test 划分
  inject_fixtures.py   已实现：注入 exact/near/泄漏/boilerplate 陷阱 + ground truth
  baselines.py         已实现：char / word / HF tokenizers BPE 三个对照基线
  metrics.py           已实现：长度分布、tokens/char、tokens/word、UNK 率、round-trip
  audit_pipeline.py    已实现：exact 指纹 + 调用你的 MinHash + 探针式子串泄漏 + 清洗
  plotting.py          已实现：8 张结果图
  report.py            已实现：对照 ground truth 评分 + 数据卡渲染
  run_experiments.py   已实现：一条命令复现全部实验
  oracle.py            已实现：brute-force Jaccard，只用于打标签和评分，不参与检测
notebooks/tokenizer_demo.ipynb   叙事 + 对照实验 + TODO 驱动 cell + 失败分析/我的问题模板
tests/                 手算夹具 + 统计性质 + 端到端验收（核心未实现时自动 skip）
data/raw|corpus|fixtures         原始下载 / 清洗后语料 / 注入标签
outputs/               data_audit.csv、data_card.md、metrics.csv、figures/ 等
notes/                 精读四问、失败分析模板
```

## 环境

base conda 环境（`D:\miniconda3`）已装好全部依赖。换机器时：

```bash
pip install -r requirements.txt
```

网络受限时走代理（Gutenberg/维基文库直连也可，PyPI 与 HF 建议走代理）：

```bash
export HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897
```

## 运行命令

```bash
python -m src.prepare_data          # 1) 语料：下载+清洗+切分+划分（有缓存则秒级）
python -m src.inject_fixtures       # 2) 注入重复/泄漏夹具，写 ground truth
python -m src.run_experiments       # 3) 全部对照实验 + 8 张图 + CSV + 数据卡
python -m pytest tests -q           # 4) 验收测试（核心未实现时 15 项 skip）
jupyter lab notebooks/tokenizer_demo.ipynb   # 或 VS Code / Qoder 直接打开
```

快速迭代（子采样，十几秒）：`python -m src.run_experiments --quick`
只跑 exact+子串层：`python -m src.run_experiments --no-near`
注意顺序：`prepare_data` 会重建语料并**清掉注入**，所以注入必须在它之后。

## 对照实验设计

| # | 实验 | 自变量 | 控制量 | 指标 | 图 |
|---|---|---|---|---|---|
| 1 | 切分策略对照 | char / word / toy-BPE / HF-BPE | 同一语料、同一 train-only 拟合 | 长度分布 mean/p50/p90/p99/max、tokens/char、tokens/word、UNK 率 | fig1, fig2, fig4 |
| 2 | merge 预算消融 | num_merges ∈ {500,1000,2000,4000} | 语料、预分词方案 | tokens/char 曲线、vocab_size | fig3 |
| 3 | 语言对照（stretch） | en / zh | 同一 tokenizer | tokens/字、tokens/词、压缩方向 | fig4 |
| 4 | 去重前后 | 清洗前 / 清洗后 | 检测阈值 0.8 | train 重复率、test 污染率、文档数/字节 | fig5, fig6 |
| 5 | LSH 参数扫描 | threshold ∈ {0.5..0.9}；(bands,rows) 组合 | num_perm=128 | 实测 recall/precision vs 理论 S 曲线 `1-(1-s^r)^b` | fig7 |
| 6 | 夹具相似度分布 | 注入类型 / 随机负样本对 | char-5 shingle | oracle Jaccard 直方图 | fig8 |
| 7 | 预分词消融（你写 BPE 后） | 有无 `</w>`、中文按字/按词 | num_merges 固定 | 长度分布 + round-trip + UNK | 自行加 cell |

## 验收清单（当前实测，基线层）

- [x] 语料 4.83MB / 3827 docs（train 3062 / test 765），注入后 3965 docs
- [x] 长度分布：char / word / HF-BPE×2 已产出；toy-BPE 待你实现后自动加入
- [x] exact_dup_train 召回 40/40 = 100%
- [x] leak_exact 召回 20/20 = 100%；leak_span 召回 15/15 = 100%（子串层）
- [x] leak_near 召回 14/15 = 93.3%（仅子串层；MinHash 层应补到 100%）
- [x] train 重复率 1.27% → 0.00%；test 污染率 6.13% → 0.00%
- [x] boilerplate 陷阱误报 0/12；对照组误报 0/30
- [ ] near_dup light/medium 召回 ≥ 90% ← **等你的 MinHash**
- [ ] toy-BPE 进入对照表、round-trip = 1.0 ← **等你的 BPE**
- [ ] `notes/failure_analysis.md` 与「我的问题」填写完成

## 已经能看到的三个现象（先自己解释，再写进失败分析）

1. **词级基线在中文上崩溃**：`text.split()` 把整段当一个词，zh 的 tokens/word ≈ 0.006、UNK 率 19.7%。
2. **小词表 byte-level BPE 对中文是膨胀**：hf_bpe_1000 的 zh tokens/char = 1.75 > 1（一个汉字 3 字节）。
3. **文档级近重复检不出片段泄漏**：leak_span 的 oracle Jaccard 只有 0.26–0.30，远低于 0.8 阈值。

## 时间分配建议（计划给 3–4 小时）

| 时段 | 做什么 |
|---|---|
| 40min | 精读两篇 + 填 `notes/paper_*.md` 四问 |
| 60min | 写 `core_bpe.py`，跑 `pytest tests/test_bpe_core.py` 直到不 skip |
| 60min | 写 `core_minhash.py`，跑 `pytest tests/test_minhash_core.py` |
| 30min | 全量 `run_experiments` + 执行 notebook + 回填失败分析与我的问题 |

## 常见坑

- **内置 `hash()` 有进程盐**：MinHash 签名必须用 hashlib/固定种子，否则跨进程不可复现（有专门测试抓这个）。
- **Wikimedia 限流 429**：`prepare_data` 已带退避与 2s 间隔；仍被限就等一分钟重跑，缓存会续传。
- **Windows 控制台 GBK**：所有入口已 `utf8_stdout()`；自己写脚本打印中文时记得同样处理。
- **`pytest` 要用 `python -m pytest`**：直接 `pytest` 不会把项目根加入 sys.path。
- 中文为繁体（维基文库源文本），数据卡已注明；`--nfkc` 是可选消融。