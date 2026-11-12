# 失败分析（提交内容之一，至少 4 条）

每条格式：**现象 → 定位 → 根因 → 修复或边界说明**。
下面给出的是管线已经观测到的*线索*和数字，**不要直接抄**：自己复现、补上你的定位过程与结论。
写完后把对应结论同步进 `outputs/data_card.md` 第 9 节。

---

## 线索 1 · 词级基线在中文上崩溃（已观测）
- 现象：`word` 基线 zh 的 tokens/word ≈ 0.006、tokens/char ≈ 0.005、UNK 率 19.7%；en 正常（tokens/word = 1.0）。
- 定位入口：`src/baselines.py::WordTokenizer.encode` 的 `text.split()`；`outputs/metrics.csv`。
- 待你回答：根因是什么？这算 bug 还是方法边界？如果给中文加 jieba 预分词，词级基线的结论会变吗？
> 你的分析：

## 线索 2 · 小词表 byte-level BPE 对中文是膨胀（已观测）
- 现象：hf_bpe_1000 的 zh tokens/char = 1.754（>1，即比字符级还长）；hf_bpe_4000 降到 1.159；en 同词表是 0.403 / 0.313。
- 定位入口：`outputs/merges_sweep.csv`、fig3、fig4。
- 待你回答：byte-level 预分词 + 汉字 3 字节如何共同导致膨胀？merge 预算到多少才翻转为压缩？你的 toy BPE（字符级 base）会不会有同样问题？
> 你的分析：

## 线索 3 · 文档级近重复检不出片段泄漏（已观测）
- 现象：leak_span 的 oracle char-5 Jaccard 仅 0.26–0.30，阈值 0.8 下 MinHash 必然漏；靠子串层才 15/15。
- 定位入口：`data/fixtures/manifest.csv`、fig8、`src/audit_pipeline.py::substring_leaks`。
- 待你回答：为什么 Jaccard 这么低（300/1200 字符）？论文的 ExactSubstr 为什么能检出？你的探针式实现还漏什么？
> 你的分析：

## 线索 4 · 子串层漏了 1 篇 leak_near（已观测）
- 现象：仅 exact+子串层时 leak_near 召回 14/15 = 93.3%；`outputs/acceptance_scores.csv` 的 missed_ids 给出那一篇。
- 定位入口：把该 doc 与 origin 对齐，看 5 个 100 字符探针是否都被扰动打断。
- 待你回答：这是探针长度/数量的问题还是扰动率的问题？MinHash 层补上后能否到 100%？
> 你的分析：

## 线索 5 · boilerplate 陷阱的阈值敏感性（已观测）
- 现象：陷阱组两两 char-5 Jaccard ≈ 0.30–0.33；阈值 0.8 时误报 0/12。
- 待你回答：把 threshold 降到 0.3 会发生什么（跑 `lsh_threshold_sweep` 或改 cfg 重跑）？这说明阈值选择本质上是在权衡什么？
> 你的分析：

## 线索 6 · 你的 BPE / MinHash 实现中的失败（待你填）
- 现象：
- 定位：
- 根因：
- 修复或边界：
> （例如：tie-break 不一致导致手算夹具失败；hash() 盐导致跨进程签名不同；中文预分词选择导致 round-trip 失败；
>  全量重统计太慢；band 切分 off-by-one……）

---

## 我的问题（提交内容之一）
- 我还没弄懂的概念：
- 我的实现与论文不一致之处及理由：
- 换成语料类型（代码/对话/多语言）后我预期失效的结论：
- 带到 Week 2（char-LM）的假设：