# 数据卡 · Toy BPE + Data Audit（LLM Atlas V3 · 第 1 周）

- 生成日期：2026-11-18
- 用途：Tokenizer 对照实验与数据去重/污染检测的教学语料，**不用于训练发布模型**
- 随机种子：语料 20261112／夹具 20261112／审计 0
- 复现：`python -m src.prepare_data && python -m src.inject_fixtures && python -m src.run_experiments`

## 1. 规模

- 总量：**4.83 MB**（UTF-8），4,427,815 字符，3827 篇文档
- 划分：train 3062 篇 / test 765 篇（doc-level random split, stratified by source; chunks never overlap）
- 注入夹具后：train 3150 篇 / test 815 篇

| 语言 | 文档数 | 字符数 | UTF-8 字节 |
|---|---|---|---|
| en | 3656 | 4,261,572 | 4,335,406 |
| zh | 171 | 166,243 | 495,611 |

## 2. 来源与许可

| 来源 | 语言 | 文档数 | 清洗后字符 | 许可 |
|---|---|---|---|---|
| Pride and Prejudice | en | 627 | 732,756 | Public domain (US) |
| Moby Dick; Or, The Whale | en | 1056 | 1,234,029 | Public domain (US) |
| The Adventures of Sherlock Holmes | en | 487 | 568,561 | Public domain (US) |
| A Tale of Two Cities | en | 656 | 766,077 | Public domain (US) |
| Frankenstein; Or, The Modern Prometheus | en | 364 | 424,703 | Public domain (US) |
| The Adventures of Tom Sawyer | en | 341 | 396,888 | Public domain (US) |
| Alice's Adventures in Wonderland | en | 125 | 145,893 | Public domain (US) |
| 紅樓夢/第001回 | zh | 8 | 7,746 | Underlying text public domain |
| 紅樓夢/第002回 | zh | 7 | 6,078 | Underlying text public domain |
| 紅樓夢/第003回 | zh | 9 | 8,583 | Underlying text public domain |
| 紅樓夢/第004回 | zh | 7 | 5,902 | Underlying text public domain |
| 紅樓夢/第005回 | zh | 7 | 7,542 | Underlying text public domain |
| 紅樓夢/第006回 | zh | 8 | 7,429 | Underlying text public domain |
| 紅樓夢/第007回 | zh | 8 | 7,445 | Underlying text public domain |
| 紅樓夢/第008回 | zh | 6 | 6,533 | Underlying text public domain |
| 紅樓夢/第009回 | zh | 5 | 5,651 | Underlying text public domain |
| 紅樓夢/第010回 | zh | 5 | 5,096 | Underlying text public domain |
| 紅樓夢/第011回 | zh | 7 | 5,877 | Underlying text public domain |
| 紅樓夢/第012回 | zh | 4 | 4,271 | Underlying text public domain |
| 紅樓夢/第013回 | zh | 5 | 5,161 | Underlying text public domain |
| 紅樓夢/第014回 | zh | 6 | 5,555 | Underlying text public domain |
| 紅樓夢/第015回 | zh | 5 | 5,053 | Underlying text public domain |
| 紅樓夢/第016回 | zh | 8 | 7,566 | Underlying text public domain |
| 紅樓夢/第017回 | zh | 8 | 7,777 | Underlying text public domain |
| 紅樓夢/第018回 | zh | 8 | 8,414 | Underlying text public domain |
| 紅樓夢/第019回 | zh | 9 | 9,214 | Underlying text public domain |
| 紅樓夢/第020回 | zh | 6 | 5,317 | Underlying text public domain |
| 三國演義/第001回 | zh | 5 | 4,554 | Underlying text public domain |
| 三國演義/第002回 | zh | 6 | 5,766 | Underlying text public domain |
| 三國演義/第003回 | zh | 5 | 4,808 | Underlying text public domain |
| 三國演義/第004回 | zh | 4 | 3,847 | Underlying text public domain |
| 狂人日記 | zh | 5 | 5,102 | Underlying text public domain |
| 孔乙己 | zh | 3 | 3,045 | Underlying text public domain |
| 故鄉 | zh | 5 | 5,508 | Underlying text public domain |
| 論語/學而第一 | zh | 1 | 832 | Underlying text public domain |
| 論語/為政第二 | zh | 1 | 1,011 | Underlying text public domain |

- 英文：Project Gutenberg，美国公有领域；已剥离 Gutenberg 商标头尾。
- 中文：维基文库，原文公有领域（红楼梦/三国演义/鲁迅/论语），排版版本 CC BY-SA 4.0，需署名。
- 中文为**繁体**（维基文库源文本未做繁简转换）；`--nfkc` 消融会改变这一点，见 meta.json。

## 3. 预处理

- 文档切分：按段落聚合到约 1200 字符，上限 2400，**块间不重叠**
- Unicode NFKC：False
- 中文页面抽取：MediaWiki API `action=parse` + `<p>` 过滤，剔除导航/版权/姊妹计划等 boilerplate
- 精确去重指纹：sha256(小写化 + 空白折叠)

## 4. 注入的重复与泄漏夹具（ground truth）

| 类型 | 数量 | 说明 |
|---|---|---|
| exact_dup_train | 40 | train 内逐字复制 |
| near_dup_light/medium/heavy | 12 | 词/字级 replace-delete-insert，编辑率 3%/10%/25% |
| leak_exact | 20 | train 文档逐字进入 test |
| leak_near | 15 | train 文档轻度扰动后进入 test |
| leak_span | 15 | train 文档中 300 字符连续片段进入 test |
| boilerplate_trap | 12 | 共享长头但正文不同，**已知不重复**，用于测误报 |

- 陷阱组两两 char-5 Jaccard：[0.3176, 0.2994, 0.3305, 0.3123, 0.3218, 0.3083]
- 注入文档使用与真实文档同一编号空间的 id，并打乱顺序；检测器不读标签文件。

## 5. 去重与污染检测

- 近重复：MinHash num_perm=128, k=5, mode=char, bands=32, rows=4, threshold=0.8
- 近重复阶段状态：**ok**
- 子串泄漏：每篇 test 取 5 个 100 字符探针，在 train 全文中查找
- 检出（各层有重叠）：exact 60 篇，near 101 篇，substring 50 篇，跨划分 near 对 37

## 6. 清洗前后

| 指标 | 清洗前 | 清洗后 |
|---|---|---|
| train 文档数 | 3150 | 3086 |
| test 文档数 | 815 | 764 |
| train 重复率 | 2.03% | 0.00% |
| test 污染率 | 6.26% | 0.00% |
| train 字节 | 3977357 | 3898472 |

## 7. 验收得分（对照 ground truth）

| 夹具类型 | 应有 | 检出 | 召回 / 误报 |
|---|---|---|---|
| exact_dup_train | 40 | 40 | 100.0% |
| near_dup_train | 36 | 24 | 66.7% |
| boilerplate_trap | 12 | 0 | 误报数（越低越好） |
| leak_exact | 20 | 20 | 100.0% |
| leak_near | 15 | 15 | 100.0% |
| leak_span | 15 | 15 | 100.0% |

近重复按扰动强度：
| 强度 | 应有 | 检出 | 召回 |
|---|---|---|---|
| light | 12 | 12 | 100.0% |
| medium | 12 | 12 | 100.0% |
| heavy | 12 | 0 | 0.0% |

- 对照组误报：0 / 30
- 被标记但非注入：1 篇（**精确率下界**，可能是语料本身的真实重复）

## 8. Tokenizer 对照

| tokenizer | lang | vocab | tokens/char | tokens/word | mean len | p99 len | UNK率 |
|---|---|---|---|---|---|---|---|
| char | en | 3670 | 1.000 | 5.665 | 1164.3 | 1200 | 0.00% |
| char | zh | 3670 | 1.000 | 1.489 | 967.3 | 1214 | 0.15% |
| word | en | 58033 | 0.177 | 1.000 | 205.5 | 233 | 1.10% |
| word | zh | 58033 | 0.005 | 0.007 | 4.7 | 23 | 19.73% |
| hf_bpe_1000 | en | 1000 | 0.403 | 2.285 | 469.6 | 565 | 0.00% |
| hf_bpe_1000 | zh | 1000 | 1.754 | 2.611 | 1696.2 | 2304 | 0.00% |
| hf_bpe_4000 | en | 4000 | 0.314 | 1.779 | 365.7 | 457 | 0.00% |
| hf_bpe_4000 | zh | 4000 | 1.159 | 1.725 | 1121.0 | 1589 | 0.00% |
| toy_bpe_500 | en | 4217 | 0.570 | 3.228 | 663.4 | 751 | 0.00% |
| toy_bpe_500 | zh | 4217 | 0.994 | 1.480 | 961.3 | 1213 | 0.15% |
| toy_bpe_1000 | en | 4717 | 0.517 | 2.929 | 601.9 | 687 | 0.00% |
| toy_bpe_1000 | zh | 4717 | 0.958 | 1.426 | 926.5 | 1182 | 0.15% |
| toy_bpe_2000 | en | 5717 | 0.473 | 2.680 | 550.9 | 635 | 0.00% |
| toy_bpe_2000 | zh | 5717 | 0.926 | 1.379 | 895.7 | 1159 | 0.16% |
| toy_bpe_4000 | en | 7717 | 0.437 | 2.478 | 509.3 | 587 | 0.00% |
| toy_bpe_4000 | zh | 7717 | 0.875 | 1.303 | 846.8 | 1108 | 0.17% |

## 9. 已知局限与失败模式

1. **词级基线在中文上失效**：中文无空格，`text.split()` 把整段当成一个词，UNK 率极高。
2. **小词表 byte-level BPE 对中文是膨胀而非压缩**：一个汉字 3 字节，merges 不足时 tokens/char > 1。
3. **文档级近重复检测漏掉片段泄漏**：leak_span 的 Jaccard 只有 0.26–0.30，低于阈值，必须靠子串层。
4. **探针式子串检测不完备**：短于探针长度或跨探针边界的泄漏会漏；论文用后缀数组做全子串去重。
5. **精确率只是下界**：语料本身存在未标注的真实重复。
6. **体裁与语言不平衡**：英文以 19 世纪小说为主，中文以古典小说为主，中文仅占约 10% 字节、4% 文档。
7. **boilerplate 陷阱**：共享长头会把 Jaccard 抬到 0.3 左右，阈值设得过低就会误杀。
8. 中文 tokens/word 依赖 jieba 分词；未安装时该列为空而非假数字。

## 10. 伦理与使用约束

- 全部来源为公有领域或 CC BY-SA，可自由用于教学；转发中文部分需署名维基文库。
- 语料含 19 世纪文学中的历史偏见与刻板表述，未做内容过滤，不适合直接做对齐数据。
- 本数据卡描述的清洗只针对重复与划分污染，**不代表内容质量过滤**。
