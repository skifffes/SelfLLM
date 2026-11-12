# 精读 · Deduplicating Training Data Makes Language Models Better
Lee et al., 2021 · arXiv:2107.06499

## 四个问题（读完必填）

1. **它解决的核心问题是什么？**
   （提示：重复 → 记忆放大 / 泛化受损 / 评测污染被低估；区分 exact、near、substring 三种重复。）
   > 你的回答：

2. **它改进的是模型参数、Prompt、工具还是系统？**
   （提示：纯数据管线。为什么作者强调"去重比调模型更便宜"？）
   > 你的回答：

3. **实验是否真的支持作者的结论？**
   （提示：去重后 train/test overlap 下降多少？perplexity 与 memorization 指标怎么变？
   NearDup 的阈值/shingle 选择会不会改变结论？C4 与 RealNews 的差异说明什么？）
   > 你的回答：

4. **缩小到 toy 规模，我能复现哪一个机制？**
   （提示：MinHash 签名估计 Jaccard 的概率性质；LSH 的 S 曲线；验证层为什么不可省。）
   > 你的回答：

## 关键公式推导（写代码前先推）

- 为什么 `P(minhash_i(A) == minhash_i(B)) = Jaccard(A, B)`？
  （提示：把哈希值看成随机排列，最小值落在交集里的概率。）
  > 你的推导：
- b 个 band、每 band r 行时，候选概率 `P(s) = 1 - (1 - s^r)^b`。
  取 num_perm=128, b=32, r=4：s=0.8 时 P≈？s=0.3 时 P≈？
  > 你的计算：
- 阈值近似 `(1/b)^(1/r)` 是怎么来的？它和 verify 层的 threshold 是什么关系？
  > 你的回答：

## 三种重复与我的检测层对应

| 论文方法 | 我的实现 | 能检出 | 检不出 |
|---|---|---|---|
| ExactSubstr（后缀数组） | 探针式子串查找 | | |
| NearDup（MinHash+LSH） | core_minhash.py | | |
| 文档级 exact | sha256 指纹 | | |

## 还没弄懂的地方
>