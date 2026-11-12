# SelfLLM · LLM Atlas 16 周实践工程

一套「读论文 → 写最小复现 → 做对照实验 → 写失败分析」的 16 周 LLM 全生命周期实践仓库。
每周一个独立可运行的工程目录，全部必做部分以 **RTX 5060 8GB 本地可执行** 为目标；A40 仅作为授权且空闲时的选做资源。

- 计划原文：[`LLM_Atlas_16周计划与论文目录对齐_V3.html`](./LLM_Atlas_16周计划与论文目录对齐_V3.html)（离线可开，含 137 篇文献目录与进度勾选）
- 主线：数据 → 预训练 → 后训练 → 推理服务 → 感知与行动 → 反馈与改进

## 目录总览

| 周 | 目录 | 项目 | 阶段 | 主要产出 | 状态 |
|---|---|---|---|---|---|
| W01 | [`01_toy_bpe_data_audit`](./01_toy_bpe_data_audit) | Toy BPE + Data Audit | DATA · P0 | `tokenizer_demo.ipynb` + `data_audit.csv` + 数据卡 | 已实现 |
| W02 | [`02_minigpt_char_lm`](./02_minigpt_char_lm) | MiniGPT 字符级 LM | PRETRAIN · P0 | 代码 + 张量形状说明 + loss 曲线 + 生成样例 | 待开始 |
| W03 | [`03_micro_scaling_mixture`](./03_micro_scaling_mixture) | Micro Scaling + Mixture | PRETRAIN · P0 | 配置表 + loss/step/速度/显存曲线 + 1 页结论 | 待开始 |
| W04 | [`04_mini_train_profiler`](./04_mini_train_profiler) | Mini Train Profiler | TRAIN · P0 | 实验脚本 + 结果 CSV + 两张图 + 内存预算表 | 待开始 |
| W05 | [`05_tiny_adapter_lora`](./05_tiny_adapter_lora) | Tiny Adapter（0.6B LoRA + 数据消融） | POSTTRAIN · P0 | 训练配置/Adapter + held-out 测试表 + 失败分析 | 待开始 |
| W06 | [`06_reasoning_arena`](./06_reasoning_arena) | Reasoning Arena | REASON · P0 | 评测 JSONL + Accuracy/Token/Latency 图 + 错误归因 | 待开始 |
| W07 | [`07_verifier_arena`](./07_verifier_arena) | Verifier Arena | REASON · P0 | Verifier 单测 + pass@N 曲线 + 投机案例 | 待开始 |
| W08 | [`08_context_serving_bench`](./08_context_serving_bench) | Context & Serving Bench | SERVE · P1 | 可复现 Benchmark + Latency/VRAM + 位置准确率图 | 待开始 |
| W09 | [`09_paper_rag`](./09_paper_rag) | Paper RAG | RETRIEVE · P1 | 索引 + Retrieval/Answer 评测 + 引用样例 | 待开始 |
| W10 | [`10_memory_vault`](./10_memory_vault) | Memory Vault | MEMORY · P1 | SQLite + memory CRUD + 20 题测试报告 | 待开始 |
| W11 | [`11_mini_multimodal_desk`](./11_mini_multimodal_desk) | Mini Multimodal Desk | MULTIMODAL · P0 | 可运行 demo + 误差样例 + 耗时/显存记录 | 待开始 |
| W12 | [`12_video_to_plan`](./12_video_to_plan) | Video-to-Plan | MULTIMODAL · P0 | 帧索引 + 时序检索 + 动作状态图 | 待开始 |
| W13 | [`13_stateful_sandbox_agent`](./13_stateful_sandbox_agent) | Stateful Sandbox Agent | AGENT · P1 | Sandbox Agent + Trace + Retry/Resume 演示 + 20 题评测 | 待开始 |
| W14 | [`14_injection_defense_lab`](./14_injection_defense_lab) | Injection Defense Lab | SECURITY · P1 | 攻防对照 JSONL + 权限矩阵 + 风险登记册 | 待开始 |
| W15 | [`15_four_layer_update`](./15_four_layer_update) | Four-Layer Update | IMPROVE · P1 | continual-learning notebook + 4 层改进矩阵 + forgetting 曲线 | 待开始 |
| W16 | [`16_mini_evolution_v2`](./16_mini_evolution_v2) | Mini Evolution v2 | IMPROVE · P0 | 版本化候选档案 + 评测脚本 + 3–5 页研究报告 | 待开始 |

## 每周任务速查

| 周 | 目标 | 关键动作 | 验收底线 |
|---|---|---|---|
| 01 | 理解 BPE/SentencePiece、去重、数据质量与评测污染 | 小 BPE 对照四种切分；Exact/MinHash 近重复；清洗前后重复率/污染率 | 长度分布准确；给定重复/泄漏样本能检出 |
| 02 | Decoder-only、Causal Mask、LayerNorm、自回归训练 | 手写 Embedding/Attention/MLP/LN/残差；无位置编码与去 Mask 对照 | 能现场解释 Q/K/V、Mask、残差与 train/eval 差异 |
| 03 | ICL 与 Scaling，数据规模与模型规模的选择 | 0.5M/1.5M/3M 三配置；两种语料配比；固定切分与种子 | 区分「小模型观察」与「正式 Scaling Law」 |
| 04 | Adam、混合精度、Checkpointing、ZeRO/并行原理 | FP32 vs BF16/FP16、Checkpointing on/off，记录显存与 step/s | 有真实显存记录与公平对照，不支持时明确标注 |
| 05 | 数据质量、SFT、偏好优化与训练失败信号 | 100–200 条短任务，含低质量对照组；Qwen3-0.6B 短序列 LoRA | 固定测试集比较，不靠聊天样例证明变强 |
| 06 | CoT、Self-Consistency、ToT、PRM | 30 题三难度；Direct/CoT/3-sample SC；固定预算 | 同题同模型，长输出不等于正确推理 |
| 07 | RLVR/GRPO 与评价器，识别验证泄漏与奖励投机 | 40 题确定性校验；N=1/3/5 Best-of-N；5 个边缘测试 | 报告误判率与独立测试集 |
| 08 | 有效上下文、量化、KV Cache、Prefill/Decode | 256→2048 tokens 的 TTFT 与 decode tok/s；事实放头/中/尾 | 写明 warmup、后端、量化格式、并发数 |
| 09 | BM25/向量检索/Rerank/证据引用 | 30–60 篇摘要建索引；BM25 vs 小 Embedding；混合或轻量 rerank | 有 Gold evidence、Recall@K、来源归属 |
| 10 | 区分 Context / Memory / 参数知识 / 长期记录 | SQLite event/fact/source/time/version；20 段跨会话任务 | 撤回信息不再出现；过期事实不覆盖最新事实 |
| 11 | CLIP、Whisper、ImageBind、LLaVA 的输入/表征/输出 | CLIP 零样本图文检索 + Whisper tiny/base 转写 | 素材许可清晰，报告错误案例 |
| 12 | 视频时序、视觉定位、动作 token、世界模型 | 抽帧 + 事件标注 + CLIP 检索 + 确定性状态机 | 区分感知/预测/规划/控制，不声称复现 VLA |
| 13 | ReAct、工具定义、状态机、Trace、Workflow 边界 | 3 个白名单工具 + JSON schema 校验；SQLite checkpoints | 最多 6 步、有 allowlist 与回滚、副作用幂等 |
| 14 | 间接 Prompt Injection、工具权限、防御评测 | 只读工具输出内嵌无害越权请求；来源标记与输出隔离 | 越权率为 0，同时报告正常任务正确率 |
| 15 | 外部记忆 / 程序策略 / 参数更新 / 遗忘与知识编辑 | toy 双任务 A/B + Replay；四类更新对象对照 | 明确区分参更新与配置改写；Replay 不使用未来数据 |
| 16 | 把数据→模型→检索→Agent→反馈串成可评估系统 | 30 开发 + 30 隔离测试；≥5 种候选配置；只测一次独立测试 | 有隔离测试集、可复现命令、回滚与安全边界 |

完整的目标、精读论文、资源与 stretch 项见计划 HTML 的「16 周学习计划」与「文献对照目录」两节。

## 环境

```bash
# 先检查显卡与空闲显存
nvidia-smi

# 依赖按需分批安装（不要一次装完所有大框架）
pip install transformers datasets accelerate peft
pip install tokenizers scikit-learn rank-bm25
pip install sentence-transformers opencv-python-headless

# 本地小模型推理也可用 Ollama / llama.cpp：Qwen3-0.6B、Qwen3-1.7B (Q4)
```

各周目录自带 `requirements.txt`（已实现的周）时，以该目录内说明为准：

```bash
cd 01_toy_bpe_data_audit
pip install -r requirements.txt
```

网络受限时走本地代理：

```powershell
$env:HTTPS_PROXY = "http://127.0.0.1:7897"
$env:HTTP_PROXY  = "http://127.0.0.1:7897"
```

```bash
export HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897
```

## 算力预算

| 级别 | 资源 | 约定 |
|---|---|---|
| LEVEL 00 | CPU / 小模型 | 0.3M–5M MiniGPT、BM25、SQLite/FTS5、规则 Verifier |
| LEVEL 01 | RTX 5060 8GB（默认） | 0.6B–1.7B Qwen3，量化优先 4-bit，上下文先 1K–4K |
| LEVEL 02 | 单张 A40 48GB（严格选做） | 仅在与同事确认空闲后短时使用，目标显存 ≤16GB，可中断可恢复 |

共享资源保护：每次先看 `nvidia-smi`、确认使用许可与空闲显存；不抢占具身实验、不开多卡训练、不默认占满单卡 48GB。显存与性能受驱动、精度、CUDA/PyTorch 版本影响，以上均为保守预算，非保证值。

## 每周交付节奏

| 时段 | 做什么 |
|---|---|
| 第 1 天 | 读主论文，列出研究问题 |
| 第 2 天 | 完成关键概念笔记与最小复现 |
| 第 3 天 | 实现一个可靠的基线 |
| 第 4 天 | 加入对照或消融实验 |
| 第 5 天 | 完成评测、错误分析与提交 |

每篇论文读完后写四问：它解决的核心问题是什么？改进的是参数、Prompt、工具还是系统？实验是否真的支持结论？缩小到 0.6B–1.7B 我能复现哪个机制？

## 通用验收原则

- 同一个固定测试集比较，不用几条对话样例证明「变强」。
- 对照组必须公平：固定数据切分、种子、训练步数、设备与显存峰值。
- 明确区分「小规模实验的观察」与「论文规模的结论」，不越界断言。
- 失败分析优先于成功指标：记录反例、误判、泄漏与投机路径。
- 安全相关实验全部离线合成，不针对真实账户或服务。

## 仓库约定

- 目录名与代码、变量、文件名统一使用英文与下划线；中文仅出现在笔记与文档正文。
- 数据、模型权重、checkpoint 与 `outputs/` 产物不入库（见 `.gitignore`）。
- 每个周目录自带 `README.md` 描述任务、命令与验收清单；空目录以 `.gitkeep` 占位。
- 提交信息用 `w03: add compare-script` 形式标注所属周次。

## 参考资源

- Transformer / Tokenizer：<https://huggingface.co/docs/transformers/main/custom_tokenizers>
- PyTorch AMP：<https://docs.pytorch.org/docs/stable/amp.html>
- PyTorch Activation Checkpointing：<https://docs.pytorch.org/docs/stable/checkpoint.html>
- Model Context Protocol 规范：<https://modelcontextprotocol.io/specification>
- vLLM 推理服务文档（非必装）：<https://docs.vllm.ai/en/latest/>
- OWASP LLM Top 10：<https://owasp.org/www-project-top-10-for-large-language-model-applications/>

## 说明

计划版本 V3（2026-11-11），137 篇文献中 73 篇进入正式课程、64 篇为扩展阅读。学习进度以计划页的勾选与导出的 JSON 备份为准，本仓库只跟踪代码与笔记。
