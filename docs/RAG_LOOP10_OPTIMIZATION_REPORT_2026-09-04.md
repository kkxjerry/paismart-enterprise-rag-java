# RAG 10轮循环优化结果

范围：P0 Evidence→Prompt 十轮迭代；随后做干净提交的500题回放和20题真实 Qwen Flash 配对生成。代码分支 `develop`。候选实现提交 `c7730c8`。完整机器数据见 `results/2026-09-04-rag-loop10-monitor.json`。

## 1. 固定门禁

Retrieval 本轮冻结，没有重新调参；沿用同一份已保存 Evidence，因此以下为既有回归门禁，不冒充本轮重测：Hit@1 **89.15%**，Hit@5 **95.96%**，Hit@10 **98.09%**，MRR@10 **0.9217**，Miss **9/470**，ACL/source violation **0**。

工程：Python **104/104**；A40 Java **88/88**、BUILD SUCCESS。后续改动只涉及 Python 工具，因此 Java 未重复构建。

## 2. 十轮结果

每轮固定10000字符、最多12个上下文。480道事实题中固定抽160题做循环：126 tune + 34 guard。只有通过 tune 改善且 guard 不明显回退的改动才进入下一轮。

| 轮次 | 改动 | 接受 | Tune Lexical | Req Coverage | Exact* | List | Condition | Tune回退率 | Guard Lexical | Guard回退率 |
|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 降低同文档惩罚 | 否 | 66.09% | 67.00% | 46.30% | 55.20% | 56.49% | 24.60% | 65.55% | 20.59% |
| 2 | 加强 document rank | 否 | 66.06% | 67.32% | 46.76% | 55.36% | 57.30% | 25.40% | 65.43% | 17.65% |
| 3 | 加 evidence_score prior | 否 | 66.32% | 68.31% | 46.76% | 55.15% | 59.01% | 23.02% | 66.35% | 14.71% |
| 4 | 加 query_coverage prior | **是** | **66.42%** | **68.62%** | 46.76% | 55.46% | 59.01% | **22.22%** | 66.32% | 14.71% |
| 5 | 减弱长Span惩罚 | **是** | **66.42%** | **68.69%** | 46.30% | **55.66%** | 58.74% | **21.43%** | 66.16% | 14.71% |
| 6 | 2400→2000整块阈值 | 否 | 66.42% | 68.69% | 46.30% | 55.66% | 58.74% | 21.43% | 66.16% | 14.71% |
| 7 | 扩大相邻段落窗口 | 否 | 66.42% | 68.69% | 46.30% | 55.66% | 58.74% | 21.43% | 66.16% | 14.71% |
| 8 | 增强List提示 | 否 | 66.42% | 68.69% | 46.30% | 55.66% | 58.74% | 21.43% | 66.16% | 14.71% |
| 9 | 增强Duration提示 | 否 | 66.42% | 68.69% | 46.30% | 55.66% | 58.74% | 21.43% | 66.16% | 14.71% |
| 10 | 减弱Requirement饱和 | 否 | 66.42% | 68.69% | 46.30% | 55.66% | 58.74% | 21.43% | 66.16% | 14.71% |

`*` 十轮执行时 Exact 还是旧的数字/时间口径；轮后发现名称、ID、标题、path 漏评，已扩展 typed exact scorer 并加回归测试。最终指标统一用新口径重算。

结论：10轮只有 **第4、5轮**有效。第6–10轮在这组数据上没有产生可测变化。继续微调 Packing 参数已经进入小收益区。

## 3. 全480题：Legacy → v2 → v3

| 指标 | Legacy | query-spans-v2 | query-spans-v3 | v3 vs Legacy |
|---|---:|---:|---:|---:|
| Prompt Lexical Recall | 63.494% | 65.964% | **66.062%** | **+2.568pp** |
| Requirement Evidence Coverage | 61.214% | 67.358% | **67.766%** | **+6.552pp** |
| Typed Exact Value Recall（323题） | 54.043% | **58.620%** | 58.595% | **+4.552pp** |
| List Item Recall | 49.800% | 55.416% | **55.490%** | **+5.690pp** |
| Condition / Exception Recall | 52.038% | 58.028% | **58.409%** | **+6.371pp** |
| Gold Doc retained | 95.745% | 95.532% | **95.745%** | 0 |
| 平均Prompt字符 | 9997.6 | 9656.4 | **9652.0** | **-3.46%** |
| 平均Contexts | 8.35 | 7.65 | **7.63** | -0.73 |
| Packing回退率 | 0%基线 | 23.13% | **22.71%** | 未达门槛 |
| Severe回退率 | 0%基线 | 3.13% | **2.92%** | -0.21pp vs v2 |
| Win / Regression / Tie | — | 218 / 111 / 151 | **221 / 109 / 150** | 小幅改善 |

v3 相对 v2 是小改进，不是跨阶段提升：Requirement Coverage +0.41pp、Condition +0.38pp、回退率 -0.42pp；Typed Exact 基本持平（-0.026pp）。

## 4. 真实 Qwen Flash 20题配对

固定20题，其中显式包含已审计的 `qst_0026/qst_0174/qst_0065/qst_0434`，其余按固定seed选取。两臂交替串行，每臂20次模型调用，temperature=0，max output=1024，retries=0；Mapper/Verifier/Secondary Retrieval关闭，只隔离 Evidence Packing。0错误。

| 指标 | Legacy | v3 | Δ |
|---|---:|---:|---:|
| Prompt Lexical Recall | 61.90% | **66.68%** | **+4.78pp** |
| Prompt Typed Exact（16题） | 53.13% | **66.25%** | **+13.13pp** |
| Prompt Requirement Coverage | 57.67% | **70.04%** | **+12.38pp** |
| Prompt List Recall（5题） | 54.00% | **68.50%** | **+14.50pp** |
| Prompt Condition Recall（6题） | 16.67% | **58.33%** | **+41.67pp** |
| Answer Lexical Recall | 46.02% | **51.29%** | **+5.28pp** |
| Answer Typed Exact（16题） | 47.08% | **62.71%** | **+15.63pp** |
| Requirement Completion | 36.54% | **46.08%** | **+9.54pp** |
| Answer List Completeness（5题） | 14.50% | **41.00%** | **+26.50pp** |
| Answer Condition Accuracy（6题） | 16.67% | **28.33%** | **+11.67pp** |
| Answer Negation Accuracy（3题） | **0%** | **0%** | 0 |
| Answer Fact Coverage | 36.54% | **46.08%** | **+9.54pp** |
| Gold Answer F1 | 32.88% | **38.48%** | **+5.60pp** |
| Citation ID validity | 100% | 100% | 0 |
| Citation semantic support | 未测 | 未测 | — |
| 总Token | 60,074 | **59,445** | **-1.05%** |
| Prompt Token | 57,286 | **56,283** | -1.75% |
| Completion Token | **2,788** | 3,162 | +13.41% |
| Cached Token | 768 | 1,280 | +512 |
| Mean latency | **1844.6ms** | 1900.4ms | +55.8ms |
| P50 latency | **1726.9ms** | 1805.6ms | +78.7ms |
| P95 latency | **2897.4ms** | 3252.0ms | +354.5ms |
| 模型调用 | 20 | 20 | 0 |
| Retry Token | 0 | 0 | 0 |

Answer lexical配对：**9 win / 6 regression / 5 tie**。Typed Exact重新评分后：16题中 **5 win / 2 regression / 9 tie**。Requirement Completion：**6 win / 2 regression / 12 tie**。List与Condition都没有配对回退；Negation 3题两边都是0，说明该能力仍是明确红项。

## 5. 代表性回退归因

- `qst_0180`：**Evaluation Miss**。v3回答正确 `traffic_escrow`，但旧完整性指标因为没复述机制描述判低；已将名称/ID/title/path加入 typed exact scorer。
- `qst_0218`：**Generation source-mixing**。正确 `Add to Canary` 已在Prompt，但模型混入另一卡片的 safe rollback。
- `qst_0222`：**Generation source-preference**。正确 `Operational Flows and Policy Gallery` 已在Prompt，模型却优先选择相似的 `Process Canvas` 页面。Typed Exact真实回退。
- `qst_0395`：**Generation causal precision**。两边都没有稳定给出“对canonical logical payload、压缩前签名”的精确server fix，v3还增加了未经充分支持的细节。

所以现在不再是“继续把Packing调高一点”能解决的主要问题。P1应转为 target-artifact-aware / Requirement-level Generation，P2同步做 typed scorer 人工校准。

## 6. Promotion Gate

| Gate | 要求 | 当前 | 结论 |
|---|---:|---:|:---:|
| Packing regression | ≤15% | **22.71%** | ❌ |
| Typed answer gain | ≥+3pp | **+15.63pp Exact** | ✅（20题） |
| Requirement completion gain | ≥+5pp | **+9.54pp** | ✅（20题） |
| Condition/negation不回退 | 不回退 | Condition↑，Negation 0→0 | ⚠️ |
| Citation ID validity | 100% | **100%** | ✅ |
| Semantic citation precision/recall | ≥95% / ≥90% | **未测** | ⚪ |
| Token / Fast | ≤1.5× | **0.990×** | ✅ |
| Mean额外延迟 | ≤+1.0s | **+0.056s** | ✅ |
| Secondary Retrieval | trigger/recovery门禁 | **未跑** | ⚪ |
| Verifier false deletion | ≤2% | **未跑** | ⚪ |
| Typed scorer-human agreement | ≥95% | **未校准** | ⚪ |

**决定：`query-spans-v3` 保持 opt-in，不切默认。** 原因不是平均质量不升，而是 Packing 回退率仍高于15%，Negation与Semantic Citation未闭环。

下一循环优先级：**P1 Generation → P2 100–200题人工typed校准 → P3 Citation semantic support → P4 Secondary Retrieval → P5 Verifier**。Retrieval继续冻结。
