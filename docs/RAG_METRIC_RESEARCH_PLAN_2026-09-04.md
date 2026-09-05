# RAG 后续指标与研究计划

日期：2026-09-04。目标：以后每轮优化都用同一张表判断是否真的变好，不再只看 Hit@10、单一词法召回或组件数量。

## 1. 当前完整基线

| 维度 | 已验证数据 | 当前判断 |
|---|---:|---|
| Document Retrieval Hit@1 | 89.15% | 冻结，回归门禁 |
| Hit@5 | 95.96% | 冻结，回归门禁 |
| Hit@10 | 98.09% | 冻结，回归门禁 |
| MRR@10 | ≈0.9217 | 冻结，回归门禁 |
| Retrieval Miss | 9/470 | 不再围绕少量 Miss 全局加规则 |
| ACL/source violation | 0 | 必须持续为0 |
| 全局 rerank Hit→Miss | Single 12 / Neighbor 8 / Parent 7 | 全局 rerank 不采用 |
| Parent rerank 额外延迟 | ≈2.69s/题 | 不采用全局方案 |
| Java Evidence 原口径 | 83.0671%，470题 | 口径不同，只作历史参考 |
| 同一 Python scorer：完整 Evidence | 77.0110%，480题 | Evidence 上界代理 |
| ↓ per-document cap | 74.8132% | 损失 2.1978pp |
| ↓ context-count cap | 70.5134% | 再损失 4.2998pp |
| ↓ 历史实际 Prompt | 65.8470% | 再损失 4.6664pp |
| 历史 Verifier 前 Answer | 53.4218% | Generation 主要瓶颈之一 |
| 历史 Verifier 后 Answer | 53.1896% | Verifier -0.2322pp |
| Fast 路由未实查 Evidence | 312/500；事实可评题 304/480 | Requirement coverage 不能直接信 |
| Citation ID validity | 100% | 只证明 ID 存在，不证明语义支持 |
| Secondary Retrieval | 500题实际触发 0 次 | 尚无质量证据 |
| Source-aware 全来源 Hit@1 | 89.15% → 87.23% | 回退，不采用 |
| Source-aware 全来源 Hit@10 | 98.09% → 97.87% | 回退 |
| Source-aware 全来源 Evidence | ≈82.82% → 81.50% | 回退 |
| Slack/Linear 特化 Hit@1 | +0.21pp | 可保留候选 |
| Slack/Linear 特化 MRR@10 | +0.16pp | 可保留候选 |
| Slack/Linear 特化 Hit@10 | 不变 | 可保留候选 |
| Slack/Linear 特化 Evidence | ≈+0.25pp | 小幅收益 |
| P1 EvidenceBuilder Context Fact Recall | 71.99% → 73.24% | +1.25pp |
| P1 EvidenceBuilder Context Fact Coverage | 79.70% → 81.15% | +1.45pp |
| P1 EvidenceBuilder Prompt chars | ≈+39% | 成本明显增加 |
| P1 EvidenceBuilder 固定小模型答案 | 未提升，略回退 | Evidence 更多≠Answer 更好 |
| 旧 Qwen Plus Evidence 增强 AFR | 50.09% → 54.12% | +4.03pp，真实完成 |
| 旧 Plus Answer Coverage | 42.42% → 47.73% | +5.31pp |
| 旧 Plus Gold Answer F1 | 39.93% → 43.54% | +3.61pp |
| 旧 Plus Gold Doc Citation | 89.36% → 93.40% | +4.04pp |
| 旧 Plus Citation Precision | 100% | 仍只是 ID 合法性口径 |
| 旧 Plus Quality 额外延迟 | ≈+2.14s/题 | 成本项 |
| 旧 Plus Fast / Quality Token | ≈225万 / 729万 | Quality ≈3.24× Fast |
| 典型 Flash Fast / Adaptive AFR | ≈52.9% → 53.5% | 仅≈+0.6pp |
| Flash Fast / Adaptive Coverage | ≈45.3% → 44.5% | -0.8pp |
| Flash Fast / Adaptive Context Recall | ≈63.8% → 65.9% | +2.1pp |
| Flash Fast / Adaptive Gold F1 | ≈39.4% → 40.4% | +1.0pp |
| Flash Adaptive Token | ≈Fast 2.6× | 收益/成本不成立 |
| query-spans 固定 Fast Prompt Recall | 63.4940% → 65.9642% | +2.4702pp，词法代理 |
| query-spans v1 | 65.0052% | 中间版本 |
| query-spans v2 paired outcomes | 218升 / 111降 / 151平，480题 | 回退率 23.13%，仍不能默认上线 |
| query-spans 平均 Prompt chars | 9997.594 → 9652.388 | -345.206，约 -3.45% |
| qst_0026 | 关键时间缺失 → 时间+snapshot+DB size 条件均进入 Prompt | 已救回 Evidence→Prompt |
| qst_0174 | 2/11 字段 → 11/11 字段进入 Prompt | 已救回 Evidence→Prompt |
| 新真实 Flash 生成 A/B | 0/20题，0模型调用 | 尚未完成，不能报 Answer 提升 |
| 新 Adaptive Plus 500 / judge100 | 未完成 | 不下结论 |
| Python tests | 103/103 | 工程门禁通过 |
| A40 Java tests | 88/88，BUILD SUCCESS | 工程门禁通过 |

注意：77.0110→65.8470→53.4218 是同一 Python scorer 的历史漏斗；63.4940→65.9642 是固定 Fast Packing 消融。两组实验配置不同，不能直接拼成一条新漏斗。

## 2. 后续真正优化的指标

### P0 Evidence → Prompt：当前第一优先级

主指标不再是单一 lexical recall，而是五个 typed 指标：

| 指标 | 含义 | 当前值 | 下一轮验收 |
|---|---|---:|---|
| Requirement Evidence Coverage | 每个 Requirement 是否至少保留一段真正支持它的 Evidence | 尚未建立人工基线 | 100–200题人工校准；候选相对 legacy ≥+5pp |
| Exact Value Recall in Prompt | 数字、日期、时间、ID、阈值是否进入 Prompt | 未全量测 | 不允许任一类型相对 legacy 回退 >1pp |
| List Item Recall in Prompt | 字段/步骤/账号集合保留比例 | 未全量测；qst_0174 已 2/11→11/11 | 相对 legacy ≥+5pp；完整列表题优先看 exact completeness |
| Condition/Exception Recall | only if、unless、depends on、例外、否定是否一起保留 | 未全量测；qst_0026 已救回 | 相对 legacy ≥+5pp |
| Packing Regression Rate | 配对题中候选比 legacy 更差比例 | 当前词法代理 111/480=23.13% | 降到 ≤15% 后才考虑默认切换 |

辅助指标必须继续记录：Prompt lexical recall 当前 63.4940%→65.9642%；Prompt chars 当前 9997.594→9652.388；Prompt Token 实际值尚未测。字符数不能代替 Token。

### P1 Prompt → Answer：Packing 冻结后立即优化

| 指标 | 为什么需要 | 当前基线 | 下一轮验收 |
|---|---|---:|---|
| Exact Value Accuracy | 修复 qst_0065 这类“值全对但词法分低” | 未建立；qst_0065 两时间实际答对但 AFR=30.95% | 建 typed scorer；候选相对 baseline ≥+3pp |
| Requirement Completion | 多问型问题实际回答了几个 Requirement | 现有 Fast coverage 不可信 | 多 Requirement 题 ≥+5pp |
| List Completeness | 11项到底答了几项 | 未全量测 | 不允许完整列表题只答部分却判 complete |
| Condition/Exception Accuracy | 数值对但限定条件错也算错 | 未全量测 | 相对 baseline 不回退 |
| Negation Accuracy | not/never/only/unsupported 等方向是否正确 | 未全量测 | 相对 baseline 不回退 |
| Answer Fact Recall | 保留历史连续性 | 历史 53.4218% verifier前 | 只作辅助，不再单独决定上线 |
| Answer Fact Coverage | 保留历史连续性 | Flash Fast≈45.3%，Adaptive≈44.5% | 只作辅助 |
| Gold Answer F1 | 保留历史连续性 | Flash Fast≈39.4%，Adaptive≈40.4% | 只作辅助 |
| Abstention Accuracy | 该拒答才拒答、不该拒答不拒答 | 未建立可靠语义基线 | 在人工校准集单列 |

Generation 研究问题：一次结构化调用按 Requirement 输出短答案+引用，是否比当前整题自由生成提高 completeness，而不把调用数变成 N 个 Requirement=N 次模型调用。

### P2 Evaluation：必须和 Generation 同期做

建立100–200题人工校准集，至少覆盖：数字、时间、日期、字段列表、步骤列表、条件、例外、否定、冲突、多 Requirement、不可回答。

自动评分必须报告：

- typed exact-value accuracy；
- list precision / recall / exact-complete rate；
- condition/exception accuracy；
- requirement completion；
- abstention accuracy；
- 自动评分与人工标签一致率。

验收目标：typed scorer 对人工标签一致率 ≥95%；若达不到，不能拿自动分数作为发布依据。词法 AFR/F1 继续保留，但降级为趋势指标。

### P3 Citation Semantic Support

| 指标 | 当前 | 目标 |
|---|---:|---|
| Citation ID validity | 100% | 保持100% |
| Claim→Citation Support Precision | 未证明 | 人工校准后 ≥95% |
| Claim→Citation Support Recall | 未证明 | 人工校准后 ≥90% |
| Unsupported Claim Rate | 未可靠测 | ≤5% |
| Citation Wrong-owner / parser corruption | 已修 qst_0434 类错误 | 回归测试必须为0 |

不要再把 Citation Precision=100% 写成“引用正确率100%”。

### P4 Secondary Retrieval

只针对“Requirement 在完整当前 Evidence 中确实缺失”的题触发。

必须记录：eligible missing requirements、attempted、trigger recall、false-trigger rate、added contexts、recovered requirements、最终答案救回、额外 Token、额外延迟。

当前真实基线：500题 attempted=0，added=0。

建议验收门槛：Missing Requirement trigger recall ≥80%；非 Missing 误触发 ≤10%；触发后 Requirement Evidence Recovery ≥50%。这三项是后续工程目标，不是已有结果。

### P5 Verifier

Verifier 只处理高风险 Claim，不允许默认重写整份答案。

必须记录：unsupported detection recall、supported false-deletion rate、pre/post typed correctness、pre/post completeness、Token、latency。

当前历史代理：Answer Fact Recall 53.4218%→53.1896%，即 -0.2322pp；qst_0434 曾出现正确主结论被误删。

上线门槛：supported false-deletion ≤2%；typed correctness 不下降；Requirement Completion 不下降。否则关闭或只在更窄风险条件触发。

## 3. 永久回归门禁，不作为主要优化目标

- Hit@1：89.15% 基线，下降 >0.5pp 即失败。
- Hit@10：98.09% 基线，必须 ≥98.0%。
- MRR@10：≈0.9217，下降 >0.005 即失败。
- ACL/source violation：必须 0。
- 全局 Hit→Miss：任何新 rerank 都要逐题报告，不能只报平均分。
- Python tests：103/103 或更多且全绿。
- Java A40 clean verify：88/88 或更多且全绿。
- Run 必须记录 commit、dirty state、input/output SHA256、模型、Prompt hash、题集、预算、Token、latency、错误数。

## 4. Cost / Latency Gate

历史已经证明“质量多一点、成本翻倍”不可接受：Flash Adaptive≈2.6× Fast；旧 Plus Quality 729万/225万≈3.24×，额外≈2.14s/题。

后续每个候选都同时报告：prompt/completion/total/cached Token，模型调用次数，mean/p50/p95 latency，失败重试 Token。

建议 promotion gate：在 typed answer correctness 有明确提升的前提下，默认路径 total Token ≤Fast 1.5×，mean latency 增量 ≤1.0s；超过该门槛只能进入 Quality/Deep 路由，不能成为默认 Fast。该门槛是后续产品约束，不是已有实验结果。

## 5. 固定研究顺序

1. 冻结 Retrieval。
2. 建100–200题 typed 人工校准集，先把评测做可信。
3. 固定 query-spans-v2，跑同一固定题集 Flash 真实生成 A/B，先回答“65.96% Prompt 是否转成 Answer 提升”。
4. 若 Prompt 提升但 Answer 不升，进入 Requirement-level Generation；不继续调 Packing。
5. Generation 收口后，再让 Missing Requirement 定向 Secondary Retrieval 参加 A/B。
6. 最后验证高风险 Verifier；不允许用 Verifier 掩盖上游缺证据。
7. 通过开发集后，必须增加未参与调参的新 query/document/template holdout，再做 Plus 正式验收。

## 6. 每轮报告固定输出

每次实验必须一页内回答：

- 版本：commit / dirty / input SHA256 / model / prompt hash；
- 样本：总题数、事实可评题、人工校准题、错误题；
- Retrieval：Hit@1/5/10、MRR、Miss、ACL violation；
- Evidence/Prompt：Requirement Evidence Coverage、Exact/List/Condition Recall、lexical recall、chars、Token、回退题数；
- Answer：Exact Value Accuracy、Requirement Completion、List Completeness、Condition/Negation、AFR、Coverage、F1、Abstention；
- Citation：ID validity、semantic precision/recall、unsupported rate；
- Secondary Retrieval：eligible/attempted/recovered/false-trigger；
- Verifier：pre/post typed correctness、false deletion；
- Cost：model calls、Token、mean/p50/p95 latency、retry cost；
- Pair outcome：wins / regressions / ties；
- 结论：promote / reject / continue research，以及失败题 qid。

只有 typed correctness、completeness、citation support 和成本同时过门禁，才允许把一个候选写成“RAG质量提升”。
