# A40 P0 / P1 复现与全量验收

> A40 运行日期：2026-09-03（服务器时区）  
> 被测代码：`035a7d9a395f4450c78f3cc4ec0bb9b0010d5eda`  
> 数据：EnterpriseRAG-Bench，10,722 篇文档、115,406 个 Chunk、500 题，其中 470 题可计算文档检索指标。

## 结论

本轮严格按照“先在修复前提交复现，再验证修复”的顺序执行。

```text
六项已报告缺陷：A40 复现 6 / 6
六项对应修复：A40 验证 6 / 6
P0：接受
P1 实现：保留为可选实验链路
P1 当前 6000 Token 配置：不接受为默认配置
```

P1 能在不改变文档排名和 ACL 结果的前提下提高 Evidence 覆盖与 gold 文档引用率，但当前
配置把平均 Prompt 从约 17.2K 字符扩大到约 24.0K 字符，并未提高固定 Generator 的答案
质量。Qwen3.5 Judge 的同题 DeepEval Correctness 也基本持平。因此不能把“Evidence 更多”
直接写成“答案更好”。

## 1. A40 环境与固定输入

```text
Host：a40-48g-3
Java：17.0.19
Elasticsearch：8.10.4
vLLM：0.17.0
DeepEval：4.1.8
Embedding：Qwen/Qwen3-Embedding-4B
Generator：Qwen/Qwen2.5-VL-3B-Instruct
Judge：Qwen/Qwen3.5-9B，disable_thinking=true
```

固定输入哈希：

| 输入 | SHA-256 |
|---|---|
| `docs.jsonl` | `72744bf266042fd675221795be501dd2d062ee5789fb24e23bb40ac5456c5f3b` |
| `questions.json` | `0a9542fe3bef39fb668f46491a59b93ebf939f8c441bc132550292d2b22dd3d1` |
| `acl_docs.jsonl` | `e1900ff6567bdf484bd22fade263a0ed495a0476f4443c07e6a35d837bcb244a` |

本轮没有停止或修改其他用户的训练任务和既有 E5 服务。运行结束后，本轮启动的
Qwen3 Embedding、Qwen2.5 Generator 和 Qwen3.5 Judge 进程已全部停止。

## 2. 六项缺陷：先复现，再保留修复

修复前提交：`b9bd312`。修复提交：`035a7d9`。

| 缺陷 | 修复前 A40 结果 | 修复后 A40 结果 |
|---|---|---|
| no-gold 问题仍产生 Evidence 答案分数 | 复现 | 指标改为 `null`，不进入聚合 |
| Token 裁剪只取 Chunk 开头，尾部查询证据丢失 | 复现 | 使用查询词聚焦窗口，尾部证据保留 |
| 同一 `docId` 残留不同版本 Chunk 未标记冲突 | 复现 | 输出 conflict group 与 `preserve_and_mark` |
| 输出路径可以覆盖输入数据或 ExperimentConfig | 复现 | 配置解析阶段拒绝运行 |
| 同维度但不同 Embedding 模型的索引可以被使用 | 复现 | 运行前同时校验维度和 mapping `_meta.embedding_model` |
| RunManifest 把非 Git 目录当 clean，且无未跟踪文件/输出哈希 | 复现 | Git 状态为 unknown；记录 untracked 与输出 SHA-256 |

复现结果：

```text
REPRODUCTION_RESULT=6/6
```

修复验证结果：

```text
FIX_VERIFICATION_RESULT=6/6
```

远端证据：

```text
/srv/paismart-develop-experiment/repro-p0p1/b9bd312/a40-bug-reproduction.log
SHA-256: a1691b2ae0e919461c9c562e2cdf735feccab5ccc076ada2ed28f2f88030b30c

/srv/paismart-develop-experiment/repro-p0p1/035a7d9/a40-fix-verification.log
SHA-256: a2cbc34449348b695e0fd7a26bdf71725a6ffed80d5e6ea7007142a7e715b72a
```

由于六项都已在修复前代码上稳定复现，本轮保留全部六项修复；没有基于静态推测新增修复。

## 3. Qwen3 2048 维兼容问题

A40 实测结果：

```text
原生 /v1/embeddings：返回 2560 维
请求 dimensions=2048：HTTP 400
Java local 请求中的 dimension=2048：被服务忽略，仍返回 2560 维
```

因此，历史 2048 维索引不能通过修改 Java 参数复用。新增：

```text
tools/qwen3_embedding_adapter.py
```

适配器明确执行：

```text
上游原生 2560 维
→ 取前 2048 维
→ L2 归一化
→ 返回 OpenAI-compatible 2048 维响应
```

它不是模型原生降维。版本化 ExperimentConfig 仍必须写明：

```text
native_dimension = 2560
stored_dimension = 2048
dimension_strategy = first_2048_then_l2_normalize
```

## 4. 检索控制组

为了给历史索引增加 P1 所需 mapping 字段，又不重新构建 HNSW 图，本轮使用 Elasticsearch
段级 `_clone`，随后只追加 `documentVersion`、`documentHash` 和 `sourceUpdatedAt` mapping。

稳定控制组与段级克隆索引的 Top50 排名逐题一致：

```text
500 / 500 exact Top50 match
```

稳定控制组指标：

| 指标 | 结果 |
|---|---:|
| Hit@1 | 89.15% |
| Hit@5 | 95.96% |
| Hit@10 | 98.09% |
| Hit@20 | 98.09% |
| MRR@10 | 0.9219 |
| All gold@10 | 92.34% |
| Source filter violation | 0 |

说明：第一次对源索引设置 write block 后触发了 Lucene 段提交，ANN 的部分近邻顺序随之改变；
同一源索引复跑后与段级 clone 完全一致。这个现象属于 ANN 索引状态变化，不是
EvidenceBuilder 重排。采用稳定复跑结果作为控制组。

普通 `_reindex` 会重建 HNSW 图，造成 256 题 Top50 顺序变化，不能用于“冻结排名”的
Evidence A/B。该无效试验索引已删除，段级 clone 索引保留用于审计：

```text
knowledge_base_benchmark_qwen3_4b_2048_evidence_clone_20260903_v1
```

## 5. Java EvidenceBuilder：500题结果

EvidenceBuilder 与稳定控制组的 Top50 文档排名逐题完全一致：

```text
Top50 ranking differences：0 / 500
document ranking changed：0
Hit → Miss：0
Miss → Hit：0
retrieval ACL violation：0
evidence ACL violation：0
```

Java summary 自身指标：

| 指标 | 结果 |
|---|---:|
| Evidence fact token recall | 82.81% |
| Evidence fact coverage | 93.56% |
| Evidence gold answer token recall | 82.75% |
| 平均 Evidence Token | 4,504.95 |
| 平均 Evidence Span | 29.14 |
| 平均候选 Chunk | 63.42 |
| 平均 Evidence 延迟 | 208.29 ms |
| P95 Evidence 延迟 | 267.64 ms |
| 平均总延迟 | 477.98 ms |
| P95 总延迟 | 661.17 ms |

这些 Java 内部指标采用 Java 自身 token 规则，不能直接与旧 Python 上下文指标比较。

## 6. 同口径 Evidence A/B

用现有生成评测脚本相同的 token 规则、相同 470 道可评测题和相同 24,000 字符 Prompt
上限，重新计算旧 `lexical_multi_chunk` 与 Java EvidenceBuilder：

| 指标 | lexical_multi_chunk | Java EvidenceBuilder | 变化 |
|---|---:|---:|---:|
| Context fact token recall | 71.99% | 73.24% | +1.25 pp |
| Context fact coverage | 79.70% | 81.15% | +1.45 pp |
| Gold answer token recall | 73.21% | 74.14% | +0.93 pp |
| 平均 Prompt 字符 | 17,196 | 23,957 | +39.3% |
| 平均 Context 数 | 10.00 | 29.09 | +190.9% |

结论：Evidence 覆盖确实提升，但提升是用明显更大的上下文换来的。

## 7. 固定 Generator：500题 A/B

固定条件：

```text
Generator：Qwen/Qwen2.5-VL-3B-Instruct
temperature：0
Prompt：同一脚本、同一系统指令
max_context_chars：24,000
max_tokens：512
concurrency：4
```

两侧500题最终各有4题持续产生非法 JSON。错误题不同，因此配对结果只使用双方都成功的
492题，避免把结构化输出失败算成 Evidence 差异。

| 指标 | 旧上下文 | Java Evidence | 变化 |
|---|---:|---:|---:|
| Answer fact token recall | 40.79% | 40.30% | -0.49 pp |
| Answer fact coverage | 28.53% | 27.49% | -1.04 pp |
| Gold answer token F1 | 35.52% | 34.86% | -0.66 pp |
| Context fact token recall | 71.98% | 73.25% | +1.27 pp |
| Context fact coverage | 79.86% | 81.29% | +1.43 pp |
| Citation coverage | 97.15% | 100.00% | +2.85 pp |
| Gold doc citation rate | 73.78% | 80.89% | +7.11 pp |
| Grounded answer proxy | 19.31% | 18.50% | -0.81 pp |
| 平均生成延迟 | 2,298 ms | 3,017 ms | +719 ms |
| 平均 Prompt 字符 | 17,224 | 23,959 | +39.1% |

结论：模型更常引用正确文档，但答案事实覆盖没有提升，反而略有下降。当前 Generator 无法
稳定利用新增证据。

## 8. DeepEval Correctness：同题50题

从双方成功的492题中，按 question type 轮询抽取同一50题：

```text
Judge：Qwen/Qwen3.5-9B
DeepEval：4.1.8
thinking：关闭
Correctness threshold：0.5
Judge error：0
```

| 指标 | 旧上下文 | Java Evidence |
|---|---:|---:|
| Correctness 平均分 | 0.496 | 0.494 |
| 通过率 | 46% | 48% |
| 提升题数 | - | 9 |
| 回退题数 | - | 10 |
| 不变题数 | - | 31 |

按类型观察到：

```text
intra_document_reasoning：+0.12
semantic：+0.06
high_level：+0.033
project_related：+0.02
conflicting_info：-0.10
constrained：-0.05
completeness：-0.033
```

50题只能作为方向性语义评测，但结果已经足够否定“当前配置稳定提升最终答案”的说法。

## 9. 最终决策

### P0：接受

ExperimentConfig、RunManifest、输入/输出哈希、Git 状态、索引模型校验和输出覆盖保护均已
在 A40 验证。

### P1 实现：保留为可选实验能力

它满足：

```text
不改变文档排名
不引入 ACL 泄漏
能输出 Chunk 级引用、路线贡献和版本冲突信息
能提高 Evidence Recall 与 gold 文档引用率
```

### 当前 P1 配置：不设为默认

以下配置不应成为线上默认：

```text
top_documents = 10
chunks_per_document = 3
token_budget = 6000
```

原因：

```text
Evidence Recall 仅提高约 1.25 pp
Prompt 增加约 39%
Evidence 阶段增加约 208 ms
Generator 增加约 719 ms
答案词法指标略降
DeepEval Correctness 基本持平
```

下一轮只应测试更小、更有条件触发的 Evidence 预算，而不是继续无条件增加 Context。该工作
属于 P2，不在本轮修改范围内。

## 10. 远端产物

根目录：

```text
/srv/paismart-develop-experiment/repro-p0p1/full500/runs
```

关键产物：

```text
historical-index-baseline-rerun-*.json/jsonl
segment-clone-baseline-*.json/jsonl
java-evidence-segment-clone-*.json/jsonl
qwen25-old-lexical-500-final*.json/jsonl
qwen25-java-evidence-500-final*.json/jsonl
deepeval-qwen25-old-paired50*.json/jsonl
deepeval-qwen25-new-paired50*.json/jsonl
deepeval-paired-stratified50-qids.json
```

完整聚合结果及关键 SHA-256：

```text
results/2026-09-03-a40-p0-p1-validation-summary.json
```
