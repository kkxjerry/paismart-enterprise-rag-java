# P0 / P1：ExperimentConfig、RunManifest 与 Java EvidenceBuilder

## 当前状态

P0 和 P1 已进入 Java 主评测链路，并有单元测试覆盖。

```text
P0：版本化 ExperimentConfig
    + 最终参数合并
    + RunManifest
    + Git/运行环境/输入 SHA-256/索引 mapping 留痕

P1：四路代表 Chunk 留存
    + Top 文档内部有界检索
    + 互补 Chunk 选择
    + 全局与单文档 Token 预算
    + EvidenceSpan / Chunk 级 citation
    + 版本冲突显式标记
```

A40固定500题验收已经完成。六项已报告缺陷均在修复前提交 `b9bd312` 复现，并在
`035a7d9` 验证修复，结果为 `6/6`。EvidenceBuilder 与稳定控制组的 Top50 文档排名
500/500逐题一致，ACL越界为0。它提高了 Evidence Recall，但没有提高固定 Generator 的
最终答案质量，因此实现保留为可选实验链路，当前6000 Token配置不设为默认。完整结果见
[`A40_P0_P1_VALIDATION_2026-09-03.md`](A40_P0_P1_VALIDATION_2026-09-03.md)。

## P0：一个配置真正驱动一次运行

正式配置：

```text
config/experiments/enterpriserag-qwen3-evidence-v1.json
```

它使用独立的 `knowledge_base_benchmark_qwen3_4b_2048_evidence_v2` 索引和
`config/elasticsearch-evidence-2048.json`。历史 v1 mapping 保持不变；v2 必须重新导入
相同语料，并先证明文档排名复现，才能比较 Evidence 和答案质量。

原生 2560 维对照配置：

```text
config/experiments/enterpriserag-qwen3-native2560-evidence-v1.json
```

该对照当前标记为 `not_yet_run`，用于验证 2048 维兼容截取是否影响排序与 Evidence，
不是新的已验证结果。

sample 配置：

```text
config/experiments/sample-evidence-v1.json
```

sample 执行方式：

```bash
java -jar target/paismart-enterprise-rag.jar evaluate \
  --config config/experiments/sample-evidence-v1.json
```

固定 500 题首次运行前，需要建立 P1 独立索引。以下命令假设 18084 前已经有明确的
2048 维兼容适配器：

```bash
java -jar target/paismart-enterprise-rag.jar create-index \
  --es-url http://127.0.0.1:19200 \
  --index knowledge_base_benchmark_qwen3_4b_2048_evidence_v2 \
  --embedding-model Qwen/Qwen3-Embedding-4B \
  --embedding-dimension 2048 \
  --bm25-k1 2.2 \
  --bm25-b 1.0

java -jar target/paismart-enterprise-rag.jar import \
  --docs data/enterpriserag/docs.jsonl \
  --acl-docs data/enterpriserag/acl_docs.jsonl \
  --es-url http://127.0.0.1:19200 \
  --index knowledge_base_benchmark_qwen3_4b_2048_evidence_v2 \
  --embedding-url http://127.0.0.1:18084/v1/embeddings \
  --embedding-api-format local \
  --embedding-model Qwen/Qwen3-Embedding-4B \
  --embedding-dimension 2048 \
  --embedding-batch-size 128 \
  --checkpoint runs/enterpriserag-qwen3-evidence-v2-import-checkpoint.json

java -jar target/paismart-enterprise-rag.jar evaluate \
  --config config/experiments/enterpriserag-qwen3-evidence-v1.json
```

CLI 参数可以覆盖配置文件，但 Manifest 记录的是覆盖后的最终参数：

```bash
java -jar target/paismart-enterprise-rag.jar evaluate \
  --config config/experiments/sample-evidence-v1.json \
  --evidence-token-budget 800
```

配置文件包含四部分。运行前还会拒绝两类危险配置：输出文件互相重名，以及输出路径覆盖
配置文件、问题集、文档、ACL 或 mapping 等已声明输入。

```json
{
  "schema_version": 1,
  "name": "experiment-name",
  "base_dir": "../..",
  "arguments": {},
  "inputs": {},
  "metadata": {}
}
```

- `arguments`：真正驱动 Java 评测的最终参数来源。
- `inputs`：需要进入审计记录的数据、ACL、mapping 等文件。
- `metadata`：模型原生维度、兼容转换、vLLM 版本和基线说明。
- `base_dir`：相对路径的统一解析根目录，避免从不同工作目录运行时指向不同文件。

API Key、Password、Secret、Bearer Token 不允许写入版本化配置。它们只能通过环境变量或
临时 CLI 参数注入，并会在 Manifest 中替换为 `<redacted>`。

## RunManifest 记录什么

评测开始前先写 `status=running`，成功后变为 `completed`，异常时变为 `failed`。即使检索
中断，也不会只留下半个 details 文件却不知道运行条件。

Manifest 包含：

```text
run_id
ExperimentConfig 路径与 SHA-256
CLI 覆盖后的最终参数及其 SHA-256
Git root / commit / branch / dirty 状态；dirty 时记录 tracked diff 哈希和未跟踪文件哈希，非 Git 目录明确记为 unknown
Java、OS、CPU 和工作目录
问题、文档、ACL、mapping 的路径、大小、mtime、SHA-256
ES index、vector dimension、similarity、text analyzer、mapping _meta
运行前校验实际 index 的向量维度、Embedding 模型声明和 Evidence 必需字段
summary/details/evidence 输出位置、大小和 SHA-256
成功或失败状态
```

Importer 的 checkpoint 签名也补入了：

```text
ACL 文件路径、大小和 mtime
Embedding API format
Embedding query instruction
```

因此更换 ACL、调用格式或 instruction 后，旧 checkpoint 不会被误当成同一次导入继续使用。

## P1：EvidenceBuilder 所在位置

```text
Dense route --------- representative Chunk ---+
Original BM25 ------- representative Chunk ---+
Keyword BM25 -------- representative Chunk ---+--> 文档级 weighted RRF
English BM25 -------- representative Chunk ---+          |
                                                          | 文档排名冻结
                                                          v
                                            Top 文档内部 lexical search
                                                          |
                                                          v
                                      retained route chunks + lexical chunks
                                                          |
                                                          v
                                                Java EvidenceBuilder
                                                          |
                                                          v
                                          EvidenceSpan[] + contexts JSONL
```

ES 第一阶段仍然按 `benchmarkDocId` collapse，保持原检索基线。不同之处是：融合前不再把
每条路线选中的代表 Chunk 丢掉，而是保存：

```text
route
route rank
route weight
raw score
RRF contribution
chunk ES id
```

融合后只对 Top 文档执行一次有界查询，使用同一份来源/ACL filter，通过 `inner_hits` 每篇
取有限数量的 lexical 候选 Chunk，不扫描每篇全部子 Chunk。

## Evidence 选择规则

每个候选 Chunk 的基础信息包括：

```text
query token coverage
文档内 lexical score
该 Chunk 承载的多路 RRF contribution
父文档排名 prior
```

选择第二、第三个 Chunk 时再加入：

```text
尚未覆盖的 query token 奖励
与已选 Chunk 的重复惩罚
相邻 Chunk 的小幅连续性奖励
```

当 Chunk 超过单条证据预算时，不再固定截取开头，而是在 Chunk 内寻找覆盖查询词最多的
连续窗口；查询词相同的窗口优先让命中位置靠近窗口中心。这样不会因为 Token 裁剪把位于
Chunk 中后部的关键证据删除。

这不是模型 rerank，也不会改变父文档顺序。它解决的是：同一正确文档中，哪些 Chunk 应该
进入 Prompt。

默认边界：

```text
evidence_top_documents = 10
evidence_candidate_chunks_per_document = 8
evidence_chunks_per_document = 3
evidence_token_budget = 6000
evidence_per_document_token_budget = 1200
evidence_max_chunk_tokens = 512
evidence_redundancy_penalty = 0.35
```

Evidence 采用按文档轮询分配预算：先让多个高排名文档各获得第一条证据，再分配第二、第三条，
避免第一篇长文档耗尽全部 Token。

## 版本与冲突

Importer 新增：

```text
documentVersion
documentHash
sourceUpdatedAt
contentHash
```

其中：

- `documentHash` 对标题和完整正文计算 SHA-256。
- `contentHash` 对具体 Chunk 计算 SHA-256。
- `documentVersion`、`sourceUpdatedAt` 从数据源字段或 metadata 读取。

当 Top 文档中出现相同 `sourcePath`，且候选 Chunk 的版本或整文哈希不同，EvidenceBuilder
会同时覆盖“不同文档 ID 的多个版本”和“同一 docId 残留旧 Chunk”两种情况：

```text
保留两边证据
给相关 EvidenceSpan 写入同一个 conflict_group
输出 evidence_conflicts
resolution = preserve_and_mark
```

当前不会自动判断哪个版本是真相。自动时序裁决和业务版本优先级属于后续能力，不能在没有
可靠源版本语义时靠字符串猜测。

## 输出契约

一次启用 Evidence 的评测输出：

```text
summary.json
  文档检索指标 + Evidence recall/coverage/token/latency/ACL 指标

details.jsonl
  文档排名 + 每条路线贡献 + Evidence 选择诊断

contexts.jsonl
  每题一个 contexts 数组，每个 EvidenceSpan 一个 citation_id
  可直接交给根项目 scripts/eval_generation_qwen_vllm.py

manifest.json
  配置、代码、输入、索引和运行状态
```

`contexts.jsonl` 中每条证据包含：

```text
citation_id
doc_id / chunk_id / chunk_es_id
title / source_type / source_path / classification
text / token_count / evidence_score / query_coverage
route_signals
document_version / document_hash / content_hash / source_updated_at
conflict_group
```

## 新增验收指标

summary 除原有 `Hit@K`、`MRR@10` 外，新增：

```text
evidence_fact_token_recall_avg
evidence_fact_coverage_avg
evidence_gold_answer_token_recall_avg
avg_evidence_tokens
avg_evidence_spans
avg_evidence_candidate_chunks
evidence_source_filter_violation_count
evidence_conflict_case_count
avg/p95_evidence_latency_ms
avg/p95_retrieval_latency_ms
avg/p95_latency_ms
```

`document_ranking_changed_by_evidence_count`、`document_hit_to_miss_count` 和
`document_miss_to_hit_count` 固定为 `0`，用于明确 P1 没有偷偷重排文档。Evidence 层的
事实增减由固定排名下的 context A/B 单独比较。

A40在相同检索排名、相同Generator和相同问题集下完成了正式A/B：

```text
Context fact recall：71.99% -> 73.24%
Context fact coverage：79.70% -> 81.15%
平均Prompt字符：17,196 -> 23,957
Evidence平均延迟：+208.29 ms
Qwen2.5配对答案Fact Recall：40.79% -> 40.30%
DeepEval Correctness同题50题：0.496 -> 0.494
```

因此证据覆盖提升成立，但“最终答案稳定提升”不成立。当前大预算配置不作为默认，后续只能
通过更小预算或低置信条件触发继续验证。

## 本阶段明确不做

```text
不使用 question_type 触发 Evidence
不根据 gold/source benchmark 标签改变 Evidence 算法
不做低置信 rerank
不做 BGE parent-child
不改变文档级 RRF 排名
不自动裁决冲突版本
不把 Evidence Recall 提升误写成最终答案质量提升
```

低置信路由和 bounded parent-child rerank 属于 P2。
