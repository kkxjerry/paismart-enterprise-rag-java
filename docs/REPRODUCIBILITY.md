# 复现与指标口径

## 固定环境

已记录实验环境：

```text
JDK: 17
Elasticsearch: 8.10.4
vLLM verified runtime: 0.17.0
Corpus: 10,722 documents / 115,406 chunks
Questions: 500 total / 470 evaluable / 30 no-gold
Qwen native dimension: 2560
Historical Qwen compatibility index: knowledge_base_benchmark_qwen3_4b_2048_v1
P1 evidence index: knowledge_base_benchmark_qwen3_4b_2048_evidence_v2
Cloud index: knowledge_base_benchmark_text_embedding_v4_2048_v1
```

为了避免覆盖现有索引，公开复现时应使用带自己后缀的新索引名。

## 全量导入参数

```text
chunk_size=1200
chunk_overlap=200
document_batch_size=20
embedding_batch_size=32（按服务显存/限流调整）
bulk_size=100
embedding_model=Qwen/Qwen3-Embedding-4B
native_embedding_dimension=2560
historical_stored_dimension=2048
historical_dimension_strategy=first_2048_then_L2_normalize
```

Importer 每批输出一行 JSON 进度。checkpoint schema 当前为 `version=2`；
`runs/import-checkpoint.json` 记录下一物理行、累计文档与 Chunk，失败详情写入
`runs/import-failures.jsonl`。checkpoint 签名包含 docs、ACL、
索引、模型、维度、Embedding API format、query instruction 和 Chunk 参数。不要在导入
中途删除 checkpoint，除非改用一个全新索引重新开始。

原生 2560 维与历史 2048 维必须使用不同索引。当前 vLLM 不接受任意修改输出维度；复用
2048 索引时必须经过显式适配器，不能只把 CLI 的 `embedding_dimension` 改成 2048。

## P0 配置驱动

固定 500 题 P0/P1 兼容维度配置：

```bash
java -jar target/paismart-enterprise-rag.jar evaluate \
  --config config/experiments/enterpriserag-qwen3-evidence-v1.json
```

原生 2560 维对照配置：

```bash
java -jar target/paismart-enterprise-rag.jar evaluate \
  --config config/experiments/enterpriserag-qwen3-native2560-evidence-v1.json
```

原生对照尚未运行，不得把配置文件的存在当成实验结果。

运行会写 `summary`、`details`、`contexts` 和 `manifest`。Manifest 中的输入 SHA-256、
Git commit、最终参数和实际 ES mapping 应作为一次正式实验的身份，而不是只依赖输出文件名。

## 500 题四路参数

```text
retrieval_mode=hybrid
retriever_k=50
dense_chunk_candidates=500
dense_num_candidates=2500
rrf_k=10
dense_weight=0.75
bm25_weight=0.50
keyword_bm25_enabled=true
keyword_bm25_weight=1.25
english_bm25_enabled=true
english_bm25_weight=1.50
top_k=50
Qwen query instruction=Given an enterprise search query, retrieve relevant passages that answer the query
```

## 指标定义

- `questions_total`：本次输入问题总数。
- `questions_evaluable`：至少有一个 `expected_doc_id` 的问题数。
- `hit@K`：任意一个 gold 文档出现在前 K 个文档的题目比例。
- `mrr@10`：第一个 gold 文档在 Top10 内的 reciprocal rank 均值；未命中为 0。
- `expected_doc_hit@10`：本项目中与 `hit@10` 同义，保留用于统一跨数据集字段。
- `all_expected_docs_hit@10`：所有 gold 文档都进入 Top10 的题目比例。
- `no_gold_doc_count`：没有标准文档的问题数，不进入 Hit/MRR 分母。
- `avg_retrieval_latency_ms` / `p95_retrieval_latency_ms`：四路检索与文档融合延迟。
- `avg_evidence_latency_ms` / `p95_evidence_latency_ms`：文档内候选查询与 Evidence 选择延迟。
- `avg_latency_ms` / `p95_latency_ms`：检索加 Evidence 的总延迟。
- `avg_document_context_tokens`：旧单代表 Chunk 上下文的正则 token 近似值。
- `avg_evidence_tokens`：EvidenceBuilder 实际输出的正则 token 近似值。
- `avg_context_tokens`：启用 Evidence 时等于 Evidence Token；未启用时等于旧文档上下文 Token。
- `evidence_fact_token_recall_avg`：仅在有 gold 文档的可评测问题上，answer facts 的 token 在 Evidence 中出现的平均比例。
- `evidence_fact_coverage_avg`：仅在可评测问题上，token recall 至少 0.60 的 answer facts 比例。
- `evidence_gold_answer_token_recall_avg`：仅在可评测问题上，gold answer token 在 Evidence 中出现的平均比例；no-gold 问题不进入该指标。
- `evidence_source_filter_violation_count`：Evidence 是否越过问题的离线 source scope。
- `document_ranking_changed_by_evidence_count`：P1 固定为 0，证明 Evidence 没有修改文档排名。
- `by_question_type` / `by_source_type`：相同口径的分层结果。

`answer_contains@10` 在纯检索实验中为 `null`；它不是漏测，而是此阶段没有生成答案。

## 公平对比规则

1. 固定 questions、docs、ACL 和 Chunk。
2. 换 Embedding 时新建索引，不能用旧向量。
3. 对照实验一次只改一类变量。
4. 报 470 题分母和 30 道 no-gold，不能只报 500 的模糊“准确率”。
5. 同时保留 summary、逐题 details、contexts 和 manifest；总分提升后检查 question/source slice。
6. Evidence A/B 必须复用完全相同的文档排名和 Generator，只替换上下文构建方式。
7. 运行前必须校验实际索引的向量维度、mapping `_meta.embedding_model` 和 Evidence 字段；同维度但不同模型的索引不能混用。
8. 配置、输入和输出路径必须隔离，输出不得覆盖问题、文档、ACL、mapping 或 ExperimentConfig。
9. 延迟比较需说明本地 GPU还是远程 API，以及是否预热。
10. 2048 兼容实验与原生 2560 对照必须分别报告，不能只写“Qwen3 维度”。
