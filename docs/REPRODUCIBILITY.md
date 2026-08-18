# 复现与指标口径

## 固定环境

已记录实验环境：

```text
JDK: 17
Elasticsearch: 8.10.4
Corpus: 10,722 documents / 115,406 chunks
Questions: 500 total / 470 evaluable / 30 no-gold
Qwen index: knowledge_base_benchmark_qwen3_4b_2048_v1
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
embedding_dimension=2048
```

Importer 每批输出一行 JSON 进度。`runs/import-checkpoint.json` 记录下一物理行、累计
文档与 Chunk；失败详情写入 `runs/import-failures.jsonl`。不要在导入中途删除 checkpoint，
除非改用一个全新索引重新开始。

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
- `avg_latency_ms` / `p95_latency_ms`：每题完整检索链路延迟。
- `avg_context_tokens`：Top10 标题+Chunk 的正则 token 近似值，不是模型 tokenizer 精确值。
- `by_question_type` / `by_source_type`：相同口径的分层结果。

`answer_contains@10` 在纯检索实验中为 `null`；它不是漏测，而是此阶段没有生成答案。

## 公平对比规则

1. 固定 questions、docs、ACL 和 Chunk。
2. 换 Embedding 时新建索引，不能用旧向量。
3. 对照实验一次只改一类变量。
4. 报 470 题分母和 30 道 no-gold，不能只报 500 的模糊“准确率”。
5. 同时保留 summary 与逐题 details；总分提升后检查 question/source slice。
6. 延迟比较需说明本地 GPU还是远程 API，以及是否预热。
