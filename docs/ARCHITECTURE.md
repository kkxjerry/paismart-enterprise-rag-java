# 架构与代码入口

## 离线索引链路

1. `ElasticsearchIndexCommand` 创建隔离索引，正文使用 `standard` analyzer，另保留
   `english` 子字段；向量字段维度必须与模型一致。
2. `EnterpriseRagImporter` 流式读取 `docs.jsonl` 和 ACL，不一次性加载 114 MiB 正文。
3. `TextChunker` 用 1,200 字符窗口和 200 字符 overlap，优先在段落、句号和空格切分。
4. `EmbeddingClient` 区分 document/query 角色，支持本地、OpenAI 兼容和 DashScope
   原生请求格式，并检查每个向量的数量与维度。
5. Importer 用确定性 `_id=docId:chunkId` Bulk 写 ES；每个文档批次原子更新 checkpoint。

断点只在整个“分块、Embedding、Bulk 写入”批次成功后前移。中途失败时同一批会重跑，
确定性 ID 让写入保持幂等。改变数据文件、索引、模型、维度或 Chunk 参数后，旧 checkpoint
会被签名检查拒绝。

## 在线检索链路

`EnterpriseRagJavaBenchmark` 对每个问题并列执行候选路线：

| 路线 | 查询 | 索引字段 |
|---|---|---|
| Dense | Qwen query embedding + ES KNN | `vector` |
| Original BM25 | 原始自然语言问题 | `title`, `textContent` |
| Keyword BM25 | 去停用词后的关键词 | `title`, `textContent` |
| English BM25 | 原始问题 | `title.english`, `textContent.english` |

每条路线先按 `benchmarkDocId` collapse，避免一个长文档的多个 Chunk 挤占 TopK。
同一份 tenant/source/ACL filter 应用于每条路线，然后用 weighted RRF 融合“排名”，
而不是直接相加不可比较的 cosine 和 BM25 原始分数。

已验证四路参数：

```text
retriever_k=50
dense chunk candidates=500
ANN num_candidates=2500
RRF k=10
weights: dense=0.75, original BM25=0.50,
         keyword BM25=1.25, English BM25=1.50
```

## 评测链路

Evaluator 同时写两类产物：

- `summary.json`：总体、`by_question_type`、`by_source_type` 聚合指标。
- `details.jsonl`：每题的标准文档、最终排名、各阶段延迟，便于错误分析。

检索阶段不接 LLM。这样标准文档没有召回时，不会被模型生成出来的流畅答案掩盖。

## 关键代码

- `PaiSmartRagCli`：可执行 Jar 的命令分发。
- `ElasticsearchIndexCommand`：动态 mapping。
- `EnterpriseRagImporter`：流式、断点、并发 Embedding、Bulk。
- `EnterpriseRagJavaBenchmark`：四路召回、ACL filter、指标。
- `Bm25QueryRewriter`：关键词路线。
- `ReciprocalRankFusion`：加权 RRF。

## 没有合入的方案

全局 `cross-encoder/ms-marco-MiniLM-L-6-v2` rerank 被实验否决：它会救回部分困难题，
但破坏更多已经正确的 BM25 排名，Hit@10 低于不 rerank 的 weighted Hybrid。因此本仓库
保留结果记录，不把失败方案放进默认在线链路。
