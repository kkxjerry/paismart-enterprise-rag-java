# 数据格式

## `docs.jsonl`

每行一篇原始文档：

```json
{
  "doc_id": "dsid_...",
  "title": "Runbook title",
  "text": "Full document text",
  "source_path": "enterpriserag:confluence:dsid_...",
  "source_dataset": "onyx-dot-app/EnterpriseRAG-Bench",
  "source_type": "confluence"
}
```

`doc_id` 和 `text` 必填。Java Importer 会生成 Chunk ID、hash、模型版本和索引时间。

## `acl_docs.jsonl`

每行描述一篇文档的权限：

```json
{
  "doc_id": "dsid_...",
  "tenant_id": "tenant_redwood",
  "classification": "internal",
  "allowed_group_ids": ["role:employee", "source:confluence"],
  "denied_group_ids": []
}
```

当前固定 benchmark 用 `tenant_redwood` 与 `source:<source_type>` 模拟 source-aware ACL。
它是评测夹具，不是完整生产鉴权模型。

## `questions.json`

顶层可以是数组，也可以是带 `questions` 的对象：

```json
{
  "source_dataset": "onyx-dot-app/EnterpriseRAG-Bench",
  "questions": [
    {
      "id": "qst_0001",
      "question": "What are the upload limits?",
      "question_type": "basic",
      "source_types": ["github"],
      "expected_doc_ids": ["dsid_..."],
      "gold_answer": "10 MiB and 50 MiB"
    }
  ]
}
```

本项目只用 `expected_doc_ids` 计算检索指标。`gold_answer` 不参与检索分数；30 道
没有 expected doc 的题计入 `questions_total` 和 `no_gold_doc_count`，但不进入 Hit/MRR
分母。

## ES Chunk mapping

关键字段：

```text
benchmarkDocId keyword      原文档 ID，collapse 和 gold 对齐
chunkId       integer      文档内序号
title         text         standard + keyword + english
textContent   text         standard + english
sourceType    keyword      数据源过滤
tenantId      keyword      租户过滤
allowedGroupIds/deniedGroupIds keyword ACL
vector        dense_vector 模型维度，cosine
```

`config/elasticsearch-2048.json` 是已验证 mapping 快照；CLI 的 `create-index` 会按参数
动态生成同等结构。
