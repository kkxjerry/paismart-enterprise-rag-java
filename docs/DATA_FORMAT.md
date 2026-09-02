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
  "source_type": "confluence",
  "document_version": "v2",
  "source_updated_at": "2026-08-17T12:00:00Z"
}
```

`doc_id` 和 `text` 必填。`document_version`、`source_updated_at` 可选，也可以放在
`metadata.version` / `metadata.updated_at`。Java Importer 会生成 Chunk ID、整文
`documentHash`、Chunk `contentHash`、模型版本和索引时间。

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
      "gold_answer": "10 MiB and 50 MiB",
      "answer_facts": [
        "The per-file limit is 10 MiB.",
        "The total request limit is 50 MiB."
      ]
    }
  ]
}
```

`expected_doc_ids` 只用于检索评测，不参与排序。`gold_answer` 和 `answer_facts` 只用于
计算 Evidence token recall/coverage，也不参与 Evidence 选择。30 道没有 expected doc
的题计入 `questions_total` 和 `no_gold_doc_count`，但不进入 Hit/MRR 分母。

## ES Chunk mapping

关键字段：

```text
benchmarkDocId keyword      原文档 ID，collapse 和 gold 对齐
chunkId       integer      文档内序号
title         text         standard + keyword + english
textContent   text         standard + english
sourceType    keyword      数据源过滤
sourcePath    keyword      原来源位置与版本冲突分组
tenantId      keyword      租户过滤
allowedGroupIds/deniedGroupIds keyword ACL
documentVersion keyword    来源版本，可选
documentHash  keyword      标题+整文 SHA-256
sourceUpdatedAt date       来源更新时间，可选
contentHash   keyword      单 Chunk SHA-256
vector        dense_vector 模型维度，cosine
```

`config/elasticsearch-2048.json` 是历史已验证 mapping 快照，不做原地修改。
`config/elasticsearch-evidence-2048.json` 是加入版本字段的 P1 兼容 schema，
`config/elasticsearch-evidence-2560.json` 是 Qwen3 原生维度对照 schema；CLI 的
`create-index` 会按参数动态生成同等的 Evidence 结构。
