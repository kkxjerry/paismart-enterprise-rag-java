# Adaptive RAG：从 Evidence 到可验证答案

本阶段在既有四路检索和 Java EvidenceBuilder 之后增加一个自适应控制层。目标不是让每道题都走更长链路，而是根据线上可得信号选择 Fast、Quality 或 Deep，并把复杂问题拆成可验证的 Requirement—Evidence 映射。

## 总链路

```text
用户身份 / ACL
→ Java 四路检索与 EvidenceBuilder
→ 确定性特征提取
→ Fast / Quality / Deep Router
→ Requirement—Evidence 映射（按需）
→ 低置信二次检索（按需）
→ 动态 Evidence 预算
→ 生成
→ Claim-Citation Verifier（按需）
→ 答案、引用、诊断和 Token/延迟指标
```

线上路由不读取 `question_type`、gold 文档、gold answer 或 `answer_facts`。这些字段只允许在离线评测和失败归因中使用。

## 1. 逐题失败归因

入口：

```text
tools/rag_failure_attribution.py
tools/adaptive_rag/attribution.py
```

每个 answer fact 按实际经过的层级归入：

```text
R1_RETRIEVAL_MISS
正确文档没有进入 Top10

R2_EVIDENCE_MISS
文档进入 Top10，但完整 Java Evidence 中没有该事实

R3_WINDOW_MISS
完整 Evidence 中有该事实，但最终 Prompt 窗口没有

R4_GENERATION_MISS
Prompt 中有该事实，但答案没有

R5_CITATION_OR_CONFLICT_ERROR
答案事实存在，但引用、冲突或权限处理错误

OK / UNANSWERABLE_OK
没有检测到上述失败，或明确不可回答题正确拒答
```

归因结果同时按问题类型、Router 模式和事实层级输出，不能用一个总体分数代替失败位置。

## 2. Fast / Quality / Deep Router

入口：

```text
tools/adaptive_rag/features.py
tools/adaptive_rag/controller.py
```

Router 只使用真实请求中可得到的特征：

```text
问题是否包含多个要求
日期、版本、ID、数值等精确约束数量
候选文档和来源数量
Dense / BM25 路线是否一致
Top1路线支持数与RRF margin
Evidence查询覆盖率
Evidence冲突数量
```

模式边界：

```text
Fast
小预算直接生成，不调用 Requirement Mapper 和 Verifier。

Quality
调用 Requirement Mapper，按支持关系重排 Evidence，必要时扩大预算。

Deep
处理多要求、跨来源、冲突或二次检索；生成后进入条件式 Verifier。
```

`--force-mode` 仅用于离线消融，不应由线上用户直接指定。

## 3. Requirement—Evidence 映射

入口：

```text
tools/adaptive_rag/requirements.py
```

输出不是一个笼统的 `selected_citations`，而是：

```json
{
  "answerability": "partial",
  "requirements": [
    {
      "id": "R1",
      "requirement": "说明故障原因",
      "status": "supported",
      "citations": ["S1"],
      "search_query": ""
    },
    {
      "id": "R2",
      "requirement": "说明最终修复版本",
      "status": "missing",
      "citations": [],
      "search_query": "final fix version"
    }
  ]
}
```

Validator 会根据每个 Requirement 的实际状态重新计算 answerability，拒绝模型把存在 missing Requirement 的计划标成完全 answerable。

## 4. 动态 Evidence 预算

入口：

```text
tools/adaptive_rag/budget.py
```

当前默认边界：

```text
Fast：约10K字符
Quality：约11K起，覆盖不足时扩展到16K
Deep：16K起，复杂或冲突问题最多约28K
```

预算由 Requirement 数量、支持状态、冲突和 Router 模式决定。它不是按照 benchmark 类型分配，也不会无上限增加上下文。

## 5. 条件式 Claim-Citation Verifier

入口：

```text
tools/adaptive_rag/verifier.py
```

Verifier 只在以下高风险情形运行：

```text
Deep模式
Requirement缺失或冲突
引用数量较多
跨来源答案中出现精确数值、日期或版本
```

它输出每个 Claim 的：

```text
supported
partial
unsupported
conflicting
```

并可返回修正版答案。Fast 单引用答案不会因为包含一个数字就额外调用模型。

## 6. 低置信二次检索

入口：

```text
tools/adaptive_rag/retrieval.py
src/main/java/.../RagSearchServer.java
```

只有 Requirement Mapper 明确产生 missing requirement 和 `search_query` 时才触发。二次检索继承第一次请求的：

```text
tenant
groups
classification
source scope
```

结果合并后重新分配 citation，不允许覆盖或伪造原始引用。

## 7. 来源感知 Chunking

入口：

```text
src/main/java/.../SourceAwareChunker.java
```

支持：

```text
Jira / Linear：Summary、Root Cause、Workaround、Resolution等字段
Slack / Fireflies：说话人、时间与对话窗口
Confluence / Google Drive：标题层级、段落、列表和表格边界
GitHub：Issue/PR结构、标题与代码块
```

每个 Chunk 保存：

```text
chunkKind
sectionPath
speaker
threadId
eventTime
chunkingStrategy
chunkingFingerprint
```

默认导入仍使用 `fixed`。`source-aware` 必须建立新索引并独立验收，不能原地修改历史索引。

## 8. 索引生命周期和在线 API

Java CLI 新增：

```text
sync-index
index-lifecycle
serve-search
```

### 增量同步

`sync-index` 区分：

```text
完全未变化：跳过Embedding
仅ACL/版本/来源元数据变化：只更新元数据
正文、sourceType、模型或分块策略变化：重新分块和Embedding
文档缩短：清理旧Chunk
源文档删除：可选删除传播
```

每个 Chunk 额外保存：

```text
documentGeneration
documentChunkCount
aclHash
sourceRevision
```

同步器只有在以下条件全部满足时才把文档视为完整：

```text
观察到的Chunk数等于documentChunkCount
所有Chunk只有一个generation
generation与当前正文、sourceType、分块指纹和Embedding模型一致
```

这样可识别 Bulk 中断后只落下一部分 Chunk，以及新旧 generation 尚未清理完的状态。旧索引只有在内容和 Chunk 数量完全匹配时才允许无向量 metadata backfill。

### 蓝绿索引

`index-lifecycle` 支持：

```text
status
promote
compare-and-set promote / rollback
```

错误的 `expected-current-index` 会拒绝切换，避免并发发布互相覆盖。

### Search API

`serve-search` 提供：

```text
GET  /health
GET  /metrics
POST /v1/search
```

ACL 在 Elasticsearch 查询中执行，缓存键包含 tenant、groups、classification、source scope、query、index 和上下文上限。不同 Principal 不能共享缓存结果。

### Answer API

```text
tools/adaptive_rag_api.py
```

提供：

```text
GET  /health
GET  /metrics
POST /v1/search
POST /v1/answer
```

Python API 调用 Java Search API，再运行 Router、Requirement、动态预算、生成和条件 Verifier。云模型 Key 只存在于 Python 服务环境中，不传给 Java 检索节点。

当前认证方式定位为受信网关后的 service-to-service bearer。生产接入时，tenant 和 groups 必须由可信身份层注入，不能允许终端用户自行声明。

## Flash 优化口径

调参使用 `qwen-flash`。分层50题最终优化版相对全 Fast 10K 控制组：

```text
升级到Quality/Deep：14/50（28%）
Answer Fact Recall：49.89% → 52.60%
Answer Fact Coverage：36.12% → 44.91%
Context Fact Recall：63.84% → 65.95%
Gold Answer F1：37.72% → 39.03%
不可回答题拒答：80% → 100%
平均延迟：2.16s → 4.50s
```

该50题只用于低成本迭代，不作为最终结论。最终配置必须在500题上重跑，并由 `qwen-plus` 做独立生成和盲评验收。

## 运行

Flash 优化：

```bash
python tools/adaptive_rag_pipeline.py \
  --contexts runs/java-evidence-contexts.jsonl \
  --output runs/adaptive-flash.jsonl \
  --summary-output runs/adaptive-flash-summary.json \
  --profile optimize \
  --mapper-model qwen-flash \
  --generator-model qwen-flash \
  --verifier-model qwen-flash \
  --verifier-mode conditional
```

Plus 验收：

```bash
python tools/adaptive_rag_pipeline.py \
  --contexts runs/java-evidence-contexts.jsonl \
  --output runs/adaptive-plus.jsonl \
  --summary-output runs/adaptive-plus-summary.json \
  --profile validate \
  --mapper-model qwen-plus \
  --generator-model qwen-plus \
  --verifier-model qwen-plus \
  --verifier-mode conditional
```

逐题归因：

```bash
python tools/rag_failure_attribution.py \
  --contexts runs/java-evidence-contexts.jsonl \
  --answers runs/adaptive-plus.jsonl \
  --output runs/adaptive-plus-attribution.json \
  --strict
```

配对比较：

```bash
python tools/adaptive_rag_compare.py \
  --baseline runs/fast-plus.jsonl \
  --candidate runs/adaptive-plus.jsonl \
  --output runs/adaptive-plus-comparison.json
```

## 明确边界

```text
不使用benchmark oracle做在线路由
不把source-aware直接覆盖历史索引
不把更多Evidence等同于更好答案
不在所有问题上运行Verifier
不允许二次检索绕过ACL
不把service-to-service bearer描述成完整终端用户认证
不宣称增量Bulk具有数据库事务级原子性
```
