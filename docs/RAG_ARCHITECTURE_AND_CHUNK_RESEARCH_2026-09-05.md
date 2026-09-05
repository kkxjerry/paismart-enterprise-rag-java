# RAG 架构与 Chunk 改造研究

日期：2026-09-05。基线：develop / bfd02a8。本文只给可落地结论，不把外部系统的论文结果直接当成本项目收益。

## 1. 结论

当前最值得改变的不是把 `chunk_size=1200` 调成 800 或 1600，而是把一个 Chunk 同时承担的四个职责拆开：

```text
检索命中单元      小：句子、命题、字段、消息、代码符号
上下文解释单元    中：段落窗口、完整列表、完整消息、函数、Issue section
引用核验单元      精确：原文 offset 对应的连续 Span
生成组织单元      按 Requirement 和文档分组，不再平铺所有 S*
```

推荐目标链路：

```text
现有强文档检索（冻结）
→ Top 文档内部的 leaf/span 检索
→ parent / sentence-window 动态扩展
→ overlap 去重和 Requirement 分槽
→ 每个 Requirement 先生成结构化事实与引用
→ 确定性组装答案
→ 高风险 Claim 才进入语义校验
```

这比“继续调全局 Rerank”“统一换成语义 Chunk”“把更多 Chunk 塞进 Prompt”更符合当前数据。

## 2. 当前 Chunk 的真实问题

实际500题 Evidence 有14,570个上下文。其中文本长度中位数1194字符、P95为1199，70.45%的上下文长度至少1190字符。九种来源最终都接近1200字符，说明主体仍是定长切割，而不是按信息单元切割。

| 现象 | 当前数据 | 影响 |
|---|---:|---|
| 接近1200字符硬上限 | 10,265 / 14,570，70.45% | Chunk size 主导边界，结构只起弱作用 |
| `chunk_kind`、`section_path`为空 | 11,529 / 14,570，79.13% | 除Slack和Linear外，生成端几乎看不到结构 |
| `speaker`、`thread_id`为空 | 12,936 / 14,570，88.79% | 邮件、会议和对话关系无法显式利用 |
| `event_time`为空 | 14,570 / 14,570，100% | 时间题只能依赖正文词法匹配 |
| 相邻Chunk存在精确重叠 | 5,202 / 6,127对，84.90% | 200字符 overlap 被重复计入排序和Prompt |
| 可识别的重复字符 | 1,034,326 / 16,441,793，6.29% | 约6.3%的Evidence字符是可直接消除的重复 |
| 每题每文档的Evidence数 | 中位数=P95=最大值=3 | Java固定三块；问题需要一块或五块都按三块处理 |

当前只对Slack和Linear启用了结构化索引。Confluence、Google Drive、GitHub、Gmail、HubSpot、Jira、Fireflies在这批contexts里的结构字段全部为空。`SourceAwareChunker`虽然声明支持部分来源，但全来源切换曾使Hit@10从98.09%降到97.87%，说明问题不是“打开source-aware开关”，而是每种来源需要独立解析策略和独立验收。

代码层还有四个具体限制：

1. `TextChunker`按字符而不是模型Token控制大小，并选择窗口中最靠后的分隔符；它不知道列表、表格、条件从句和代码符号是否完整。
2. `SourceAwareChunker.normalize`会把连续空格和Tab压成单空格。对GitHub代码、YAML和对齐表格，搜索文本可以规范化，但引用原文不能这样处理。
3. `EvidenceBuilder`在每个Top文档内最多选3块，再以轮询方式分配全局预算。这保证来源公平，却不能保证每个Requirement有答案；query-spans只能在这三块中重排，无法救回未被Java选出的第4块。
4. Prompt把不同文档的S*平铺。`qst_0218`和`qst_0222`已经显示：正确证据进入Prompt后，模型仍会把两个相似产品卡片或两个相似Confluence页面混成一个答案。

## 3. 外部系统中真正值得借鉴的部分

| 系统/研究 | 可借鉴机制 | 本项目用法 | 不应照搬 |
|---|---|---|---|
| Anthropic Contextual Retrieval | 给Chunk加50–100 token的文档内语境，再做Dense和BM25 | 先用确定性前缀：标题、heading path、来源、时间、speaker、artifact identity；只对歧义块考虑LLM前缀 | 其49%/67%失败率下降来自其他数据和模型，不能当作本项目预期值 |
| Sentence Window / Haystack Auto-Merge | 小leaf负责命中，返回相邻窗口或父节点 | 在Top10文档内部检索句子/命题；命中后返回完整段落、列表、消息或函数 | 不要把整篇父文档无条件塞入Prompt |
| Dense X Retrieval | 原子命题适合细粒度命中 | 建立`retrieval_key`命题视图，但引用和生成仍回到连续原文Span | 不用生成命题替代原文作为证据 |
| Late Chunking | 先编码长文档，再对token表示分块池化，保留文档语境 | 做独立可行性POC；先确认Qwen3 Embedding/vLLM链路能否输出token级隐藏状态 | 当前OpenAI兼容embedding API只返回单向量，不能假设直接支持 |
| LongRAG / RAPTOR | 大单元或层级摘要适合跨段、多步和整体理解 | 仅给高层问题、跨文档综合和长链路问题增加父级/摘要检索 | 当前只有10/500为`high_level`，不能全局替换精确事实检索 |
| GraphRAG / LazyGraphRAG | 实体关系和社区摘要适合全局sensemaking | 单独的global route，覆盖主题、关系、跨项目总结 | 不用于字段、时间、ID、阈值等局部精确问题 |
| RECOMP / LongLLMLingua | 相关内容抽取、压缩和位置优化 | 用抽取式Span、去重、anchor前置；保留原文offset | 不用抽象摘要作为唯一事实来源和引用来源 |
| IRCoT / CRAG | 缺什么再检索，推理与检索交替 | 只对多步问题和明确Missing Requirement触发 | 不把每题都变成多轮Agent，旧Adaptive的2.6倍Token已经证明代价 |
| RAGChecker / ARES | 分开诊断检索、上下文、答案和忠实度；以少量人工标注校准 | 延续typed scorer，并补Claim支持与人工一致率 | 不再用单一词法AFR代表答案正确率 |

相关研究也给出一个重要反结论：2026年的Chunking复现实验指出，最优方法依任务而变；结构化简单方法在in-corpus检索中可能优于LLM-guided方案，而contextualized chunking也可能损害in-document retrieval。因此这里应做分来源、分查询类型A/B，不寻找一个“最先进万能Chunker”。

## 4. 建议的新数据模型

每份文档保留一棵可追踪的层级，而不是只存一串1200字符块：

```text
Document
├── Section / Thread / Message / Issue field / Code symbol
│   ├── Paragraph / Turn / List group / Table row group / Statement
│   │   └── Sentence or Proposition leaf
```

建议新增字段：

```text
node_id, parent_id, level, start_char, end_char
raw_text                 # 引用和生成使用，绝不破坏格式
search_text              # 可规范化、可加context prefix
section_path, artifact_id, speaker, event_time
node_kind, sibling_index, previous_id, next_id
exact_anchors            # ID、版本、日期、时间、百分比、路径等
```

索引至少分成两种视图：

```text
leaf_index：句子/命题/字段，负责Dense+BM25命中
parent_store：完整段落/列表/消息/函数，负责返回给LLM
```

文档级排名继续使用当前四路RRF。leaf检索只在已授权的Top文档ID内执行，这样既保住98.09%的Hit@10，也避免再做已被证伪的全局rerank。

## 5. 分来源 Chunk 设计

| 来源 | retrieval leaf | 返回给LLM的parent/window | 必须保留 |
|---|---|---|---|
| Confluence / Drive | 句子、列表项、表格行、heading+正文命题 | 完整小节或完整列表/表格片段 | heading层级、页面标题、canonical/template标记 |
| Gmail | 当前邮件段落、主题、行动项 | 当前消息；必要时加被回复的最近一条消息 | sender、date、thread、quoted-reply边界；去签名和重复引用 |
| Slack | 单条消息或同speaker短turn | 同线程前后3–8条消息 | speaker、timestamp、reply关系、thread_id |
| Fireflies | summary/topic/decision/next_step条目 | 同一topic或相邻决定窗口 | 会议主题、参与人、时间、next_steps；不要只按`Speaker:`解析。当前样本实际是summary/topics/next_steps结构，且Fireflies是唯一平均词法回退的来源（-0.56pp） |
| Jira / Linear | 字段、acceptance criterion、comment、root cause条目 | 完整section或相关comment窗口 | 字段名、issue版本、评论时间、状态变化 |
| GitHub | symbol、docstring、配置项、错误信息 | 完整函数/类/配置块 | AST/符号路径、行号、代码缩进；raw_text与search_text分离 |
| HubSpot | CRM字段、note、timeline event | 同一note或一组相关timeline事件 | account/contact/deal identity、event time，不把结构化字段拍平成正文 |

优先级不按来源数量简单排序：Confluence有114题且40题回退，应先解决相似页面混淆和heading身份；Fireflies只有25题，但当前平均回退且结构完全丢失，应作为Chunk parser的第一批实验；Gmail有55题、当前Requirement Coverage提升8.73pp，说明消息级结构可能进一步放大收益。

## 6. 生成层也必须同步改变

只改Chunk无法解决`qst_0218`、`qst_0222`的source mixing。Prompt应由平铺S*改成：

```text
R1：需要回答什么
  canonical anchor：来自一个明确artifact的直接答案
  support：同文档的限定条件
  alternatives/conflicts：其他文档，必须明确标注，不得静默混合
R2：...
```

模型先返回：

```json
{
  "requirements": [
    {
      "id": "R1",
      "facts": ["..."],
      "citations": ["S3@420:610"],
      "source_document": "doc-id",
      "status": "supported|missing|conflicting"
    }
  ]
}
```

最终自然语言由同一次调用的第二字段或确定性代码组装。单一artifact问题中，一条Requirement默认只能有一个canonical document；跨文档引用必须是补充或冲突，不能共同生成一个未在任何单一来源出现的混合事实。

同时把高价值证据放在Prompt开头，缺失/冲突说明放结尾，普通背景放中间。长上下文研究表明，相关信息位于中部时模型利用率可能显著下降；当前简单按选择顺序平铺没有处理这一风险。

## 7. 接下来应该做的实验顺序

### E1：不重建索引，先验证层级检索思想

把Java给出的每个候选Chunk在查询时拆成sentence/list/field leaf，在Top10文档内重新排序，再回填相邻句或完整结构。与v3固定同题、同10K字符、同模型对比。

必须新增：leaf recall、parent expansion completeness、overlap waste、source contamination。成功门槛：Requirement Evidence Coverage再提升至少3pp，Packing回退率从22.71%降到15%以内，Gold Doc retained不下降。

### E2：Prompt按Requirement和文档分组

不改Evidence集合，只改变组织和生成Schema，专门验证`qst_0218`、`qst_0222`及相似artifact题。主指标：source identity accuracy、source contamination rate、Requirement Completion、Condition/Negation Accuracy。该实验比继续微调Packing权重更优先。

### E3：新建parent-child实验索引

保留原索引不动。先选Fireflies、Confluence、Gmail三个来源建立leaf+parent索引，并记录offset、层级和exact anchors。分别验收，不做全来源一次切换。

### E4：确定性Context Prefix

Embedding和BM25的`search_text`加入标题、heading path、artifact类型、speaker/date等短前缀；`raw_text`保持原样。先与无前缀parent-child索引对照，再决定是否引入LLM生成的chunk context。

### E5：命题索引与Late Chunking POC

命题只作为retrieval key，命中后必须返回原文parent；Late Chunking先验证接口、吞吐、显存和索引重建成本。任一方案不能稳定提高Top-doc内部answer-bearing leaf recall，就停止。

### E6：只给Global/Deep Route增加RAPTOR或GraphRAG

从10道high_level题开始，再人工筛选project_related/completeness子集。不得影响普通字段、时间、ID查询，也不得进入默认Fast路径。

## 8. 新增的必须监控指标

除现有指标外，再加入：

```text
hard_cap_saturation_rate       当前70.45%
structure_metadata_coverage    当前20.87%
overlap_waste_ratio            当前6.29%
answer_bearing_leaf_recall@K
parent_expansion_completeness
boundary_integrity_rate
canonical_source_accuracy
source_contamination_rate
exact_span_citation_precision/recall
index_nodes_per_document
index_storage_ratio
inner_doc_retrieval_p50/p95
```

已有硬门禁继续保留：Hit@10≥98%、ACL violation=0、默认Token≤Fast×1.5、平均额外延迟≤1秒、Citation semantic precision≥95%、Verifier false deletion≤2%。

## 9. 明确不建议做

1. 不再全局搜索一个统一Chunk size；当前来源结构差异太大。
2. 不直接全量启用SourceAwareChunker；历史全来源实验已经回退。
3. 不把GraphRAG、RAPTOR或Agent式多轮检索用于全部500题。
4. 不用LLM摘要替代原始引用Span。
5. 不先上全局Cross-Encoder/ColBERT再说；文档检索已经强，应先解决Top文档内部证据定位与生成混源。多向量检索可作为E3/E5的候选，而不是第一步。
6. 不以平均AFR上涨掩盖来源回退；每次必须按source、question_type和typed fact报告。

## 10. 参考

- Anthropic, Contextual Retrieval: https://www.anthropic.com/engineering/contextual-retrieval
- Late Chunking: https://arxiv.org/abs/2409.04701
- Dense X Retrieval: https://arxiv.org/abs/2312.06648
- LongRAG: https://arxiv.org/abs/2406.15319
- RAPTOR: https://arxiv.org/abs/2401.18059
- RECOMP: https://arxiv.org/abs/2310.04408
- IRCoT: https://aclanthology.org/2023.acl-long.557/
- Lost in the Middle: https://arxiv.org/abs/2307.03172
- RAGChecker: https://arxiv.org/abs/2408.08067
- ARES: https://arxiv.org/abs/2311.09476
- Haystack AutoMergingRetriever: https://docs.haystack.deepset.ai/docs/automergingretriever
- LlamaIndex Sentence Window: https://docs.llamaindex.ai/en/v0.10.34/examples/node_postprocessor/MetadataReplacementDemo/
- Microsoft GraphRAG: https://www.microsoft.com/en-us/research/publication/from-local-to-global-a-graph-rag-approach-to-query-focused-summarization/
- RAGFlow: https://github.com/infiniflow/ragflow
- Chunking taxonomy/reproduction: https://arxiv.org/abs/2602.16974
