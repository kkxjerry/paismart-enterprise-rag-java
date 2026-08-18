# 面试讲法

## 60 秒项目介绍

“我做的是一个独立 Java 企业 RAG 检索与评测项目。数据用 EnterpriseRAG-Bench，
包含 Slack、Jira、Confluence、GitHub 等企业来源。我先把 10,722 篇文档按 1,200 字符、
200 overlap 切成 115,406 个 Chunk，调用 Qwen3-Embedding-4B 生成 2048 维向量，写入
Elasticsearch，同时保留 BM25 字段和 ACL 元数据。在线检索不是只做向量，而是 Dense、
原始 BM25、关键词 BM25、English BM25 四路并行，再用 weighted RRF 融合。固定 500 题中
470 题有标准文档，Java Hit@10 从早期 86.81% 提升到 98.09%，MRR@10 从 0.7493 提升到
0.9217。这里 98.09% 是标准文档进入 Top10，不是答案正确率。全局 rerank 实验反而降分，
所以我没有为了堆组件把它留在默认链路。”

## 面试官追问

### BM25 和 Dense 谁在前？

它们不是串行前后关系，而是两个并行检索路线。离线阶段各自需要索引：BM25 建倒排索引，
Dense 建向量索引。查询阶段两路各取候选，再融合；rerank 若启用，才是在候选召回之后。

### 为什么 BM25 这么强？

企业问题含项目代号、数字、接口名和配置键，词面匹配非常重要。该数据上 source/ACL
BM25-only Hit@10 95.11%，高于当时 Dense-only 92.98%，所以纯向量方案不成立。

### 为什么用 RRF？

BM25 score 与 cosine score 不在同一尺度。RRF 只依赖每路 rank，把各路贡献写成
`weight/(k+rank)`，避免人工归一化原始分数。

### 为什么 rerank 没用？

全局 reranker 对部分困难题有效，但对更多已正确的 BM25 排序造成回退。实验中 weighted
Hybrid Hit@10 96.38%，全局 L6 rerank 只有 94.68%，所以应按可观测信号定向启用，
而不是默认全局启用。

### 98.09% 为什么这么高？

第一，指标是 Top10 文档召回而非答案正确。第二，题目提供 source type，我们把它作为
benchmark 的检索范围，降低了跨来源噪声。第三，这是同一个固定 test 集上的多轮优化，
存在过拟合风险，必须在 held-out 数据上再验证。

### 如何保证权限安全？

tenant、source、allowed groups 和 denied groups 在 Dense 与所有 BM25 查询中使用同一
pre-filter。不能先全库召回再在 Java 里删，因为那会污染 TopK，也可能暴露标题等元数据。

### 为什么选择 Elasticsearch？

它同时提供 BM25、dense_vector KNN、metadata filter 和 collapse，适合在一个服务里完成
企业 Hybrid。代价是 2048 维索引约 4.5 GB，ANN 参数和 mapping 需要单独管理。

### 项目最重要的工程点是什么？

不是某个模型，而是可复现：索引隔离、断点导入、确定性 Chunk ID、向量维度校验、
固定 500 题、逐题 details、按题型/来源拆分指标，以及失败实验也留痕。

### 下一步怎么做？

先建立 held-out 企业问题集，避免继续对固定 500 题调参；再针对 Confluence/Fireflies
验证 parent-child retrieval。检索冻结后才接 LLM，分别评测答案正确性、忠实度、引用和
no-gold 拒答，不能把这些混成一个“准确率”。
