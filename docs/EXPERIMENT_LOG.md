# Java 检索优化实验记录

## 固定评测口径

- 数据：EnterpriseRAG-Bench 10,722 文档。
- 问题：test 500 题，其中 470 题有 gold 文档。
- Chunk：1,200 字符、200 overlap，共 115,406 Chunk。
- 指标：文档级 Hit@K、MRR@10；不接 LLM。
- 正文：Elasticsearch `standard` analyzer；English 路线使用 `english` 子字段。
- 每轮必须保留 summary 和逐题 details，不能只报一个总百分比。

## 指标变化

| 顺序 | 主要变化 | Hit@1 | Hit@10 | Hit@20 | MRR@10 |
|---:|---|---:|---:|---:|---:|
| 1 | E5 Dense baseline | 52.77% | 72.55% | 77.45% | 0.5979 |
| 2 | BM25 baseline | 45.32% | 74.68% | 85.74% | 0.5478 |
| 3 | Dense + BM25，RRF k=60 | 65.11% | 81.49% | 84.26% | 0.7099 |
| 4 | Hybrid + MiniLM-L6 rerank | 68.30% | 82.77% | 84.26% | 0.7350 |
| 5 | Top50 candidates + L6 rerank | 68.51% | 86.81% | 90.85% | 0.7493 |
| 6 | source/ACL E5 Dense | 74.89% | 92.98% | 95.53% | 0.8153 |
| 7 | source/ACL BM25 | 78.72% | 95.11% | 97.23% | 0.8470 |
| 8 | tuned BM25 k1=2.2 / b=1.0 | 83.19% | 95.74% | 97.23% | 0.8771 |
| 9 | weighted Hybrid：Dense 0.5 / BM25 1.0 | 86.38% | 96.38% | 97.45% | 0.9018 |
| 10 | weighted Hybrid + tuned BM25 | 86.81% | 96.60% | 97.66% | 0.9046 |
| 11 | E5 四路 weighted RRF | 88.51% | 97.87% | 98.30% | 0.9174 |
| 12 | 四路 + 全局 L6 rerank，失败分支 | 82.13% | 94.47% | 97.02% | 0.8636 |
| 13 | **Qwen3 + 四路 weighted RRF** | **89.15%** | **98.09%** | **98.09%** | **0.9217** |
| 对照 | 百炼 text-embedding-v4 + 四路 | 88.94% | 97.45% | 98.09% | 0.9194 |

从最初 Dense baseline 到最终 Qwen3 四路，Java Hit@10 提升 `25.54pp`；若从已经扩大
候选池的通用 Hybrid + rerank 阶段算，提升 `11.28pp`。最大的单次跃升不是换模型，
而是把 source/ACL 元数据过滤正确地放到每条召回路线，减少跨数据源噪声。

## 每一步为什么有效

### source/ACL filter

EnterpriseRAG 同时模拟 GitHub、Jira、Slack、Confluence、Gmail 等来源。问题已有来源范围时，
先过滤再 ANN/BM25 搜索，比召回后删除更准确，也不会返回无权限候选。Dense Hit@10
从早期链路提升到 92.98%。

注意：本 benchmark 用 `source_types` 构造范围，是为了验证检索。生产必须从登录用户
ACL 获取范围，不能读取题目 gold 标签。

### BM25 与 Dense 并行

Dense 擅长语义改写，BM25 擅长项目代号、数字、接口名和原文关键词。BM25-only 的
Hit@10 95.11% 高于当时 Dense 92.98%，证明企业语料不能只做向量检索。

### RRF 而不是原始分数相加

cosine 与 BM25 score 尺度不同，直接线性相加没有稳定含义。RRF 只使用每条路线的排名，
`weight/(k+rank)` 后求和。`k=10` 在该固定集上比 60 更强调头部结果。

### 关键词和 English BM25

关键词路线去掉英文问句停用词，English analyzer 路线增加 stemming。它们补充原始 BM25，
与 Qwen3 Dense 组成四路，最终 Hit@10 98.09%。

### 换 Qwen3 Embedding

Qwen3 2048 维相对 E5 384 维主要改善语义问题和最终融合排序。它不是独立带来全部提升：
Qwen3 Dense Hit@10 仍只有 95.96%，必须与 BM25 路线融合才能达到 98.09%。

## 被否决的尝试

全局 MiniLM L6 rerank：Top50 rerank 后 Hit@10 94.68%，低于 weighted Hybrid 的
96.38%。原因是 reranker 与英文企业文档、长 Chunk 的匹配不稳定，并破坏已正确的
BM25 头部排序。结论是 rerank 需要可观测的定向路由，不能默认全局开启。

## 云端模型对照

保持语料、Chunk、索引参数和融合参数不变，只替换向量：

- 本地 Qwen3 四路：Hit@10 98.09%，平均 255.44 ms。
- 百炼 text-embedding-v4 四路：Hit@10 97.45%，平均 581.53 ms。
- 差距只有 0.64pp，云端可作为不维护 GPU 时的替代；主要代价是网络延迟。
- 给百炼 query 添加当前 Qwen instruction 会降分，因此最终用 plain query。

## 剩余风险

1. 只有一个 500 题 test 集；多次调参后存在 benchmark overfitting 风险。
2. 98.09% 是任一 gold 文档进入 Top10；多文档题的 all-gold@10 为 92.34%。
3. 30 道 no-gold 题没有在检索层实现可靠拒答，不能混入 Hit/MRR 分母。
4. 最终答案正确性、忠实度和引用格式需另接 LLM eval，不能由检索指标替代。
5. Qwen3 索引约 4.5 GB，维度、索引体积和 GPU 成本高于 E5。
