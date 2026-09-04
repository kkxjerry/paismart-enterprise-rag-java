# Enterprise RAG 项目交接说明（内部版）

> 快照时间：2026-09-04（America/Los_Angeles；A40 服务器已进入 2026-09-05）  
> 用途：交给新的开发者、Reviewer 或外部专家继续检查质量链路和代码。  
> 安全说明：本文只记录密钥名称、加载位置和服务入口，不包含任何 API Key、SSH 私钥或 Bearer Token 明文。不要把本文发到公开渠道。

## 0. 先读结论

当前项目已经具备较完整的企业 RAG 工程骨架：权限过滤、多路检索、EvidenceBuilder、配置和 Manifest、增量索引、蓝绿 alias、在线 Search/Answer API、自适应 Fast/Quality/Deep、Requirement—Evidence、动态预算和条件式引用验证。

但它还不能被描述为“高质量答案系统已经验收通过”。当前真实状态是：

```text
文档级检索：强，Hit@10 = 98.09%
Evidence 构建：较强，完整 Java Evidence 的事实召回约 82%～83%
最终 Prompt：仍丢失大量事实，约 64%～66%
最终答案：中等，Flash 的事实召回约 53%
自适应增益：较小且不稳定
新 Adaptive 链路的 Qwen Plus 500题验收：没有成功运行
当前完整 Python/Java 测试：失败，需要先修复
```

最重要的接手原则：

1. 不要继续围绕少数 Retrieval Miss 加 benchmark 正则。
2. 不要把 98.09% Hit@10 当成回答准确率。
3. 优先解决 Evidence → Prompt → Answer 的事实损失。
4. 先修正测试、文档和实验结论，再继续做新的模型实验。

---

## 1. 项目定位与演进背景

项目最早是一个可解释的 Python Mini RAG：

```text
Loader → Chunk → Embedding → FAISS / pgvector → Retrieval → 可选 Rerank → LLM → Citation → Eval
```

随后切换到公开的 EnterpriseRAG-Bench，并形成两条主线：

```text
Python：离线实验、消融、评测、失败归因、生成实验
Java：Elasticsearch 索引、ACL、多路检索、EvidenceBuilder、在线 Search API
```

现在最准确的项目定位是：

> 一个面向企业多源知识库的权限感知 RAG 引擎与评测平台，而不是普通的“向量库 + Chat UI”示例，也还不是完整生产知识库产品。

当前优化目标已经从：

```text
让正确文档进入 Top10
```

转为：

```text
让每个问题要求对应的正确证据进入有限 Prompt
→ 逐项生成完整答案
→ 每个 Claim 有真实支持它的 Citation
→ 权限、版本和拒答同时正确
```

---

## 2. 代码与工作区位置

### 2.1 总工作区

```text
/Users/zhouguichao/Downloads/minirag_demo
```

主要内容：

```text
ragdemo/                       最早的教学级 Python RAG
scripts/                       大型 Python 检索、生成和评测实验
configs/retrieval_loop066.yaml Python Loop066 参数记录
runs/                          Python 历史实验输出
java-enterprise-rag/           历史 Java 实验目录
PaiSmart-main/                 历史 PaiSmart 应用副本
external/、github/             外部/独立代码目录
```

### 2.2 当前活跃 Java 仓库

```text
/Users/zhouguichao/Downloads/minirag_demo/github/paismart-enterprise-rag-java
```

Git 状态快照：

```text
remote： https://github.com/kkxjerry/paismart-enterprise-rag-java.git
branch： develop
HEAD：   0a8e981
状态：   develop 比 origin/develop ahead 5
worktree：干净
```

最近五个主要提交：

```text
0a8e981  feat: add adaptive RAG control, lifecycle and search API
666711e  feat: add qwen plus evidence enhancement and generation
b5b1cc6  test: validate P0 P1 on A40
035a7d9  fix: harden reproducible evidence pipeline
b9bd312  feat: add reproducible evidence builder pipeline
```

### 2.3 历史 PaiSmart 代码

旧记录中还出现过：

```text
/Users/zhouguichao/Desktop/cv/面试/王二/PaiSmart-main
/Users/zhouguichao/Downloads/minirag_demo/PaiSmart-main
```

这些是历史迁移来源，不是当前独立 Java 主线。接手开发应优先使用 `github/paismart-enterprise-rag-java`，除非明确要把能力重新合入完整 PaiSmart 应用。

---

## 3. 服务器、SSH、目录和端口

### 3.1 A40 连接

本机 SSH alias：

```text
alias：      a40
user：       root
host：       10.147.20.180
port：       25022
ProxyJump：  a40-jump
SSH key：    ~/.ssh/server_key
```

跳板机：

```text
alias： a40-jump
host：  10.147.20.54
port：  22
user：  root
```

连接命令：

```bash
ssh a40
```

SSH 配置包含本地 `7890` 转发。若本机 7890 已被占用，SSH 会提示 `Address already in use`，通常不影响普通 SSH 命令，但会导致该次 LocalForward 建立失败。

### 3.2 A40 硬件快照

```text
主机名：a40-48g-3
GPU：3 × NVIDIA A40，单卡约 48GB
```

2026-09-05 A40 本地时间快照：

```text
GPU0：约 32GB，占用且利用率 100%
GPU1：约 38GB，占用且利用率 100%
GPU2：约 44.6GB，占用，当前利用率 0%
```

GPU0、GPU1 上有其他任务，不要清理。GPU2 当前承载 Qwen3 Embedding 服务；停止前必须确认没有别人在复用。

### 3.3 A40 关键目录

```text
主实验根目录：
/srv/paismart-develop-experiment

Adaptive 实验与运行资产：
/srv/paismart-develop-experiment/repro-adaptive-v1

二进制与脚本：
/srv/paismart-develop-experiment/repro-adaptive-v1/bin

Adaptive 运行结果：
/srv/paismart-develop-experiment/repro-adaptive-v1/runs

P0/P1 复现记录：
/srv/paismart-develop-experiment/repro-p0p1

EnterpriseRAG 数据：
/root/minirag_demo/data/eval_1000/enterpriserag

Qwen3 Embedding 模型：
/opt/models/Qwen3-Embedding-4B

Elasticsearch 数据与日志：
/srv/paismart-develop-experiment/runtime/elasticsearch
/srv/paismart-develop-experiment/runtime/es-data
/srv/paismart-develop-experiment/runtime/es-logs
```

### 3.4 服务端口

| 端口 | 服务 | 状态/用途 |
|---:|---|---|
| 19200 | Elasticsearch 8.x | 主检索索引 |
| 18085 | vLLM Qwen3-Embedding-4B | 原生 2560 维 |
| 18084 | `qwen3_embedding_adapter.py` | 2560 → 前2048维截取 → L2归一化 |
| 18083 | E5 embedding service | 历史 E5 路径 |
| 18086 | fake embedding | 生命周期/故障测试专用，不可用于真实评测 |
| 18090 | Java `serve-search` | 当前进程指向生命周期测试索引，不是正式 benchmark 服务 |
| 18091 | 本机 Python Answer API | 当前本机仍有进程，调用本机 28090 SSH tunnel |

当前 A40 `18090` 的 Java 进程使用：

```text
index = paismart_adaptive_lifecycle_test_v1
embedding = fake-embedding / 2 dimensions
```

因此不能直接把它当作真实 EnterpriseRAG Search API。需要重新启动并指向正式 alias 或正式索引。

---

## 4. API Key、模型提供方与认证

### 4.1 云模型 Key

当前代码实际调用的是：

```text
Alibaba Cloud Model Studio / DashScope OpenAI-compatible API
API base：https://dashscope.aliyuncs.com/compatible-mode/v1
```

不是 ModelScope 开源模型推理服务。模型 alias 包括：

```text
qwen-flash
qwen-plus
qwen3.7-flash（历史实验）
```

默认环境变量：

```text
DASHSCOPE_API_KEY
```

当前本机登录 shell 状态：

```text
DASHSCOPE_API_KEY：存在
MODELSCOPE_API_KEY：不存在，也没有被当前 qwen-plus 链路使用
```

密钥只应放在 shell、进程管理器或 Secret Manager 中，不要写入：

```text
Git
.env.example
ExperimentConfig
RunManifest
日志
交接文档
```

`.env.example` 只保留空变量名。

安全检查命令，不打印实际值：

```bash
python3 - <<'PY'
import os
for name in ["DASHSCOPE_API_KEY", "MODELSCOPE_API_KEY", "RAG_SEARCH_API_KEY", "RAG_ANSWER_API_KEY"]:
    print(name, bool(os.getenv(name)))
PY
```

### 4.2 Search / Answer API Key

```text
RAG_SEARCH_API_KEY：Java Search API 的 Bearer Key
RAG_ANSWER_API_KEY：Python Answer API 的 Bearer Key
```

当前本机登录 shell 中这两个变量均未设置。历史进程可能继承了临时会话中的值，但明文没有记录，也不应从进程环境中导出。

接手人应生成新的 Key，并分别注入进程：

```bash
export RAG_SEARCH_API_KEY='<GENERATE_NEW_SECRET>'
export RAG_ANSWER_API_KEY='<GENERATE_ANOTHER_NEW_SECRET>'
```

### 4.3 身份边界

Search/Answer API 只适合放在受信网关之后：

```text
Gateway 验证用户身份
→ Gateway 注入 tenant / groups / classifications / source scope
→ RAG 服务执行 ACL filter
```

不能让终端用户在请求体中自由声明自己属于哪个 tenant 或 group。即使 API 有 Bearer Token，若所有用户共享同一个服务 Token，依然可能伪造 Principal。

---

## 5. 本机开发环境

```text
项目虚拟环境：/Users/zhouguichao/Downloads/minirag_demo/.venv
虚拟环境 Python：3.12.13
系统 python3：3.14.6
Maven：3.9.9
POM 编译目标：Java 17
```

本机 Java 环境存在不一致：

```text
java -version：OpenJDK 18
mvn -version 使用的 Java：23.0.2
```

建议接手时固定：

```text
JDK 17
Python 3.12
```

并修正 `~/.m2/settings.xml`。当前 Maven 会警告：

```text
Expected root element 'settings' but found 'mirrors'
```

---

## 6. 数据集与索引

### 6.1 EnterpriseRAG-Bench 固定数据

```text
文档：10,722
历史固定 Chunk：115,406
问题：500
可计算文档检索指标的问题：470
有答案事实、可计算生成指标的问题：480
info_not_found 问题：20
```

470、480、20 是不同评测口径，不能互相替代：

```text
470：有 expected_doc_ids，可算 Hit@K
480：有 answer_facts，且不是 info_not_found
20：专门用于拒答评测
```

A40 数据文件：

```text
/root/minirag_demo/data/eval_1000/enterpriserag/docs.jsonl
/root/minirag_demo/data/eval_1000/enterpriserag/questions.json
/root/minirag_demo/data/eval_1000/enterpriserag/acl_docs.jsonl
/root/minirag_demo/data/eval_1000/enterpriserag/by_source/
```

### 6.2 主要 Elasticsearch 索引

| 索引 | Chunk 数 | 用途/结论 |
|---|---:|---|
| `knowledge_base_benchmark_qwen3_4b_2048_v1` | 115,406 | 历史 Qwen3 四路基线 |
| `knowledge_base_benchmark_qwen3_4b_2048_evidence_clone_20260903_v1` | 115,406 | P1 排名冻结/证据实验控制组 |
| `knowledge_base_benchmark_qwen3_4b_2048_mixed_control_20260903_v1` | 115,406 | 来源切块消融控制组 |
| `knowledge_base_benchmark_qwen3_4b_2048_mixed_slack_20260903_v1` | 115,386 | 仅 Slack source-aware |
| `knowledge_base_benchmark_qwen3_4b_2048_mixed_slack_linear_20260903_v1` | 115,401 | Slack + Linear source-aware，当前较优候选 |
| `knowledge_base_benchmark_qwen3_4b_2048_sourceaware_20260903_v2` | 124,843 | 全来源 source-aware；全局指标回退，不应设默认 |

当前 alias：

```text
paismart_adaptive_selected_current_rc
→ knowledge_base_benchmark_qwen3_4b_2048_mixed_slack_linear_20260903_v1

paismart_adaptive_lifecycle_current
→ paismart_adaptive_lifecycle_test_v1
```

`*_preview_*`、`paismart_adaptive_*test*`、`*crash*` 属于预检或故障注入资产，可以在确认无人复用后删除。

### 6.3 Qwen3 Embedding 维度

模型原生输出：

```text
2560 dimensions
```

历史索引：

```text
2048 dimensions
```

A40 已验证：

```text
vLLM 不支持通过 dimensions=2048 直接修改输出
Java 请求里的 dimension=2048 会被忽略或上游拒绝
```

历史兼容链路必须显式执行：

```text
取前2048维
→ L2 normalize
```

工具：

```text
tools/qwen3_embedding_adapter.py
```

---

## 7. 当前系统架构

```text
数据源
→ Source-aware 或 Fixed Chunking
→ Qwen3 Embedding
→ Elasticsearch
   ├─ Dense KNN
   ├─ Original BM25
   ├─ Keyword BM25
   └─ English BM25
→ Weighted RRF
→ ACL / tenant / classification / source filter
→ 文档聚合
→ Java EvidenceBuilder
→ Fast / Quality / Deep Router
→ Requirement—Evidence Mapper（按需）
→ Requirement 二次检索（按需）
→ 动态 Evidence 预算
→ Qwen 生成
→ Claim-Citation Verifier（按需）
→ Answer API + 逐题评测
```

### 7.1 Java 代码地图

```text
src/main/java/com/yizhaoqi/smartpai/benchmark/
  PaiSmartRagCli.java                  CLI入口
  ElasticsearchIndexCommand.java      创建隔离索引
  EnterpriseRagImporter.java          全量分块、Embedding、Bulk导入
  EnterpriseRagSynchronizer.java      增量同步、ACL更新、旧Chunk/删除传播
  EnterpriseRagJavaBenchmark.java     四路检索、评测、在线检索核心
  ExperimentConfig.java               配置驱动实验
  RunManifest.java                    代码/输入/索引/产物留痕
  SourceAwareChunker.java              分来源切块
  IndexLifecycleCommand.java          alias status/promote/CAS
  RagSearchServer.java                Java Search API
  SearchPrincipal.java                tenant/group/classification身份
  EmbeddingClient.java                Embedding请求

src/main/java/com/yizhaoqi/smartpai/service/
  ReciprocalRankFusion.java           Weighted RRF
  Bm25QueryRewriter.java              Keyword query rewrite
  EvidenceBuilder.java                文档内Evidence选择、引用、Token预算
```

### 7.2 Python 代码地图

```text
tools/adaptive_rag_pipeline.py        离线Adaptive 500题运行入口
tools/adaptive_rag_api.py             在线Answer API
tools/rag_failure_attribution.py      R1～R5逐题失败归因
tools/adaptive_rag_compare.py         配对比较、Bootstrap、成本/延迟
tools/qwen_plus_rag_pipeline.py       旧版Qwen Plus证据增强与生成
tools/qwen_plus_pair_judge.py         盲化A/B Judge

tools/adaptive_rag/
  features.py                         Router特征和Fast/Quality/Deep决策
  requirements.py                     Requirement—Evidence映射
  retrieval.py                        二次检索和Evidence合并
  budget.py                           动态Prompt预算
  controller.py                       总控制器
  verifier.py                         Claim-Citation验证
  attribution.py                      失败分层逻辑
```

### 7.3 总工作区历史 Python 脚本

```text
/Users/zhouguichao/Downloads/minirag_demo/scripts/
```

包括：

```text
acl_overlay_eval.py                   早期企业检索大脚本
export_retrieved_contexts.py          旧上下文导出
enterprise_retrieval_ablation.py      检索消融
eval_generation_qwen_vllm.py          生成评测
eval_citation_claims.py               Citation评测
eval_answerability_gate.py            Answerability评测
```

这些脚本仍有历史价值，但不要再把其中旧链路当作当前 Java/Adaptive 主链。

---

## 8. 已完成的优化与真实结论

### 8.1 Python 66轮检索优化

起点：

```text
Dense-only Hit@10 = 58.51%
```

加入以下能力：

```text
BM25
ACL/source filter
RRF / weighted RRF
按来源候选深度
受控 rerank
parent-child / proximity
窄条件 fallback
```

Python Loop066：

```text
Hit@1 = 81.91%
Hit@10 = 98.72%
MRR@10 = 0.8764
Miss = 6
```

注意：Loop066 包含 benchmark 定制路由和窄规则，只能作为离线上界和实验参考，不能直接照搬上线。

### 8.2 Java/Elasticsearch 主线

优化路径：

```text
E5 Dense/BM25
→ Equal Hybrid
→ Weighted Hybrid
→ tuned BM25
→ Dense + Original BM25 + Keyword BM25 + English BM25
→ Qwen3-Embedding-4B
```

最终历史四路基线：

```text
Hit@1 = 89.15%
Hit@5 = 95.96%
Hit@10 = 98.09%
MRR@10 ≈ 0.9217
470道检索可评测题中 Miss = 9
ACL/source violation = 0
```

### 8.3 Rerank 实验

全局 rerank 被否决：

```text
MiniLM/BGE 等 CrossEncoder 在强 BM25/RRF 基线上产生更多 Hit→Miss
```

已验证的主要现象：

```text
单Chunk rerank：Hit→Miss 12
Neighbor：8
Parent：7
Parent延迟约2.69秒/题
```

结论：

```text
不做全局Top50 × 全Chunk × max pooling
只允许低置信、受限、每篇2～3个Chunk的bounded rerank
```

### 8.4 P0：可复现实验

新增：

```text
ExperimentConfig
RunManifest
Git commit / branch / dirty状态
输入SHA-256
mapping和Embedding元数据
输出产物SHA-256
running / completed / failed状态
```

P0方向成立，但当前新一轮结果资产仍有 `/tmp`、A40、Git summary 三处割裂，尚未完全收口。

### 8.5 P1：Java EvidenceBuilder

实现：

```text
保留四路代表Chunk和RRF贡献
Top文档内部二次检索
每篇选择多个互补Chunk
Chunk级Citation
版本冲突标记
Token预算
```

A40同口径结果：

```text
Context Fact Recall：71.99% → 73.24%（+1.25pp）
Context Fact Coverage：79.70% → 81.15%（+1.45pp）
Prompt字符：约 +39%
固定小模型答案质量：没有提升，略有回退
```

结论：Evidence 更多不等于答案更好，P1保留但大预算配置不能默认开启。

### 8.6 2026-09-03 Qwen Plus Evidence顺序增强

这是已真实完成的独立实验，不是后来的 Adaptive Controller。

固定16K生成Prompt：

```text
Fast：Java Evidence原始顺序 → qwen-plus
Quality：qwen-plus选择S* → 选中证据前置 → 保留原排序回退 → qwen-plus生成
```

500题结果：

```text
Answer Fact Recall：50.09% → 54.12%（+4.03pp）
Answer Fact Coverage：42.42% → 47.73%（+5.31pp）
Gold Answer F1：39.93% → 43.54%（+3.61pp）
Gold Doc Citation：89.36% → 93.40%（+4.04pp）
Citation Precision：100%
```

代价：

```text
平均额外延迟约2.14秒
Fast总Token约225万
Quality总Token约729万
```

可信汇总：

```text
results/2026-09-03-qwen-plus-enhancement-generation-summary.json
```

### 8.7 Adaptive 八项能力

已实现代码：

```text
1. 逐题失败归因
2. Fast/Quality/Deep Router
3. Requirement—Evidence映射
4. 动态Evidence预算
5. 条件Claim-Citation Verifier
6. 低置信二次检索
7. 来源感知Chunking
8. 索引生命周期与在线API
```

Flash 500题的典型结果：

```text
Fast Answer Fact Recall：约52.9%
Adaptive Answer Fact Recall：约53.5%
Fast Answer Fact Coverage：约45.3%
Adaptive Answer Fact Coverage：约44.5%
Fast Context Fact Recall：约63.8%
Adaptive Context Fact Recall：约65.9%
Fast Gold Answer F1：约39.4%
Adaptive Gold Answer F1：约40.4%
Adaptive Token约为Fast的2.6倍
```

说明：

```text
Adaptive明显改善了进入Prompt的Evidence，
但没有稳定提高答案完整性，部分指标回退，收益远小于成本增加。
```

### 8.8 Source-aware Chunking

全来源一起切换时：

```text
Hit@1：89.15% → 87.23%
Hit@10：98.09% → 97.87%
Evidence Recall：约82.82% → 81.50%
```

因此全来源方案被否决。

只对 Slack + Linear 启用后：

```text
Hit@1：+0.21pp
MRR@10：+0.16pp
Hit@10：不变
Evidence Recall：约+0.25pp
```

这是小幅、可接受但非决定性的增益。当前仍应保留固定切块作为回滚基线。

---

## 9. 当前结果中哪些可以相信

### 9.1 可以相信

```text
Java Qwen3四路检索 98.09% Hit@10
P0/P1 A40复现记录
2026-09-03 Qwen Plus Evidence顺序增强结果
Source-aware全量和Slack/Linear消融结果
Flash 500题运行确实完成过500/500
生命周期、ACL隔离和故障注入的方向性验证
```

### 9.2 不能直接相信

以下文件把新的 Adaptive Qwen Plus 验收写成 `passed`：

```text
results/2026-09-04-adaptive-rag-implementation-summary.json
```

但实际日志显示：

```text
plus-adaptive-full500：使用 --profile accept，CLI不支持，立即退出
plus-pair-judge100：找不到 Plus candidate 文件，立即退出
/tmp/adaptive_rag/adaptive-final-acceptance.json：不存在
```

实际 CLI 只支持：

```text
--profile optimize
--profile validate
```

因此：

> 新 Adaptive Controller 的 Qwen Plus 500题和100题盲评尚未完成。2026-09-04 的 `passed` 声明必须撤销或更正。

`docs/ADAPTIVE_RAG.md` 也仍写着错误命令：

```text
--profile accept
```

正确值是：

```text
--profile validate
```

---

## 10. 当前核心瓶颈

### 10.1 最大质量损失：Evidence → Prompt

现有实验的大致漏斗：

```text
正确文档进入Top10：98.09%
完整Java Evidence事实召回：约82%～83%
最终Prompt事实召回：约64%～66%
最终答案事实召回：约53%
```

这些指标口径不同，不能简单相减，但足以说明主要损失发生在：

```text
完整Evidence → 有限Prompt
```

当前预算仍以整Chunk为单位，一个1200字符Chunk可能只有一两句真正相关，却占用全部预算。

### 10.2 Prompt → Answer 仍继续丢失

即使事实进入Prompt，模型仍可能：

```text
只回答前几个Requirement
漏掉条件、例外和否定
压缩长列表
混淆不同来源
为保证引用而写得过度保守
```

一次性让模型完成完整SOP或跨文档长答案，稳定性不足。

### 10.3 低置信二次检索没有参加500题离线结果

多个500题汇总中：

```text
secondary_retrieval_attempted = 0
secondary_retrieval_added_contexts = 0
```

原因是离线运行没有传真实 Principal 和 Search API。代码实现存在，但质量验收没有真正覆盖它。

这意味着 Adaptive 当前只能重新排列已有 Evidence，不能解决真正的 Evidence Miss。

### 10.4 Requirement Mapper 是新的单点失败

Flash Mapper可能：

```text
少拆Requirement
过度拆分
把完整列表合并过粗
错误判断supported
选择过多Citation
返回超过max requirements / max selected
```

历史上出现过：

```text
requirements exceed limit
selected citations exceed limit
unknown citation
长输出多次重试
```

虽然已加入部分压缩/归一化，但需要系统评测 Planner 本身，而不是只看最终答案。

### 10.5 Verifier成本高且可能伤害Recall

Verifier的职责是删除或修正不受支持的Claim，不会主动补齐遗漏事实。

当前还存在：

```text
Verifier结构化输出错误
supported claim无citation
pass结果包含unsupported claim
个别500题 verifier status=error
```

如果触发门槛不准，会增加成本、延迟，并可能把正确但表达不同的内容删掉。

### 10.6 指标仍主要是词法代理

`Answer Fact Recall`使用答案与标准事实的Token集合重合度，不等于人工正确率。它会低估：

```text
同义表达
更简洁但正确的回答
术语变体
语序变化
```

但低分不能完全归因于指标，因为Context Recall和Coverage同样显示事实确实被丢失。

目前缺少：

```text
稳定人工标注集
Claim级支持关系人工校准
独立Judge校准
按Requirement完整性评分
```

### 10.7 实验资产和结论没有完全绑定

当前结果分散在：

```text
Git results/
本机 /tmp/adaptive_rag/
本机 /tmp/qwen_plus_rag/
A40 /srv/paismart-develop-experiment/repro-adaptive-v1/runs/
```

部分文件没有与当前Git commit、Prompt hash、模型返回版本和配置一一绑定。新的接手人应先建立一份可信 Run Registry。

---

## 11. 当前代码健康状态

截至本交接文档生成时，Git工作区干净，但完整测试不是绿色。

### 11.1 Python测试失败

命令：

```bash
python3 -m unittest discover -s tools -p 'test_*.py'
```

结果：

```text
56 tests discovered
1 import error
```

错误：

```text
tools/test_prompt_injection_guards.py
无法从 tools/adaptive_rag/requirements.py 导入 UNTRUSTED_EVIDENCE_RULE
```

这说明测试和当前Prompt实现已经漂移。之前“Prompt Injection测试通过”的声明不适用于当前HEAD。

### 11.2 Java完整构建失败

命令：

```bash
mvn clean verify
```

失败阶段：

```text
testCompile
```

主要错误来自：

```text
EnterpriseRagGenerationConsistencyTest.java
```

测试仍依赖已经变化或不存在的接口：

```text
private completeGeneration(...)
needsGenerationBackfill()
GenerationObservation
generationObservations(...)
Set/List构造参数不一致
```

主代码可编译不代表完整构建通过。当前不能宣称 release-ready。

### 11.3 其他环境问题

```text
~/.m2/settings.xml 根元素错误
本机Java运行时版本不一致
```

接手后的第一项工作应先让：

```text
Python tests 全绿
mvn clean verify 全绿
```

---

## 12. 推荐给接手人的优化顺序

### 阶段一：先恢复事实可信度

1. 修复 Python Prompt Injection 测试与实现的漂移。
2. 修复 Java generation consistency 测试和同步器接口漂移。
3. 修正 `docs/ADAPTIVE_RAG.md` 的 `--profile accept`。
4. 更正 `results/2026-09-04-adaptive-rag-implementation-summary.json`，撤销未真实运行的 Plus `passed`。
5. 建立统一 Run Registry：commit、config、Prompt hash、输入hash、模型alias、返回模型、产物hash。

这一阶段没有完成前，不应再报告新的质量结论。

### 阶段二：真正打穿 Evidence → Prompt

当前最值得做的不是继续扩大上下文，而是：

```text
Requirement
→ 在完整Java Evidence中寻找支持它的句子/窗口
→ 每个Requirement保留2～3个局部EvidenceSpan
→ Span保留原Chunk ID、字符范围和Citation
```

目标：

```text
最终Prompt Fact Recall 从约66%提高到75%～80%
平均Prompt Token不显著上升
```

### 阶段三：让二次检索真正参加评测

离线500题要启动真实 Search API，并为每题构造与benchmark授权一致的 Principal：

```text
Requirement missing
→ 生成独立子查询
→ 使用相同tenant/group/classification/source scope检索
→ 合并新Evidence
→ 重新映射Requirement
```

必须报告：

```text
attempted题数
新增Evidence题数
Evidence Miss救回数
原有Hit→Miss
ACL leak
额外延迟和Token
```

### 阶段四：复杂问题按Requirement生成

不要再一次性生成完整长答案。建议：

```text
每个supported Requirement单独生成短答案
→ 每段强制绑定Citation
→ missing Requirement明确说明缺失
→ conflicting Requirement并列双方
→ 最后只做结构合并，不重新创造事实
```

这比继续调整总Prompt更可能提高完整性。

### 阶段五：重新设计Verifier

建议把Verifier拆成：

```text
确定性检查：引用是否合法、是否在输入、句子是否有引用
模型检查：只核对高风险数字、日期、版本、否定、跨来源冲突
```

不要让模型Verifier重写所有答案。优先输出 Claim status，由Generator按明确错误做局部修复。

### 阶段六：建立人工校准集

从500题中选择100～200题，人工标注：

```text
Requirement列表
每个Requirement的gold EvidenceSpan
答案必须包含的事实
允许的同义表达
是否应拒答
冲突处理方式
Claim-Citation支持关系
```

没有这套校准集，继续使用同模型家族Judge只能得到弱证据。

### 阶段七：正确完成Qwen Plus验收

使用当前CLI支持的：

```text
--profile validate
```

至少运行：

```text
Qwen Plus Fast 500
Qwen Plus Adaptive 500
配对比较
100题盲评
人工抽查20～50题
```

接受门槛建议：

```text
500/500无结构错误
Citation Precision = 100%
ACL leak = 0
Answer Fact Recall至少+2pp，且Bootstrap下界尽量>0
Answer Fact Coverage不回退
Gold Answer F1不回退
unanswerable abstain ≥95%
平均Token和P95延迟在预设预算内
```

### 阶段八：泛化验证

当前Router和Prompt已经在固定500题上多次迭代。最终必须增加：

```text
按模板/文档隔离的holdout
新的真实查询集
跨数据集问题
不同来源占比
不同租户ACL组合
```

否则无法判断提升是泛化还是benchmark适配。

---

## 13. 复现命令

### 13.1 本机进入主仓库

```bash
cd /Users/zhouguichao/Downloads/minirag_demo/github/paismart-enterprise-rag-java
git checkout develop
git status
```

### 13.2 当前测试命令

```bash
/Users/zhouguichao/Downloads/minirag_demo/.venv/bin/python \
  -m unittest discover -s tools -p 'test_*.py'

mvn clean verify
```

当前两条命令都需要先修复前述问题。

### 13.3 A40 Embedding服务

```bash
ssh a40

# 原生2560维
/root/miniconda3/envs/vllm/bin/vllm serve /opt/models/Qwen3-Embedding-4B \
  --runner pooling \
  --served-model-name Qwen/Qwen3-Embedding-4B \
  --host 127.0.0.1 \
  --port 18085 \
  --max-model-len 8192

# 2048维兼容适配器
python3 /srv/paismart-develop-experiment/repro-adaptive-v1/bin/qwen3_embedding_adapter.py \
  --host 127.0.0.1 \
  --port 18084 \
  --upstream-url http://127.0.0.1:18085/v1/embeddings \
  --target-dimension 2048
```

服务已存在时不要重复启动。

### 13.4 Java固定检索与Evidence评测

```bash
java -jar target/paismart-enterprise-rag.jar evaluate \
  --config config/experiments/enterpriserag-qwen3-evidence-v1.json
```

正式A/B必须确认配置中的索引名、数据路径和实际A40目录一致。

### 13.5 Flash低成本迭代

```bash
python tools/adaptive_rag_pipeline.py \
  --contexts /tmp/qwen_plus_rag/java-evidence-contexts.jsonl \
  --output /tmp/adaptive_rag/flash-adaptive.jsonl \
  --summary-output /tmp/adaptive_rag/flash-adaptive-summary.json \
  --profile optimize \
  --mapper-model qwen-flash \
  --generator-model qwen-flash \
  --verifier-model qwen-flash \
  --verifier-mode conditional
```

### 13.6 正确的Qwen Plus验收命令

```bash
python tools/adaptive_rag_pipeline.py \
  --contexts /tmp/qwen_plus_rag/java-evidence-contexts.jsonl \
  --output /tmp/adaptive_rag/plus-adaptive.jsonl \
  --summary-output /tmp/adaptive_rag/plus-adaptive-summary.json \
  --profile validate \
  --mapper-model qwen-plus \
  --generator-model qwen-plus \
  --verifier-model qwen-plus \
  --verifier-mode conditional
```

不要使用：

```text
--profile accept
```

### 13.7 失败归因

```bash
python tools/rag_failure_attribution.py \
  --contexts /tmp/qwen_plus_rag/java-evidence-contexts.jsonl \
  --answers /tmp/adaptive_rag/plus-adaptive.jsonl \
  --output /tmp/adaptive_rag/plus-adaptive-attribution.json \
  --strict
```

### 13.8 配对比较

```bash
python tools/adaptive_rag_compare.py \
  --baseline /tmp/adaptive_rag/plus-fast.jsonl \
  --candidate /tmp/adaptive_rag/plus-adaptive.jsonl \
  --output /tmp/adaptive_rag/plus-comparison.json
```

### 13.9 盲评

```bash
python tools/qwen_plus_pair_judge.py \
  --baseline /tmp/adaptive_rag/plus-fast.jsonl \
  --candidate /tmp/adaptive_rag/plus-adaptive.jsonl \
  --output /tmp/adaptive_rag/plus-judge100.jsonl \
  --summary-output /tmp/adaptive_rag/plus-judge100-summary.json \
  --model qwen-plus \
  --limit 100 \
  --stratified
```

---

## 14. 重要实验产物

### Git中可信的聚合结果

```text
results/2026-08-14-enterpriserag-java-qwen3-embedding-4b-2048-four-route-500-summary.json
results/2026-09-03-a40-p0-p1-validation-summary.json
results/2026-09-03-qwen-plus-enhancement-generation-summary.json
```

### 需要更正的结果

```text
results/2026-09-04-adaptive-rag-implementation-summary.json
```

### A40 Adaptive结果

```text
/srv/paismart-develop-experiment/repro-adaptive-v1/runs/mixed-control-500/
/srv/paismart-develop-experiment/repro-adaptive-v1/runs/mixed-slack-500/
/srv/paismart-develop-experiment/repro-adaptive-v1/runs/mixed-slack-linear-500/
/srv/paismart-develop-experiment/repro-adaptive-v1/runs/sourceaware-retrieval-500/
/srv/paismart-develop-experiment/repro-adaptive-v1/generation-recovery-v3/
```

### 本机临时结果

```text
/tmp/qwen_plus_rag/java-evidence-contexts.jsonl
/tmp/adaptive_rag/
/tmp/adaptive_rag/final-v2/
```

`/tmp` 不是长期存储。接手前应把必要产物复制到版本化实验目录，并生成SHA-256。

---

## 15. 不要继续做的事

```text
不要围绕固定9道Java Miss继续加正则
不要使用question_type、gold doc或answer_facts参与线上Router
不要默认开启全局CrossEncoder rerank
不要通过扩大到超长Prompt掩盖Evidence选择问题
不要覆盖历史fixed-chunk基线索引
不要把source-aware全来源方案设为默认
不要把终端用户提交的tenant/group直接当可信身份
不要把实际Key写进文档、命令历史、配置或Git
不要在Plus真实运行前写“验收通过”
```

---

## 16. 希望接手人最终交付什么

请接手人不要只给架构建议，最好交付以下内容：

```text
1. 修复后的全绿测试和构建
2. 更正后的可信结果登记
3. Evidence→Prompt损失的逐题统计和主要模式
4. Requirement级二次检索的真实500题A/B
5. Sentence/Span级Evidence压缩实现
6. Requirement逐项生成实现
7. Flash迭代结果
8. 正确的Qwen Plus 500题验收
9. 100题盲评与至少20题人工复核
10. Token、延迟、ACL、拒答和质量的最终决策表
```

最值得回答的核心问题不是“还能换什么模型”，而是：

> 为什么完整Java Evidence已经覆盖约83%的事实，最终Prompt只保留约66%，而答案只有约53%；怎样在不显著增加成本的前提下，把每个Requirement对应的正确Evidence稳定送入Prompt并逐项生成？
