# RAG 逐题诊断：为什么检索强，答案仍然差

审计日期：2026-09-04。审计起点：develop / 8eefad8。

## 结论

不能把 Hit@10=98.09% 和 Answer Fact Recall≈53% 理解为“模型把98分的检索结果答成53分”。两者不是同一指标，也不在同一批可评测题上。

本次找到了三类不同问题：Python 的证据预算会删除已经检索到的关键答案；词法评测既会低估正确答案，也会高估不含关键值的上下文；引用分句与校验逻辑还会破坏已经生成的答案。它们必须分别验证，不能全部归结为千问能力不够。

本次只复算既有500题结果并做确定性回放，没有新增云模型调用，没有新的500题生成A/B，也没有新的Plus验收。不能用下面的回放结果宣称答案质量已经提升。

## 证据来源与口径

完整证据：/tmp/adaptive_rag/mixed-slack-linear-contexts.jsonl

历史答案：/tmp/adaptive_rag/flash-release-adaptive-full500.jsonl

完整证据 SHA-256：b7663976eee8e2499de4b4d3ffe609f0310e107f34dc3a77c5e6d8a8d2496c3b

历史答案 SHA-256：090c70ef0428e0760ac7c01a6fe9544c15604c83310758a5ed1c39a051aae660

历史答案 run_signature：44c93cfc5e89f06a9eff9cae2e6bec85018fd640e4ef0c3b8e9bd61557f0bc31

500题中，Python事实评测覆盖480题；Java原Evidence指标覆盖470题。Java scoreEvidence 在 expectedDocIds 为空时不评分，分词、停用词和数字单位处理也与Python不同。

用同一Python评分器、同一480题复算：

完整Evidence：77.0110%

经过每文档块数上限：74.8132%

再经过总块数上限：70.5134%

实际进入Prompt：65.8470%

Verifier前答案：53.4218%

Verifier后答案：53.1896%

Java原口径为83.0671%，但分母是470题，不能与以上数字直接相减。

这些仍然只是词法重合指标，不是语义正确率，也不是严格的事实覆盖上界。窗口阶段下降11.164个百分点，生成阶段下降12.425个百分点，校验阶段下降0.232个百分点，仅描述该词法指标。不能直接把这些差值解释为各阶段真正漏掉了多少事实。

## 三个实质性反例

### qst_0026：答案已在Evidence里，却被Fast预算删除

问题：客户私有部署升级失败、有预先快照时，通常需要多久恢复？

正确文档的 S21 明确写出：恢复通常是几十分钟，具体取决于快照机制和数据库大小。

Java Evidence 已包含 S1、S11、S21。但 Fast 设置每文档最多2块，留下讨论回滚前提和支持安排的 S1、S11，删除真正给出时间的 S21。S21进入Prompt的字符数是0。

最终答案变成“支持团队没有给出典型恢复时间”。这是已检索到的关键证据被中间层移除，不是应该继续优化文档检索，也不是仅靠要求模型更仔细就能修复。

相关代码：tools/adaptive_rag/budget.py 的 DynamicEvidenceBudget.decide、prioritize_contexts。

### qst_0174：Prompt词法分90.4%，实际必需字段列表却不在里面

问题：contract conformance results ledger 每条记录必须包含哪些字段？

正确文档 S21 原文完整列出11项：timestamp、service_a、service_a_version、service_b、service_b_version、endpoint、dcts_version、test_case_id、result(pass|fail)、failure_reason、run_id。

同样因为每文档最多2块，S21被删除，Prompt只保留同一文档的 S1、S11。这些上下文包含大量 conformance、ledger、required 等共享词，词法得分仍达90.4040%。模型最终拒答。

因此，不能因为“Prompt得分高，答案为0”就认定是Generation Miss。真正的问题是字段列表缺失，词法分把周边相关内容误当成了事实支持。

相关代码：tools/qwen_plus_rag_pipeline.py 的 token_recall、fact_scores；tools/adaptive_rag/attribution.py 的词法归因也需要结合人工证据定位解释。

### qst_0065：两个时间回答正确，答案分却只有30.95%

问题要求事故内部宣布时间和缓解时间。标准答案是16:09 UTC和16:34 UTC，模型也准确给出两个时间，并引用S11。

但 Answer Fact Recall 是30.9524%。答案没有重复标准事实中的大量叙述词，加上词法处理差异，导致正确答案得分很低。

这说明当前“约53%”不能称为真实答案正确率。也不意味着系统其实全部正确：前两个案例已经证明存在真实的证据丢失。

## 另外两个已复现的代码错误

### 指定引用ID存在，但引用正文被截断

原预算只检查 selected_citations 的编号是否出现。只要 S26 的开头进入Prompt，就被视为已保留，哪怕支持答案的后半段已被切掉。

480题中有442题存在末块截断，不等于442题都答错。其中Mapper指定的块被截断有2题：qst_0361的S26、qst_0372的S29；指定引用编号整项缺失为0。

本次修复：比较指定引用的实际文本是否完整；必要时在已有maximum_chars内扩容；仍不完整则明确输出 selected_citations_truncated。

两题回放均从13,000字符扩到原有16,000字符上限，指定块恢复完整，未再有指定引用缺失或截断。没有重新生成答案，也没有证明这两题最终评分一定上升。达到硬上限后如何降级或阻断仍需独立策略。

### 句号后的引用被错误分配给下一句

qst_0434 的答案形式是“Fintech最多。 [S3] 后续解释……”。原分句器会把第一句切成没有引用，把[S3]放进下一句。Verifier随后将第一句判为unsupported并删除，最终答案丢了直接回答问题的主结论。

该题草稿本身还有故事数量算错的问题：标准答案是3篇，草稿描述成3个case study加1个one-pager。因此，不能把整个失败都归结为解析错误，也不能声称修好引用就答对整题。

本次新增共享 split_cited_segments，生成端的引用规范化、未引用诊断和Verifier分句使用同一规则，保留句号后引用的归属，并覆盖vs.、小数和列表序号等回归用例。

## 当前指标还会制造两种假象

Citation Precision=100%只表示引用ID存在于给定上下文，不表示引用真正支持句子。原来的自动补引用逻辑仍属格式补救，不是语义验证。

Fast默认把整个问题包装为supported的R1，不实际检查证据。该轮500题有312题走Fast，事实可评的480题中有304题。因此Requirement覆盖率不能当成独立的答案完整性指标。Verifier删掉内容后，final covered_requirements 仍继承校验前字段，也不能当成已复核结果。

这两类统计目前保留兼容实现，本次没有通过换分词或修改阈值制造分数上涨。

## 已执行的改动与验证

新增 tools/rag_trace_audit.py：同题同评分器阶段审计、重复qid和混合run检查、单题证据检查、预算回放、输入哈希记录。

新增 tools/adaptive_rag/claims.py：共享引用感知分句器。

修复 tools/adaptive_rag/budget.py：指定证据截断检测、受限扩容、显式截断字段。

修复controller与verifier引用分句；保留既有句子补引用兼容行为，并明确其不证明语义支持。

补入实际模型Prompt的untrusted-evidence指令规则，修复UNTRUSTED_EVIDENCE_RULE导入问题；测试改为检查运行时Prompt，而不是检查源码里有没有几个词。这不是提示注入防御完整有效性的证明。

撤销 results/2026-09-04-adaptive-rag-implementation-summary.json 的两项Plus passed声明，标记not_completed并保留失败日志路径。docs/ADAPTIVE_RAG.md 的 --profile accept 已改为 --profile validate。其他历史验证声明不视为本轮重新验证。

Python全量测试：70项，0 failure，0 error；其中新增14项回归与审计完整性测试。git diff --check通过。

Java未通过本轮验证：多次Maven执行超时，最后一次停在resources:3.3.1:resources。不能把历史交接中的Java编译错误当成本轮已重新复现，也不能声称Java已修好。

未修改服务器、索引、API Key或正式服务；没有运行新生成A/B、人工校准集或Plus验收。

## 接下来最值得做的改动

第一，建立关键值/字段级校准集。对数字、时间、字段列表、条件和否定分别记录“是否在完整证据中、是否在实际Prompt中、答案是否正确给出”，保留词法指标作为辅助，不再用单一0.6阈值判断语义支持。先覆盖本次反例，再扩到有代表性的校准集，不能只优化这几题。

第二，在固定Token预算下做Span级证据选择对照。先给每个Requirement找到包含实际答案值的连续原文，再补必要上下文。保留doc/chunk/offset与引用关系，不能只保相关词；表格、条件、否定、例外和跨句依赖需要作为整体。避免“每篇文档2块”的机械规则删除第三块里唯一的答案。原Fast策略本次未替换，这仍是待验证的主要质量改动。

第三，再验证生成完整性。先用一次结构化生成同时返回各Requirement的短答案与证据引用，按关键值/字段检查缺项；只对未完成部分补检索或补生成。不要默认给每个Requirement都增加一次模型调用，也不要默认让Verifier重写整份答案。

对照必须冻结题集、Evidence输入、模型与预算，按qid报告救回、回退、拒答、关键值准确性、引用支持、Token和延迟。先通过语义校准与小规模配对，再做真正的Plus 500题验收。历史Adaptive总Token约为Fast的2.59倍，词法答案召回仅高约0.94个百分点，不能以组件数量代替效果。

## 复现

在项目根目录，使用项目Python运行：

python -m unittest discover -s tools -p 'test_*.py'

python tools/rag_trace_audit.py --contexts /tmp/adaptive_rag/mixed-slack-linear-contexts.jsonl --answers /tmp/adaptive_rag/flash-release-adaptive-full500.jsonl

python tools/rag_trace_audit.py --contexts /tmp/adaptive_rag/mixed-slack-linear-contexts.jsonl --answers /tmp/adaptive_rag/flash-release-adaptive-full500.jsonl --inspect-qid qst_0026 --inspect-qid qst_0174

python tools/rag_trace_audit.py --contexts /tmp/adaptive_rag/mixed-slack-linear-contexts.jsonl --answers /tmp/adaptive_rag/flash-release-adaptive-full500.jsonl --replay-budget-qid qst_0361 --replay-budget-qid qst_0372

结果摘要：results/2026-09-04-rag-trace-audit-summary.json。
