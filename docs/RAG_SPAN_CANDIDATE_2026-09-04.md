# RAG Evidence 选择候选：query-spans

状态：已写入本地 develop 工作区；未提交、未 push、未同步 A40、未替换在线服务。读取到的 develop 提交指针仍为 209fc0046caeb706c8dea9b29f3844d15b1a8bd5。没有检查整个工作区是否干净，也没有覆盖 Java 文件。

本轮针对 Evidence → Prompt 的截断与每文档两块上限，不调整检索索引、不新增 Mapper/Verifier 调用、不修改生成模型。实验性排序是词法相关度与通用答案形态启发式，不是语义事实校验，不保证真实题目质量提升。

## 修改

新增 tools/adaptive_rag/evidence_spans.py：按问题与 Requirement 对全部授权输入块排序，以软性来源多样性惩罚替代每文档硬删除。小于等于 2400 字符的块、已映射的必需引用、检测出的列表/表格/代码保留整块。较长普通正文使用命中段落及紧邻段落的连续原文窗口。不会把不连续文本拼接成一段，也不在预算末尾硬切正文。

保留 citation_id、doc_id 和原有 chunk 元数据；evidence_span 记录输入 chunk 内 Python 字符偏移及原长，不声称是文档全局或字节偏移。selection_trace 逐引用记录纳入/淘汰及原因。lexical_affinity_not_support 只能解释排序，不能作为 Requirement coverage 或 answerability 指标。

修改 tools/adaptive_rag/budget.py：添加 legacy/query-spans 两种策略，沿用原有 initial/maximum 字符预算及总上下文数，仍在既有上限内处理必需引用扩容。标题、引用、来源元数据、偏移头和分隔符均计入字符预算。字符预算不等于相同 Token 消耗，成本尚未实测。

修改 tools/adaptive_rag/controller.py：添加配置 evidence_strategy，默认 legacy。候选路径若必需引用无法完整放入预算，返回显式 budget 错误并保留选择 Trace，而不是让模型依据一半冲突证据作答。没有把预算错误伪装成用户问题不可回答。

修改 tools/adaptive_rag_pipeline.py：添加 --evidence-strategy legacy|query-spans，默认 legacy。策略写入 run_signature 和汇总，签名 schema 由 3 升至 4，避免不同选择器的 resume 结果混用。

新增 tools/test_rag_evidence_spans.py。

## 实际验证范围

在助手隔离 Python 3.13.5 环境中，对新增选择器和测试文件执行：

python -m unittest tools.test_rag_evidence_spans -v

结果：21 tests，全部通过。其中一个用例含固定随机种子的 200 组预算/原文位置约束检查。文件内容与写入 Mac 的两个新增文件一致。

覆盖第三块证据、完整字段列表、相邻限定条件、Unicode 偏移、输入不变、必需与冲突引用、缺失引用、超大块、表格/代码、重复与非法引用、空输入、元数据预算、禁止评分标签参与排序等。

这些都是合成合约用例，不是 qst_0026/qst_0174 原始 Trace 回放，不是原有70项测试的重跑，也不是 Mac/A40 全项目集成验收。没有执行新 CLI 的真实模型 A/B，没有新的500题答案、正确率、费用或延迟结论。

DevSpaces 的命令执行调用两次均返回安全状态无法判定的拦截。本轮没有绕过拦截；只使用获准的文件读写修改 Mac 代码，新增纯算法在独立环境验证。没有读取或转移凭据。阻塞原因不是“Mac 没有 Key”。

## 启用与验收边界

现有 adaptive_rag_pipeline.py 命令加入 --evidence-strategy query-spans 才会启用候选；不加参数继续 legacy。正式对照用新输出路径，固定输入、题集、模型、路由及各阶段配置。初轮可用固定 Fast 路由和关闭 Verifier 单独隔离证据选择变量，再恢复自适应流程。不要把不同 Mapper 随机输出误算成选择器收益。

先检查 qst_0026 的恢复时间及限定条件、qst_0174 的完整11字段是否进入 Prompt，再逐项检查最终答案。必须同时记录救回与回退；关键词覆盖上升不等于事实正确。已有历史词法召回数字没有在本轮被重新解释为真实正确率。

候选限制：长列表放不下时整块排除并留痕，未实现跨块结构识别；段落窗口不能证明更远处限定条件无关；词法启发式可能选择主题相关但不含正确答案的块。默认不切换，待真实配对验证后再决定是否采用。
