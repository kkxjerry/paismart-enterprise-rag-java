# RAG 本轮合并、验证与优化

代码已直接提交到 develop，没有创建工作树、没有 push、没有改动在线服务。不是只交付补丁文件。

代码提交：197c9b7 集成 query-spans 候选、Java Generation 测试修复及既有验收记录；7ad823c 修正选择器忽略检索排名的问题，加入配对回放/生成入口和集成回归测试。

## 实际验证

Mac：`/Users/zhouguichao/Downloads/minirag_demo/.venv/bin/python -m unittest discover -s tools -p 'test_*.py'`，103 项通过，无跳过。包含选择器、Controller 到真实生成消息构造、必需冲突证据放不下时禁止生成、Mapper 成本保留、标准答案不进入消息、选择集合稳定性、无效样本不计入质量分母等检查。

A40：在 `/srv/paismart-develop-experiment/repro-adaptive-v1/builds/209fc00-generation-test-fix` 重新执行离线 clean verify，88 项通过，0 failure/error/skip，BUILD SUCCESS。使用原生 JDK17 与既有 Maven 镜像配置；不是 Mac 上的构建，也没有本地 Docker。Mac/A40 的 GenerationConsistencyTest.java SHA256 相同：7997f94b79fadc871e83e0fa296989da93b0407d31465eab807e944f31554fcc。构建产物仍在独立目录，未替换在用 Jar；shade 既有警告仍存在。

## 真实 Evidence 的两轮迭代

使用原500题完整授权 Evidence，不使用合成题替代。固定 Fast 证据预算10000字符、最多12块、同一个确定性 Requirement 计划，关闭 Mapper、Verifier、二次检索，只比较证据选择。字符预算不是Token计费，不能由字符数声称费用降低。标准答案只在选择完成之后评分。

第一轮 v1：不再硬删同文档第三块，但纯词法排序仍把真正相关文档挤掉。qst_0026 恢复时间进入 Prompt；qst_0174 完整字段列表仍然没进 Prompt。不能仅凭原21项合成测试就认为真实问题已解决。

第二轮 v2：把 Java 已有 document_rank 作为软性先验并保留来源多样性惩罚。没有把 gold doc ID、gold answer 或特定 qid 塞进选择器，也没有提高最大字符数。现在 qst_0026 的“tens of minutes”及快照/数据库大小限定条件进入 Prompt；qst_0174 的11个字段完整进入 Prompt，而旧选择器只有2个字段名。这里证明的是证据进入 Prompt，不是模型最终回答正确。

同一480道事实可评测题的 Prompt 词法召回：旧选择器63.4940%，v1 65.0052%，v2 65.9642%。最终 v2 相对旧选择器218题上升、111题下降、151题持平。500题全部执行成功，但另外20题不进入事实分母。开发中旧计数曾混入这20题，最终统计已修正并新增回归用例；不能直接拿旧全500计数与新480计数对比。

v2 的平均渲染字符数9652.388，旧选择器9997.594，双方上限均10000。仍有111题词法回退，没有据此切换默认策略。词法指标不等于事实准确率，本批数据已用于调试，不能当未见 holdout。

7ad823c 提交后，在干净工作区重新跑103项Python测试及500题配对回放，结果与开发回放的平均值一致。

## 真实云生成边界

实时确认 Mac 登录 Shell 中 DASHSCOPE_API_KEY 存在，未输出值。尝试20题、每题两臂各一次的 qwen-flash 真实生成，执行工具在命令运行前拦截，原因是无法判定安全状态。没有通过其他路径绕过，没有读出或转移 Key，没有新模型答案、Token费用或答案正确率结果。不能把以上回放数字称为生成 A/B 提升。

## 留痕与后续入口

机器摘要：`results/2026-09-04-rag-evidence-cycle-summary.json`。

真实配对数据：`/tmp/adaptive_rag/cycle-final-replay-v2/`，包含 metadata.json、pairs.jsonl、summary.json。metadata 固定输入/实现/Runner 的 SHA256、提交及工作区状态、题集、预算、模型阶段开关。原始证据与 Trace 不提交 Git。

第一轮与开发第二轮分别保留在 `/tmp/adaptive_rag/cycle-197c9b7-replay-v1/`、`/tmp/adaptive_rag/cycle-197c9b7-replay-v2/`，不覆盖失败证据。

现有 Adaptive 默认保持 legacy，明确加 `--evidence-strategy query-spans` 才启用候选。独立诊断入口：

```sh
python tools/rag_evidence_experiment.py \
  --contexts /tmp/adaptive_rag/mixed-slack-linear-contexts.jsonl \
  --output-dir /tmp/adaptive_rag/NEW-UNIQUE-REPLAY \
  --phase replay
```

同一入口的 `--phase live --limit 20 --qid qst_0026 --qid qst_0174 --qid qst_0065 --qid qst_0434` 为真实生成对照；必须在已有安全凭据的进程执行，用新输出目录。其余题目由固定种子选取，不依答案分数挑题。两臂顺序交替、串行、每请求输出1024、零重试，不运行额外模型评审；保存实际请求消息、原始/规范化输出、调用ID、真实模型名、usage及错误，并逐臂落盘。它是证据选择消融入口，不冒充完整 Adaptive 验收。

下一阶段应先完成已固定题集的真实生成和逐题事实核对，处理回退，再做独立查询/文档 holdout；不直接扩大预算或把词法增益当作发布依据。
