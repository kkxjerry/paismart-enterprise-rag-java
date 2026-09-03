# Qwen Plus：Evidence 增强与答案生成

> 运行日期：2026-09-03  
> 输入：Java EvidenceBuilder 固定500题 contexts  
> 增强模型：`qwen-plus`  
> 生成模型：`qwen-plus`  
> Judge：`qwen-plus`，仅作为辅助盲评  
> 本轮没有调用本地生成模型或本地 Judge。

## 结论

最终保留的增强方式不是删掉大部分 Evidence，而是：

```text
Java EvidenceBuilder contexts
→ Qwen Plus 只选择原始 S* 引用编号
→ 被选证据移到最前
→ 其余证据保持原顺序作为回退
→ 仍按相同16K字符预算构造Prompt
→ Qwen Plus生成带引用答案
```

固定500题结果：

```text
Answer Fact Recall：50.09% → 54.12%  （+4.03 pp）
Answer Fact Coverage：42.42% → 47.73%（+5.31 pp）
Context Fact Recall：69.25% → 71.15% （+1.89 pp）
Gold Answer F1：39.93% → 43.54%     （+3.61 pp）
Gold Doc Citation：89.36% → 93.40%  （+4.04 pp）
Grounded Answer Proxy：32.71% → 36.88%（+4.17 pp）
```

生成侧平均上下文仍约16K字符，没有通过扩大Generator Prompt换取提升。

代价是：

```text
平均总延迟：4.29 s → 6.44 s
新增平均延迟：约2.14 s
500题新增计入产物的Token：约504万
```

因此最终提供两档：

```text
Fast：generate-only，默认
Quality：enhance-generate，显式开启
```

## 1. 实际调用的模型服务

本机登录环境已有：

```text
DASHSCOPE_API_KEY
```

代码通过阿里云百炼兼容接口调用：

```text
API base：https://dashscope.aliyuncs.com/compatible-mode/v1
model：qwen-plus
```

密钥只从环境变量读取：

```text
--api-key-env DASHSCOPE_API_KEY
```

以下内容均不会进入输出：

```text
密钥值
Authorization Header
本地Shell配置
```

提交前又检查了脚本和全部实验产物，未发现密钥内容。

## 2. 为什么增强器不改写Evidence

增强模型只允许返回：

```json
{
  "answerability": "answerable",
  "selected_citations": ["S1", "S11"],
  "conflict_citations": []
}
```

它不能返回新的事实摘要。

选择完成后，Generator读取的仍然是Java EvidenceBuilder输出的原始Chunk文本。这样保持：

```text
引用ID不变
原始证据文本不变
doc_id和chunk_id不变
ACL边界不变
不会把模型摘要当成新的事实来源
```

所有模型返回的引用都要经过程序校验：

```text
必须符合S1、S2等格式
必须存在于当前输入
不能伪造未知引用
引用数组与答案内联引用会统一
```

## 3. 三种增强方案的实际对比

### 方案A：只保留模型选中的Evidence

50题结果：

```text
平均Generator上下文：3,855字符
Answer Fact Recall：52.49%
Context Fact Recall：55.00%
Gold Answer F1：42.19%
```

它虽然极大压缩Prompt，但丢失了大量补充事实，因此拒绝。

### 方案B：选中Chunk加同文档兄弟Chunk

50题结果：

```text
成功：49/50
平均Generator上下文：5,656字符
Answer Fact Recall：49.66%
Context Fact Recall：58.86%
Gold Answer F1：39.72%
```

同文档补全并不能稳定恢复多文档信息，也被拒绝。

### 方案C：选中证据前置，原始排序回退

规则：

```text
Qwen Plus选择S*
→ Selected Evidence
→ Original Remaining Evidence
→ 统一按16K字符截断
```

这不是硬裁剪，而是对有限Prompt预算内的Evidence顺序做Query-aware调整。

该方案在50题和500题上都表现最好，因此保留。

## 4. 500题A/B条件

控制组：

```text
Java EvidenceBuilder原始顺序
→ 16K字符Prompt
→ Qwen Plus生成
```

候选组：

```text
Java EvidenceBuilder原始顺序
→ Qwen Plus查看最多48K字符候选
→ 最多返回8个S*，实际平均2.80个
→ Selected-first + Original-fallback
→ 相同16K字符Prompt
→ Qwen Plus生成
```

共同条件：

```text
questions：500
answer-evaluable：480
info_not_found：20
temperature：0
初始max output tokens：768
并发：8
Generator Prompt上限：16,000字符
```

两组最终均为：

```text
500/500成功
非法引用：0
Citation Precision：100%
```

## 5. 500题总结果

| 指标 | 原始顺序 | Qwen Plus增强 | 变化 |
|---|---:|---:|---:|
| Answer Fact Token Recall | 50.09% | 54.12% | +4.03 pp |
| Answer Fact Coverage | 42.42% | 47.73% | +5.31 pp |
| Context Fact Token Recall | 69.25% | 71.15% | +1.89 pp |
| Context Fact Coverage | 72.78% | 77.08% | +4.30 pp |
| Gold Answer Token F1 | 39.93% | 43.54% | +3.61 pp |
| Gold Answer Token Recall | 47.61% | 51.85% | +4.23 pp |
| Citation Coverage | 88.80% | 92.60% | +3.80 pp |
| Citation Precision | 100% | 100% | 0 |
| Gold Doc Citation Rate | 89.36% | 93.40% | +4.04 pp |
| Grounded Answer Proxy | 32.71% | 36.88% | +4.17 pp |
| Unanswerable Abstain Accuracy | 95% | 95% | 0 |
| Unanswerable Caveat Accuracy | 100% | 100% | 0 |
| 平均Generator上下文 | 15,989.61 chars | 15,989.89 chars | +0.28 chars |
| 平均Generator延迟 | 4,143 ms | 4,319 ms | +176 ms |
| 平均端到端延迟 | 4,295 ms | 6,436 ms | +2,141 ms |

注意：增强没有改变文档检索结果，`Hit@10`两侧均为98.09%。

## 6. 逐题配对，而不是只看平均数

### Answer Fact Recall

```text
提升：194题
回退：109题
不变：177题
平均变化：+4.03 pp
Bootstrap 95% CI：[+2.88, +5.35] pp
```

### Answer Fact Coverage

```text
提升：115题
回退：53题
不变：312题
平均变化：+5.31 pp
Bootstrap 95% CI：[+2.97, +7.78] pp
```

### Context Fact Recall

```text
提升：165题
回退：29题
不变：286题
平均变化：+1.89 pp
Bootstrap 95% CI：[+1.48, +2.39] pp
```

### Gold Answer F1

```text
提升：222题
回退：152题
不变：106题
平均变化：+3.61 pp
Bootstrap 95% CI：[+2.50, +4.79] pp
```

### Gold文档引用

```text
新增正确引用：20题
丢失正确引用：1题
不变：449题
```

这些结果说明提升不是由少数极端样本拉高。

## 7. 运行波动检查

`temperature=0`不代表云端Alias完全确定。

因此额外比较了两次独立的原始顺序50题运行：

```text
Answer Fact Recall平均漂移：-0.66 pp
Answer Fact Recall平均绝对逐题变化：4.23 pp
Gold F1平均漂移：-0.53 pp
Context指标变化：0
```

全量增强的Answer Fact Recall提升为4.03 pp，明显高于两次控制运行的平均漂移。

更重要的是，Context指标只由输入排序和截断决定，不受生成采样影响；其全量提升为确定性的
`+1.89 pp`。

## 8. Qwen Plus盲化100题评审

从500题中按question type轮询抽取100题，包含10道`info_not_found`。

盲化方式：

```text
候选作为A：50题
候选作为B：50题
Judge看不到baseline/candidate名称
```

结果：

```text
Qwen Plus增强胜：34
原始顺序胜：18
平局：48
Judge Error：0
```

评分：

| 评分 | 原始顺序 | 增强 |
|---|---:|---:|
| Correctness | 7.25 | 7.68 |
| Completeness | 6.84 | 7.52 |
| Directness | 7.30 | 7.56 |
| Correctness/Completeness均值 | 7.05 | 7.60 |

这是参考答案驱动的同模型盲评，仍可能存在同模型家族偏差，因此只作为辅助证据，不替代500题
确定性指标和人工校准。

## 9. 长答案结构化输出修复

第一次全量控制组中，以下三道`completeness`题连续产生不完整JSON：

```text
qst_0440
qst_0442
qst_0445
```

在768输出Token下失败，改成1536后3/3成功，确认是输出长度截断。

客户端现在会：

```text
检测finish_reason=length
或检测不完整JSON
→ 输出预算翻倍
→ 最大4096
→ 重新请求
→ 累加各次Token使用量
```

最终结果：

```text
控制组：500/500成功
增强组：500/500成功
增强组有6题从768自动扩大到1536
```

不是简单把所有请求都固定设成高上限。

## 10. 一个未保留的错误假设

全量结果中，增强器把29题标为`insufficient`，其中11题属于可计算答案指标的问题。

最初怀疑是Generator被`insufficient`提示诱导拒答，于是暂时移除了该提示，并只重跑这29题。

结果：

```text
11道可回答题仍全部拒答
原始顺序Generator在这11题上也全部拒答
```

因此“提示导致误拒答”没有被复现，该改动已经撤销，没有留在最终代码中。

## 11. 为什么24K增强输入没有采用

把增强阶段候选输入从48K降到24K后，分层50题结果：

```text
Answer Fact Recall：51.80%
Context Fact Recall：69.25%
Gold Answer F1：40.23%
```

同批原始顺序Answer Fact Recall为51.63%，收益只有约0.17 pp；明显弱于48K增强，因此没有
把24K设为质量模式配置。

这也说明增强器需要看到比Generator更完整的候选池，才能把后部关键证据提前到16K窗口内。

## 12. Token与延迟

最终产物记录的Token：

```text
Fast生成：2,248,938
Quality增强：5,018,278
Quality生成：2,272,187
Quality合计：7,290,465
比Fast增加：5,041,527
```

这些是最终成功产物中记录的Token，不等同于整个开发过程的账单，因为探索实验和被重试覆盖的
旧请求没有全部包含在最终JSONL中。

延迟：

```text
Fast端到端平均：4.29 s
Quality端到端平均：6.44 s
Quality额外平均：2.14 s
```

因此推荐：

```text
默认Fast
低置信、多跳、完整性要求高的请求显式使用Quality
```

本轮没有实现自动低置信路由，它属于下一阶段。

## 13. 使用方式

### Fast：只生成

```bash
python tools/qwen_plus_rag_pipeline.py \
  --contexts runs/java-evidence-contexts.jsonl \
  --output runs/qwen-plus-fast.jsonl \
  --summary-output runs/qwen-plus-fast-summary.json \
  --pipeline generate-only \
  --model qwen-plus \
  --generation-max-input-chars 16000 \
  --concurrency 8
```

### Quality：增强后生成

```bash
python tools/qwen_plus_rag_pipeline.py \
  --contexts runs/java-evidence-contexts.jsonl \
  --output runs/qwen-plus-quality.jsonl \
  --summary-output runs/qwen-plus-quality-summary.json \
  --pipeline enhance-generate \
  --model qwen-plus \
  --enhance-max-input-chars 48000 \
  --enhance-max-selected 8 \
  --selection-expansion rerank \
  --generation-max-input-chars 16000 \
  --concurrency 8
```

密钥由登录环境提供：

```bash
export DASHSCOPE_API_KEY=...
```

命令行不需要传入密钥值。

使用`--resume`时，工具会校验：

```text
contexts SHA-256
qid文件SHA-256
Enhancement/Generation Prompt SHA-256
model、pipeline、temperature
Context和Token预算
selection_expansion
```

签名不同会直接拒绝复用，避免把其他配置的成功行混入本次结果。

### 盲化配对评审

```bash
python tools/qwen_plus_pair_judge.py \
  --baseline runs/qwen-plus-fast.jsonl \
  --candidate runs/qwen-plus-quality.jsonl \
  --output runs/qwen-plus-pair-judge.jsonl \
  --summary-output runs/qwen-plus-pair-judge-summary.json \
  --model qwen-plus \
  --limit 100 \
  --stratified \
  --include-unanswerable
```

## 14. 远端产物

A40审计目录：

```text
/srv/paismart-develop-experiment/repro-p0p1/full500
```

完整产物：

```text
runs/qwen-plus-raw-full500.jsonl
runs/qwen-plus-raw-full500-summary.json
runs/qwen-plus-rerank-full500.jsonl
runs/qwen-plus-rerank-full500-summary.json
runs/qwen-plus-pair-judge-stratified100.jsonl
runs/qwen-plus-pair-judge-stratified100-summary.json
```

代码：

```text
tools/qwen_plus_rag_pipeline.py
tools/qwen_plus_pair_judge.py
```

关键SHA-256保存在：

```text
results/2026-09-03-qwen-plus-enhancement-generation-summary.json
```

结果文件同时区分：

```text
runtime script hash：实际生成500题和100题盲评时的脚本
finalized script hash：增加Fast默认、路径保护和resume签名后的提交版本
```

这些收尾保护没有修改实验Prompt、显式Quality参数、评分算法或盲评逻辑。

## 15. 当前边界

```text
qwen-plus是滚动Alias，后续服务端版本可能变化
temperature=0仍不能保证跨时间完全一致
盲评Judge与Generator使用同一模型家族
增强阶段Token成本较高
当前只实现Fast/Quality显式切换，尚未实现自动路由
Benchmark source_types仍属于离线范围，不是生产身份权限
```

因此可以确认的是：

> 在本次固定500题、固定Java EvidenceBuilder上下文和相同16K生成预算下，Qwen Plus进行
> “原始证据ID选择并前置、其余证据回退”能稳定改善当前确定性代理指标，并得到同模型盲评支持。

不能把它外推成所有数据集、所有时间点或所有模型版本都必然提升。
