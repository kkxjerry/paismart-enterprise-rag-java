# 验证记录

## 2026-08-18 本地构建

```text
mvn clean verify
Tests run: 21, Failures: 0, Errors: 0, Skipped: 0
Fat Jar: target/paismart-enterprise-rag.jar
CLI help: passed
Sample/result JSON parse: passed
Secret pattern scan: passed
Python data preparation syntax: passed
```

本机 Maven 输出一条用户级 `~/.m2/settings.xml` 根节点格式警告，但不影响项目编译、
测试或打包；该用户文件不属于本仓库。

## 2026-08-18 A40 端到端 smoke

验证环境：

```text
Elasticsearch 8.10.4: 127.0.0.1:19200
Qwen3-Embedding-4B vLLM: 127.0.0.1:18084
独立索引: paismart_enterpriserag_java_github_smoke_20260818_v1
业务索引: 未修改
```

执行了完整路径：

```text
create-index -> Java chunk -> Qwen3 2048-d embedding -> ES bulk
             -> checkpoint resume -> Dense + 3 BM25 routes -> weighted RRF -> metrics
```

最终索引 3 篇 sample 文档、3 个 Chunk。3 道问题中 2 道有 gold 文档，两道均在
Top1 命中；1 道 no-gold 不计入 Hit/MRR。`source_filter_violation_count=0`。

这次 smoke 的 100% 只证明新仓库的真实 HTTP、Embedding、ES 和评测链路可运行，
不用于证明检索质量。质量结论仍以固定 500 题的 470 道可评测问题为准。

## Smoke 中发现并修复的问题

1. JDK `HttpClient` 与当前 vLLM 部署协商时，服务端把 Embedding 请求体识别为空。
   修复：客户端和请求显式固定 HTTP/1.1。
2. Elasticsearch `_refresh` 不接受 `{}` body。
   修复：POST refresh 使用空 body。
3. Checkpoint 的小整数由 Jackson 从 LongNode 读回 IntNode，严格节点比较误判签名变化。
   修复：签名逐字段按规范化文本值比较，并补回归测试。

三个问题都在独立 smoke 索引发现。Importer 只在完整批次成功后前移 checkpoint，因此
Embedding 失败没有产生半批数据；refresh 失败后重启也正确识别已完成的 3 篇文档，
没有重复生成向量。

产物：

- `results/2026-08-18-a40-sample-import-summary.json`
- `results/2026-08-18-a40-sample-retrieval-summary.json`

## 2026-08-18 独立 Jar 全量 500 题复测

新仓库生成的 Jar 直接连接既有 Qwen3 全量只读索引，重新执行固定四路参数：

```text
questions_total=500
questions_evaluable=470
hit@1=89.15%
hit@10=98.09%
mrr@10=0.9217
all_expected_docs_hit@10=92.34%
avg_latency_ms=245.17
p95_latency_ms=362.44
source_filter_violation_count=0
details lines=500
```

Hit/MRR 与迁移前结果完全一致；延迟从原记录 255.44 ms 变为 245.17 ms，属于同一服务
不同运行时负载下的波动。逐题 details 保留在 A40，Git 只保存 summary：

- `results/2026-08-18-github-standalone-qwen3-four-route-500-summary.json`
