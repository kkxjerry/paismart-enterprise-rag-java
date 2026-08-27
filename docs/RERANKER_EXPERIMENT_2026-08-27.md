# EnterpriseRAG Larger Reranker Experiment

## Goal

Test whether the previous global rerank regression was specific to the small
`cross-encoder/ms-marco-MiniLM-L-6-v2` model, and repeat the experiment on both
the E5 and Qwen3-Embedding four-route candidates.

Only the reranker changed. Questions, gold documents, Top50 candidate rankings,
Elasticsearch chunks, passage construction, and metrics remained fixed.

## Runtime Safety

All three A40 GPUs were occupied by existing vLLM and embedding services. Free
memory was approximately 4.8, 4.4, and 10.2 GiB, which was not considered safe
for `Qwen3-Reranker-4B`. No GPU process was stopped or modified.

The experiment therefore used the already cached, substantially larger
`BAAI/bge-reranker-base` model on CPU:

```text
CUDA_VISIBLE_DEVICES=""
CPU cores: 72
Host memory: 251 GiB
Model cache: 1.1 GiB
candidate_k: 50 documents
passage: title + one selected chunk, max 1,800 characters
batch_size: 16
questions: 500 total, 470 evaluable
```

The E5 and Qwen3 candidate sets ran concurrently. GPU memory usage was unchanged
before and after the experiment.

## Results

| Pipeline | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR@10 | Avg rerank |
|---|---:|---:|---:|---:|---:|---:|
| E5 four-route, no rerank | **88.51%** | **95.53%** | **97.87%** | **98.30%** | **0.9174** | - |
| E5 + MiniLM-L6 | 82.13% | 91.49% | 94.47% | 97.02% | 0.8636 | 87.75 ms, GPU |
| E5 + BGE base | 83.40% | 93.19% | 95.53% | 96.81% | 0.8782 | 6,363.70 ms, CPU |
| Qwen3 four-route, no rerank | **89.15%** | **95.96%** | **98.09%** | **98.09%** | **0.9217** | - |
| Qwen3 + BGE base | 79.57% | 92.98% | 96.17% | 97.45% | 0.8551 | 6,437.75 ms, CPU |

### E5 paired comparison

Compared with no rerank:

```text
Hit@10: 97.87% -> 95.53% (-2.34 pp)
Hit@1:  88.51% -> 83.40% (-5.11 pp)
Recovered at Top10: 4
Regressed at Top10: 15
Net: -11 questions
Exact paired McNemar p: 0.0192
```

Compared with MiniLM-L6, BGE improved Hit@10 by 1.06 pp and produced a paired
net gain of five questions. That BGE-versus-MiniLM difference was not
statistically significant in this sample (`p=0.3833`). It is a better reranker
than MiniLM for these fixed passages, but still worse than preserving the
four-route ranking.

### Qwen3 paired comparison

```text
Hit@10: 98.09% -> 96.17% (-1.91 pp)
Hit@1:  89.15% -> 79.57% (-9.57 pp)
Recovered at Top10: 3
Regressed at Top10: 12
Net: -9 questions
Exact paired McNemar p: 0.0352
```

The regression remained concentrated in semantic questions:

```text
E5 semantic Hit@10:    93.60% -> 88.00% (-5.60 pp)
Qwen3 semantic Hit@10: 94.40% -> 89.60% (-4.80 pp)
Fireflies Hit@10:      88.00% -> 84.00% in both runs
```

## Interpretation

Increasing reranker size improved the result relative to MiniLM, but did not
fix the main interface problem. Four-route retrieval combines Dense, original
BM25, keyword BM25, and English BM25 evidence at document level. The global
reranker discards those ensemble signals and represents each document using
only one selected chunk.

The earlier paired audit found that 14 of 17 MiniLM regressions had less than
0.5 lexical recall of the gold answer facts in the single passage passed to the
reranker. A larger CrossEncoder cannot reliably recover evidence that is in a
different chunk. The new BGE results show the same shape: better than MiniLM,
but still demoting more correct documents than it recovers, especially on long
semantic and meeting/document sources.

## Decision

Reject global `BAAI/bge-reranker-base` with the current single-chunk document
representation. Keep the no-rerank Qwen3 four-route result as the accepted
pipeline.

This experiment does not prove that all rerankers are ineffective. A future
Qwen3 reranker experiment should wait for safe GPU capacity and should first
change the rerank input to parent-child or multi-chunk document evidence. That
would test model quality without repeating the known single-chunk bottleneck.

That representation follow-up has now been completed with the same BGE model;
see [`MULTICHUNK_RERANKER_EXPERIMENT_2026-08-27.md`](MULTICHUNK_RERANKER_EXPERIMENT_2026-08-27.md).

## Artifacts

- `results/2026-08-27-enterpriserag-e5-four-route-bge-reranker-base-cpu-500-summary.json`
- `results/2026-08-27-enterpriserag-qwen3-four-route-bge-reranker-base-cpu-500-summary.json`
- E5 details SHA-256:
  `d8176417977a74610c43a1e94addbaedcc3f3debfc69efe6f3856f07083b8815`
- Qwen3 details SHA-256:
  `1a35266d67ad37a2a2f84d4bfab5a3aa1defea505bba1dffbd23f3d2521fd990`

The two 500-line detail files remain in the experiment workspace and are not
committed because they total approximately 18 MiB.
