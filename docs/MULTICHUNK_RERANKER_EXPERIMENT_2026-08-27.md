# EnterpriseRAG Multi-Chunk and Parent-Child Reranker Experiment

## Goal

Keep the reranker and frozen Qwen3 four-route Top50 candidates unchanged, and
test whether replacing the single representative chunk reduces cases where a
gold document was in the original Top10 but was pushed out by reranking.

Fixed components:

```text
Dataset: EnterpriseRAG-Bench, 500 questions / 470 evaluable
Retriever: Qwen3-Embedding-4B four-route weighted RRF
Candidates: the same frozen Top50 document ranking
Reranker: BAAI/bge-reranker-base
Chunk text cap: 1,800 characters
CrossEncoder max length: 512 tokens
Batch size: 16
```

## Representations

Three reranker inputs were compared:

1. `single-chunk`: score the one chunk selected by first-stage retrieval.
2. `neighbor-max`: score the selected chunk and its immediate previous/next
   chunks separately; use the highest child score as the parent document score.
3. `parent-max`: score every child chunk of the candidate parent document
   separately; use the highest child score as the parent document score.

Chunks were scored separately instead of concatenated. This prevents a
multi-chunk string from being silently truncated back to 512 tokens.

## GPU Safety

At execution time GPU0 and GPU2 were free. The existing GPU1 process was left
untouched. Each BGE reranker process used approximately 2.2 GiB and returned to
idle after completion. No model service was stopped.

## Results

| Representation | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR@10 | All gold@10 | Avg rerank |
|---|---:|---:|---:|---:|---:|---:|---:|
| No rerank | **89.15%** | 95.96% | **98.09%** | 98.09% | **0.9217** | **92.34%** | - |
| Single chunk | 79.57% | 92.98% | 96.17% | 97.45% | 0.8551 | 87.87% | 302 ms |
| Neighbor max | 83.19% | 94.68% | 97.23% | 98.09% | 0.8794 | 89.79% | 784 ms |
| Parent max | **87.87%** | **95.96%** | **97.45%** | **98.51%** | **0.9114** | **91.06%** | 2,691 ms |

Parent-child recovered most of the first-rank and MRR damage. It also raised
Hit@20 above the no-rerank result, but Hit@10 remained 0.64 pp below the accepted
four-route ranking.

## Paired Hit-to-Miss Analysis

The counts below compare each reranked result directly with the original
no-rerank Qwen3 Top10:

| Representation | Miss -> Hit | Hit -> Miss | Net | Exact paired p-value |
|---|---:|---:|---:|---:|
| Single chunk | 3 | 12 | -9 | 0.0352 |
| Neighbor max | 4 | 8 | -4 | 0.3877 |
| Parent max | 4 | 7 | -3 | 0.5488 |

The practical reduction in Hit-to-Miss cases was:

```text
Single -> Neighbor: 12 -> 8, 33% fewer
Single -> Parent:   12 -> 7, 42% fewer
```

Compared directly with single chunk:

```text
Neighbor: 5 questions improved, 0 worsened, p=0.0625
Parent:   9 questions improved, 3 worsened, p=0.1460
```

Therefore the direction is strong and consistent with the representation
hypothesis, but the reduction is not statistically significant at the 0.05
level on this 470-question paired benchmark. The correct statement is
"materially reduced, not yet statistically proven."

## Slice Effects

| Representation | Basic Hit@10 | Semantic Hit@10 | Misc Hit@10 |
|---|---:|---:|---:|
| No rerank | 99.43% | 94.40% | 100.00% |
| Single chunk | 98.29% | 89.60% | 95.00% |
| Neighbor max | 98.29% | **93.60%** | 95.00% |
| Parent max | **100.00%** | 91.20% | **100.00%** |

Neighbor scoring almost eliminated the semantic regression while parent scoring
restored the basic and miscellaneous slices. Parent max still introduced seven
new semantic regressions, showing that max pooling across many child chunks can
promote a spurious high-scoring child.

## Interpretation

The experiment confirms that the single-chunk interface caused a substantial
part of the rerank regression:

- Neighbor evidence fixed five single-chunk misses without introducing a new
  single-chunk success-to-failure transition.
- Full parent evidence fixed nine single-chunk misses, although it introduced
  three different failures.
- Parent-child improved Hit@1 by 8.30 pp and MRR@10 by 0.0563 relative to the
  same BGE single-chunk reranker.

However, global parent max still does not beat the original weighted RRF at
Top10 and adds about 2.7 seconds per question. More evidence helps, but an
unbounded maximum over all children introduces ranking noise and cost.

## Decision

Keep the no-rerank Qwen3 four-route pipeline as the accepted default.

If reranking is revisited, use a routed and bounded parent-child design rather
than global max pooling:

```text
low-confidence document candidates only
-> preselect 2-3 child chunks per parent
-> CrossEncoder child scoring
-> top-two score aggregation or RRF, not unbounded max
```

This should preserve the observed Hit-to-Miss reduction without paying for all
387 child pairs per question or favoring one accidental high child score.

## Artifacts

- `results/2026-08-27-enterpriserag-qwen3-bge-single-chunk-gpu-500-summary.json`
- `results/2026-08-27-enterpriserag-qwen3-bge-neighbor-max-gpu-500-summary.json`
- `results/2026-08-27-enterpriserag-qwen3-bge-parent-max-gpu-500-summary.json`
- Evaluation script SHA-256:
  `b2a023e953f3d35a3dee6052ca30b9363c4ebbf2d3a258abd1fc4b4686e60651`
- Single details SHA-256:
  `d8e0d020629a1287bf674624d593b1c9d2889c10bf1feb9a46f719a3bade8ccb`
- Neighbor details SHA-256:
  `2b1a867893a36691fda5760f3261a1d73ae046de8e7a2b1e6a1ae95438d4d6e4`
- Parent details SHA-256:
  `cc6b64de7f364367e1ee0bfbb0d68b3c7ce3a8cff7d23f3b3f8b028230c31e4e`

The three 500-line detail files remain in the experiment workspace.
