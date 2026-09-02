# Cross-Benchmark RAG and Reranker Evaluation

## Goal

Test whether the EnterpriseRAG rerank regression generalizes to other retrieval
benchmarks, using one common indexing and evaluation protocol.

The additional datasets cover passage search, Chinese retrieval, long-tail
retrieval, and multi-hop multi-gold retrieval:

| Dataset | Parent documents | Questions | Multi-gold questions |
|---|---:|---:|---:|
| MS MARCO Passage | 101,064 | 1,000 | 57 |
| MIRACL zh | 100,966 | 393 | 244 |
| LoTTE pooled search | 102,549 | 1,000 | 781 |
| BEIR HotpotQA | 101,976 | 1,000 | 1,000 |
| EnterpriseRAG | 10,722 | 500 | 93 |

## Unified Protocol

```text
chunk size: 1,200 characters
chunk overlap: 200 characters
embedding: Qwen/Qwen3-Embedding-4B
stored dimension: 2,048
Elasticsearch: dense vector + standard BM25 + keyword BM25 + English BM25
Dense ANN: num_candidates=2,500, chunk k=500
each route: Top50 documents
fusion: weighted RRF(k=10), weights 0.75 / 0.50 / 1.25 / 1.50
evaluation: Hit@1/5/10/20, MRR@10, all-gold@10
reranker: BAAI/bge-reranker-base
rerank candidates: frozen Top50
```

The four public passage benchmarks have no ACL metadata, so ACL filtering was
disabled rather than fabricated. EnterpriseRAG retains its separate ACL-safe
result and is compared by within-dataset deltas, not by claiming that absolute
scores have identical difficulty.

### Embedding compatibility note

The actual experiment log used vLLM 0.17.0. That service returns Qwen3's native
2,560 dimensions and rejects
the `dimensions=2048` parameter. To remain compatible with the existing
EnterpriseRAG 2,048-dimensional index, a local adapter kept the first 2,048
dimensions and L2-normalized them. This is a controlled compatibility setting,
not a claim that 2,048 dimensions are optimal for these four datasets.

## Index Build

| Dataset | Documents | Indexed chunks | ES size | Import time |
|---|---:|---:|---:|---:|
| MS MARCO | 101,064 | 101,066 | 3.4 GiB | 34.4 min |
| MIRACL zh | 100,966 | 101,046 | 3.3 GiB | 34.4 min |
| LoTTE | 102,549 | 120,620 | 4.0 GiB | 49.9 min |
| HotpotQA | 101,976 | 102,340 | 3.3 GiB | 46.4 min |

Total: 406,555 parent rows and 425,072 indexed chunks. MS MARCO, MIRACL, and
HotpotQA remained almost entirely one child per parent. LoTTE produced 18,071
additional chunks and is the main non-enterprise dataset where neighbor and
parent-child representations can materially differ.

## Hit@10 Results

| Dataset | No rerank | BGE single | BGE neighbor | BGE parent |
|---|---:|---:|---:|---:|
| EnterpriseRAG | **98.09%** | 96.17% | 97.23% | 97.45% |
| MS MARCO | 83.80% | **99.10%** | 99.10% | 99.10% |
| MIRACL zh | 65.14% | **95.42%** | 95.42% | 95.42% |
| LoTTE | 77.50% | 90.20% | **90.50%** | **90.50%** |
| HotpotQA | 98.50% | **99.70%** | 99.70% | 99.70% |

Five-dataset macro Hit@10:

```text
No rerank: 84.61%
Single:    96.12%
Neighbor:  96.39%
Parent:    96.43%
```

The macro number is descriptive only; the datasets have different question
counts, gold structures, languages, and difficulty.

## Ranking Quality

| Dataset | Metric | No rerank | Single | Neighbor | Parent |
|---|---|---:|---:|---:|---:|
| MS MARCO | Hit@1 | 54.30% | 87.30% | 87.30% | 87.30% |
| MS MARCO | MRR@10 | 0.6237 | 0.9194 | 0.9194 | 0.9194 |
| MIRACL zh | Hit@1 | 43.77% | 74.55% | 74.55% | 74.55% |
| MIRACL zh | MRR@10 | 0.5019 | 0.8255 | 0.8255 | 0.8255 |
| LoTTE | Hit@1 | 49.10% | 62.90% | 63.20% | 63.30% |
| LoTTE | MRR@10 | 0.5830 | 0.7241 | 0.7276 | 0.7286 |
| HotpotQA | Hit@1 | 90.00% | 96.60% | 96.60% | 96.60% |
| HotpotQA | MRR@10 | 0.9302 | 0.9794 | 0.9794 | 0.9794 |

## Multi-Gold Coverage

| Dataset | All gold@10, no rerank | Single | Neighbor | Parent |
|---|---:|---:|---:|---:|
| MS MARCO | 82.30% | 98.60% | 98.60% | 98.60% |
| MIRACL zh | 38.42% | 72.26% | 72.26% | 72.26% |
| LoTTE | 39.90% | 49.10% | 50.30% | **50.40%** |
| HotpotQA | 69.40% | **85.50%** | 85.50% | 85.50% |

HotpotQA shows that reranking can improve not only any-gold recall but also the
probability that both supporting documents survive in Top10.

## Paired Top10 Changes

Each row compares the reranked Top10 directly with the frozen no-rerank Top10.

| Dataset | Representation | Miss -> Hit | Hit -> Miss | Net |
|---|---|---:|---:|---:|
| EnterpriseRAG | Single | 3 | 12 | -9 |
| EnterpriseRAG | Neighbor | 4 | 8 | -4 |
| EnterpriseRAG | Parent | 4 | 7 | -3 |
| MS MARCO | Single / Neighbor / Parent | 154 | 1 | +153 |
| MIRACL zh | Single / Neighbor / Parent | 122 | 3 | +119 |
| LoTTE | Single | 145 | 18 | +127 |
| LoTTE | Neighbor / Parent | 146 | 16 | +130 |
| HotpotQA | Single / Neighbor / Parent | 12 | 0 | +12 |

The improvements on all four additional benchmarks are statistically decisive
under an exact paired McNemar test. EnterpriseRAG is the only dataset where
global reranking has a negative paired result.

## Latency

Average GPU rerank latency per question:

| Dataset | Single | Neighbor | Parent |
|---|---:|---:|---:|
| MS MARCO | 218 ms | 228 ms | 237 ms |
| MIRACL zh | 426 ms | 444 ms | 456 ms |
| LoTTE | 389 ms | 493 ms | 530 ms |
| HotpotQA | 335 ms | 351 ms | 355 ms |
| EnterpriseRAG | 302 ms | 784 ms | 2,691 ms |

EnterpriseRAG is much more expensive under parent-child scoring because each
candidate parent contains many child chunks. The other datasets are mostly
already passage-level.

## Interpretation

The cross-benchmark result changes the correct project conclusion:

> The BGE reranker is not generally ineffective. It improves four public
> passage-retrieval benchmarks and only regresses on EnterpriseRAG.

The EnterpriseRAG failure is explained by the combination of:

1. A very strong, source-aware, enterprise-tuned first-stage ranking already at
   98.09% Hit@10.
2. Long parent documents represented by one selected chunk in the original
   reranker interface.
3. Exact identifiers, dates, policy revisions, meeting transcripts, and
   conflicting versions that are poorly summarized by one generic passage
   score.
4. Global reranking overriding four independent retrieval signals that were
   tuned for the enterprise corpus.

MS MARCO and MIRACL show the opposite regime: the frozen Enterprise-tuned RRF
weights and the 2,048-dimensional compatibility embedding leave many gold
documents below rank 10 but inside Top50. BGE is highly effective at promoting
those already-recalled candidates. Therefore the large gains should not be
misrepresented as proof that the shared first-stage parameters are optimal.

## Multi-Chunk Conclusion

Neighbor and parent-child inputs matter only when the parent has multiple
children:

- MS MARCO, MIRACL, and HotpotQA are almost entirely one passage per parent, so
  all three representations produce the same ranking.
- LoTTE gains 0.30 pp Hit@10 and 1.30 pp all-gold@10 from parent evidence.
- EnterpriseRAG reduces Hit-to-Miss by 33-42%, but global parent max still does
  not beat the no-rerank ranking.

The next production-shaped reranker should therefore be routed and bounded:

```text
low-confidence candidates only
-> preselect 2-3 children per parent
-> BGE child scoring
-> bounded top-two aggregation or RRF
```

## Decision

- Keep no-rerank Qwen3 four-route as the EnterpriseRAG default.
- Accept BGE single-passage reranking as a strong baseline for MS MARCO,
  MIRACL, and HotpotQA.
- Use bounded parent-child reranking for LoTTE and future long-document tests.
- Do not claim one global rerank policy is optimal across all RAG datasets.

## Artifacts

The 16 aggregate summaries are committed under `results/2026-08-29-*.json`.
Per-question detail files remain in the A40 experiment workspace.
