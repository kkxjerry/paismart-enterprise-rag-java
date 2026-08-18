# Result artifacts

Only aggregate JSON summaries are committed. Per-question details contain 500 rows per run and remain
in the experiment workspace.

| File marker | Meaning |
|---|---|
| `2026-08-13-...hybrid-rerank...` | Early Top20 generic Hybrid + global rerank baseline |
| `2026-08-14-...hybrid-rerank...k50...` | Expanded Top50 candidate rerank, Hit@10 86.81% |
| `...source-acl-weighted-hybrid-bm25-tuned...` | Best 384-d E5 Java retrieval |
| `...qwen3...dense...` | Qwen3-Embedding-4B Dense-only ablation |
| `...qwen3...four-route...` | Qwen3 four-route Hybrid result |
| `...text-embedding-v4...` | DashScope cloud embedding comparison |
| `2026-08-18-github-standalone...` | Fresh 500-question rerun with this standalone Jar |
| `2026-08-18-a40-sample...` | Three-document end-to-end smoke |

Read `../docs/REPRODUCIBILITY.md` before comparing percentages. In particular, `hit@10` is document
retrieval recall over 470 evaluable questions, not final answer accuracy.
