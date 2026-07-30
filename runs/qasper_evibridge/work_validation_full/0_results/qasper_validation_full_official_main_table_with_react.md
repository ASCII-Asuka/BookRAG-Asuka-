# Qasper validation full official main comparison with reasoning baselines

- Dataset: Qasper validation full, 281 documents, 1005 questions.
- Metrics: official Qasper `Answer F1` and `Evidence F1` only.
- Evidence F1 uses the dynamic evidence submission policy and is evaluated by the Qasper official evaluator.
- IRCoT interleaves reasoning and BM25 paragraph retrieval.
- ReAct-RAG uses the paper-style `Thought -> Action -> Observation` loop with local-corpus `Search`, `Lookup`, and `Finish` actions. The Qasper prompt uses six fixed demonstrations sampled from the Qasper training split.

| Method | Answer F1 | Evidence F1 |
|---|---:|---:|
| Abstract-only | 0.1074 | 0.1214 |
| BM25 paragraph | 0.3500 | 0.2420 |
| Dense paragraph | 0.3529 | 0.2884 |
| Hybrid BM25+Dense RRF | 0.3580 | 0.2967 |
| BM25 + Reranker | 0.3650 | 0.3241 |
| RAPTOR | 0.3604 | 0.2885 |
| Full-document Long-context LLM | 0.3954 | 0.4905 |
| LongRAG-style | 0.3942 | 0.4442 |
| IRCoT | 0.3010 | 0.3275 |
| ReAct-RAG (local corpus) | 0.2157 | 0.1769 |
| EviBridge-RAG full | 0.4233 | 0.3138 |
