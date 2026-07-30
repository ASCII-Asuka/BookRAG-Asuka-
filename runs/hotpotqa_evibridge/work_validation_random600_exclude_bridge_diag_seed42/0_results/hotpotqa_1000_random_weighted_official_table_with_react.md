# HotpotQA validation 1000-question weighted comparison with reasoning baselines

Data: first200 + second200 + random600, weighted as first200 * 0.2 + second200 * 0.2 + random600 * 0.6.

Metrics are official HotpotQA metrics. Strict Supporting Fact EM and Joint EM are omitted because they are nearly all zero in the current paragraph-title mapping and are not used in the main table.

ReAct-RAG uses the paper-style `Thought -> Action -> Observation` loop with local-corpus `Search`, `Lookup`, and `Finish` actions. Its HotpotQA prompt uses the six demonstrations from the original ReAct HotpotQA setup.

| Method | Answer EM | Answer F1 | Supporting Fact F1 | Joint F1 |
|---|---:|---:|---:|---:|
| BM25 paragraph | 0.4610 | 0.5753 | 0.2861 | 0.1870 |
| Dense paragraph | 0.3880 | 0.4789 | 0.2338 | 0.1503 |
| Hybrid BM25+Dense RRF | 0.4650 | 0.5820 | 0.2915 | 0.1922 |
| RAPTOR | 0.4430 | 0.5573 | 0.2573 | 0.1737 |
| Full-document Long-context LLM | 0.5580 | 0.6966 | 0.7186 | 0.5385 |
| LongRAG-style | 0.5440 | 0.6789 | 0.5989 | 0.4385 |
| IRCoT | 0.4290 | 0.5394 | 0.4592 | 0.2959 |
| ReAct-RAG (local corpus) | 0.3450 | 0.4679 | 0.3857 | 0.2034 |
| BM25 + Reranker | 0.5540 | 0.6779 | 0.3392 | 0.2431 |
| EviBridge-RAG | 0.5130 | 0.6462 | 0.3386 | 0.2360 |
