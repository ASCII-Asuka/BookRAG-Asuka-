# Closed-corpus KG²RAG baseline

This adapter evaluates the NAACL 2025 KG²RAG method on the repository's locked
Qasper and HotpotQA subsets. The upstream source is pinned for provenance under
`runs/third_party/KG2RAG`; it remains ignored because it is GPL-3.0. The adapter
implements the paper's generic `(head, relation, tail, source chunk)` graph so
Qasper paragraph IDs and HotpotQA title/sentence mappings remain auditable.

The baseline never uses answers or annotated evidence for OpenIE, expansion,
ranking, fallback, or parameter selection.

## Private model service

Before running, set the following variables in the current PowerShell session:

- `OPENAI_API_KEY`
- `KG2RAG_LLM_BASE_URL`
- `KG2RAG_EMBEDDING_BASE_URL`
- `KG2RAG_RERANKER_BASE_URL`
- optionally `KG2RAG_GENERATION_BASE_URL`

Values must come from the same private service configuration used by the
EviBridge experiments. Do not add them to `config/kg2rag_official.yaml` or a
tracked launcher. TLS certificate verification is always enabled.

## Source and environment

```powershell
powershell -ExecutionPolicy Bypass -File Scripts/baselines/kg2rag/setup_environment.ps1
```

The adapter itself runs in the project's `.venv`; `.venv-kg2rag` records an
isolated dependency snapshot and allows upstream inspection without changing
the main environment.

## Smoke test

```powershell
.venv/Scripts/python.exe -m Scripts.baselines.kg2rag.prepare `
  --dataset-config Scripts/cfg/Qasper.yaml `
  --output-root runs/kg2rag/qasper/prepared-smoke `
  --limit-scopes 2

.venv/Scripts/python.exe -m Scripts.baselines.kg2rag.run `
  --prepared-root runs/kg2rag/qasper/prepared-smoke `
  --config config/kg2rag_official.yaml `
  --output-root runs/kg2rag/qasper/retrieval-smoke `
  --stage all
```

Repeat with `Scripts/cfg/HotpotQA.yaml` and four scopes. Then finalize with:

```powershell
.venv/Scripts/python.exe -m Scripts.baselines.kg2rag.finalize `
  --retrieval-manifest runs/kg2rag/qasper/retrieval-smoke/manifest.json `
  --config config/kg2rag_official.yaml
```

## Full experiment

The launcher is deliberately foreground-only and rejects a second active copy.
Start it from an external hidden PowerShell process after both model endpoints
pass a minimal TLS-verified health check:

```powershell
powershell -ExecutionPolicy Bypass -File Scripts/baselines/kg2rag/run_full_experiments.ps1
```

Progress is written to `runs/kg2rag/full_experiment_status.json`. Per-stage
cache identities include the corpus hash, public config hash, upstream revision,
and all model names. Restarting the same command resumes validated scopes and
queries; a config change creates a different run directory.

## Results

Evaluator-compatible outputs are written under each document's
`eval_<dataset>_kg2rag_official` directory. Qasper is evaluated with Answer F1,
Evidence Precision/Recall/F1, and Recall@1/2/3/5/10/20. HotpotQA is evaluated
with Answer EM/F1, SP Precision/Recall/F1, and Joint F1. `token_cost.json`
contains online retrieval and generation only; `kg2rag_index_stats.json`
contains offline OpenIE and index cost.
