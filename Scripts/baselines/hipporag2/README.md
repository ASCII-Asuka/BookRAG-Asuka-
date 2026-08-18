# Official HippoRAG 2 adapter

This directory keeps the paper baseline isolated from the repository's existing
`hipporag` strategy. It uses the official HippoRAG 2 retrieval implementation
at the revision recorded in `official-source-lock.json`; it does not train a
dataset-specific model and does not route through `main.py`.

## 1. Create the isolated environment

From the repository root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File Scripts/baselines/hipporag2/setup_environment.ps1
```

The official source, environment records, graphs, caches, and outputs remain
under ignored paths. The script installs the official package with `--no-deps`
and installs the API-oriented dependency set without `vllm`. It enables
Python's UTF-8 mode because the pinned upstream `setup.py` otherwise reads its
README using the Windows legacy code page. `datasets` and `mteb`, used by the
unused local GritLM benchmark path, are also omitted from this API-only setup.

Set private endpoints only in the shell or a private launcher:

```powershell
$env:OPENAI_API_KEY = "..."
$env:HIPPORAG2_LLM_BASE_URL = "http://host:port/v1"
$env:HIPPORAG2_EMBEDDING_BASE_URL = "http://host:port/v1/embeddings"
$env:HIPPORAG2_GENERATION_BASE_URL = "http://host:port/v1"
```

## 2. Prepare fixed dataset scopes

Run preparation with the main project environment. Existing dataset configs
and manifests remain authoritative, including question order and hashes.

```powershell
python -m Scripts.baselines.hipporag2.prepare `
  --dataset-config Scripts/cfg/Qasper.yaml `
  --output-root runs/hipporag2/qasper/prepared

python -m Scripts.baselines.hipporag2.prepare `
  --dataset-config Scripts/cfg/HotpotQA.yaml `
  --output-root runs/hipporag2/hotpotqa/prepared
```

For a mapping-only smoke test, add `--limit-scopes 2`.

## 3. Run official indexing and retrieval

Use the isolated Python 3.10 interpreter:

```powershell
.venv-hipporag2/Scripts/python.exe -m Scripts.baselines.hipporag2.run_official `
  --prepared-root runs/hipporag2/qasper/prepared `
  --config config/hipporag2_official.yaml `
  --output-root runs/hipporag2/qasper/retrieval
```

Repeat with the HotpotQA prepared root. The runner checkpoints every completed
query. A scope is reused only when corpus hash, public configuration hash,
official revision, and both model names match. Unknown or duplicate source IDs
are fatal. Empty graph facts are retained as an explicitly labelled dense
fallback.

Official HippoRAG hashes chunks by text. The adapter therefore adds a stable,
non-semantic source marker only to the indexed copy of each passage so repeated
paragraph text cannot collapse two source IDs. Generation and evaluation still
use the unmodified paragraph or title/sentence text.

## 4. Finalize with the unified generator

Run this stage in the main project environment:

```powershell
python -m Scripts.baselines.hipporag2.finalize `
  --retrieval-manifest runs/hipporag2/qasper/retrieval/manifest.json `
  --config config/hipporag2_official.yaml
```

Final outputs are written to
`<working_dir>/<doc_uuid>/eval_<dataset>_hipporag2_official/`. Retrieval and
generation tokens are combined in `token_cost.json`; offline OpenIE/index cost
is kept separately in `hipporag2_index_stats.json`.

## 5. Evaluate

```powershell
python Eval/evaluation.py -d Scripts/cfg/Qasper.yaml --method hipporag2_official --max_workers 1
python Eval/evaluation.py -d Scripts/cfg/HotpotQA.yaml --method hipporag2_official --max_workers 1
```

The retrieval files retain the top 20 passages, so Qasper Recall@1/2/3/5/10/20
can be computed from the same complete run used for answer, evidence, token, and
time results.
