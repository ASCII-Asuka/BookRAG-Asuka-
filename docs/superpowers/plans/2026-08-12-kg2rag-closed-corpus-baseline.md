# KG²RAG Closed-Corpus Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable, gold-free KG²RAG baseline that evaluates Qasper validation full and HotpotQA fixed-1000 with the repository's unified models, evidence mappings, metrics, and online cost protocol.

**Architecture:** Keep the pinned GPL-3.0 upstream source under ignored `runs/third_party/KG2RAG`, while the repository contains a clean-room adapter implementing the paper's generic `(head, relation, tail, source_chunk)` equations for Qasper and HotpotQA. The adapter is split into `prepare`, `index`, `retrieve`, and `finalize` stages so offline OpenIE cost, online retrieval cost, final generation cost, cache identity, and recovery can be verified independently.

**Tech Stack:** Python 3.10+, PyYAML, requests/httpx-compatible OpenAI endpoints, NumPy, NetworkX, existing `DocumentTree`, Qwen3-VL-8B, Qwen3-Embedding-0.6B, BAAI/bge-reranker-v2-m3, unittest/pytest.

---

Project policy requires direct work on the current dev branch and forbids automatic commits. Therefore every task ends with a diff/test checkpoint rather than a `git commit` step.

## File map

- Create `Scripts/baselines/kg2rag/__init__.py`: package marker and public method suffix.
- Create `Scripts/baselines/kg2rag/common.py`: JSON/hash helpers, configuration loading, entity normalization, budget enforcement, citation validation, and cost accounting.
- Create `Scripts/baselines/kg2rag/prepare.py`: convert locked Qasper/HotpotQA inputs into closed-corpus scopes.
- Create `Scripts/baselines/kg2rag/model_clients.py`: TLS-verified OpenAI-compatible chat, embedding, and reranker clients with retries and usage tracking.
- Create `Scripts/baselines/kg2rag/index.py`: gold-free OpenIE extraction, triple validation, graph persistence, and offline statistics.
- Create `Scripts/baselines/kg2rag/retrieval.py`: semantic seeds, one-hop graph expansion, maximum-spanning-tree filtering, DFS organization, component reranking, and top-20 candidate export.
- Create `Scripts/baselines/kg2rag/run.py`: scope/query orchestration, cache identity, checkpointing, and resume logic.
- Create `Scripts/baselines/kg2rag/finalize.py`: unified generation and evaluator-compatible output writing.
- Create `Scripts/baselines/kg2rag/setup_environment.ps1`: pin and verify official source without committing it.
- Create `Scripts/baselines/kg2rag/run_full_experiments.ps1`: single-process staged launcher with status/log files.
- Create `Scripts/baselines/kg2rag/README.md`: exact smoke, full-run, recovery, evaluation, and cost commands.
- Create `Scripts/baselines/kg2rag/official-source-lock.json`: repository, full revision, license, and paper link.
- Create `config/kg2rag_official.yaml`: public model names and method parameters only.
- Create `tests/test_kg2rag_official_adapter.py`: deterministic unit and integration tests.
- Modify `.gitignore`: ignore `.venv-kg2rag`, official source, graphs, caches, logs, and outputs while keeping source lock/config/tests tracked.

### Task 1: Pin upstream source and public configuration

**Files:**
- Create: `Scripts/baselines/kg2rag/__init__.py`
- Create: `Scripts/baselines/kg2rag/official-source-lock.json`
- Create: `Scripts/baselines/kg2rag/setup_environment.ps1`
- Create: `config/kg2rag_official.yaml`
- Modify: `.gitignore`
- Test: `tests/test_kg2rag_official_adapter.py`

- [ ] **Step 1: Write the failing source-lock/config test**

```python
def test_source_lock_and_public_config_are_pinned():
    import json
    from pathlib import Path
    import yaml

    lock = json.loads(Path("Scripts/baselines/kg2rag/official-source-lock.json").read_text("utf-8"))
    cfg = yaml.safe_load(Path("config/kg2rag_official.yaml").read_text("utf-8"))
    assert lock["repository"] == "https://github.com/nju-websoft/KG2RAG.git"
    assert lock["revision"] == "7d626c77b7af30b55aa3f960cde755b9549a0616"
    assert lock["license"] == "GPL-3.0"
    assert cfg["method_suffix"] == "kg2rag_official"
    assert cfg["seed_topk"] == 10
    assert cfg["expansion_hops"] == 1
    assert cfg["retrieval_topk"] == 20
    assert cfg["generation_max_blocks"] == 10
    assert cfg["generation_max_tokens"] == 4000
    serialized = Path("config/kg2rag_official.yaml").read_text("utf-8").lower()
    assert "api_key" not in serialized
    assert "siliconflow" not in serialized
```

- [ ] **Step 2: Run the test and verify that the files are missing**

Run: `python -m pytest tests/test_kg2rag_official_adapter.py::test_source_lock_and_public_config_are_pinned -v`

Expected: FAIL because the KG²RAG source lock and public config do not exist.

- [ ] **Step 3: Add the pinned lock and public configuration**

`official-source-lock.json` must contain exactly the public provenance fields:

```json
{
  "repository": "https://github.com/nju-websoft/KG2RAG.git",
  "revision": "7d626c77b7af30b55aa3f960cde755b9549a0616",
  "license": "GPL-3.0",
  "paper": "https://aclanthology.org/2025.naacl-long.449/",
  "upstream_entrypoint": "code/kg_rag_distractor.py"
}
```

`config/kg2rag_official.yaml` must use these values:

```yaml
method_suffix: kg2rag_official
source_revision: 7d626c77b7af30b55aa3f960cde755b9549a0616
seed_topk: 10
retrieval_topk: 20
expansion_hops: 1
generation_max_blocks: 10
generation_max_tokens: 4000
supporting_evidence_topk: 4
openie:
  model_name: Qwen/Qwen3-VL-8B-Instruct
  temperature: 0.0
  max_output_tokens: 1024
  max_attempts: 3
embedding:
  model_name: Qwen/Qwen3-Embedding-0.6B
  batch_size: 16
reranker:
  model_name: BAAI/bge-reranker-v2-m3
  batch_size: 16
generation:
  model_name: Qwen/Qwen3-VL-8B-Instruct
  temperature: 0.1
  model_context_tokens: 32768
  max_output_tokens: 512
```

The setup script must clone into `runs/third_party/KG2RAG`, checkout the full revision, verify `git rev-parse HEAD`, create `.venv-kg2rag`, and install only adapter dependencies. Every `Start-Process` call must use `-WindowStyle Hidden`.

- [ ] **Step 4: Add narrow ignore rules**

Append only these patterns if absent:

```gitignore
.venv-kg2rag/
runs/third_party/KG2RAG/
runs/kg2rag/
```

- [ ] **Step 5: Run the source-lock/config test**

Run: `python -m pytest tests/test_kg2rag_official_adapter.py::test_source_lock_and_public_config_are_pinned -v`

Expected: PASS.

- [ ] **Step 6: Review without committing**

Run: `git diff -- .gitignore Scripts/baselines/kg2rag config/kg2rag_official.yaml tests/test_kg2rag_official_adapter.py`

Expected: only KG²RAG-specific additions; no API URL, key, existing baseline change, or run artifact.

### Task 2: Implement deterministic helpers, provenance, citations, and cost accounting

**Files:**
- Create: `Scripts/baselines/kg2rag/common.py`
- Test: `tests/test_kg2rag_official_adapter.py`

- [ ] **Step 1: Write failing helper tests**

```python
def test_entity_normalization_is_deterministic_and_non_generative():
    from Scripts.baselines.kg2rag.common import normalize_entity
    assert normalize_entity("  Café—Model. ") == "café model"
    assert normalize_entity("Café Model") == "café model"

def test_cache_identity_changes_with_corpus_config_revision_and_models():
    from Scripts.baselines.kg2rag.common import build_cache_identity
    base = build_cache_identity("c", "p", "r", "openie", "embed", "rerank")
    assert base != build_cache_identity("c2", "p", "r", "openie", "embed", "rerank")
    assert base != build_cache_identity("c", "p2", "r", "openie", "embed", "rerank")

def test_budget_never_splits_or_skips_over_budget_chunk():
    from Scripts.baselines.kg2rag.common import apply_generation_budget
    ranked = [{"source_id": "a", "content": "1234"}, {"source_id": "b", "content": "12"}]
    assert [x["source_id"] for x in apply_generation_budget(ranked, 10, 3, len)] == []

def test_online_cost_excludes_offline_openie():
    from Scripts.baselines.kg2rag.common import combine_online_cost
    cost = combine_online_cost(
        retrieval={"prompt_tokens": 5, "completion_tokens": 1, "time": 0.4},
        generation={"prompt_tokens": 7, "completion_tokens": 2, "time": 0.6},
    )
    assert cost["rag_cost"]["total_tokens"] == 15
    assert cost["time"] == 1.0
```

- [ ] **Step 2: Run tests and verify missing imports**

Run: `python -m pytest tests/test_kg2rag_official_adapter.py -k "normalization or cache_identity or budget or online_cost" -v`

Expected: FAIL because `common.py` does not exist.

- [ ] **Step 3: Implement the exact helper interfaces**

`common.py` must expose `canonical_json`, `sha256_json`, `sha256_file`,
`load_json`, `write_json`, `safe_filename`, `normalize_entity`,
`scope_corpus_sha256`, `build_cache_identity`, `apply_generation_budget`,
`validate_qasper_citations`, `validate_hotpot_citations`, and
`combine_online_cost`. Use the same argument names and return shapes exercised
by the tests in this task. The core deterministic implementation is:

```python
def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()

def normalize_entity(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = re.sub(r"[\u2010-\u2015]+", " ", text)
    text = text.strip(string.whitespace + string.punctuation)
    return re.sub(r"\s+", " ", text)

def build_cache_identity(corpus_sha256: str, config_sha256: str,
                         source_revision: str, openie_model: str,
                         embedding_model: str, reranker_model: str) -> str:
    return sha256_json({
        "corpus_sha256": corpus_sha256,
        "config_sha256": config_sha256,
        "source_revision": source_revision,
        "openie_model": openie_model,
        "embedding_model": embedding_model,
        "reranker_model": reranker_model,
    })

def apply_generation_budget(ranked_results, max_blocks, max_tokens, token_counter):
    selected, used = [], 0
    for item in ranked_results:
        if len(selected) >= max_blocks:
            break
        cost = max(1, int(token_counter(str(item.get("content") or ""))))
        if used + cost > max_tokens:
            break
        selected.append(dict(item))
        used += cost
    return selected
```

`normalize_entity` must use Unicode NFKC, lowercase, replace Unicode dashes with spaces, remove leading/trailing punctuation, collapse whitespace, and never call a model. Citation validators must accept only IDs present in ranked results and fall back only to the top-ranked legal candidates.

- [ ] **Step 4: Run helper tests**

Run: `python -m pytest tests/test_kg2rag_official_adapter.py -k "normalization or cache_identity or budget or online_cost or citation" -v`

Expected: PASS.

- [ ] **Step 5: Review without committing**

Run: `git diff -- Scripts/baselines/kg2rag/common.py tests/test_kg2rag_official_adapter.py`

Expected: deterministic helpers only; no filesystem path, endpoint, key, gold field access, or model call in citation fallback.

### Task 3: Prepare locked Qasper and HotpotQA scopes

**Files:**
- Create: `Scripts/baselines/kg2rag/prepare.py`
- Test: `tests/test_kg2rag_official_adapter.py`

- [ ] **Step 1: Write failing scope tests**

```python
def test_qasper_groups_questions_by_paper_and_preserves_paragraph_ids(fake_qasper_tree):
    from Scripts.baselines.kg2rag.prepare import qasper_chunks_from_tree
    chunks = qasper_chunks_from_tree(fake_qasper_tree, "paper-1")
    assert chunks[0]["metadata"]["paragraph_id"] == 7
    assert chunks[0]["metadata"]["qasper_evidence_text"] == "Original paragraph."
    assert chunks[0]["content"].startswith("[Section: Methods]")

def test_hotpot_scope_contains_only_current_question_titles(fake_hotpot_tree):
    from Scripts.baselines.kg2rag.prepare import hotpot_chunks_from_tree
    chunks = hotpot_chunks_from_tree(fake_hotpot_tree, "question-1")
    assert {c["metadata"]["hotpot_title"] for c in chunks} == {"A", "B"}
    assert chunks[0]["metadata"]["hotpot_sentences"][0]["sent_id"] == 0

def test_prepare_rejects_duplicate_question_or_source_ids():
    from Scripts.baselines.kg2rag.prepare import _group_rows
    rows = [{"doc_uuid": "p", "question_id": "q"}, {"doc_uuid": "p", "question_id": "q"}]
    with pytest.raises(ValueError, match="duplicate"):
        _group_rows(rows)
```

- [ ] **Step 2: Run the preparation tests**

Run: `python -m pytest tests/test_kg2rag_official_adapter.py -k "qasper_groups or hotpot_scope or prepare_rejects" -v`

Expected: FAIL because `prepare.py` is missing.

- [ ] **Step 3: Implement preparation using existing dataset configs and trees**

The CLI must be:

```powershell
python -m Scripts.baselines.kg2rag.prepare `
  --dataset-config Scripts/cfg/Qasper.yaml `
  --output-root runs/kg2rag/qasper/prepared `
  --limit-scopes 2
```

Each scope JSON must contain:

```json
{
  "dataset": "qasper",
  "scope_id": "paper-id",
  "corpus_sha256": "sha256",
  "chunks": [{"source_id": "paragraph-id", "content": "retrieval text", "metadata": {}}],
  "queries": [{"question_id": "id", "question": "text", "dataset_row": {}}]
}
```

`prepare_scopes()` must load `DocumentTree` from `<working_dir>/<doc_uuid>/tree.pkl`, group Qasper by paper, enforce one HotpotQA row per scope, retain the authoritative dataset order, write a manifest containing dataset SHA and scope hashes, and never copy `answer`, `supporting_facts`, `evidence_*`, or `gold_*` into the index input passed to OpenIE.

- [ ] **Step 4: Run preparation tests**

Run: `python -m pytest tests/test_kg2rag_official_adapter.py -k "qasper or hotpot or prepare" -v`

Expected: PASS.

- [ ] **Step 5: Run mapping-only smoke preparation**

Run:

```powershell
python -m Scripts.baselines.kg2rag.prepare --dataset-config Scripts/cfg/Qasper.yaml --output-root runs/kg2rag/qasper/prepared-smoke --limit-scopes 2
python -m Scripts.baselines.kg2rag.prepare --dataset-config Scripts/cfg/HotpotQA.yaml --output-root runs/kg2rag/hotpotqa/prepared-smoke --limit-scopes 4
```

Expected: Qasper manifest has 2 scopes; HotpotQA manifest has 4 scopes; all source IDs are unique within scope.

### Task 4: Add TLS-verified model clients and gold-free OpenIE indexing

**Files:**
- Create: `Scripts/baselines/kg2rag/model_clients.py`
- Create: `Scripts/baselines/kg2rag/index.py`
- Test: `tests/test_kg2rag_official_adapter.py`

- [ ] **Step 1: Write failing client and OpenIE tests**

```python
def test_clients_keep_tls_verification_enabled(mock_post):
    from Scripts.baselines.kg2rag.model_clients import OpenAICompatibleClient
    client = OpenAICompatibleClient("https://api.example/v1", "secret", timeout=30)
    client.embed("embed-model", ["text"])
    assert mock_post.call_args.kwargs.get("verify", True) is True
    assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer secret"

def test_openie_accepts_only_three_nonempty_string_fields():
    from Scripts.baselines.kg2rag.index import parse_triples
    response = '{"triples":[["A","rel","B"],["A","","C"],["A","rel",true]]}'
    assert parse_triples(response) == [["A", "rel", "B"]]

def test_openie_payload_contains_no_question_or_gold_fields(fake_scope):
    from Scripts.baselines.kg2rag.index import openie_input
    payload = openie_input(fake_scope["chunks"][0])
    lowered = str(payload).lower()
    assert "question" not in lowered
    assert "answer" not in lowered
    assert "supporting" not in lowered
```

- [ ] **Step 2: Run the client/OpenIE tests**

Run: `python -m pytest tests/test_kg2rag_official_adapter.py -k "clients_keep or openie" -v`

Expected: FAIL because the modules do not exist.

- [ ] **Step 3: Implement retrying clients**

`OpenAICompatibleClient` must read private values only from `OPENAI_API_KEY`, `KG2RAG_LLM_BASE_URL`, `KG2RAG_EMBEDDING_BASE_URL`, and `KG2RAG_RERANKER_BASE_URL`. Requests must use certificate verification, connect/read timeouts, bounded exponential backoff for 408/429/5xx/TLS connection resets, and must not log headers or environment values.

The public interfaces are `chat_json(model, messages, temperature,
max_tokens) -> (payload, usage)`, `embed(model, texts) -> (vectors, usage)`,
and `rerank(model, query, documents) -> (scores, usage)`. The implementation
must validate response cardinality: one embedding per input text and one score
per reranker document; mismatches raise `ValueError` before a result is saved.

- [ ] **Step 4: Implement OpenIE indexing and provenance**

The OpenIE prompt must demand strict JSON:

```text
Extract informative factual triples stated directly in the supplied text.
Return JSON only: {"triples": [["head", "relation", "tail"], ["head2", "relation2", "tail2"]]}.
Do not infer facts not stated in the text and do not use outside knowledge.
```

`index_scope(scope, config, client, output_dir)` must extract each chunk independently, retry malformed responses twice, normalize matching keys while retaining original triple strings, attach `source_id` to every triple, construct entity-to-triple and chunk-to-triple indexes, and write:

```json
{
  "scope_id": "id",
  "cache_identity": "sha",
  "triples": [{"head": "A", "relation": "r", "tail": "B", "head_key": "a", "tail_key": "b", "source_id": "p1"}],
  "chunk_embeddings": {"source_ids": ["p1"], "vectors_file": "chunk_embeddings.npy"},
  "index_stats": {"chunks": 1, "entities": 2, "triples": 1, "offline_tokens": 0, "offline_seconds": 0.0}
}
```

- [ ] **Step 5: Run client/OpenIE tests**

Run: `python -m pytest tests/test_kg2rag_official_adapter.py -k "client or openie or triple or index" -v`

Expected: PASS, with malformed triple rows rejected and no gold fields in prompts.

### Task 5: Implement paper-faithful semantic seeds, one-hop expansion, MST filtering, and DFS organization

**Files:**
- Create: `Scripts/baselines/kg2rag/retrieval.py`
- Test: `tests/test_kg2rag_official_adapter.py`

- [ ] **Step 1: Write failing algorithm tests**

```python
def test_one_hop_expansion_finds_chunk_connected_by_entity():
    from Scripts.baselines.kg2rag.retrieval import expand_one_hop
    triples = [
        {"head_key": "a", "tail_key": "b", "source_id": "seed"},
        {"head_key": "b", "tail_key": "c", "source_id": "bridge"},
        {"head_key": "c", "tail_key": "d", "source_id": "two-hop"},
    ]
    result = expand_one_hop(["seed"], triples)
    assert result["source_ids"] == {"seed", "bridge"}

def test_mst_keeps_maximum_weight_edges_and_dfs_is_deterministic():
    from Scripts.baselines.kg2rag.retrieval import organize_components
    triples = [
        {"head_key": "a", "tail_key": "b", "relation": "ab", "source_id": "p1"},
        {"head_key": "b", "tail_key": "c", "relation": "bc", "source_id": "p2"},
        {"head_key": "a", "tail_key": "c", "relation": "ac", "source_id": "p3"},
    ]
    components = organize_components(triples, {"p1": 0.9, "p2": 0.8, "p3": 0.1})
    assert {e["source_id"] for e in components[0]["mst_edges"]} == {"p1", "p2"}
    assert components[0]["ordered_source_ids"] == ["p1", "p2"]

def test_retrieval_returns_unique_monotonic_top20(fake_index, fake_clients):
    from Scripts.baselines.kg2rag.retrieval import retrieve
    result = retrieve("question", fake_index, fake_clients, seed_topk=10, retrieval_topk=20, expansion_hops=1)
    ids = [x["source_id"] for x in result["ranked_results"]]
    assert len(ids) == len(set(ids))
    assert len(ids) <= 20
    assert [x["rank"] for x in result["ranked_results"]] == list(range(1, len(ids) + 1))
```

- [ ] **Step 2: Run algorithm tests**

Run: `python -m pytest tests/test_kg2rag_official_adapter.py -k "one_hop or mst or retrieval_returns" -v`

Expected: FAIL because `retrieval.py` is missing.

- [ ] **Step 3: Implement cosine seed retrieval and one-hop expansion**

The seed ranking must use normalized query/chunk embeddings and stable source-order tie-breaking. `expand_one_hop` must:

1. collect entities in triples whose `source_id` is a seed;
2. include triples touching those entities;
3. include their source chunks;
4. stop after exactly one hop;
5. return seed IDs, expanded IDs, entity keys, and auditable expansion edges.

- [ ] **Step 4: Implement graph organization**

Use `networkx.MultiGraph`. Every triple edge stores `relation`, `source_id`, and `weight=chunk_similarity[source_id]`. For each connected component, call `nx.maximum_spanning_tree`; select the highest-weight edge as the root anchor; perform deterministic DFS with neighbors sorted by descending edge weight then normalized entity key; emit unique source IDs in first-encounter order.

Rerank each component using a comma-separated triple representation. Sort by reranker score, component maximum semantic score, then stable component ID. Append unused seed/expanded chunks by semantic score so the output can reach 20 candidates without inventing graph evidence. Generation later uses only the budgeted first 10.

- [ ] **Step 5: Run algorithm tests and upstream-toy compatibility check**

Run: `python -m pytest tests/test_kg2rag_official_adapter.py -k "one_hop or mst or dfs or retrieval" -v`

Expected: PASS. A toy Hotpot-style graph must select the same source set as upstream `KGRetrievePostProcessor` plus `GraphFilterPostProcessor`; ordering differences must be explained only by deterministic DFS replacing upstream set iteration.

### Task 6: Add resumable index/retrieval orchestration

**Files:**
- Create: `Scripts/baselines/kg2rag/run.py`
- Test: `tests/test_kg2rag_official_adapter.py`

- [ ] **Step 1: Write failing resume and failure tests**

```python
def test_matching_cache_is_reused_but_config_change_creates_new_run(tmp_path, fake_scope, fake_clients):
    from Scripts.baselines.kg2rag.run import run_scopes
    prepared, config, output = write_fake_run_inputs(tmp_path, fake_scope)
    first = run_scopes(str(prepared), str(config), str(output), clients=fake_clients)
    second = run_scopes(str(prepared), str(config), str(output), clients=fake_clients)
    assert second["scopes"][0]["reused"] is True
    changed = run_scopes(
        str(prepared), str(config), str(output), clients=fake_clients,
        config_override={"seed_topk": 5},
    )
    assert changed["run_id"] != first["run_id"]

def test_scope_failure_is_recorded_and_prevents_complete_manifest(tmp_path, fake_scope, failing_openie):
    from Scripts.baselines.kg2rag.run import run_scopes
    prepared, config, output = write_fake_run_inputs(tmp_path, fake_scope)
    with pytest.raises(RuntimeError, match="scope"):
        run_scopes(str(prepared), str(config), str(output), clients=failing_openie)
    status = json.loads((output / "status.json").read_text("utf-8"))
    assert status["status"] == "failed"
```

- [ ] **Step 2: Run orchestration tests**

Run: `python -m pytest tests/test_kg2rag_official_adapter.py -k "cache_is_reused or scope_failure" -v`

Expected: FAIL because `run.py` is missing.

- [ ] **Step 3: Implement stage-aware recovery**

The CLI must accept `--prepared-root`, `--config`, `--output-root`, and `--stage {index,retrieve,all}`. Before reusing a scope, verify corpus SHA, config SHA, source revision, OpenIE model, embedding model, reranker model, expected chunk IDs, and serialized graph files. Completed queries are checkpointed independently so a restart does not repeat OpenIE, embeddings, or completed retrieval.

The runner must write `status.json` atomically with `status`, `stage`, `completed_scopes`, `total_scopes`, `completed_queries`, `total_queries`, `last_progress_at`, and sanitized `last_error`. It must never include environment values or Authorization headers.

- [ ] **Step 4: Run orchestration tests**

Run: `python -m pytest tests/test_kg2rag_official_adapter.py -k "cache or resume or failure or status" -v`

Expected: PASS.

### Task 7: Finalize unified answers, evidence, outputs, and online costs

**Files:**
- Create: `Scripts/baselines/kg2rag/finalize.py`
- Test: `tests/test_kg2rag_official_adapter.py`

- [ ] **Step 1: Write failing finalize tests**

```python
def test_finalize_writes_qasper_outputs_and_legal_fallback(tmp_path, fake_qasper_run):
    from Scripts.baselines.kg2rag.finalize import finalize_runs
    manifest, config, output_dir = fake_qasper_run
    summary = finalize_runs(
        str(manifest), str(config), generator=fake_generator_with_bad_id,
        token_counter=len,
    )
    retrieval = json.loads((output_dir / "query_001/retrieval_res.json").read_text("utf-8"))
    assert retrieval["citation_validation"]["fallback_used"] is True
    assert retrieval["supporting_block_ids"] == [7]

def test_finalize_rejects_hotpot_fact_outside_retrieved_scope(tmp_path, fake_hotpot_run):
    from Scripts.baselines.kg2rag.finalize import finalize_runs
    manifest, config, output_dir = fake_hotpot_run
    finalize_runs(
        str(manifest), str(config), generator=fake_generator_with_illegal_fact,
        token_counter=len,
    )
    retrieval = json.loads((output_dir / "query_001/retrieval_res.json").read_text("utf-8"))
    assert retrieval["supporting_evidence"][0]["hotpot_sent_id"] == 0
    assert retrieval["citation_validation"]["fallback_used"] is True

def test_finalize_is_idempotent(tmp_path, fake_run):
    from Scripts.baselines.kg2rag.finalize import finalize_runs
    manifest, config, output_dir, generator = fake_run
    finalize_runs(str(manifest), str(config), generator=generator, token_counter=len)
    first = output_hash(output_dir)
    finalize_runs(str(manifest), str(config), generator=generator, token_counter=len)
    assert output_hash(output_dir) == first
```

- [ ] **Step 2: Run finalize tests**

Run: `python -m pytest tests/test_kg2rag_official_adapter.py -k "finalize" -v`

Expected: FAIL because `finalize.py` is missing.

- [ ] **Step 3: Implement evaluator-compatible finalization**

For each query, enforce the 10-block/4000-token budget before prompting the generator. The generator schema is:

```json
{
  "answer_short": "concise answer",
  "answer_rationale": "brief evidence-grounded explanation",
  "supporting_block_ids": ["qasper paragraph IDs"],
  "supporting_facts": [["Hotpot title", 0]]
}
```

Write `result.json`, `retrieval_res.json`, `evidence_chain.json`, document-level `final_results.json`, `token_cost.json`, and `kg2rag_index_stats.json`. `retrieval_res.json` must retain semantic seeds, expanded chunks, expansion edges, component triples, MST edges, DFS order, reranker scores, the complete top-20 ranking, citation audit, retrieval time, retrieval usage, and dense fallback reason.

`token_cost.json` must combine online retrieval and final generation only. OpenIE usage and indexing time remain in `kg2rag_index_stats.json`.

- [ ] **Step 4: Run finalize tests**

Run: `python -m pytest tests/test_kg2rag_official_adapter.py -k "finalize or citation or cost" -v`

Expected: PASS.

### Task 8: Add documentation, full-run launcher, and smoke verification

**Files:**
- Create: `Scripts/baselines/kg2rag/README.md`
- Create: `Scripts/baselines/kg2rag/run_full_experiments.ps1`
- Test: `tests/test_kg2rag_official_adapter.py`

- [ ] **Step 1: Add launcher contract test**

```python
def test_full_launcher_is_single_process_and_contains_both_datasets():
    text = Path("Scripts/baselines/kg2rag/run_full_experiments.ps1").read_text("utf-8")
    assert "Qasper.yaml" in text
    assert "HotpotQA.yaml" in text
    assert "full_experiment_status.json" in text
    assert "Start-Process" not in text
    assert "OPENAI_API_KEY" not in text
```

- [ ] **Step 2: Implement the launcher**

The script must run prepare, index, retrieve, finalize, and evaluation sequentially; rely on stage-level checkpointing; write sanitized stdout/stderr and `full_experiment_status.json` under `runs/kg2rag`; exit non-zero on any incomplete scope or query; and never start a second copy of itself.

- [ ] **Step 3: Document exact commands and private environment names**

The README must include setup, health checks, smoke, full run, stop/restart, result paths, and metric commands. It may name environment variables but must not contain their values or private endpoints.

- [ ] **Step 4: Run all adapter tests**

Run: `python -m pytest tests/test_kg2rag_official_adapter.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Run existing regression tests**

Run:

```powershell
python -m pytest tests/test_hipporag2_official_adapter.py tests/test_qasper_official_eval.py tests/test_hotpotqa_eval.py tests/test_evibridge_wiring.py -v
```

Expected: all selected existing tests PASS.

- [ ] **Step 6: Run service health checks with TLS verification**

Use the private configuration already used by EviBridge to send one embedding request, one JSON chat request, and one reranker request. Expected: HTTP success and structurally valid responses. Do not print keys, Authorization headers, or private base URLs.

- [ ] **Step 7: Run data-adapter smoke experiment**

Run two Qasper paper scopes and four HotpotQA scopes through every stage. Expected:

- every scope has a non-empty source map;
- every triple has a legal source ID;
- every final citation is legal within the current scope;
- no request payload contains gold fields;
- candidate ranks are unique and at most 20;
- generation contexts satisfy both budgets;
- offline and online token totals are separated.

### Task 9: Run full experiments and produce result summaries

**Files:**
- Runtime only under ignored `runs/kg2rag/`
- No paper modification in this task

- [ ] **Step 1: Start the unique background launcher**

From the repository root, start one hidden PowerShell process that invokes `Scripts/baselines/kg2rag/run_full_experiments.ps1`. First confirm no existing launcher or KG²RAG Python process is active. Preserve TLS verification and current private model configuration.

- [ ] **Step 2: Monitor scope and query coverage**

Track Qasper 281 scopes/1005 questions and HotpotQA 1000 scopes/1000 questions from status and per-scope result files. Do not restart while a process is active. On transient service failure, allow bounded client retries; on process failure, preserve logs and resume from validated checkpoints.

- [ ] **Step 3: Evaluate Qasper**

Run:

```powershell
python Eval/evaluation.py -d Scripts/cfg/Qasper.yaml --method kg2rag_official --max_workers 1
```

Expected: 1005/1005 predictions, no missing or duplicate question IDs, and Answer F1 plus Evidence Precision/Recall/F1.

- [ ] **Step 4: Evaluate HotpotQA**

Run:

```powershell
python Eval/evaluation.py -d Scripts/cfg/HotpotQA.yaml --method kg2rag_official --max_workers 1
```

Expected: 1000/1000 predictions and Answer EM/F1, SP Precision/Recall/F1, and Joint F1.

- [ ] **Step 5: Compute retrieval and efficiency summaries**

Compute Qasper Recall@1/2/3/5/10/20 from the same complete run and assert monotonic non-decrease. Aggregate online `Tokens/Question` and `Seconds/Question` for both datasets; aggregate OpenIE/index cost separately. Validate that all denominators equal the complete question counts.

- [ ] **Step 6: Final verification without editing the paper**

Run: `git status --short`

Expected: only intended adapter/config/test/design/plan files plus pre-existing user changes; no official source, model cache, index, result, log, API key, or private endpoint is tracked. Report result file paths and metrics to the user, but do not modify `main_revised.tex` or another manuscript file.

## Self-review record

- Spec coverage: source provenance, closed-corpus construction, both dataset scopes, official seed/expansion/organization mechanisms, uniform models/budgets, evidence mappings, Recall@k, metrics, offline/online cost separation, recovery, smoke, full coverage, and no-paper-change requirements are each assigned to a task.
- Placeholder scan: the plan contains no deferred implementation decisions or placeholder function bodies; test invocations use explicit fixture-provided paths and clients.
- Type consistency: `source_id`, `paragraph_id`, `hotpot_title`, `hotpot_sentences`, `ranked_results`, `cache_identity`, `retrieval usage`, and generation schema names are consistent from prepare through finalize.
- Design note: the upstream executable is HotpotQA-specific and couples chunks to document-title/sentence identifiers. The adapter therefore implements the paper's generic `G={(h,r,t,c)}` formulation so Qasper paragraph IDs remain valid, while preserving the upstream one-hop expansion, maximum-spanning-tree filtering, component reranking, and context organization semantics.
