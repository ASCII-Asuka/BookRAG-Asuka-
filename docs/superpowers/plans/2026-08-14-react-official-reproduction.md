# ReAct Official Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an isolated, official-control-flow ReAct reproduction on Qasper validation full and HotpotQA fixed-1000 using the unified Qwen model and only each question's fixed local corpus.

**Architecture:** Add a standalone adapter under `Scripts/baselines/react_official/` that reads the already locked HippoRAG 2 prepared manifests, exposes dataset-specific closed-corpus `Search`/`Lookup` environments, executes the official seven-step ReAct loop, and converts trajectories to the repository's existing evaluator-compatible outputs. The adapter will not modify `Core/rag/react_*`, overwrite prior ReAct results, access live Wikipedia, or use gold evidence for retrieval or fallback.

**Tech Stack:** Python 3.10+, `requests`, `PyYAML`, repository Qasper/HotpotQA evaluators, `pytest`/`unittest`, PowerShell orchestration.

---

## File map

- `Scripts/baselines/react_official/common.py`: deterministic JSON I/O, hashing, safe filenames, token/cost aggregation, and run identity.
- `Scripts/baselines/react_official/model_client.py`: TLS-verified OpenAI-compatible text completion client with bounded retries and client-side stop truncation.
- `Scripts/baselines/react_official/prompts.py`: locked official HotpotQA prompt and validated Qasper-train demonstrations.
- `Scripts/baselines/react_official/environment.py`: closed-corpus HotpotQA and Qasper `Search`/`Lookup` environments and evidence provenance.
- `Scripts/baselines/react_official/runner.py`: official seven-step ReAct control loop and one-shot action repair.
- `Scripts/baselines/react_official/run.py`: prepared-manifest execution, cache validation, recovery, and raw trajectory manifests.
- `Scripts/baselines/react_official/finalize.py`: evaluator-compatible per-query result, retrieval, evidence-chain, and token-cost outputs.
- `Scripts/baselines/react_official/diagnostics.py`: coverage and trajectory-quality summaries.
- `Scripts/baselines/react_official/official-source-lock.json`: official repository, commit, license, and archive identity.
- `Scripts/baselines/react_official/run_full_experiments.ps1`: sequential Qasper/HotpotQA run, finalize, evaluation, and diagnostics.
- `Scripts/baselines/react_official/README.md`: reproducibility and recovery commands.
- `config/react_official_repro.yaml`: public non-secret experiment parameters.
- `tests/test_react_official_repro.py`: adapter unit, mapping, recovery, and regression tests.

### Task 1: Package skeleton, immutable source lock, and public configuration

**Files:**
- Create: `Scripts/baselines/react_official/__init__.py`
- Create: `Scripts/baselines/react_official/common.py`
- Create: `Scripts/baselines/react_official/official-source-lock.json`
- Create: `config/react_official_repro.yaml`
- Test: `tests/test_react_official_repro.py`

- [ ] **Step 1: Write failing identity and source-lock tests**

```python
def test_source_lock_and_public_config_are_fixed_and_secret_free():
    lock = load_json(REPO / "Scripts/baselines/react_official/official-source-lock.json")
    config_text = (REPO / "config/react_official_repro.yaml").read_text("utf-8")
    assert lock["repository"] == "https://github.com/ysymyth/ReAct"
    assert lock["commit"] == "6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9"
    assert lock["license"] == "MIT"
    assert "api_key" not in config_text.lower()
    assert "api_base" not in config_text.lower()

def test_run_identity_changes_when_corpus_or_config_changes():
    first = build_run_identity("corpus-a", "config-a", "commit", "model")
    assert first == build_run_identity("corpus-a", "config-a", "commit", "model")
    assert first != build_run_identity("corpus-b", "config-a", "commit", "model")
```

- [ ] **Step 2: Run the tests and verify the package is missing**

Run: `python -m pytest tests/test_react_official_repro.py -k "source_lock or run_identity" -q`

Expected: FAIL with `ModuleNotFoundError` or missing source-lock/config files.

- [ ] **Step 3: Implement deterministic helpers and fixed public metadata**

```python
def build_run_identity(corpus_sha256: str, config_sha256: str,
                       source_revision: str, model_name: str) -> str:
    return sha256_json({
        "corpus_sha256": corpus_sha256,
        "config_sha256": config_sha256,
        "source_revision": source_revision,
        "model_name": model_name,
    })
```

Set `method_suffix: react_official_repro`, `temperature: 0.0`, `max_output_tokens: 100`, `max_steps: 7`, and `max_supporting_evidence: 4` in the public YAML. Store no endpoint or credential.

- [ ] **Step 4: Run the identity tests**

Run: `python -m pytest tests/test_react_official_repro.py -k "source_lock or run_identity" -q`

Expected: all selected tests PASS.

- [ ] **Step 5: Review checkpoint**

Run: `git diff --check -- Scripts/baselines/react_official config/react_official_repro.yaml tests/test_react_official_repro.py`

Expected: no whitespace errors. Do not commit; the repository policy leaves commits to the user.

### Task 2: Official completion parameters and stop behavior

**Files:**
- Create: `Scripts/baselines/react_official/model_client.py`
- Modify: `tests/test_react_official_repro.py`

- [ ] **Step 1: Add failing tests for request parameters, TLS, retry bounds, and local stop truncation**

```python
def test_completion_uses_official_parameters_and_tls(monkeypatch):
    response = fake_response("Thought 1: x\nAction 1: Search[x]\nObservation 1: leaked")
    session = FakeSession(response)
    client = CompletionClient("key", "https://service/v1", timeout=30, max_attempts=2,
                              session=session)
    text, usage = client.complete("model", "prompt", stop="\nObservation 1:")
    assert text.endswith("Search[x]")
    assert session.last_json["temperature"] == 0.0
    assert session.last_json["max_tokens"] == 100
    assert session.last_json["stop"] == ["\nObservation 1:"]
    assert session.last_verify is True
```

- [ ] **Step 2: Run the focused tests**

Run: `python -m pytest tests/test_react_official_repro.py -k completion -q`

Expected: FAIL because `CompletionClient` does not exist.

- [ ] **Step 3: Implement the minimal client**

The client posts to `<base>/completions` with `model`, `prompt`, `temperature=0`, `max_tokens=100`, and a one-element `stop` list. It accepts both legacy `choices[0].text` and OpenAI-compatible `choices[0].message.content`, preserves TLS verification, retries only network/HTTP/JSON failures up to `max_attempts`, and truncates at the same stop marker if the service ignores it.

- [ ] **Step 4: Run completion-client tests**

Run: `python -m pytest tests/test_react_official_repro.py -k completion -q`

Expected: all selected tests PASS.

### Task 3: Closed-corpus Search and Lookup environments

**Files:**
- Create: `Scripts/baselines/react_official/environment.py`
- Modify: `tests/test_react_official_repro.py`

- [ ] **Step 1: Add failing HotpotQA environment tests**

```python
def test_hotpot_search_and_lookup_never_leave_current_scope():
    env = HotpotEnvironment.from_chunks(hotpot_chunks())
    search = env.search("Alpha")
    assert search.observation.startswith("Alpha")
    assert {item["hotpot_title"] for item in search.evidence} == {"Alpha"}
    first = env.lookup("founder")
    second = env.lookup("founder")
    assert first.evidence[0]["hotpot_sent_id"] < second.evidence[0]["hotpot_sent_id"]
    assert "Outside Scope" not in search.observation
```

- [ ] **Step 2: Add failing Qasper section/paragraph provenance tests**

```python
def test_qasper_search_keeps_original_paragraph_ids_and_text():
    env = QasperEnvironment.from_chunks(qasper_chunks())
    result = env.search("Methods")
    assert result.evidence[0]["paragraph_id"] == "p-1"
    assert result.evidence[0]["content"] == "original paragraph"
    assert "[Section: Methods]" not in result.evidence[0]["content"]
```

- [ ] **Step 3: Run environment tests**

Run: `python -m pytest tests/test_react_official_repro.py -k "hotpot_search or qasper_search" -q`

Expected: FAIL because the environment classes do not exist.

- [ ] **Step 4: Implement exact-title/section matching, BM25 fallback, and sequential Lookup**

Use only chunks passed to the environment. Exact normalized titles/section paths win; otherwise rank pages by deterministic BM25 over the current scope. `Lookup[keyword]` advances through matching atomic sentences/paragraphs on the active page and returns the next match on repeated calls. Every observation carries legal provenance records.

- [ ] **Step 5: Run environment tests**

Run: `python -m pytest tests/test_react_official_repro.py -k "environment or hotpot_search or qasper_search or lookup" -q`

Expected: all selected tests PASS.

### Task 4: Locked prompts and official seven-step loop

**Files:**
- Create: `Scripts/baselines/react_official/prompts.py`
- Create: `Scripts/baselines/react_official/runner.py`
- Modify: `tests/test_react_official_repro.py`

- [ ] **Step 1: Add failing prompt-integrity and parsing tests**

```python
def test_hotpot_prompt_contains_six_official_examples_and_action_contract():
    assert HOTPOT_PROMPT.count("Question:") == 6
    assert "Search[entity]" in HOTPOT_PROMPT
    assert "Lookup[keyword]" in HOTPOT_PROMPT
    assert "Finish[answer]" in HOTPOT_PROMPT

def test_parse_action_accepts_only_three_official_actions():
    assert parse_action("Search[Alpha]").name == "Search"
    with pytest.raises(ValueError):
        parse_action("Browse[Alpha]")
```

- [ ] **Step 2: Add failing loop/repair/step-limit tests**

```python
def test_runner_repairs_once_and_stops_on_finish():
    model = FakeModel(["Thought 1: inspect", "Search[Alpha]", "Thought 2: done\nAction 2: Finish[answer]"])
    result = ReActRunner(model, max_steps=7).run("question", FakeEnvironment())
    assert result.answer == "answer"
    assert result.repair_calls == 1
    assert model.stops == ["\nObservation 1:", "\n", "\nObservation 2:"]

def test_runner_forces_empty_finish_after_seven_steps():
    model = FakeModel([f"Thought {i}: continue\nAction {i}: Search[x]" for i in range(1, 8)])
    result = ReActRunner(model, max_steps=7).run("question", FakeEnvironment())
    assert result.answer == ""
    assert result.termination_reason == "step_limit"
    assert len(result.steps) == 7
```

- [ ] **Step 3: Run prompt and runner tests**

Run: `python -m pytest tests/test_react_official_repro.py -k "prompt or parse_action or runner" -q`

Expected: FAIL because prompt/runner modules do not exist.

- [ ] **Step 4: Implement prompt builders and the official loop**

The main call stops at `\nObservation i:`. If `\nAction i:` cannot be parsed, keep the first Thought line and make exactly one repair call stopped by newline. Execute only `Search`, `Lookup`, or `Finish`; invalid actions are recorded and observed as a controlled message, without adding semantic retries. At step seven, return `Finish[]` semantics with `termination_reason="step_limit"`.

- [ ] **Step 5: Run prompt and runner tests**

Run: `python -m pytest tests/test_react_official_repro.py -k "prompt or parse_action or runner" -q`

Expected: all selected tests PASS.

### Task 5: Prepared-manifest execution and resumable raw trajectories

**Files:**
- Create: `Scripts/baselines/react_official/run.py`
- Modify: `tests/test_react_official_repro.py`

- [ ] **Step 1: Add failing cache and coverage tests**

```python
def test_run_reuses_only_complete_matching_identity(tmp_path):
    first = run_prepared(prepared_manifest(tmp_path), config_path(tmp_path), tmp_path / "out",
                         model=FakeModel.finishing("yes"))
    second = run_prepared(prepared_manifest(tmp_path), config_path(tmp_path), tmp_path / "out",
                          model=FailIfCalledModel())
    assert first["question_count"] == second["question_count"] == 1
    assert second["cached_question_count"] == 1

def test_run_rejects_duplicate_question_ids(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        run_prepared(duplicate_manifest(tmp_path), config_path(tmp_path), tmp_path / "out",
                     model=FakeModel.finishing("yes"))
```

- [ ] **Step 2: Run orchestration tests**

Run: `python -m pytest tests/test_react_official_repro.py -k "run_reuses or duplicate_question" -q`

Expected: FAIL because `run_prepared` does not exist.

- [ ] **Step 3: Implement one-scope-at-a-time execution and atomic checkpoints**

Read `runs/hipporag2/full_prepared/<dataset>/manifest.json` and each listed scope. Save one `scope_result.json` per scope under a run-identity directory. A cached question is reusable only if identity, question ID, trajectory, answer field, evidence trace, and cost fields validate. Unknown/duplicate IDs and manifest-count mismatches fail immediately. Atomic `.tmp` replacement prevents partial JSON reuse.

- [ ] **Step 4: Run orchestration tests**

Run: `python -m pytest tests/test_react_official_repro.py -k "run_reuses or duplicate_question or coverage" -q`

Expected: all selected tests PASS.

### Task 6: Evidence ordering and evaluator-compatible finalization

**Files:**
- Create: `Scripts/baselines/react_official/finalize.py`
- Modify: `tests/test_react_official_repro.py`

- [ ] **Step 1: Add failing evidence-order tests**

```python
def test_observed_evidence_orders_lookup_before_search_and_deduplicates():
    ranked = rank_observed_evidence(trajectory_with_duplicate_observations())
    assert [item["source_id"] for item in ranked] == ["lookup-p", "search-p"]

def test_empty_trace_submits_empty_evidence_without_gold_fallback():
    result = finalize_query(empty_trajectory(), prepared_query())
    assert result["supporting_block_ids"] == []
```

- [ ] **Step 2: Add failing dataset mapping/output tests**

```python
def test_finalize_writes_qasper_and_hotpot_legal_evidence(tmp_path):
    qasper = finalize_fixture("qasper", tmp_path)
    hotpot = finalize_fixture("hotpotqa", tmp_path)
    assert qasper["supporting_block_ids"] == ["paragraph-1"]
    assert hotpot["supporting_facts"] == [["Title", 0]]
    assert (tmp_path / "query_001/retrieval_res.json").exists()
    assert (tmp_path / "query_001/evidence_chain.json").exists()
```

- [ ] **Step 3: Run finalization tests**

Run: `python -m pytest tests/test_react_official_repro.py -k "observed_evidence or empty_trace or finalize" -q`

Expected: FAIL because finalization functions do not exist.

- [ ] **Step 4: Implement trace-only evidence conversion and output writers**

Rank first-observed `Lookup` atoms before first-observed `Search` atoms, deduplicate by Qasper paragraph ID or HotpotQA title/sentence ID, reject provenance not present in the prepared scope, and never fall back to gold evidence. Write `result.json`, `retrieval_res.json`, `evidence_chain.json`, `token_cost.json`, per-scope `final_results.json`, and aggregate cost with method suffix `react_official_repro`.

- [ ] **Step 5: Run finalization tests**

Run: `python -m pytest tests/test_react_official_repro.py -k "observed_evidence or empty_trace or finalize" -q`

Expected: all selected tests PASS.

### Task 7: Diagnostics and full adapter regression tests

**Files:**
- Create: `Scripts/baselines/react_official/diagnostics.py`
- Modify: `tests/test_react_official_repro.py`

- [ ] **Step 1: Add a failing trajectory-statistics test**

```python
def test_diagnostics_reports_official_failure_rates():
    report = summarize_trajectories([successful(), repaired(), invalid(), step_limited()])
    assert report["query_count"] == 4
    assert report["repair_call_rate"] == pytest.approx(0.25)
    assert report["invalid_action_rate"] == pytest.approx(0.25)
    assert report["step_limit_rate"] == pytest.approx(0.25)
```

- [ ] **Step 2: Run the diagnostics test**

Run: `python -m pytest tests/test_react_official_repro.py -k diagnostics -q`

Expected: FAIL because `summarize_trajectories` does not exist.

- [ ] **Step 3: Implement deterministic summaries**

Report query coverage, mean steps, mean model calls, repair-call rate, invalid-action rate, step-limit rate, Search/Lookup counts, prompt/completion/total tokens, and mean online seconds per question. The CLI writes JSON beside the raw manifest and prints no endpoint or credential.

- [ ] **Step 4: Run all adapter and pre-existing ReAct tests**

Run: `python -m pytest tests/test_react_official_repro.py tests/test_react_runner.py tests/test_react_env.py tests/test_react_prompt.py -q`

Expected: all tests PASS and existing local ReAct behavior remains unchanged.

### Task 8: Official source acquisition, orchestration, and documentation

**Files:**
- Create: `Scripts/baselines/react_official/run_full_experiments.ps1`
- Create: `Scripts/baselines/react_official/README.md`
- Modify: `.gitignore`
- Modify: `tests/test_react_official_repro.py`

- [ ] **Step 1: Add failing script-safety tests**

```python
def test_orchestration_uses_locked_source_and_contains_no_secrets():
    script = (REPO / "Scripts/baselines/react_official/run_full_experiments.ps1").read_text("utf-8")
    assert "6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9" in script
    assert "OPENAI_API_KEY=" not in script
    assert "--prepared-root" in script
    assert "react_official_repro" in script
```

- [ ] **Step 2: Run script-safety tests**

Run: `python -m pytest tests/test_react_official_repro.py -k orchestration -q`

Expected: FAIL because the script is missing.

- [ ] **Step 3: Implement safe acquisition and sequential full-run script**

The PowerShell script clones the official repository to ignored `runs/third_party/ReAct` or downloads the exact commit archive if Git fails, verifies `HEAD`/archive SHA recorded in the source lock, loads the existing private service configuration without printing it, executes Qasper then HotpotQA, finalizes and evaluates each dataset, and writes `runs/react_official_repro/full_experiment_status.json`. It must be restart-safe and must not overwrite old `react` outputs.

- [ ] **Step 4: Document exact smoke/full/recovery commands**

Document the official-vs-adapted boundary, fixed local-corpus restriction, input manifests, public/private configuration split, per-stage outputs, and evaluation commands. State explicitly that the Qwen model substitutes for the unavailable original `text-davinci-002` while control flow remains official.

- [ ] **Step 5: Run static and regression verification**

Run: `python -m pytest tests/test_react_official_repro.py -q`

Run: `git diff --check -- .gitignore Scripts/baselines/react_official config/react_official_repro.yaml tests/test_react_official_repro.py`

Expected: all tests PASS; no whitespace errors.

### Task 9: Bounded smoke experiments

**Files:**
- Runtime outputs only: `runs/react_official_repro/smoke/`

- [ ] **Step 1: Run an official-source integrity smoke**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File Scripts/baselines/react_official/run_full_experiments.ps1 -Stage source-smoke`

Expected: locked commit is present and official HotpotQA prompt/control source is readable.

- [ ] **Step 2: Run Qasper two-paper smoke**

Run: `python -m Scripts.baselines.react_official.run --prepared-root runs/hipporag2/full_prepared/qasper --config config/react_official_repro.yaml --output-root runs/react_official_repro/smoke/qasper --max-scopes 2`

Expected: two valid scope results, no out-of-scope provenance, and every query has a trajectory/cost record.

- [ ] **Step 3: Run HotpotQA four-question smoke**

Run: `python -m Scripts.baselines.react_official.run --prepared-root runs/hipporag2/full_prepared/hotpotqa --config config/react_official_repro.yaml --output-root runs/react_official_repro/smoke/hotpotqa --max-bridge 2 --max-comparison 2`

Expected: four valid scope results with legal title/sentence evidence only.

- [ ] **Step 4: Finalize and evaluate smoke outputs**

Run both dataset finalizers with `--allow-partial-smoke`, then verify the emitted method directory suffix is `react_official_repro` and no old `eval_*_react` directory timestamp changed.

### Task 10: Full dual-dataset experiment, evaluation, and result handoff

**Files:**
- Runtime outputs: `runs/react_official_repro/full/`
- Evaluator outputs under locked Qasper/HotpotQA working directories with suffix `react_official_repro`

- [ ] **Step 1: Start the resumable sequential full run**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File Scripts/baselines/react_official/run_full_experiments.ps1 -Stage full`

Expected: status changes through Qasper run/finalize/evaluate, then HotpotQA run/finalize/evaluate, without duplicate worker processes.

- [ ] **Step 2: Verify complete prediction coverage**

Run: `python -m Scripts.baselines.react_official.diagnostics --root runs/react_official_repro/full/qasper --expected-queries 1005`

Run: `python -m Scripts.baselines.react_official.diagnostics --root runs/react_official_repro/full/hotpotqa --expected-queries 1000`

Expected: Qasper `1005/1005`, HotpotQA `1000/1000`, no duplicate question IDs, and all evidence provenance is legal.

- [ ] **Step 3: Run unified official evaluators**

Run: `python Scripts/eval/qasper_official.py --dataset-config runs/_datasets/qasper/qasper_validation_full1005_unified_0417263dd.yaml --method react_official_repro --answer-source output --dynamic-evidence-topk`

Run: `python Eval/evaluation.py -d runs/_datasets/hotpotqa/fixed1000/hotpotqa_distractor_validation_fixed1000_seed42.yaml --method react_official_repro --max_workers 1`

Expected: Qasper Answer F1/Evidence P/R/F1 and HotpotQA Answer EM/F1, SP P/R/F1, Joint F1 files are produced with no missing predictions.

- [ ] **Step 4: Verify cost and diagnostic consistency**

Confirm every query's online token/time cost comes from the same run identity as its answer and trajectory. Aggregate mean steps, calls, repair-call rate, invalid-action rate, step-limit rate, tokens/question, and seconds/question.

- [ ] **Step 5: Final verification checkpoint**

Run: `python -m pytest tests/test_react_official_repro.py tests/test_react_runner.py tests/test_react_env.py tests/test_react_prompt.py -q`

Run: `git status --short`

Expected: all tests PASS; only intentional adapter/config/test/document changes plus pre-existing user changes are present. Do not update the paper or replace old ReAct table rows until the user explicitly requests it.
