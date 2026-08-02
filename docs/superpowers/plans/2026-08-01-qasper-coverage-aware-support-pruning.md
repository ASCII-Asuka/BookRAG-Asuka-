# Qasper Coverage-Aware Supporting Evidence Pruning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a weak-only, coverage-aware EviBridge supporting-evidence pruner and prove its effect on a locked Qasper-100 replay and fresh-inference set before evaluating the 664-question holdout.

**Architecture:** Keep retrieval, typed PPR, the ten-block answer context, the LLM, and the answer prompt unchanged. Put deterministic support selection in a pure module, call the existing answer-conditioned reranker once, and persist complete candidate diagnostics. Add separate manifest/replay tools so cached drafts and candidates can be evaluated without an LLM, then wire the accepted policy into a distinct EviBridge method suffix with answer regeneration disabled.

**Tech Stack:** Python 3, unittest, Pydantic, YAML, existing EviBridge index and reranker provider, Qasper official evaluator, paired bootstrap.

**Execution constraint:** Work directly on `dev`; do not create a worktree. Runtime datasets, score caches, and experiment outputs stay under ignored `runs/` paths.

---

### Task 1: Implement the pure coverage-aware support pruner

**Files:**
- Create: `Core/rag/evibridge_support_pruner.py`
- Create: `tests/test_evibridge_support_pruner.py`

- [ ] **Step 1: Write failing tests for anchors, dynamic caps, and no-padding behavior**

Create tests around this public interface:

```python
from Core.rag.evibridge_support_pruner import prune_supporting_evidence


def candidate(block_id, text, score, block_type="paragraph"):
    return {
        "block_id": block_id,
        "block_type": block_type,
        "text": text,
        "section_id": "s",
        "score_parts": {"answer_conditioned_rerank_score": score},
    }


class CoverageAwareSupportPrunerTests(unittest.TestCase):
    def test_retains_legal_anchor_without_padding(self):
        result = prune_supporting_evidence(
            question="Which model is best?",
            draft_answer="Model B is best.",
            intent="fact",
            subqueries=[],
            candidates=[candidate(1, "Model B is best.", 0.9), candidate(2, "Noise.", 0.8)],
            anchor_ids=[1],
            allowed_types={"paragraph", "table", "caption", "figure"},
            max_items=2,
            min_normalized_relevance=0.5,
            redundancy_overlap_threshold=0.75,
        )
        self.assertEqual(result["final_ids"], [1])
        self.assertEqual(result["stopping_reason"], "no_marginal_coverage")

    def test_comparison_can_add_distinct_facets_up_to_three(self):
        result = prune_supporting_evidence(
            question="Compare Method A and Method B.",
            draft_answer="Method B performs better.",
            intent="comparison",
            subqueries=["Method A score", "Method B score"],
            candidates=[
                candidate(1, "Method A score is 80.", 0.95),
                candidate(2, "Method B score is 84.", 0.90),
                candidate(3, "Unrelated setup details.", 0.85),
            ],
            anchor_ids=[],
            allowed_types={"paragraph"},
            max_items=3,
            min_normalized_relevance=0.4,
            redundancy_overlap_threshold=0.75,
        )
        self.assertEqual(result["final_ids"], [1, 2])
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
python -m unittest tests.test_evibridge_support_pruner -v
```

Expected: import failure because `evibridge_support_pruner.py` does not exist.

- [ ] **Step 3: Add failing edge-case tests**

Cover all of these in the same test module with the stated exact outcomes:

- `test_rejects_bridge_auxiliary_type`: an entity candidate is absent from
  `final_ids` and has rejection reason `ineligible_type`.
- `test_rejects_redundant_contained_paragraph`: the second contained paragraph
  is absent and has rejection reason `redundant`.
- `test_deduplicates_anchor_ids_in_first_seen_order`: anchors `[2, 1, 2]`
  produce final prefix `[2, 1]`.
- `test_unanswerable_short_circuits_to_empty`: `unanswerable=True` produces an
  empty result and stopping reason `unanswerable`.
- `test_non_finite_scores_are_rejected`: `float("nan")` and `float("inf")`
  candidates have rejection reason `non_finite_score`.
- `test_tied_scores_have_deterministic_rank_normalization`: two repeated calls
  produce identical normalized scores, ordering, and final IDs.
- `test_no_anchor_uses_top_eligible_candidate_as_safe_fallback`: when every
  candidate has zero lexical coverage, only the first ranked eligible candidate
  is selected.
- `test_every_candidate_has_accept_or_reject_reason`: diagnostic IDs equal the
  candidate IDs and every diagnostic has a non-empty decision reason.

Verify that entity, title, summary, and patch blocks never appear in `final_ids`
when they are not in `allowed_types`.

- [ ] **Step 4: Implement the minimal pure module**

The module must expose `tokenize_support_text(text: str) -> set[str]`,
`support_overlap(left: set[str], right: set[str]) -> float`,
`normalize_candidate_relevance(candidates: Sequence[Mapping[str, Any]]) ->
list[dict[str, Any]]`, `support_budget_for_intent(intent: str) -> int`, and
`prune_supporting_evidence` with the exact keyword arguments used in Step 1.
`support_budget_for_intent` returns three for comparison/multi-hop and two for
all other intents.

Use a fixed utility of `0.65 * normalized_relevance + 0.35 * marginal_coverage`.
Tokenization is lowercase alphanumeric English tokens with a module-local fixed
stop-word set. Redundancy is `intersection / min(left_size, right_size)`.
Greedy ties resolve by reranker rank and then integer block ID.

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run:

```powershell
python -m unittest tests.test_evibridge_support_pruner -v
```

Expected: all pruner tests pass.

- [ ] **Step 6: Commit the pure selector**

```powershell
git add -- Core/rag/evibridge_support_pruner.py tests/test_evibridge_support_pruner.py
git commit -m "feat: add coverage-aware evidence pruner"
```

### Task 2: Add configuration and wire the pruner into EviBridge

**Files:**
- Modify: `Core/configs/rag/evibridge_config.py`
- Modify: `Core/rag/evibridge_rag.py`
- Modify: `tests/test_evibridge_modules.py`
- Modify: `tests/test_evibridge_wiring.py`
- Create: `config/evibridge_support_pruned.yaml`

- [ ] **Step 1: Write failing configuration tests**

Add assertions that the new config loads as follows:

```python
self.assertEqual(rag.support_selection_policy, "coverage_prune")
self.assertEqual(rag.support_min_normalized_relevance, 0.5)
self.assertEqual(rag.support_redundancy_overlap_threshold, 0.75)
self.assertFalse(rag.regenerate_on_support_expansion)
self.assertEqual(rag.method_suffix, "evibridge_support_pruned")
```

Also assert that `EviBridgeRAGConfig()` defaults to
`support_selection_policy="fill_budget"` so existing configurations and method
suffixes remain compatible.

- [ ] **Step 2: Run config tests and verify RED**

Run:

```powershell
python -m unittest tests.test_evibridge_wiring.EviBridgeWiringTests.test_support_pruned_config_enables_coverage_policy -v
```

Expected: failure because the new config fields and YAML file are absent.

- [ ] **Step 3: Add validated config fields and suffix logic**

Add these fields to `EviBridgeRAGConfig`:

```python
support_selection_policy: Literal["fill_budget", "coverage_prune"] = "fill_budget"
support_min_normalized_relevance: float = Field(default=0.5, ge=0.0, le=1.0)
support_redundancy_overlap_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
```

When `support_selection_policy == "coverage_prune"` and the ablation variant is
`full`, return `evibridge_support_pruned`. Preserve the current
`evibridge_support_controller` suffix for `fill_budget + weak_only`.

- [ ] **Step 4: Create the public config variant**

Copy `config/evibridge.yaml` to `config/evibridge_support_pruned.yaml` and change
only the method settings:

```yaml
  support_completion_policy: weak_only
  support_selection_policy: coverage_prune
  support_min_normalized_relevance: 0.5
  support_redundancy_overlap_threshold: 0.75
  enable_answer_conditioned_support_rerank: true
  answer_conditioned_support_topk: 20
  regenerate_on_support_expansion: false
```

Do not insert local paths, service URLs, or credentials.

- [ ] **Step 5: Write failing controller-integration tests**

Add a test that creates a comparison demand, one legal anchor, and three ranked
candidates. Assert:

```python
self.assertEqual(retrieval_info["supporting_block_ids"], [1, 3])
self.assertEqual(retrieval_info["answer_context_block_ids"], [1, 2])
self.assertFalse(retrieval_info["answer_regeneration"]["required"])
self.assertEqual(
    retrieval_info["support_controller"]["selection_policy"],
    "coverage_prune",
)
self.assertIn("candidate_diagnostics", retrieval_info["support_controller"])
```

Use a fake reranker and fake LLM call counter. Assert one answer-generation call
and no conditional regeneration call.

- [ ] **Step 6: Run the new integration test and verify RED**

Run:

```powershell
python -m unittest tests.test_evibridge_modules.EviBridgeModuleTests.test_coverage_prune_keeps_answer_context_and_skips_regeneration -v
```

Expected: the current controller pads to its target and expands answer context.

- [ ] **Step 7: Integrate the pure pruner**

In `_control_supporting_evidence`:

1. keep current `fill_budget` behavior unchanged;
2. preserve answer-conditioned scores and ranks on every ranked candidate;
3. call `prune_supporting_evidence` only for weak questions when the policy is
   `coverage_prune`;
4. build final payloads from the ranked candidate map so score diagnostics are
   not lost;
5. keep `answer_context=selected_payload` and
   `regeneration_required=False` for the new policy;
6. merge pure-pruner diagnostics into `support_controller` without deleting
   current keys.

- [ ] **Step 8: Run focused and regression tests**

Run:

```powershell
python -m unittest tests.test_evibridge_support_pruner tests.test_evibridge_modules tests.test_evibridge_wiring -v
```

Expected: all tests pass, including existing fixed-padding and regeneration
tests.

- [ ] **Step 9: Commit config and runtime integration**

```powershell
git add -- Core/configs/rag/evibridge_config.py Core/rag/evibridge_rag.py config/evibridge_support_pruned.yaml tests/test_evibridge_modules.py tests/test_evibridge_wiring.py
git commit -m "feat: integrate coverage-aware support pruning"
```

### Task 3: Add deterministic Qasper Recall@5/@10/@20 metrics

**Files:**
- Modify: `Scripts/eval/qasper_official.py`
- Modify: `tests/test_qasper_official_eval.py`

- [ ] **Step 1: Write failing annotation-selection tests**

Test one question with two gold annotations where the first is best at @5 and
the second is best at @20. Assert that the annotation selected by @20 is reused
for all k values:

```python
recalls = evidence_recall_at_ks(
    ranked_evidence=["a", "x", "b", "c"],
    references=[{"evidence": ["a"]}, {"evidence": ["b", "c"]}],
    ks=(1, 2, 4),
)
self.assertEqual(recalls, {1: 0.0, 2: 0.0, 4: 1.0})
```

Add a tie case proving dataset-order tie-breaking.

- [ ] **Step 2: Run the focused tests and verify RED**

Expected: import failure for `evidence_recall_at_ks`.

- [ ] **Step 3: Implement Recall@k helpers and aggregation**

Add `evidence_recall_at_ks(ranked_evidence: Sequence[Any], references:
Sequence[Mapping[str, Any]], ks: Sequence[int] = (5, 10, 20)) -> Dict[int,
float]`.

Select one annotation by descending `(Recall@20, Recall@10, Recall@5)` and
stable dataset order. Reuse `paragraph_prf_score` alignment semantics. Add
per-question `evidence_recall_at_5`, `_10`, and `_20` fields and aggregate keys
`Evidence Recall@5`, `Evidence Recall@10`, and `Evidence Recall@20` when ranked
candidate evidence is supplied. Preserve existing official P/R/F1 output.

- [ ] **Step 4: Run evaluator tests and verify GREEN**

```powershell
python -m unittest tests.test_qasper_official_eval -v
```

Expected: all Qasper official evaluator tests pass.

- [ ] **Step 5: Commit retrieval diagnostics**

```powershell
git add -- Scripts/eval/qasper_official.py tests/test_qasper_official_eval.py
git commit -m "feat: report Qasper evidence recall at k"
```

### Task 4: Generate and validate the locked 100/664 manifests

**Files:**
- Create: `Scripts/eval/qasper_support_manifest.py`
- Create: `tests/test_qasper_support_manifest.py`

- [ ] **Step 1: Write failing deterministic-sampling tests**

Build a synthetic population that has enough rows for all quotas. Call:

```python
selection = build_support_optimization_split(
    rows=rows,
    demand_intents=intent_by_question_id,
    seed=42,
    target_size=100,
    answer_type_quotas={
        "extractive": 45,
        "abstractive": 25,
        "boolean": 15,
        "none": 15,
    },
    minimum_intent_counts={"comparison": 30, "multi-hop": 8},
)
```

Assert exact size, quotas, minimum intents, no overlap, full source coverage,
stable ordering across repeated calls, and failure when a quota is impossible.

- [ ] **Step 2: Run tests and verify RED**

Expected: import failure because the manifest module is absent.

- [ ] **Step 3: Implement answer typing, sampling, hashing, and validation**

Expose `qasper_answer_type(row)`, `build_support_optimization_split` with the
arguments from Step 1, `validate_support_split` with source/tuning/holdout rows
and quota arguments, and `dataset_sha256(path)`.

Use question ID as the stable identity. Read demand intent only from the frozen
`retrieval_res.json`; never read gold evidence during sampling. Emit tuning and
holdout dataset JSON files plus manifests containing source hash, seed, quotas,
question IDs, document IDs, answer types, and demand intents.

- [ ] **Step 4: Add a CLI and test malformed input failures**

The CLI accepts `--dataset`, `--retrieval-root`, `--method`, `--tuning-output`,
`--holdout-output`, `--manifest-output`, and `--seed`. Missing question IDs,
duplicate IDs, missing retrieval output, or a source count other than 764 must
fail with a non-zero exit.

- [ ] **Step 5: Run tests and verify GREEN**

```powershell
python -m unittest tests.test_qasper_support_manifest -v
```

- [ ] **Step 6: Commit manifest tooling**

```powershell
git add -- Scripts/eval/qasper_support_manifest.py tests/test_qasper_support_manifest.py
git commit -m "feat: add locked Qasper support tuning split"
```

### Task 5: Add reranker-cache and replay evaluation tooling

**Files:**
- Create: `Scripts/eval/qasper_support_replay.py`
- Create: `tests/test_qasper_support_replay.py`

- [ ] **Step 1: Write failing cache-fingerprint tests**

Define and test this interface:

```python
fingerprint = support_rerank_fingerprint(
    question=question,
    draft_answer=draft_answer,
    candidate_payload=candidates,
    model_name="BAAI/bge-reranker-v2-m3",
    api_base="https://api.siliconflow.cn/v1",
    controller_config={"answer_conditioned_support_topk": 20},
)
```

Assert that changing question, draft answer, candidate order, candidate text,
model name, endpoint identity, or relevant config changes the fingerprint.
Assert that credentials are absent from serialized cache content.

- [ ] **Step 2: Write failing replay-integrity tests**

Use fixtures with frozen `result.json`, `retrieval_res.json`, and index payloads.
Verify:

- incomplete scores fail;
- fingerprint mismatch fails;
- current and new controllers consume the same ordered scores;
- replay preserves the draft answer exactly;
- no LLM provider is constructed;
- reranker is called once on cache miss and zero times on cache hit;
- final IDs are always inside the eligible candidate/anchor whitelist.

- [ ] **Step 3: Run tests and verify RED**

```powershell
python -m unittest tests.test_qasper_support_replay -v
```

Expected: import failure because the replay module is absent.

- [ ] **Step 4: Implement candidate reconstruction and safe cache writing**

Expose `reconstruct_support_candidates(retrieval, index_payload, topk=20)`,
`support_rerank_fingerprint` with the arguments from Step 1,
`load_validated_score_cache(path, expected_fingerprint)`,
`build_score_cache_entry(reranker, replay_case, provider_identity)`, and
`replay_controller(case, scores, policy, thresholds)`.

Persist raw scores, normalized scores, rank, candidate ID, candidate-text hash,
and the fingerprint. Store endpoint origin/path only; never store headers,
tokens, or API keys. Write cache files under `runs/`.

- [ ] **Step 5: Implement threshold-grid and paired reporting**

Evaluate the fixed grid:

```python
MIN_RELEVANCE_GRID = (0.35, 0.50, 0.65)
REDUNDANCY_GRID = (0.65, 0.75, 0.85)
```

For every pair, write Answer F1, Evidence P/R/F1, base Recall@5/@10/@20,
triggered support-rerank Recall@5/@10/@20, evidence-count distribution,
stopping reasons, and paired-bootstrap intervals. Select the best passing pair
by Evidence F1, then Precision, then lower token cost, then lexicographic
threshold order. If no pair passes all replay gates, return non-zero.

- [ ] **Step 6: Run replay tests and verify GREEN**

```powershell
python -m unittest tests.test_qasper_support_replay tests.test_paired_bootstrap -v
```

- [ ] **Step 7: Commit replay tooling**

```powershell
git add -- Scripts/eval/qasper_support_replay.py tests/test_qasper_support_replay.py
git commit -m "feat: add deterministic Qasper support replay"
```

### Task 6: Generate the real split and run replay100

**Files:**
- Read: `runs/_datasets/qasper/processed/qasper_validation_recoverable_764.json`
- Read: `runs/qasper_evibridge/work_validation_full1005_support_controller_db9082b8e/`
- Read: `runs/_configs/evibridge_qasper_full1005_frozen_db9082b8e.yaml`
- Write only under: `runs/_datasets/qasper/processed/` and `runs/qasper_evibridge/work_support_pruning_replay_seed42_100/`

- [ ] **Step 1: Generate the tuning100 and holdout664 split**

Run the manifest CLI against the original 764 dataset and completed full1005
work directory. Expected: 100 tuning questions, 664 holdout questions, answer
quotas 45/25/15/15, at least 30 comparison and eight multi-hop questions, no
overlap, and a valid source hash.

- [ ] **Step 2: Build the score cache once**

Run replay with the frozen private reranker configuration. Expected: only
controller-triggered questions call the SiliconFlow reranker; no LLM request is
made; each cache entry has 20 scores or the exact available eligible count.

- [ ] **Step 3: Evaluate the two-threshold grid**

Run the grid entirely from cache. Save each configuration's per-question and
aggregate results. Expected gate:

- Evidence F1 delta at least +0.020;
- Evidence Precision delta at least +0.030;
- Evidence Recall delta at least -0.015;
- Answer F1 delta exactly 0;
- zero illegal final citations and verifier contradictions;
- zero additional LLM calls and no token increase.

- [ ] **Step 4: Stop or freeze**

If no grid point passes, stop before fresh inference and report the failure
distribution. If a point passes, write a frozen replay decision JSON containing
the thresholds, source commit, config hash, manifest hash, score-cache hash, and
metric deltas.

### Task 7: Fresh100 and conditional holdout evaluation

**Files:**
- Create private runtime config under: `runs/_configs/`
- Create tuning dataset YAML under: `runs/_datasets/qasper/`
- Write inference output under: `runs/qasper_evibridge/work_support_pruned_fresh_seed42_100/`

- [ ] **Step 1: Create private fresh100 configuration**

Copy the frozen full1005 private config under `runs/_configs/`, change only the
policy, accepted thresholds, `regenerate_on_support_expansion: false`, and the
new method suffix inputs. Preserve model, prompt, provider, retrieval, verifier,
and token settings.

- [ ] **Step 2: Run fresh end-to-end inference with progress reporting**

Run `main.py rag` over the locked tuning100 dataset. Report completed
questions/documents, errors, phase, and ETA during execution. Do not mix outputs
from the completed full1005 method directory.

- [ ] **Step 3: Validate and evaluate fresh100**

Run strict coverage validation, local official evaluation, external official
evaluation, Recall@k diagnostics, and paired bootstrap. Required gate:

- Evidence F1 delta at least +0.015;
- Answer F1 delta no worse than -0.005;
- 100/100 question coverage;
- zero illegal final citations and verifier contradictions;
- no controller-induced answer regeneration;
- token cost per question does not increase.

- [ ] **Step 4: Evaluate holdout664 once only if fresh100 passes**

Freeze the source commit and configuration first. Run inference/evaluation on
holdout664 once. Do not inspect per-question holdout outcomes for tuning.

- [ ] **Step 5: Produce the fair 764 result**

Combine tuning100 and holdout664 per-question results, verify exact 764-question
coverage, and generate Answer F1, Evidence P/R/F1, Recall@5/@10/@20, cost,
latency, and paired-bootstrap 95% confidence intervals. Require the Evidence F1
delta lower bound to be greater than zero.

- [ ] **Step 6: Decide whether the 1005 baseline campaign is authorized**

If all gates pass, freeze code/config/prompt/model/evaluator hashes and prepare
the separate 1005 baseline execution plan. If any gate fails, do not run the
1005 baselines and report the aggregate failure type without tuning against
holdout questions.

### Task 8: Final code verification

**Files:**
- Verify all tracked files from Tasks 1-5

- [ ] **Step 1: Run focused tests**

```powershell
python -m unittest tests.test_evibridge_support_pruner tests.test_evibridge_modules tests.test_evibridge_wiring tests.test_qasper_official_eval tests.test_qasper_support_manifest tests.test_qasper_support_replay tests.test_paired_bootstrap -v
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the complete repository test suite**

```powershell
python -m unittest discover -s tests -v
```

Expected: all available tests pass with zero failures and zero errors.

- [ ] **Step 3: Check tracked changes and ignored artifacts**

```powershell
git diff --check
git status --short
git check-ignore runs/qasper_evibridge/work_support_pruning_replay_seed42_100
```

Expected: no whitespace errors, no uncommitted tracked code changes after final
commits, and all runtime artifacts ignored.
