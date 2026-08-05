# EviBridge Two-Dataset Ablation200 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Keep every completed checkbox evidence-backed. Work directly on `dev`; do not create a worktree and do not commit automatically.

**Goal:** Run the eight core EviBridge ablations on frozen, stratified 200-question subsets of Qasper and HotpotQA, then produce reproducible metric tables, individual 95% confidence intervals, and paired deltas against each dataset's completed full EviBridge method.

**Architecture:** A deterministic preparation script freezes the two sample manifests and generates run-local dataset/config files. A resumable PowerShell campaign runs one dataset/variant pair at a time with two inference splits, strict coverage validation, official evaluation, and at most three missing-question retries. Dedicated statistics and table scripts reuse sliced full-method results, compute 10,000-sample bootstrap intervals, and never overwrite the completed 1005/1000-question main-table artifacts.

**Tech Stack:** Python 3, PowerShell, PyYAML, existing `main.py`, Qasper official exporter/evaluator, HotpotQA evaluator, existing EviBridge configuration loader, JSON/Markdown result artifacts.

---

## Fixed experiment contract

- Branch/workspace: modify the current `dev` checkout directly; no worktree and no automatic commit.
- Seed: `42` for sampling and all bootstrap operations.
- Variants, in execution order:
  1. `wo_multi_granularity_seeds`
  2. `wo_context_edges`
  3. `wo_semantic_edges`
  4. `wo_hierarchy_edges`
  5. `wo_typed_weights`
  6. `wo_budgeted_selector`
  7. `wo_sufficiency_verifier`
  8. `static_topk`
- Qasper sample quotas: extractive 105, free-form 53, yes/no 23, unanswerable 19.
- HotpotQA sample quotas: bridge+span 160, comparison+span 27, comparison+yes/no 13.
- Qasper full-method reference: `evibridge_support_pruned` from the completed 1005-question campaign.
- HotpotQA full-method reference: `evibridge` from the completed fixed 1000-question campaign.
- Inference concurrency: at most two splits for one dataset/variant pair; do not run dataset/variant pairs concurrently.
- Evaluation metrics:
  - Qasper: Answer F1, Evidence Precision, Evidence Recall, Evidence F1.
  - HotpotQA: Answer EM, Answer F1, SP Precision, SP Recall, SP F1, Joint F1.
- Confidence intervals: 10,000 bootstrap resamples, 95% percentile interval, seed 42.
- Full methods are sliced to the frozen 200-question manifests and are not rerun.
- No tuning, sample replacement, or variant-specific prompt/config changes after manifests are frozen.

## Task 1: Build and test deterministic sample preparation

**Files:**

- Create: `runs/_scripts/prepare_evibridge_ablation200.py`
- Create: `runs/_scripts/test_prepare_evibridge_ablation200.py`
- Read: `runs/_datasets/qasper/processed/qasper_validation.json`
- Read: `runs/_datasets/hotpotqa/fixed1000/hotpotqa_distractor_validation_fixed1000_seed42.json`
- Generate: `runs/_datasets/ablation200/qasper_ablation200_seed42.json`
- Generate: `runs/_datasets/ablation200/qasper_ablation200_seed42.manifest.json`
- Generate: `runs/_datasets/ablation200/qasper_ablation200_seed42.yaml`
- Generate: `runs/_datasets/ablation200/hotpotqa_ablation200_seed42.json`
- Generate: `runs/_datasets/ablation200/hotpotqa_ablation200_seed42.manifest.json`
- Generate: `runs/_datasets/ablation200/hotpotqa_ablation200_seed42.yaml`

### Step 1: Write failing sampler tests

Test pure functions with synthetic rows before reading local datasets:

- Qasper classification precedence: unanswerable, yes/no, extractive, free-form.
- HotpotQA classification: `type` plus yes/no versus span answer.
- Sampling is deterministic under seed 42 even when the source list order is reproduced.
- Selected rows are restored to source order after per-stratum sampling.
- Quotas must be met exactly or the script raises an error.
- Question IDs and document IDs are unique in the output.

Run:

```powershell
python runs/_scripts/test_prepare_evibridge_ablation200.py
```

Expected: failure because the preparation module does not yet exist.

### Step 2: Implement the sampler and manifest writer

Implementation requirements:

- Sort candidates by stable question ID inside each stratum before applying `random.Random(42).sample`.
- Restore source order for the final dataset JSON.
- Preserve every original record without field rewriting.
- Write UTF-8 JSON with indentation.
- Manifest fields must include source path, source SHA256, output SHA256, seed, total questions, total documents, stratum totals, selected quotas, ordered question IDs, ordered document IDs, creation timestamp, and sampler version.
- Dataset YAML files must keep the existing full-run working roots so ablation outputs can reuse indexes while using unique method suffixes.
- Refuse to overwrite an existing manifest if its selected question IDs or source hash differ unless `--force` is explicitly supplied. Formal execution must not use `--force` after inference starts.

### Step 3: Run tests and generate the frozen datasets

Run:

```powershell
python runs/_scripts/test_prepare_evibridge_ablation200.py
python runs/_scripts/prepare_evibridge_ablation200.py
```

Expected:

- All tests pass.
- Qasper output contains exactly 200 questions and the fixed 105/53/23/19 quotas.
- HotpotQA output contains exactly 200 questions and the fixed 160/27/13 quotas.
- Both manifest files contain source and output SHA256 values.

## Task 2: Generate and verify frozen ablation configurations

**Files:**

- Create: `runs/_scripts/prepare_evibridge_ablation_configs.py`
- Create: `runs/_scripts/test_prepare_evibridge_ablation_configs.py`
- Read: `runs/_configs/evibridge_qasper_support_pruned_seed42_100.yaml`
- Read: `runs/_configs/evibridge_hotpotqa_private.yaml`
- Generate: `runs/_configs/ablation200/qasper/evibridge_<variant>.yaml` for eight variants
- Generate: `runs/_configs/ablation200/hotpotqa/evibridge_<variant>.yaml` for eight variants
- Generate: `runs/_configs/ablation200/config_manifest.json`

### Step 1: Write failing resolved-config equivalence tests

For every generated configuration, resolve `extends` through `Core.configs.system_config._load_raw_config` and compare it with its dataset-specific full-method base. Permit differences only in:

- `rag_force_reprocess`: set to `false`.
- `rag.ablation_variant`: set to the requested variant.
- Derived method suffix when the Pydantic config is instantiated.

Assert that model, API backend/base, prompt-affecting fields, retrieval budgets, typed weights, support controller fields, and generation settings are otherwise identical to the correct dataset base.

Run:

```powershell
python runs/_scripts/test_prepare_evibridge_ablation_configs.py
```

Expected: failure because the generator does not yet exist.

### Step 2: Implement the configuration generator

- Generate minimal YAML files using `extends` and only the two explicit overrides.
- Validate each variant against the allowed eight-value set.
- Instantiate the final system config and verify the method suffix is `evibridge_<variant>`.
- Record base config SHA256, generated file SHA256, resolved-config SHA256, and method suffix in `config_manifest.json`.
- Do not use the generic tracked `config/evibridge_wo_*.yaml` files; they do not inherit the current dataset-specific full methods.

### Step 3: Run tests and freeze configs

Run:

```powershell
python runs/_scripts/test_prepare_evibridge_ablation_configs.py
python runs/_scripts/prepare_evibridge_ablation_configs.py
```

Expected: 16 parseable configs and one manifest with no disallowed resolved-config differences.

## Task 3: Prepare paired full-method reference slices

**Files:**

- Create: `runs/_scripts/prepare_evibridge_ablation200_references.py`
- Create: `runs/_scripts/test_prepare_evibridge_ablation200_references.py`
- Read: `runs/qasper_evibridge/work_validation_full1005_unified_0417263dd/0_results/qasper_official_full1005_evibridge_support_pruned/official_eval_detail.json`
- Read: `runs/hotpotqa_evibridge/work_validation_fixed1000_seed42/0_results/final_eval_hotpotqa_evibridge.json`
- Generate: `runs/ablation200/0_results/qasper/full/official_eval_detail.json`
- Generate: `runs/ablation200/0_results/qasper/full/point_estimates.json`
- Generate: `runs/ablation200/0_results/hotpotqa/full/eval_detail.json`
- Generate: `runs/ablation200/0_results/hotpotqa/full/point_estimates.json`

### Step 1: Write failing slice tests

- Match rows by canonical question ID, never by list index.
- Fail on missing, duplicate, or extra IDs.
- Preserve manifest order.
- Recompute macro metric means from exactly 200 rows.
- Verify metric names and values are finite and within [0, 1].

### Step 2: Implement reference slicing

- Slice the existing completed full-run details; do not invoke either model or evaluator.
- Store source path and source SHA256 beside the point estimates.
- Include the frozen manifest SHA256 and exact ordered IDs.
- Never write into the completed full1005/fixed1000 `0_results` directories.

### Step 3: Verify the reference slices

Run:

```powershell
python runs/_scripts/test_prepare_evibridge_ablation200_references.py
python runs/_scripts/prepare_evibridge_ablation200_references.py
```

Expected: exactly 200 rows per dataset with perfect ID agreement.

## Task 4: Implement strict ablation evaluation and statistics helpers

**Files:**

- Create: `runs/_scripts/evibridge_ablation200_stats.py`
- Create: `runs/_scripts/test_evibridge_ablation200_stats.py`
- Reuse: `Eval/utils/paired_bootstrap.py`
- Reuse: `runs/_scripts/bootstrap_qasper_slice.py`
- Reuse: `runs/_scripts/bootstrap_hotpotqa.py`

### Step 1: Write failing statistics tests

Use small synthetic paired details to assert:

- Per-method bootstrap point/low/high values are deterministic for a fixed seed.
- Paired deltas are candidate minus full method.
- Pairing fails if either side has different IDs.
- Qasper exposes only the four requested metrics.
- HotpotQA exposes only the six requested metrics.
- CI serialization keeps raw numeric values; Markdown formatting is handled later.

### Step 2: Implement the statistics helper

For each dataset and variant:

- Read the dedicated 200-row detail output.
- Verify exact manifest ID equality.
- Compute point estimates and 10,000-resample individual 95% CIs.
- Compute 10,000-resample paired delta 95% CIs against the full-method slice.
- Write:
  - `bootstrap_95ci.json`
  - `paired_delta_vs_full_95ci.json`
  - `metrics.json`
- Add source/detail/config/manifest SHA256 values to the metadata.

### Step 3: Run statistics unit tests

```powershell
python runs/_scripts/test_evibridge_ablation200_stats.py
```

Expected: all tests pass with deterministic intervals and strict pairing checks.

## Task 5: Implement a resumable two-dataset campaign

**Files:**

- Create: `runs/_scripts/evibridge_ablation200_campaign.ps1`
- Create: `runs/_scripts/test_evibridge_ablation200_campaign.ps1`
- Reuse: `Scripts/eval/qasper_run_validator.py`
- Reuse: `Scripts/eval/qasper_official.py`
- Reuse: `runs/_scripts/hotpotqa_unified_audit.py`
- Generate: `runs/ablation200/ablation200_campaign_status.json`
- Generate: `runs/_logs/ablation200/<dataset>/<variant>/...`

### Step 1: Add dry-run campaign tests

The script must support `-DryRun` and emit the exact command graph without starting models. Test that:

- Variant order matches the fixed contract.
- Each variant runs Qasper before HotpotQA.
- Each inference attempt starts split 1 and split 2 only.
- Strict validation precedes evaluation.
- Existing complete output skips inference but still verifies hashes and coverage.
- Missing coverage retries only missing questions, up to three attempts.
- A failed audit stops the campaign before the next dataset/variant.
- Completed full-method outputs are never selected as write targets.

### Step 2: Implement status and liveness-safe orchestration

Status JSON fields:

- state/phase/current dataset/current variant/current attempt.
- completed dataset-variant pairs out of 16.
- completed questions/documents and error count.
- active process IDs.
- sample/config hashes.
- start/update/finish timestamps and failure reason.

Inference command pattern:

```powershell
python main.py -c <variant-config> -d <dataset-200-yaml> --nsplit 2 --num <1-or-2> rag
```

Qasper validation/evaluation pattern:

```powershell
python Scripts/eval/qasper_run_validator.py --dataset-config <qasper-200-yaml> --method evibridge_<variant>
python Scripts/eval/qasper_official.py --dataset-config <qasper-200-yaml> --method evibridge_<variant> --answer-source output --dynamic-evidence-topk --output-dir <dedicated-ablation-result-dir>
```

HotpotQA validation/evaluation pattern:

```powershell
python runs/_scripts/hotpotqa_unified_audit.py --dataset <hotpotqa-200-json> --work-root <fixed1000-work-root> --method evibridge_<variant> --output <dedicated-audit-json>
python Eval/evaluation.py -d <hotpotqa-200-yaml> --method evibridge_<variant> --max_workers 1
```

After HotpotQA evaluation, copy the unique variant detail/score into its dedicated ablation result directory and record the source SHA256. The full-method `final_eval_hotpotqa_evibridge.json` must never be overwritten.

### Step 3: Add retry and failure behavior

- Keep `rag_force_reprocess: false`; normal reruns process only absent query outputs.
- After each attempt, audit against the frozen 200 IDs.
- Retry incomplete coverage no more than three total attempts.
- Treat nonzero inference exits, missing/duplicate outputs, manifest/config hash drift, or evaluation row mismatch as fatal.
- Preserve logs and status on failure; do not silently continue.

### Step 4: Run the dry-run tests

```powershell
powershell -ExecutionPolicy Bypass -File runs/_scripts/test_evibridge_ablation200_campaign.ps1
powershell -ExecutionPolicy Bypass -File runs/_scripts/evibridge_ablation200_campaign.ps1 -DryRun
```

Expected: command graph covers exactly 16 dataset-variant pairs and performs no model calls.

## Task 6: Implement final Markdown/JSON table generation

**Files:**

- Create: `runs/_scripts/evibridge_ablation200_tables.py`
- Create: `runs/_scripts/test_evibridge_ablation200_tables.py`
- Generate: `runs/ablation200/0_results/qasper/qasper_ablation200_core8.md`
- Generate: `runs/ablation200/0_results/qasper/qasper_ablation200_core8.json`
- Generate: `runs/ablation200/0_results/hotpotqa/hotpotqa_ablation200_core8.md`
- Generate: `runs/ablation200/0_results/hotpotqa/hotpotqa_ablation200_core8.json`
- Generate: `runs/ablation200/0_results/evibridge_ablation200_cross_dataset_summary.md`

### Step 1: Write failing table tests

- Each dataset table contains the full method plus exactly eight ablations.
- Qasper has exactly four metric columns; HotpotQA has exactly six.
- Every cell includes point estimate and individual 95% CI.
- Paired-delta sections include delta and paired 95% CI versus full.
- Method order is fixed and never sorted by observed score.
- Cross-dataset summary reports paired Evidence F1 delta for Qasper and paired Joint F1 delta for HotpotQA, plus a clear note that the samples are frozen 200-question subsets.

### Step 2: Implement strict table generation

- Refuse to build tables until all 16 pairs have complete status, 200 detail rows, valid bootstrap files, and matching hashes.
- Use four decimal places in Markdown and retain full precision in JSON.
- Include paths/hashes, sample composition, model/config provenance, and bootstrap settings below each table.
- Do not rank Qasper and HotpotQA methods in one combined table.

### Step 3: Run table tests

```powershell
python runs/_scripts/test_evibridge_ablation200_tables.py
```

Expected: all schema, order, count, and formatting assertions pass.

## Task 7: Run a minimal end-to-end smoke check

**Files:**

- Generate under: `runs/ablation200/smoke/`

### Step 1: Run all local unit/dry-run tests

```powershell
python runs/_scripts/test_prepare_evibridge_ablation200.py
python runs/_scripts/test_prepare_evibridge_ablation_configs.py
python runs/_scripts/test_prepare_evibridge_ablation200_references.py
python runs/_scripts/test_evibridge_ablation200_stats.py
python runs/_scripts/test_evibridge_ablation200_tables.py
powershell -ExecutionPolicy Bypass -File runs/_scripts/test_evibridge_ablation200_campaign.ps1
```

### Step 2: Smoke one variant on two questions per dataset

- Use `wo_typed_weights`, chosen because it exercises the ablation suffix without changing index dependencies.
- Build temporary two-question dataset YAMLs from the already frozen manifests.
- Run both splits, strict validators, evaluators, and statistics generation.
- Verify output suffixes are unique and no full-run main-table file timestamp/hash changes.
- Delete no material artifacts; keep smoke outputs under `runs/ablation200/smoke/` for diagnosis.

Expected: both datasets produce complete two-row details and all requested metric keys.

## Task 8: Launch and monitor the formal campaign

### Step 1: Record preflight provenance

Before the first model call, write `runs/ablation200/preflight.json` containing:

- current git branch and commit.
- dirty-worktree file list without file contents.
- two dataset manifests and SHA256 values.
- 16 config files and resolved hashes.
- source full-method detail hashes.
- Python version, model names, API bases with credentials redacted, and campaign script hash.

### Step 2: Launch the campaign

```powershell
powershell -ExecutionPolicy Bypass -File runs/_scripts/evibridge_ablation200_campaign.ps1
```

Run hidden/background only after smoke succeeds. Capture stdout/stderr in `runs/_logs/ablation200/`. Do not start another copy if status or process inspection shows a live campaign.

### Step 3: Report progress

Progress reports must include:

- completed dataset-variant pairs out of 16.
- current dataset/variant/attempt.
- completed questions and documents out of 200.
- active split/campaign process state.
- error count.
- recent-throughput ETA.

## Task 9: Final verification and handoff

### Step 1: Verify experiment integrity

Run strict checks that:

- both frozen datasets remain hash-identical to preflight.
- all 16 configs remain hash-identical to preflight.
- all 16 candidate detail files contain exactly the 200 manifest IDs.
- no final references are illegal and no evaluator output is missing.
- every bootstrap file uses 10,000 samples and seed 42.
- every paired comparison has exactly 200 matched rows.
- the completed Qasper 1005 and HotpotQA 1000 main-table/reference artifacts retain their preflight hashes.

### Step 2: Build final tables

```powershell
python runs/_scripts/evibridge_ablation200_tables.py
```

Expected: Qasper and HotpotQA each contain one full row plus eight ablation rows, and the cross-dataset summary is generated.

### Step 3: Report results without committing

Provide clickable paths to the three Markdown reports, summarize the largest paired degradations and any non-significant changes, identify failed/unstable variants if any, and explicitly state that no commit was created.

