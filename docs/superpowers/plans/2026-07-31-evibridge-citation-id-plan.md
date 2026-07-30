# EviBridge Citation ID Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove evidence-ordinal ambiguity so answer models cite only persistent EviBridge block IDs.

**Architecture:** Keep the existing answer JSON and strict citation whitelist. Change only the generation prompt's evidence labels and citation instruction, then verify the behavior with unit tests and progressively larger SiliconFlow canaries.

**Tech Stack:** Python, unittest, Pydantic, OpenAI-compatible SiliconFlow API, Qasper official evaluator.

---

### Task 1: Lock the citation-label contract with a failing test

**Files:**
- Modify: `tests/test_evibridge_modules.py`
- Test: `tests/test_evibridge_modules.py`

- [ ] **Step 1: Add the failing assertion**

Extend `test_evibridge_prompt_guides_abstractive_answers_to_concise_synthesis`:

```python
self.assertIn("[block_id=1]", prompt)
self.assertNotIn("[1] block_id=1", prompt)
self.assertIn("Copy the exact integer shown in each [block_id=...] label", prompt)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
python -m unittest tests.test_evibridge_modules.EviBridgeModuleTests.test_evibridge_prompt_guides_abstractive_answers_to_concise_synthesis -v
```

Expected: FAIL because the current prompt contains `[1] block_id=1` and lacks the exact-copy instruction.

### Task 2: Implement the unambiguous block-ID prompt

**Files:**
- Modify: `Core/rag/evibridge_rag.py:291`
- Test: `tests/test_evibridge_modules.py`

- [ ] **Step 1: Replace the evidence label**

Change the evidence rendering expression to:

```python
f"[block_id={item['block_id']}] type={item['block_type']} "
```

- [ ] **Step 2: Strengthen the citation instruction**

Add this sentence immediately after the `supporting_block_ids` constraint:

```python
"Copy the exact integer shown in each [block_id=...] label; never use evidence order numbers. "
```

- [ ] **Step 3: Run the focused test and verify GREEN**

Run the focused unittest from Task 1. Expected: PASS.

- [ ] **Step 4: Run citation-related regression tests**

Run:

```powershell
python -m unittest tests.test_evibridge_modules.EviBridgeModuleTests.test_evibridge_validates_answer_supporting_ids_and_records_diagnostics tests.test_evibridge_modules.EviBridgeModuleTests.test_evibridge_supporting_evidence_prefers_rerank_order tests.test_evibridge_modules.EviBridgeModuleTests.test_evibridge_prompt_guides_abstractive_answers_to_concise_synthesis -v
```

Expected: 3 tests PASS.

### Task 3: Verify progressively on SiliconFlow

**Files:**
- Read: `runs/_configs/evibridge_qasper_canary_private.yaml`
- Write experiment outputs only under: `runs/qasper_evibridge/work_canary_controller_seed42_50/`

- [ ] **Step 1: Run the one-question smoke**

Run:

```powershell
python main.py -c runs/_configs/evibridge_qasper_canary_private.yaml -d runs/_datasets/qasper/qasper_validation_canary_seed42_1.yaml --nsplit 1 --num 1 rag
```

Expected: complete JSON output, API 200, no truncated `"{\n"` output.

- [ ] **Step 2: Build and run a deterministic 10-question subset**

Use the first 10 rows of the existing seed-42 50-question manifest, preserving their order and question IDs. Run the same EviBridge configuration against that dataset.

- [ ] **Step 3: Evaluate the 10-question subset**

Run strict coverage validation and `Scripts/eval/qasper_official.py` with dynamic evidence top-k. Aggregate `citation_validation`, verifier consistency, tokens, and time.

Expected gate:

- zero truncated outputs;
- zero verifier state contradictions;
- invalid citation rate lower than the pre-fix 45.1%;
- Answer F1 does not fall by more than 0.005 relative to the corresponding pre-fix 10 questions.

- [ ] **Step 4: Conditionally rerun 50 questions**

Only if the 10-question gate passes, rerun the existing seed-42 50-question manifest, validate 50/50 coverage, and compare Answer/Evidence F1 with the historical and pre-fix canaries.

### Task 4: Final verification

**Files:**
- Verify: `Core/rag/evibridge_rag.py`
- Verify: `tests/test_evibridge_modules.py`

- [ ] **Step 1: Run the complete test suite**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: all tests PASS.

- [ ] **Step 2: Check the diff**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; only intended tracked source, test, and design/plan files are modified.
