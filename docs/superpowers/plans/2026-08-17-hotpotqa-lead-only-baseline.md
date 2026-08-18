# HotpotQA Lead-only Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a valid HotpotQA `lead_only` weak-information baseline, rerun the fixed-1000 experiment, and save its main metrics and online efficiency.

**Architecture:** Extend the existing vanilla RAG strategy with a tree-backed retrieval mode that emits the first non-empty sentence under every HotpotQA title. Reuse the existing answer generator, citation validation, evaluator, coverage audit, token accounting, and result directory conventions; isolate all new outputs under the `lead_only` suffix.

**Tech Stack:** Python 3, Pydantic, unittest/pytest, PowerShell, existing HotpotQA evaluator and experiment scripts.

---

### Task 1: Specify Lead-only retrieval behavior with failing tests

**Files:**
- Modify: `tests/test_bm25_baseline.py`

- [ ] **Step 1: Add a failing retrieval test**

Add `test_vanilla_hotpot_lead_only_returns_sentence_zero_for_every_title`. Construct a `DocumentTree` containing two title nodes, two non-empty text children under each title, and assert that `_retrieve()` returns exactly the two first text nodes in tree order. Assert the returned metadata includes `source=lead_only`, the original integer `node_id`, `hotpot_title`, and `hotpot_sent_id=0`.

```python
def test_vanilla_hotpot_lead_only_returns_sentence_zero_for_every_title(self):
    from Core.Index.Tree import DocumentTree, NodeType, TreeNode
    from Core.rag.vanilla_rag import VanillaRAG

    with tempfile.TemporaryDirectory() as tmp:
        tree = DocumentTree(
            meta_dict={"file_name": "hotpot.json", "file_path": "hotpotqa://question-1"},
            cfg=SimpleNamespace(save_path=tmp),
        )
        expected = []
        for title_text, sentences in [
            ("Article A", ["A lead.", "A detail."]),
            ("Article B", ["B lead.", "B detail."]),
        ]:
            title = TreeNode({"content": title_text, "page_idx": 0, "pdf_id": 0})
            title.type = NodeType.TITLE
            tree.add_node(title)
            tree.root_node.add_child(title)
            for sent_id, sentence in enumerate(sentences):
                node = TreeNode({"content": sentence, "page_idx": 0, "pdf_id": sent_id})
                node.type = NodeType.TEXT
                tree.add_node(node)
                title.add_child(node)
                if sent_id == 0:
                    expected.append(node.index_id)

        rag = VanillaRAG(
            config=SimpleNamespace(retrieval_method="lead_only", topk=1, answer_style="short"),
            llm=SimpleNamespace(config=SimpleNamespace(max_tokens=1000)),
            tree_index=tree,
        )
        results = rag._retrieve("question", top_k=1)

    self.assertEqual([item["id"] for item in results], expected)
    self.assertEqual([item["metadata"]["hotpot_title"] for item in results], ["Article A", "Article B"])
    self.assertTrue(all(item["metadata"]["hotpot_sent_id"] == 0 for item in results))
    self.assertTrue(all(item["metadata"]["source"] == "lead_only" for item in results))
```

- [ ] **Step 2: Add a failing resource-loader test**

Duplicate the existing model-independent `abstract_only` loader test with `retrieval_method="lead_only"` and assert the returned dependencies contain `tree_index` without importing ModelScope.

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```powershell
python -m pytest tests/test_bm25_baseline.py -k "lead_only" -q
```

Expected: failure because `lead_only` is not yet dispatched and accepted by the resource loader/config.

### Task 2: Implement the minimal Lead-only strategy

**Files:**
- Modify: `Core/configs/rag/vanilla_config.py`
- Modify: `Core/rag/vanilla_rag.py`
- Modify: `Core/utils/resource_loader.py`

- [ ] **Step 1: Add the config discriminator value**

Insert `"lead_only"` beside `"abstract_only"` in `VanillaConfig.retrieval_method`.

- [ ] **Step 2: Dispatch Lead-only retrieval**

Add the following branch in `VanillaRAG._retrieve()` before the full-document branch:

```python
if self.cfg.retrieval_method == "lead_only":
    return self._lead_only_context()
```

- [ ] **Step 3: Implement tree-backed lead selection**

Add `_lead_only_context()` next to `_abstract_only_context()`. It must traverse all tree nodes in their stored order, keep only text nodes whose `_hotpot_sentence_metadata()` reports `hotpot_sent_id == 0`, preserve the original node ID and text, merge section and HotpotQA metadata, and never truncate by `topk`.

```python
def _lead_only_context(self) -> List[Dict[str, Any]]:
    if self.tree_index is None:
        return []
    docs: List[Dict[str, Any]] = []
    for node in self.tree_index.get_nodes(hasRoot=False):
        text = self._node_text(node)
        if not text:
            continue
        hotpot = self._hotpot_sentence_metadata(node)
        if hotpot.get("hotpot_sent_id") != 0:
            continue
        metadata = {
            "source": "lead_only",
            "node_id": node.index_id,
            "source_node_id": node.index_id,
            "paragraph_id": node.index_id,
            "evidence_id": node.index_id,
            "node_type": self._node_type_value(node),
            "block_type": "paragraph",
        }
        metadata.update(self._section_metadata(node))
        metadata.update(hotpot)
        docs.append({"id": node.index_id, "score": 1.0, "content": text, "metadata": metadata})
    return docs
```

- [ ] **Step 4: Load only the tree dependency**

Extend the set at `Core/utils/resource_loader.py` so `lead_only` follows the same lazy tree-only dependency path as `abstract_only`, `full_document`, and `longrag`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_bm25_baseline.py -k "lead_only or abstract_only or hotpot_supporting" -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Run the complete relevant test module**

Run:

```powershell
python -m pytest tests/test_bm25_baseline.py tests/test_hotpotqa_eval.py -q
```

Expected: zero failures.

### Task 3: Prepare an isolated experiment configuration and smoke test

**Files:**
- Create: `runs/_configs/hotpotqa_fixed1000_lead_only.yaml`

- [ ] **Step 1: Create the private run config**

```yaml
extends: hotpotqa_fixed1000_baseline_common.yaml
index_type: vanilla
rag:
  retrieval_method: lead_only
```

- [ ] **Step 2: Validate configuration resolution without printing secrets**

Load the configuration through the project config loader and print only the strategy, retrieval method, dataset size, and working directory basename. Do not print serialized LLM, embedding, reranker, API key, or endpoint fields.

- [ ] **Step 3: Run a one-document smoke inference**

Run one deterministic shard of the fixed dataset:

```powershell
python main.py -c runs/_configs/hotpotqa_fixed1000_lead_only.yaml -d runs/_datasets/hotpotqa/fixed1000/hotpotqa_distractor_validation_fixed1000_seed42.yaml --nsplit 1000 --num 1 rag
```

Expected: one `eval_hotpotqa_lead_only/query_001/result.json`, one `retrieval_res.json`, and non-empty legal lead-sentence IDs.

- [ ] **Step 4: Validate the smoke artifact**

Confirm every ranked result has `hotpot_sent_id=0`, a non-empty `hotpot_title`, and an ID present in that question's `hotpot_node_facts`. Confirm citation validation did not introduce IDs outside the ranked Lead-only candidates.

### Task 4: Run and monitor the fixed-1000 experiment

**Files:**
- Create: `runs/lead_only/run_full_experiment.ps1`
- Create: `runs/lead_only/full_experiment_status.json` at runtime
- Create: `runs/lead_only/full_experiments.stdout.log` at runtime
- Create: `runs/lead_only/full_experiments.stderr.log` at runtime

- [ ] **Step 1: Create a resumable PowerShell launcher**

The launcher must use the existing dataset/config, run two inference shards, wait for both, fail if either exits non-zero, then run the strict audit, official evaluator, and 10,000-sample bootstrap. It must atomically update a JSON status file with `state`, `phase`, `active_pids`, `completed_questions`, `started_at`, and `updated_at`. It must not delete or overwrite any `abstract_only` result.

- [ ] **Step 2: Check for duplicate Lead-only processes**

Inspect process command lines for the `lead_only` config and launcher. Start nothing if an existing matching process is active.

- [ ] **Step 3: Launch the single hidden background process**

Use `Start-Process -WindowStyle Hidden` with stdout/stderr redirected inside `runs/lead_only`.

- [ ] **Step 4: Verify active progress**

Confirm the status is `running`, the PIDs exist, and the number of `eval_hotpotqa_lead_only/query_001/result.json` files increases from the smoke checkpoint.

### Task 5: Evaluate, audit, and record the final metrics and costs

**Files:**
- Create: `runs/lead_only/lead_only_results.json`
- Create: `runs/lead_only/lead_only_results.md`

- [ ] **Step 1: Verify strict coverage**

Run:

```powershell
python runs/_scripts/hotpotqa_unified_audit.py --dataset runs/_datasets/hotpotqa/fixed1000/hotpotqa_distractor_validation_fixed1000_seed42.json --work-root runs/hotpotqa_evibridge/work_validation_fixed1000_seed42 --method lead_only --output runs/lead_only/audit_lead_only.json
```

Expected: 1000 questions, 1000 documents, zero missing predictions, zero invalid supporting IDs, `complete=true`.

- [ ] **Step 2: Run official evaluation**

```powershell
python Eval/evaluation.py -d runs/_datasets/hotpotqa/fixed1000/hotpotqa_distractor_validation_fixed1000_seed42.yaml --method lead_only --max_workers 1
```

Expected: `final_eval_hotpotqa_lead_only.score.json` reports 1000 samples and zero missing predictions.

- [ ] **Step 3: Compute online efficiency from the same run**

Aggregate every `eval_hotpotqa_lead_only/token_cost.json`: sum `rag_cost.total_tokens` and `time`, divide both by exactly 1000, and reject missing/malformed files rather than silently changing the denominator.

- [ ] **Step 4: Validate supporting-fact provenance**

Across all 1000 retrieval files, require every ranked result and every submitted supporting item to have `hotpot_sent_id=0`, a non-empty title, and a node ID mapping to the identical title--sentence pair in the question result.

- [ ] **Step 5: Write the result summaries**

Save Answer EM/F1, SP P/R/F1, Joint F1, Tokens/Question, Seconds/Question, coverage, dataset hash, config path, score path, audit path, and generation timestamp in JSON and Markdown. Values must be copied programmatically from the official score and token files, not transcribed manually.

- [ ] **Step 6: Run final verification**

Run the focused and regression tests again, validate the JSON summaries can be parsed, and verify `git diff` contains no changes to Qasper results, paper files, or existing experiment outputs.

Expected: all tests pass, audit is complete, all provenance checks pass, and the summary values exactly match their source files.
