# CoSE-RAG Closed-Loop and Retrieval-Trajectory Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct auditable closed-loop retrieval statistics and two median-gain trajectory cases from the completed CoSE-RAG runs, then insert only the two approved experimental subsections into the latest paper.

**Architecture:** A pure analysis module reads the immutable Qasper and HotpotQA result/retrieval files, maps per-round selected atomic evidence to the datasets' official evidence units, validates all closed-loop invariants, and writes deterministic JSONL/CSV/JSON outputs. A separate paper renderer consumes only those outputs, creates a backup of the latest TeX source, inserts the generated table and case analysis at the approved anchor, and leaves all existing text unchanged.

**Tech Stack:** Python 3, standard library (`argparse`, `csv`, `hashlib`, `json`, `pathlib`, `statistics`), existing Qasper/HotpotQA evaluation utilities, `unittest`, XeLaTeX.

---

## File map

- Create `Scripts/analysis/closed_loop_analysis.py`: log discovery, per-round evidence mapping, metrics, invariant checks, deterministic case selection, JSONL/CSV/JSON writers.
- Create `Scripts/analysis/render_closed_loop_paper.py`: LaTeX escaping, table/case rendering, anchor-checked insertion and source backup.
- Create `tests/test_closed_loop_analysis.py`: isolated tests for evidence filtering/mapping, metrics, invariants, median-gain selection and deterministic serialization.
- Create `tests/test_closed_loop_paper.py`: LaTeX escaping, insertion-anchor uniqueness and preservation of unrelated text.
- Modify `D:/Else/Study/godot/素材/压缩包/CoSE-RAG_updated_efficiency.tex`: add only the two approved subsections after the ablation analysis and before parameter sensitivity.
- Create beside the paper: `closed_loop_query_logs.jsonl`, `closed_loop_behavior_summary.csv`, `retrieval_trajectory_cases.json`.

No Git commit is performed because the repository instructions explicitly require the user to handle commits.

### Task 1: Parse one query into an auditable round record

**Files:**
- Create: `Scripts/analysis/closed_loop_analysis.py`
- Test: `tests/test_closed_loop_analysis.py`

- [ ] **Step 1: Write failing tests for atomic/auxiliary filtering and round extraction**

Add tests using minimal in-memory payloads:

```python
import unittest

from Scripts.analysis.closed_loop_analysis import (
    atomic_items,
    build_query_log,
    context_token_count,
)


class ClosedLoopQueryTest(unittest.TestCase):
    def test_atomic_items_exclude_auxiliary_nodes(self):
        selected = [
            {"block_id": 1, "block_type": "paragraph", "text": "alpha"},
            {"block_id": 2, "block_type": "entity", "text": "bridge"},
            {"block_id": 3, "block_type": "patch", "text": "context"},
        ]
        self.assertEqual([1], [row["block_id"] for row in atomic_items(selected)])

    def test_round_two_log_records_only_new_atomic_blocks(self):
        retrieval = {
            "iterations": [
                {
                    "iteration": 1,
                    "selected": [
                        {"block_id": 1, "block_type": "paragraph", "text": "first"},
                        {"block_id": 8, "block_type": "entity", "text": "aux"},
                    ],
                    "new_answer_candidate_ids": [1],
                    "verification": {
                        "sufficient": False,
                        "missing_types": ["fact"],
                        "missing_bridge_types": ["semantic"],
                        "next_action": "expand",
                        "next_bridge": ["semantic"],
                    },
                },
                {
                    "iteration": 2,
                    "selected": [
                        {"block_id": 1, "block_type": "paragraph", "text": "first"},
                        {"block_id": 2, "block_type": "paragraph", "text": "second"},
                    ],
                    "new_answer_candidate_ids": [2],
                    "verification": {
                        "sufficient": True,
                        "missing_types": [],
                        "missing_bridge_types": [],
                        "next_action": "accept",
                        "next_bridge": [],
                    },
                },
            ],
            "stopping_reason": "sufficient",
        }
        result = {"question": "q", "qasper_question_id": "qid", "answer": []}
        row = build_query_log("qasper", result, retrieval)
        self.assertEqual([2], row["second_round_new_atomic_block_ids"])
        self.assertEqual(1, row["first_round_auxiliary_count"])
        self.assertTrue(row["y_final"])

    def test_context_token_count_matches_selector_counter(self):
        self.assertEqual(
            context_token_count([{"text": "alpha beta"}, {"text": "gamma"}]),
            sum(context_token_count([{"text": text}]) for text in ("alpha beta", "gamma")),
        )
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```powershell
python -m unittest tests.test_closed_loop_analysis.ClosedLoopQueryTest -v
```

Expected: import failure because `Scripts.analysis.closed_loop_analysis` does not yet exist.

- [ ] **Step 3: Implement the round data model and token-count adapter**

Create the module with explicit constants and deterministic helpers:

```python
from __future__ import annotations

from typing import Any, Iterable, Mapping

from Core.rag.evibridge_selector import _token_cost

ATOMIC_TYPES = frozenset({"paragraph", "table", "figure", "caption", "equation"})


def atomic_items(selected: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in selected
        if str(item.get("block_type", "")).lower() in ATOMIC_TYPES
    ]


def context_token_count(selected: Iterable[Mapping[str, Any]]) -> int:
    return sum(_token_cost(str(item.get("text") or "")) for item in selected)


def _round_summary(round_payload: Mapping[str, Any]) -> dict[str, Any]:
    selected = [dict(item) for item in round_payload.get("selected") or []]
    atomic = atomic_items(selected)
    verification = dict(round_payload.get("verification") or {})
    return {
        "iteration": int(round_payload["iteration"]),
        "sufficient": bool(verification.get("sufficient")),
        "selected_atomic_ids": sorted(int(item["block_id"]) for item in atomic),
        "selected_atomic_count": len(atomic),
        "selected_auxiliary_count": len(selected) - len(atomic),
        "context_tokens": context_token_count(selected),
        "missing_types": list(verification.get("missing_types") or []),
        "missing_bridge_types": list(verification.get("missing_bridge_types") or []),
        "next_bridge": list(verification.get("next_bridge") or []),
        "next_action": str(verification.get("next_action") or round_payload.get("next_action") or ""),
        "new_answer_candidate_ids": sorted(
            int(value) for value in round_payload.get("new_answer_candidate_ids") or []
        ),
        "selected": selected,
        "connector_paths": list(round_payload.get("connector_paths") or []),
        "connector_edges": list(round_payload.get("connector_edges") or []),
    }


def build_query_log(
    dataset: str,
    result: Mapping[str, Any],
    retrieval: Mapping[str, Any],
) -> dict[str, Any]:
    rounds = [_round_summary(item) for item in retrieval.get("iterations") or []]
    if not rounds or len(rounds) > 2:
        raise ValueError(f"invalid iteration count: {len(rounds)}")
    first, final = rounds[0], rounds[-1]
    first_ids = set(first["selected_atomic_ids"])
    final_ids = set(final["selected_atomic_ids"])
    question_id = result.get("qasper_question_id") or result.get("hotpotqa_question_id")
    return {
        "dataset": dataset,
        "question_id": str(question_id),
        "doc_uuid": str(result.get("doc_uuid") or ""),
        "question": str(result.get("question") or ""),
        "actual_rounds": len(rounds),
        "y1": first["sufficient"],
        "second_round_triggered": len(rounds) == 2,
        "y2": final["sufficient"] if len(rounds) == 2 else None,
        "y_final": final["sufficient"],
        "stopping_reason": str(retrieval.get("stopping_reason") or ""),
        "first_round_atomic_count": first["selected_atomic_count"],
        "final_atomic_count": final["selected_atomic_count"],
        "first_round_auxiliary_count": first["selected_auxiliary_count"],
        "final_auxiliary_count": final["selected_auxiliary_count"],
        "first_round_context_tokens": first["context_tokens"],
        "final_context_tokens": final["context_tokens"],
        "second_round_new_atomic_count": len(final_ids - first_ids),
        "second_round_new_atomic_block_ids": sorted(final_ids - first_ids),
        "missing_requirements": first["missing_types"],
        "suggested_bridge_types": first["missing_bridge_types"] or first["next_bridge"],
        "controlled_action": first["next_action"],
        "rounds": rounds,
    }
```

- [ ] **Step 4: Run the focused tests and verify they pass**

Run the same unittest command. Expected: all three tests pass.

### Task 2: Map round evidence and compute official evidence metrics

**Files:**
- Modify: `Scripts/analysis/closed_loop_analysis.py`
- Modify: `tests/test_closed_loop_analysis.py`

- [ ] **Step 1: Add failing metric tests**

Add tests covering exact Hotpot title--sentence matching and Qasper multi-reference selection:

```python
from Scripts.analysis.closed_loop_analysis import (
    hotpot_support_metrics,
    qasper_evidence_metrics,
)


class EvidenceMetricTest(unittest.TestCase):
    def test_hotpot_support_metrics_use_title_sentence_pairs(self):
        metrics = hotpot_support_metrics(
            predicted={("A", 0), ("B", 1)},
            gold={("A", 0), ("C", 2)},
        )
        self.assertEqual({"precision": 0.5, "recall": 0.5, "f1": 0.5}, metrics)

    def test_qasper_uses_one_best_reference_for_all_three_metrics(self):
        metrics = qasper_evidence_metrics(
            predicted={"p1", "p2"},
            gold_groups=[{"p1"}, {"p1", "p2", "p3"}],
        )
        self.assertAlmostEqual(1.0, metrics["precision"])
        self.assertAlmostEqual(2 / 3, metrics["recall"])
        self.assertAlmostEqual(0.8, metrics["f1"])
```

- [ ] **Step 2: Verify the new tests fail**

Run:

```powershell
python -m unittest tests.test_closed_loop_analysis.EvidenceMetricTest -v
```

Expected: missing metric functions.

- [ ] **Step 3: Implement mapping and metric helpers**

Use selected-item metadata without fuzzy text matching:

```python
def prf(predicted: set[Any], gold: set[Any]) -> dict[str, float]:
    if not predicted and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    overlap = len(predicted & gold)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def hotpot_support_metrics(predicted: set[tuple[str, int]], gold: set[tuple[str, int]]) -> dict[str, float]:
    return prf(predicted, gold)


def qasper_evidence_metrics(
    predicted: set[str],
    gold_groups: list[set[str]],
) -> dict[str, float]:
    candidates = [prf(predicted, group) for group in gold_groups] or [prf(predicted, set())]
    return max(candidates, key=lambda row: (row["f1"], row["precision"], row["recall"]))


def hotpot_pair(item: Mapping[str, Any]) -> tuple[str, int]:
    metadata = item.get("metadata") or {}
    title = item.get("hotpot_title") or metadata.get("hotpot_title")
    sentence_id = item.get("hotpot_sent_id")
    if sentence_id is None:
        sentence_id = metadata.get("hotpot_sent_id")
    if title is None or sentence_id is None:
        raise ValueError(f"unmapped HotpotQA atomic block {item.get('block_id')}")
    return str(title), int(sentence_id)


def qasper_paragraph_key(item: Mapping[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    value = metadata.get("qasper_paragraph_id")
    if value is None:
        value = item.get("source_node_id") or item.get("block_id")
    if value is None:
        raise ValueError("unmapped Qasper atomic block")
    return str(value)
```

Extend `build_query_log` to calculate first/final metrics from result gold fields and attach per-block source maps containing doc UUID, section, page, paragraph ID or title--sentence pair, and text.

- [ ] **Step 4: Verify metric and query tests pass**

Run:

```powershell
python -m unittest tests.test_closed_loop_analysis -v
```

Expected: all current tests pass.

### Task 3: Aggregate behavior and enforce invariants

**Files:**
- Modify: `Scripts/analysis/closed_loop_analysis.py`
- Modify: `tests/test_closed_loop_analysis.py`

- [ ] **Step 1: Add failing aggregation and invariant tests**

```python
from Scripts.analysis.closed_loop_analysis import aggregate_dataset, validate_dataset


class AggregateTest(unittest.TestCase):
    def test_trigger_subset_denominators_are_explicit(self):
        rows = [
            {"y1": True, "second_round_triggered": False, "y_final": True, "actual_rounds": 1},
            {
                "y1": False,
                "second_round_triggered": True,
                "y2": True,
                "y_final": True,
                "actual_rounds": 2,
                "second_round_new_atomic_count": 2,
                "first_round_evidence_metrics": {"f1": 0.25},
                "final_evidence_metrics": {"f1": 0.75},
            },
        ]
        summary = aggregate_dataset("qasper", rows)
        self.assertEqual((1, 2), (summary["first_accept_n"], summary["question_count"]))
        self.assertEqual(1, summary["trigger_subset_n"])
        self.assertEqual(0.5, summary["evidence_f1_absolute_gain"])

    def test_validate_rejects_new_block_outside_round_two_candidates(self):
        row = {
            "question_id": "q",
            "actual_rounds": 2,
            "y1": False,
            "second_round_triggered": True,
            "y_final": True,
            "second_round_new_atomic_block_ids": [9],
            "rounds": [{}, {"new_answer_candidate_ids": [8]}],
            "source_mapping_valid": True,
        }
        with self.assertRaisesRegex(ValueError, "round-two candidates"):
            validate_dataset("qasper", [row], expected_count=1)
```

- [ ] **Step 2: Verify the tests fail**

Run `python -m unittest tests.test_closed_loop_analysis.AggregateTest -v`.

- [ ] **Step 3: Implement aggregation and validation**

Implement counts before percentages and reject inconsistencies:

```python
def validate_dataset(dataset: str, rows: list[dict[str, Any]], expected_count: int) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"{dataset}: expected {expected_count}, found {len(rows)}")
    ids = [row["question_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{dataset}: duplicate question_id")
    for row in rows:
        if row["actual_rounds"] not in (1, 2):
            raise ValueError(f"{row['question_id']}: invalid round count")
        if row["second_round_triggered"] != (row["actual_rounds"] == 2):
            raise ValueError(f"{row['question_id']}: trigger/round mismatch")
        if row["second_round_triggered"] and row["y1"]:
            raise ValueError(f"{row['question_id']}: sufficient round one expanded")
        if row["second_round_triggered"]:
            candidates = set(row["rounds"][1]["new_answer_candidate_ids"])
            added = set(row["second_round_new_atomic_block_ids"])
            if not added <= candidates:
                raise ValueError(f"{row['question_id']}: new block outside round-two candidates")
        if not row.get("source_mapping_valid"):
            raise ValueError(f"{row['question_id']}: unmapped evidence")


def aggregate_dataset(dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    triggered = [row for row in rows if row["second_round_triggered"]]
    first_accept = sum(bool(row["y1"]) for row in rows)
    expanded_accept = sum(bool(row.get("y2")) for row in triggered)
    final_insufficient = sum(not bool(row["y_final"]) for row in rows)
    first_f1 = sum(row["first_round_evidence_metrics"]["f1"] for row in triggered) / len(triggered)
    final_f1 = sum(row["final_evidence_metrics"]["f1"] for row in triggered) / len(triggered)
    return {
        "dataset": dataset,
        "question_count": len(rows),
        "first_accept_n": first_accept,
        "first_accept_rate": first_accept / len(rows),
        "second_round_trigger_n": len(triggered),
        "second_round_trigger_rate": len(triggered) / len(rows),
        "expanded_accept_n": expanded_accept,
        "expanded_accept_rate": expanded_accept / len(triggered),
        "final_insufficient_n": final_insufficient,
        "final_insufficient_rate": final_insufficient / len(rows),
        "mean_rounds": sum(row["actual_rounds"] for row in rows) / len(rows),
        "trigger_subset_n": len(triggered),
        "mean_new_atomic_blocks": sum(row["second_round_new_atomic_count"] for row in triggered) / len(triggered),
        "first_round_evidence_f1": first_f1,
        "final_evidence_f1": final_f1,
        "evidence_f1_absolute_gain": final_f1 - first_f1,
    }
```

Also verify `first_accept_n + second_round_trigger_n == question_count`, `expanded_accept_n <= second_round_trigger_n`, and final-insufficient counts grouped by stopping reason.

- [ ] **Step 4: Run all analysis unit tests**

Run `python -m unittest tests.test_closed_loop_analysis -v`. Expected: PASS.

### Task 4: Deterministic case selection and baseline merge

**Files:**
- Modify: `Scripts/analysis/closed_loop_analysis.py`
- Modify: `tests/test_closed_loop_analysis.py`

- [ ] **Step 1: Add failing median-selection tests**

```python
from Scripts.analysis.closed_loop_analysis import choose_median_gain_case


class CaseSelectionTest(unittest.TestCase):
    def test_nearest_median_then_question_id(self):
        candidates = [
            {"question_id": "b", "evidence_gain": 0.2},
            {"question_id": "a", "evidence_gain": 0.2},
            {"question_id": "z", "evidence_gain": 0.8},
        ]
        selected, audit = choose_median_gain_case(candidates)
        self.assertEqual("a", selected["question_id"])
        self.assertEqual(3, audit["candidate_count"])
        self.assertEqual(0.2, audit["median_gain"])
```

- [ ] **Step 2: Verify the test fails**

Run `python -m unittest tests.test_closed_loop_analysis.CaseSelectionTest -v`.

- [ ] **Step 3: Implement selection and BM25+Reranker lookup**

```python
from statistics import median


def choose_median_gain_case(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not candidates:
        raise ValueError("no trajectory case candidates")
    median_gain = float(median(row["evidence_gain"] for row in candidates))
    ordered = sorted(
        candidates,
        key=lambda row: (abs(row["evidence_gain"] - median_gain), row["question_id"]),
    )
    return ordered[0], {
        "candidate_count": len(candidates),
        "median_gain": median_gain,
        "selection_rule": "nearest evidence gain to candidate median, then question_id",
    }
```

Implement dataset-specific eligibility predicates exactly as the approved spec. Resolve the baseline by matching the same question ID in `eval_qasper_bm25_rerank` or `eval_hotpotqa_bm25_rerank`; load its `result.json` and `retrieval_res.json`, retain actual answer/citations, and never rewrite a correct baseline as a failure.

- [ ] **Step 4: Run all analysis tests**

Run `python -m unittest tests.test_closed_loop_analysis -v`. Expected: PASS.

### Task 5: Discover full runs and write deterministic audit artifacts

**Files:**
- Modify: `Scripts/analysis/closed_loop_analysis.py`
- Modify: `tests/test_closed_loop_analysis.py`

- [ ] **Step 1: Add a failing filesystem fixture test**

Create a temporary document with exact method directory names and assert that ablation prefixes are excluded and query ordering is by `(dataset, question_id)`.

```python
def test_exact_hotpot_method_directory_excludes_ablation(self):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "doc" / "eval_hotpotqa_evibridge" / "query_001").mkdir(parents=True)
        (root / "doc" / "eval_hotpotqa_evibridge_wo_context_edges" / "query_001").mkdir(parents=True)
        paths = discover_query_dirs(root, "hotpotqa", "evibridge")
        self.assertEqual(1, len(paths))
        self.assertEqual("eval_hotpotqa_evibridge", paths[0].parent.name)
```

- [ ] **Step 2: Verify the filesystem test fails**

Run the focused test and expect the missing `discover_query_dirs` function.

- [ ] **Step 3: Implement CLI and stable writers**

The CLI must require explicit roots and output directory:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qasper-root", type=Path, required=True)
    parser.add_argument("--hotpot-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()
```

Discover only `root/*/eval_<dataset>_<method>/query_*`, require both `result.json` and `retrieval_res.json`, sort by the extracted question ID, validate 1005/1000 rows, and write UTF-8 with newline `\n`. JSON uses `ensure_ascii=False`, `sort_keys=True`; CSV uses a fixed column order and stores counts, denominators, rates, metric values and display strings.

- [ ] **Step 4: Run the full offline analysis**

Run:

```powershell
python Scripts/analysis/closed_loop_analysis.py `
  --qasper-root 'D:\Else\Study\git\else\BookRAG-Asuka-\runs\qasper_evibridge\work_validation_full1005_unified_0417263dd' `
  --hotpot-root 'D:\Else\Study\git\else\BookRAG-Asuka-\runs\hotpotqa_evibridge\work_validation_fixed1000_seed42' `
  --output-dir 'D:\Else\Study\godot\素材\压缩包'
```

Expected: exactly 2005 JSONL rows, two CSV dataset rows, two selected cases, no validation failures.

- [ ] **Step 5: Verify deterministic output**

Record SHA-256 for the three files, rerun the same command, and verify all three hashes are unchanged.

### Task 6: Render and insert the two LaTeX subsections

**Files:**
- Create: `Scripts/analysis/render_closed_loop_paper.py`
- Test: `tests/test_closed_loop_paper.py`
- Modify: `D:/Else/Study/godot/素材/压缩包/CoSE-RAG_updated_efficiency.tex`

- [ ] **Step 1: Add failing escape and anchor-preservation tests**

```python
from Scripts.analysis.render_closed_loop_paper import escape_latex, insert_analysis


class PaperRenderTest(unittest.TestCase):
    def test_escape_latex(self):
        self.assertEqual(r"A\&B\_1\%", escape_latex("A&B_1%"))

    def test_insert_analysis_requires_one_anchor_and_preserves_prefix_suffix(self):
        source = "before\n\\subsection{参数敏感性分析}\nafter\n"
        updated = insert_analysis(source, "NEW\n")
        self.assertEqual("before\nNEW\n\\subsection{参数敏感性分析}\nafter\n", updated)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            insert_analysis(source + source, "NEW\n")
```

- [ ] **Step 2: Verify the render tests fail**

Run `python -m unittest tests.test_closed_loop_paper -v`.

- [ ] **Step 3: Implement renderer and source backup**

Implement escaping for `& % $ # _ { } ~ ^ \\`, generate the compact behavior table and fixed-width `table*` case table from the audit files, and insert immediately before the unique `\subsection{参数敏感性分析}` anchor. Before replacing the source, write `CoSE-RAG_updated_efficiency.before_closed_loop.tex` only if it does not already exist.

The renderer refuses to run if either new label already exists, if the anchor count is not one, or if the summary/case files fail schema checks.

- [ ] **Step 4: Render the paper update**

Run:

```powershell
python Scripts/analysis/render_closed_loop_paper.py `
  --tex 'D:\Else\Study\godot\素材\压缩包\CoSE-RAG_updated_efficiency.tex' `
  --summary 'D:\Else\Study\godot\素材\压缩包\closed_loop_behavior_summary.csv' `
  --cases 'D:\Else\Study\godot\素材\压缩包\retrieval_trajectory_cases.json'
```

Expected: backup created, two labels inserted once, all original content outside the insertion point byte-equivalent.

- [ ] **Step 5: Run both test modules**

Run:

```powershell
python -m unittest tests.test_closed_loop_analysis tests.test_closed_loop_paper -v
```

Expected: PASS.

### Task 7: Cross-check artifacts and compile the paper

**Files:**
- Verify: `D:/Else/Study/godot/素材/压缩包/CoSE-RAG_updated_efficiency.tex`
- Verify: `D:/Else/Study/godot/素材/压缩包/closed_loop_query_logs.jsonl`
- Verify: `D:/Else/Study/godot/素材/压缩包/closed_loop_behavior_summary.csv`
- Verify: `D:/Else/Study/godot/素材/压缩包/retrieval_trajectory_cases.json`

- [ ] **Step 1: Recompute counts and gains independently from JSONL**

Use a read-only verification command that loads JSONL, groups by dataset, and asserts the CSV counts/rates and case IDs. Expected: Qasper 1005, HotpotQA 1000, no duplicate IDs, exact agreement.

- [ ] **Step 2: Verify no gold leakage or fabricated case content**

For each selected case, assert every displayed evidence ID/text pair exists verbatim in its source `result.json`/`retrieval_res.json`, every bridge edge exists in the selected round log, and BM25+Reranker fields originate from its actual files.

- [ ] **Step 3: Compile twice with XeLaTeX**

Run from the paper directory:

```powershell
xelatex -interaction=nonstopmode -halt-on-error 'CoSE-RAG_updated_efficiency.tex'
xelatex -interaction=nonstopmode -halt-on-error 'CoSE-RAG_updated_efficiency.tex'
```

Expected: exit code 0 on both runs; no missing reference or undefined label for `tab:closed_loop_behavior` and `tab:retrieval_trajectory_cases`.

- [ ] **Step 4: Inspect layout warnings and rendered pages**

Search the log for `Overfull \\hbox`, `undefined references`, and `multiply defined`. Render the pages containing both new tables and inspect them for overlap, clipping and unreadably small text. If the case table overflows, shorten only evidence excerpts while preserving their verbatim content and audit links; do not use `\resizebox` to shrink the case table.

- [ ] **Step 5: Final scope-diff check**

Compare the updated TeX with `CoSE-RAG_updated_efficiency.before_closed_loop.tex`. Expected: the only content change is the insertion before parameter sensitivity. Confirm the three pre-existing experiment tables and all their numeric cells are unchanged.

### Task 8: Replace the Qasper trajectory with a successful, cited median-gain case

**Files:**
- Modify: `Scripts/analysis/closed_loop_analysis.py`
- Modify: `Scripts/analysis/render_closed_loop_paper.py`
- Modify: `tests/test_closed_loop_analysis.py`
- Modify: `tests/test_closed_loop_paper.py`
- Regenerate: `D:/Else/Study/godot/素材/压缩包/retrieval_trajectory_cases.json`
- Modify only the existing analysis insertion in: `D:/Else/Study/godot/素材/压缩包/CoSE-RAG_updated_efficiency.tex`

- [x] **Step 1: Add failing selection and replacement tests**

Add a pure Qasper selection test proving that candidates must have final Answer F1 at least `0.8` and an added gold-matching block used in final supporting evidence, while the selection target remains the original strict-candidate median. Add a renderer test proving that rerendering replaces exactly the existing closed-loop/trajectory insertion and preserves the source prefix, the parameter-sensitivity section, and all following content byte-for-byte.

- [x] **Step 2: Verify the focused tests fail**

Run:

```powershell
python -m unittest tests.test_closed_loop_analysis.CaseSelectionTest tests.test_closed_loop_paper.PaperRenderTest -v
```

Expected: FAIL because the successful-and-cited selector and deterministic replacement path do not yet exist.

- [x] **Step 3: Implement the minimum selector and renderer changes**

Compute final Qasper Answer F1 with the existing official token-F1 utility. Keep the original strict candidate set for the reference median, filter successful-and-cited candidates, and expose strict count, eligible count, selected Answer F1, and cited added-gold IDs in the case audit. Make the renderer upsert the two generated subsections when their labels already exist, without touching any content outside that bounded insertion.

- [x] **Step 4: Regenerate the case audit and replace only the inserted paper block**

Expected selected Qasper question ID: `7e62a53823aba08bc26b2812db016f5ce6159565`. Verify its final answer is `IITB English-Hindi parallel corpus and ILCI English-Hindi parallel corpus`, final Answer F1 is `0.94117647`, Evidence F1 changes from `0.16666667` to `0.33333333`, and added gold block `25` is present in final supporting evidence.

- [x] **Step 5: Run regression and scope verification**

Run both unit-test modules, regenerate twice and compare hashes, validate displayed evidence and edges against immutable source logs, and compare the final TeX with the original backup. Expected: all pre-existing paper content remains byte-identical; only the previously inserted analysis block changes.
