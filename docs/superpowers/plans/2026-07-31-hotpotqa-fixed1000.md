# HotpotQA Fixed-1000 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, manifest-validated HotpotQA fixed-1000 dataset and restore exact title/sentence supporting-fact evaluation.

**Architecture:** A focused `hotpotqa_fixed_sample.py` module loads and audits the official parquet, allocates proportional strata, ranks IDs by SHA256, and writes selected raw/unified data plus a provenance manifest. Existing HotpotQA tree conversion gains exact-title-first fact resolution. Dataset configs optionally reference a manifest that is verified whenever inference or evaluation loads the config.

**Tech Stack:** Python 3.13, PyArrow Parquet, Pydantic, hashlib/JSON, unittest, existing `DocumentTree`, `EvidenceBridgeIndex`, and HotpotQA evaluator.

---

### Task 1: Fix exact title/sentence mapping

**Files:**
- Modify: `Scripts/preprocess/hotpotqa_evibridge.py:421-451`
- Test: `tests/test_hotpotqa_preprocess.py`

- [ ] **Step 1: Write the failing exact-title collision tests**

Add:

```python
def test_evidence_mapping_prefers_exact_title_over_normalized_collision(self):
    from Scripts.preprocess.hotpotqa_evibridge import _evidence_node_ids

    lookup = {
        ("Popular Science", 0): 7,
        ("Popular science", 0): 9,
    }
    self.assertEqual(_evidence_node_ids([["Popular Science", 0]], lookup), [7])

def test_evidence_mapping_rejects_ambiguous_normalized_title(self):
    from Scripts.preprocess.hotpotqa_evibridge import _evidence_node_ids

    lookup = {
        ("Popular Science", 0): 7,
        ("Popular science", 0): 9,
    }
    with self.assertRaisesRegex(ValueError, "ambiguous normalized HotpotQA title"):
        _evidence_node_ids([["POPULAR SCIENCE", 0]], lookup)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m unittest `
  tests.test_hotpotqa_preprocess.HotpotQAPreprocessTests.test_evidence_mapping_prefers_exact_title_over_normalized_collision `
  tests.test_hotpotqa_preprocess.HotpotQAPreprocessTests.test_evidence_mapping_rejects_ambiguous_normalized_title -v
```

Expected: the first test returns node 9 and the second test does not raise.

- [ ] **Step 3: Implement exact-first resolution**

Add and use:

```python
def _resolve_fact_value(
    title: Any,
    sent_id: int,
    values: Dict[Tuple[str, int], Any],
) -> Any:
    exact_key = (_clean_title(title), int(sent_id))
    if exact_key in values:
        return values[exact_key]
    normalized_title = _normalize_title(title)
    matches = [
        value
        for (candidate_title, candidate_sent_id), value in values.items()
        if candidate_sent_id == int(sent_id)
        and _normalize_title(candidate_title) == normalized_title
    ]
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous normalized HotpotQA title: {title!r}, sent_id={sent_id}"
        )
    return matches[0] if matches else None
```

Make `_evidence_node_ids()` and `_evidence_texts()` call this helper rather
than constructing a collision-prone normalized dictionary.

- [ ] **Step 4: Run focused preprocessing tests and verify GREEN**

Run:

```powershell
python -m unittest tests.test_hotpotqa_preprocess -v
```

Expected: all HotpotQA preprocessing tests pass.

- [ ] **Step 5: Commit**

```powershell
git add Scripts/preprocess/hotpotqa_evibridge.py tests/test_hotpotqa_preprocess.py
git commit -m "fix: preserve exact HotpotQA supporting titles"
```

### Task 2: Add deterministic source audit and stratified selection

**Files:**
- Create: `Scripts/preprocess/hotpotqa_fixed_sample.py`
- Create: `tests/test_hotpotqa_fixed_sample.py`

- [ ] **Step 1: Write failing sampler tests**

Create synthetic rows and assert:

```python
def make_row(question_id: str, question_type: str, valid: bool) -> dict:
    return {
        "id": question_id,
        "question": f"Question {question_id}?",
        "answer": "answer",
        "type": question_type,
        "level": "hard",
        "supporting_facts": {
            "title": ["Article"],
            "sent_id": [0 if valid else 9],
        },
        "context": {
            "title": ["Article"],
            "sentences": [["Supporting sentence."]],
        },
    }

def test_fixed_sample_uses_proportional_hamilton_quotas(self):
    rows = [
        make_row(f"b-{index}", "bridge", valid=True) for index in range(8)
    ] + [
        make_row(f"c-{index}", "comparison", valid=True) for index in range(2)
    ]
    selected, audit = select_fixed_rows(rows, sample_size=5, seed=42)
    self.assertEqual(audit["selected_strata"], {
        "bridge|hard": 4,
        "comparison|hard": 1,
    })
    self.assertEqual(len(selected), 5)

def test_fixed_sample_excludes_and_audits_out_of_range_gold_fact(self):
    rows = [
        make_row("good", "bridge", valid=True),
        make_row("bad", "bridge", valid=False),
    ]
    selected, audit = select_fixed_rows(rows, sample_size=1, seed=42)
    self.assertEqual([row["id"] for row in selected], ["good"])
    self.assertEqual(audit["excluded"][0]["question_id"], "bad")
    self.assertEqual(audit["excluded"][0]["reason"], "sentence_id_out_of_range")

def test_fixed_sample_selection_is_stable_under_input_reordering(self):
    rows = [make_row(f"id-{index}", "bridge", valid=True) for index in range(20)]
    first, _ = select_fixed_rows(rows, sample_size=5, seed=42)
    second, _ = select_fixed_rows(list(reversed(rows)), sample_size=5, seed=42)
    self.assertEqual(
        {row["id"] for row in first},
        {row["id"] for row in second},
    )
```

- [ ] **Step 2: Run the sampler tests and verify RED**

Run:

```powershell
python -m unittest tests.test_hotpotqa_fixed_sample -v
```

Expected: import fails because `hotpotqa_fixed_sample.py` does not exist.

- [ ] **Step 3: Implement source loading, audit, allocation, and hash ranking**

Implement these public functions:

```python
ALGORITHM_VERSION = "hotpotqa-fixed1000-v1"

def load_parquet_rows(path: str | Path) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq
    return [dict(row) for row in pq.read_table(path).to_pylist()]

def stable_rank(question_id: str, seed: int) -> str:
    value = f"{ALGORITHM_VERSION}\0seed={seed}\0{question_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def audit_supporting_facts(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Exact title first; unique normalized fallback; return explicit
    # missing_title, ambiguous_title, or sentence_id_out_of_range records.

def allocate_hamilton(
    counts: Dict[Tuple[str, str], int],
    sample_size: int,
) -> Dict[Tuple[str, str], int]:
    # floor proportional quotas, then assign remaining seats by descending
    # fractional remainder and lexicographic stratum name.

def select_fixed_rows(
    rows: List[Dict[str, Any]],
    sample_size: int = 1000,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # Reject duplicate/empty IDs, exclude audited invalid rows, rank within
    # each stratum, and return selected rows in original source order.
```

- [ ] **Step 4: Run sampler tests and verify GREEN**

Run:

```powershell
python -m unittest tests.test_hotpotqa_fixed_sample -v
```

Expected: all sampler tests pass.

- [ ] **Step 5: Commit**

```powershell
git add Scripts/preprocess/hotpotqa_fixed_sample.py tests/test_hotpotqa_fixed_sample.py
git commit -m "feat: add deterministic HotpotQA fixed sampler"
```

### Task 3: Write artifacts and full provenance manifest

**Files:**
- Modify: `Scripts/preprocess/hotpotqa_fixed_sample.py`
- Modify: `tests/test_hotpotqa_fixed_sample.py`

- [ ] **Step 1: Write the failing artifact test**

Use a temporary parquet with eight bridge and two comparison rows, then assert:

```python
with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    parquet_path = tmp_path / "validation.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    summary = prepare_fixed_sample(
        parquet_path=parquet_path,
        output_root=tmp_path / "fixed",
        working_dir=tmp_path / "work",
        sample_size=5,
        seed=42,
    )
    manifest = json.loads(
        Path(summary["manifest_path"]).read_text(encoding="utf-8")
    )
    self.assertEqual(manifest["question_count"], 5)
    self.assertEqual(manifest["selected_strata"], {
        "bridge|hard": 4,
        "comparison|hard": 1,
    })
    self.assertEqual(len(manifest["selected_question_ids"]), 5)
    self.assertEqual(len(manifest["source_sha256"]), 64)
    self.assertEqual(len(manifest["selected_raw_sha256"]), 64)
    self.assertEqual(len(manifest["unified_sha256"]), 64)
    self.assertEqual(manifest["mapping_coverage"], 1.0)
    self.assertTrue(Path(summary["dataset_config_path"]).exists())
    first_id = manifest["selected_question_ids"][0]
    self.assertTrue((Path(summary["working_dir"]) / first_id / "tree.pkl").exists())
```

- [ ] **Step 2: Run the artifact test and verify RED**

Run:

```powershell
python -m unittest tests.test_hotpotqa_fixed_sample.HotpotQAFixedSampleTests.test_prepare_fixed_sample_writes_hashed_manifest_and_trees -v
```

Expected: `prepare_fixed_sample` is missing.

- [ ] **Step 3: Implement artifact generation**

`prepare_fixed_sample()` writes:

```text
<output_root>/hotpotqa_distractor_validation_fixed1000_seed42.raw.json
<output_root>/hotpotqa_distractor_validation_fixed1000_seed42.json
<output_root>/hotpotqa_distractor_validation_fixed1000_seed42.manifest.json
<output_root>/hotpotqa_distractor_validation_fixed1000_seed42.yaml
```

Use `_write_unified_rows()` and `hotpotqa_row_to_tree()` from
`hotpotqa_evibridge.py`. Build each tree once and calculate mapping coverage
from `hotpot_node_facts`, `evidence_block_ids`, and all gold facts. Write paths
with forward slashes in the YAML and include:

```yaml
dataset_path: "<absolute unified path>"
working_dir: "<absolute working directory>"
dataset_name: hotpotqa
manifest_path: "<absolute manifest path>"
```

- [ ] **Step 4: Add the CLI**

Parse:

```text
--parquet PATH
--output-root PATH
--working-dir PATH
--sample-size 1000
--seed 42
```

Print the JSON summary and return nonzero on any audit or hash failure.

- [ ] **Step 5: Run artifact tests and verify GREEN**

Run:

```powershell
python -m unittest tests.test_hotpotqa_fixed_sample -v
```

Expected: all fixed-sample tests pass.

- [ ] **Step 6: Commit**

```powershell
git add Scripts/preprocess/hotpotqa_fixed_sample.py tests/test_hotpotqa_fixed_sample.py
git commit -m "feat: write HotpotQA fixed sample manifest"
```

### Task 4: Enforce manifests when dataset configs load

**Files:**
- Modify: `Core/configs/dataset_config.py`
- Create: `tests/test_dataset_manifest.py`

- [ ] **Step 1: Write failing validation tests**

Assert that a valid manifest loads and tampering fails:

```python
def test_dataset_config_validates_ordered_ids_and_sha(self):
    cfg = load_dataset_config(str(config_path))
    self.assertEqual(cfg.manifest_path, str(manifest_path))

def test_dataset_config_rejects_tampered_dataset(self):
    dataset_path.write_text("[]", encoding="utf-8")
    with self.assertRaisesRegex(ValueError, "dataset SHA256 mismatch"):
        load_dataset_config(str(config_path))

def test_dataset_config_rejects_reordered_question_ids(self):
    reordered = list(reversed(rows))
    dataset_path.write_text(json.dumps(reordered), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unified_sha256"] = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with self.assertRaisesRegex(ValueError, "selected question ID order mismatch"):
        load_dataset_config(str(config_path))
```

- [ ] **Step 2: Run manifest tests and verify RED**

Run:

```powershell
python -m unittest tests.test_dataset_manifest -v
```

Expected: `DatasetConfig` ignores `manifest_path` and tampered data loads.

- [ ] **Step 3: Implement optional manifest enforcement**

Extend the model and loader:

```python
class DatasetConfig(BaseModel):
    dataset_path: str
    working_dir: str
    dataset_name: str
    manifest_path: str = ""

def _validate_manifest(data_cfg: DatasetConfig) -> None:
    if not data_cfg.manifest_path:
        return
    manifest = json.loads(Path(data_cfg.manifest_path).read_text(encoding="utf-8"))
    dataset_bytes = Path(data_cfg.dataset_path).read_bytes()
    actual_sha = hashlib.sha256(dataset_bytes).hexdigest()
    if actual_sha != manifest["unified_sha256"]:
        raise ValueError("dataset SHA256 mismatch")
    rows = json.loads(dataset_bytes.decode("utf-8"))
    actual_ids = [
        str(row.get("hotpotqa_question_id") or row.get("question_id") or row.get("doc_uuid"))
        for row in rows
    ]
    if actual_ids != manifest["selected_question_ids"]:
        raise ValueError("selected question ID order mismatch")
    if len(rows) != int(manifest["question_count"]):
        raise ValueError("manifest question count mismatch")
```

Call `_validate_manifest(data_cfg)` before returning from
`load_dataset_config()`.

- [ ] **Step 4: Run manifest and config tests**

Run:

```powershell
python -m unittest tests.test_dataset_manifest tests.test_evibridge_wiring -v
```

Expected: all tests pass and existing configs without manifests remain valid.

- [ ] **Step 5: Commit**

```powershell
git add Core/configs/dataset_config.py tests/test_dataset_manifest.py
git commit -m "feat: validate dataset provenance manifests"
```

### Task 5: Generate and audit the real fixed-1000 artifacts

**Files generated under ignored `runs/`:**
- `runs/_datasets/hotpotqa/fixed1000/*`
- `runs/hotpotqa_evibridge/work_validation_fixed1000_seed42/*/tree.pkl`

- [ ] **Step 1: Generate the dataset**

Run:

```powershell
python Scripts/preprocess/hotpotqa_fixed_sample.py `
  --parquet runs/_datasets/hotpotqa/raw/validation-00000-of-00001.parquet `
  --output-root runs/_datasets/hotpotqa/fixed1000 `
  --working-dir runs/hotpotqa_evibridge/work_validation_fixed1000_seed42 `
  --sample-size 1000 `
  --seed 42
```

Expected summary:

```json
{
  "source_count": 7405,
  "eligible_count": 7404,
  "question_count": 1000,
  "selected_strata": {
    "bridge|hard": 799,
    "comparison|hard": 201
  },
  "mapping_coverage": 1.0
}
```

- [ ] **Step 2: Validate source and output hashes**

Run a read-only script that asserts:

```python
assert manifest["source_sha256"] == "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
assert manifest["excluded"][0]["question_id"] == "5ae61bfd5542992663a4f261"
assert manifest["question_count"] == 1000
assert len(set(manifest["selected_question_ids"])) == 1000
assert manifest["mapping_coverage"] == 1.0
```

- [ ] **Step 3: Verify all generated trees**

Assert that all 1,000 document directories contain `tree.pkl` and `tree.json`,
and that every unified row has the same number of gold facts and mapped
`evidence_block_ids`.

### Task 6: Run the fixed-manifest 20-question smoke

**Files generated under ignored `runs/`:**
- `runs/_datasets/hotpotqa/fixed1000/hotpotqa_fixed1000_smoke20.json`
- `runs/_datasets/hotpotqa/fixed1000/hotpotqa_fixed1000_smoke20.manifest.json`
- `runs/_datasets/hotpotqa/fixed1000/hotpotqa_fixed1000_smoke20.yaml`
- `runs/hotpotqa_evibridge/work_validation_fixed1000_smoke20/`

- [ ] **Step 1: Materialize the first 20 manifest IDs**

Copy the first 20 ordered rows into a separate unified dataset, write a
corresponding manifest with its own SHA256, and point its working directory to
`work_validation_fixed1000_smoke20`.

- [ ] **Step 2: Reuse trees and build EviBridge indexes**

Copy only `tree.pkl` and `tree.json` for the 20 IDs, then run:

```powershell
python main.py `
  -c runs/_configs/evibridge_hotpotqa_private.yaml `
  -d runs/_datasets/hotpotqa/fixed1000/hotpotqa_fixed1000_smoke20.yaml `
  --nsplit 1 --num 1 index --stage evibridge
```

Expected: 20 `evibridge_index.json` and 20 `evibridge_bm25.pkl` files.

- [ ] **Step 3: Run inference with bounded concurrency**

Run two shards:

```powershell
python main.py -c runs/_configs/evibridge_hotpotqa_private.yaml -d runs/_datasets/hotpotqa/fixed1000/hotpotqa_fixed1000_smoke20.yaml --nsplit 2 --num 1 rag
python main.py -c runs/_configs/evibridge_hotpotqa_private.yaml -d runs/_datasets/hotpotqa/fixed1000/hotpotqa_fixed1000_smoke20.yaml --nsplit 2 --num 2 rag
```

Expected: 20 complete `final_results.json` outputs, zero verifier state
contradictions, and no reranker failures.

- [ ] **Step 4: Run official metrics**

Run:

```powershell
python Eval/evaluation.py `
  -d runs/_datasets/hotpotqa/fixed1000/hotpotqa_fixed1000_smoke20.yaml `
  --method evibridge `
  --max_workers 1
```

Expected: score JSON contains non-omitted `sp_em`, `sp_f1`, `joint_em`, and
`joint_f1`; coverage reports 20/20 predictions.

### Task 7: Final regression and checkpoint commit

**Files:**
- All source and test files from Tasks 1-4

- [ ] **Step 1: Run focused tests**

```powershell
python -m unittest `
  tests.test_hotpotqa_preprocess `
  tests.test_hotpotqa_fixed_sample `
  tests.test_hotpotqa_eval `
  tests.test_dataset_manifest -v
```

Expected: all focused tests pass.

- [ ] **Step 2: Run full verification**

```powershell
python -m compileall -q Core Eval Scripts tests
python -m unittest discover -s tests -v
git diff --check
git status --short --untracked-files=no
```

Expected: compile succeeds, all tests pass, diff check is empty, and no
uncommitted tracked changes remain after commits.

- [ ] **Step 3: Record the experiment checkpoint**

Report the fixed manifest path and hash, smoke Answer/SP/Joint metrics,
mapping coverage, citation validity, verifier consistency, tokens/question,
and seconds/question. Label the old weighted 1000 table as historical and
non-reproducible; do not merge it with the fixed-manifest results.

### Task 8: Run the formal fixed-1000 EviBridge and BM25+Reranker comparison

**Files generated under ignored `runs/`:**
- `runs/hotpotqa_evibridge/work_validation_fixed1000_seed42/`
- `runs/hotpotqa_evibridge/work_validation_fixed1000_seed42/0_results/`

- [ ] **Step 1: Build the 1,000 EviBridge indexes**

Run the index stage in two sequential shards to bound memory:

```powershell
python main.py -c runs/_configs/evibridge_hotpotqa_private.yaml -d runs/_datasets/hotpotqa/fixed1000/hotpotqa_distractor_validation_fixed1000_seed42.yaml --nsplit 2 --num 1 index --stage evibridge
python main.py -c runs/_configs/evibridge_hotpotqa_private.yaml -d runs/_datasets/hotpotqa/fixed1000/hotpotqa_distractor_validation_fixed1000_seed42.yaml --nsplit 2 --num 2 index --stage evibridge
```

Expected: exactly 1,000 `evibridge_index.json` and 1,000
`evibridge_bm25.pkl` files.

- [ ] **Step 2: Run EviBridge with two concurrent shards**

Start only two hidden processes and write separate stdout/stderr logs:

```powershell
python main.py -c runs/_configs/evibridge_hotpotqa_private.yaml -d runs/_datasets/hotpotqa/fixed1000/hotpotqa_distractor_validation_fixed1000_seed42.yaml --nsplit 2 --num 1 rag
python main.py -c runs/_configs/evibridge_hotpotqa_private.yaml -d runs/_datasets/hotpotqa/fixed1000/hotpotqa_distractor_validation_fixed1000_seed42.yaml --nsplit 2 --num 2 rag
```

Expected: 1,000 matching `final_results.json` outputs, zero reranker
failures, and zero verifier state contradictions.

- [ ] **Step 3: Evaluate EviBridge with strict supporting metrics**

```powershell
python Eval/evaluation.py -d runs/_datasets/hotpotqa/fixed1000/hotpotqa_distractor_validation_fixed1000_seed42.yaml --method evibridge --max_workers 1
```

Require `missing_predictions == 0` and retain all Answer EM/F1,
Supporting Fact EM/F1, and Joint EM/F1 fields.

- [ ] **Step 4: Build and run BM25+Reranker on the identical manifest**

Use `runs/_configs/hotpotqa_bm25_rerank_private.yaml`, derived from
`config/qasper_bm25_rerank.yaml` with the same SiliconFlow LLM and reranker
service. Build the BM25 corpus and run two shards:

```powershell
python main.py -c runs/_configs/hotpotqa_bm25_rerank_private.yaml -d runs/_datasets/hotpotqa/fixed1000/hotpotqa_distractor_validation_fixed1000_seed42.yaml --nsplit 2 --num 1 index --stage vdb
python main.py -c runs/_configs/hotpotqa_bm25_rerank_private.yaml -d runs/_datasets/hotpotqa/fixed1000/hotpotqa_distractor_validation_fixed1000_seed42.yaml --nsplit 2 --num 2 index --stage vdb
python main.py -c runs/_configs/hotpotqa_bm25_rerank_private.yaml -d runs/_datasets/hotpotqa/fixed1000/hotpotqa_distractor_validation_fixed1000_seed42.yaml --nsplit 2 --num 1 rag
python main.py -c runs/_configs/hotpotqa_bm25_rerank_private.yaml -d runs/_datasets/hotpotqa/fixed1000/hotpotqa_distractor_validation_fixed1000_seed42.yaml --nsplit 2 --num 2 rag
python Eval/evaluation.py -d runs/_datasets/hotpotqa/fixed1000/hotpotqa_distractor_validation_fixed1000_seed42.yaml --method bm25_rerank --max_workers 1
```

Expected: both methods report 1,000/1,000 coverage under the same dataset
SHA256 and manifest ID order.

- [ ] **Step 5: Compute paired confidence intervals and replace the paper table**

Use the per-question `final_eval_hotpotqa_<method>.json` files with
`Eval/utils/paired_bootstrap.py` for:

```text
answer_em
answer_f1
sp_em
sp_f1
joint_em
joint_f1
```

Run 20,000 resamples with seed 42. Write a new fixed-1000 table containing
the manifest hash, coverage, all six official metrics, 95% confidence
intervals, tokens/question, and seconds/question. Keep the historical weighted
table in place with a visible non-reproducible label.
