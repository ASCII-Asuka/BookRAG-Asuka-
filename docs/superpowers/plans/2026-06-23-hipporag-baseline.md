# HippoRAG Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local HippoRAG-style graph retrieval baseline for Qasper split1 comparison.

**Architecture:** The baseline is an independent RAG strategy named `hipporag`. It builds a per-document passage/entity graph from the existing `DocumentTree`, seeds retrieval with BM25-like lexical overlap and query entities, runs Personalized PageRank over the graph, then generates a short answer from top paragraphs. It does not use EviBridge demand parsing, verifier, selector, or typed evidence graph.

**Tech Stack:** Python, Pydantic configs, NetworkX, existing `DocumentTree`, existing `BaseRAG`/LLM interface, Qasper official exporter.

---

### Task 1: Config and Wiring

**Files:**
- Create: `Core/configs/rag/hipporag_config.py`
- Modify: `Core/configs/rag/__init__.py`
- Modify: `Core/rag/__init__.py`
- Modify: `Core/utils/resource_loader.py`
- Test: `tests/test_hipporag_wiring.py`

- [ ] Add `HippoRAGConfig` with strategy `hipporag`, graph/retrieval top-k fields, and answer style.
- [ ] Add the config to the RAG strategy union and factory.
- [ ] Load `DocumentTree` and `save_path` for the strategy.
- [ ] Test that config parsing, resource loading, and factory dispatch work.

### Task 2: Local HippoRAG Retrieval

**Files:**
- Create: `Core/rag/hipporag_rag.py`
- Test: `tests/test_hipporag_modules.py`

- [ ] Extract paragraph documents from `DocumentTree`.
- [ ] Extract lightweight entities/keywords from passages and questions.
- [ ] Build a passage/entity NetworkX graph with mention and passage-similarity edges.
- [ ] Run PPR from query passage/entity seeds.
- [ ] Return ranked paragraph results with traceable Qasper metadata.
- [ ] Test that entity bridge evidence can outrank a purely lexical distractor.

### Task 3: Output Compatibility

**Files:**
- Modify: `Core/rag/hipporag_rag.py`
- Test: `tests/test_hipporag_modules.py`

- [ ] Generate concise JSON-style answers with `answer_short`.
- [ ] Save `retrieval_res.json` with `ranked_results`, `supporting_evidence`, `graph_diagnostics`, and seed diagnostics.
- [ ] Set `last_answer_short`, `last_answer_rationale`, `last_supporting_block_ids`, and retrieved IDs for existing inference/export logic.

### Task 4: Qasper Split1 Experiment

**Files:**
- Create: `runs/qasper_evibridge/config/hipporag_qasper_split1.yaml`
- Modify: Qasper split1 result table under `runs/qasper_evibridge/work_validation_full/0_results/`

- [ ] Reuse existing split1 dataset config.
- [ ] Run one-document smoke test if needed.
- [ ] Run Qasper split1 `rag`.
- [ ] Export official Qasper metrics with dynamic evidence top-k.
- [ ] Validate 241/241 predictions and update the split1 main comparison table.

