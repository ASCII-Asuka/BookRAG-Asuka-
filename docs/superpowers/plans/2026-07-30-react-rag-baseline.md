# ReAct-RAG Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a training-free, paper-faithful ReAct baseline over the repository's fixed local corpora and evaluate it on Qasper validation full and the weighted HotpotQA 1,000-question sample.

**Architecture:** Add a local page environment and ReAct runner as focused modules under `Core/rag/`, then expose them through `VanillaConfig(retrieval_method="react")`. `VanillaRAG` remains responsible for repository-compatible persistence while the runner owns the seven-step Thought/Action/Observation loop. Prompt assets are fixed and dataset-specific; diagnostics and official evaluation reuse existing result directories.

**Tech Stack:** Python 3.10+, Pydantic, existing `Core.utils.bm25.BM25`, existing LLM provider, `unittest`, official Qasper and HotpotQA evaluators.

---

## File Map

- Create `Core/rag/react_env.py`: local page grouping, Search/Lookup/Finish actions, source-preserving observations.
- Create `Core/rag/react_prompt.py`: official instruction, fixed HotpotQA/Qasper six-shot assets, action parser, prompt builder.
- Create `Core/rag/react_runner.py`: seven-step loop, parser repair call, trajectory and counters.
- Modify `Core/configs/rag/vanilla_config.py`: register `react` and its five configuration fields.
- Modify `Core/utils/resource_loader.py`: load the paragraph BM25 index for ReAct.
- Modify `Core/rag/vanilla_rag.py`: route ReAct generation and persist compatible output artifacts.
- Create `Scripts/eval/react_diagnostics.py`: aggregate steps, calls, actions, invalid actions, termination, tokens, and elapsed time.
- Create `tests/test_react_env.py`: environment unit tests.
- Create `tests/test_react_prompt.py`: prompt and parser tests.
- Create `tests/test_react_runner.py`: loop and repair behavior tests.
- Create `tests/test_react_wiring.py`: config, resource loading, persistence, and compatibility tests.
- Create `tests/test_react_diagnostics.py`: deterministic diagnostics aggregation tests.
- Create local ignored configs `runs/qasper_evibridge/config/react_qasper.yaml` and `runs/hotpotqa_evibridge/config/react_hotpotqa.yaml`.

## Task 1: Build the Source-Preserving Local Page Environment

**Files:**
- Create: `Core/rag/react_env.py`
- Create: `tests/test_react_env.py`

- [ ] **Step 1: Write failing tests for page grouping**

Create `tests/test_react_env.py` with document fixtures that use the actual
metadata shapes:

```python
import unittest

from Core.rag.react_env import ReactLocalEnvironment


def make_doc(node_id, content, section, score=1.0, hotpot=False):
    metadata = {
        "node_id": node_id,
        "source_node_id": node_id,
        "section_id": section,
        "section": section,
        "title_path": section,
        "qasper_evidence_text": content,
        "source": "qasper_paragraph",
        "node_type": "text",
    }
    if hotpot:
        metadata["hotpot_title"] = section
        metadata["hotpot_sent_id"] = node_id
    return {"id": node_id, "content": content, "score": score, "metadata": metadata}


class ReactEnvironmentTests(unittest.TestCase):
    def test_groups_qasper_paragraphs_by_section(self):
        docs = [
            make_doc(1, "First introduction sentence.", "Introduction"),
            make_doc(2, "Second introduction sentence.", "Introduction"),
            make_doc(3, "A result sentence.", "Results"),
        ]
        env = ReactLocalEnvironment(docs=docs, bm25=None, page_observation_units=5)

        self.assertEqual(env.page_keys, ["Introduction", "Results"])
        self.assertEqual([item.block_id for item in env.pages["Introduction"]], [1, 2])

    def test_groups_hotpot_sentences_by_title(self):
        docs = [
            make_doc(4, "Alpha first.", "Alpha", hotpot=True),
            make_doc(5, "Alpha second.", "Alpha", hotpot=True),
            make_doc(6, "Beta first.", "Beta", hotpot=True),
        ]
        env = ReactLocalEnvironment(docs=docs, bm25=None, page_observation_units=5)

        self.assertEqual(env.page_keys, ["Alpha", "Beta"])
        self.assertEqual([item.block_id for item in env.pages["Alpha"]], [4, 5])
```

- [ ] **Step 2: Run the grouping tests and verify RED**

Run:

```powershell
python -m unittest tests.test_react_env.ReactEnvironmentTests.test_groups_qasper_paragraphs_by_section tests.test_react_env.ReactEnvironmentTests.test_groups_hotpot_sentences_by_title -v
```

Expected: import failure because `Core.rag.react_env` does not exist.

- [ ] **Step 3: Implement page records and grouping**

Create `Core/rag/react_env.py` with these public records and constructor:

```python
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ReactEvidenceUnit:
    block_id: Any
    page_key: str
    content: str
    metadata: Dict[str, Any]
    corpus_index: int


@dataclass
class ReactObservation:
    text: str
    evidence: List[ReactEvidenceUnit] = field(default_factory=list)
    valid_action: bool = True
    done: bool = False
    answer: str = ""
    action_type: str = ""


class ReactLocalEnvironment:
    def __init__(self, docs, bm25, page_observation_units=5, search_topk=1):
        self.docs = list(docs)
        self.bm25 = bm25
        self.page_observation_units = max(1, int(page_observation_units))
        self.search_topk = max(1, int(search_topk))
        self.pages: Dict[str, List[ReactEvidenceUnit]] = {}
        self.page_keys: List[str] = []
        self.active_page: Optional[str] = None
        self.lookup_keyword: Optional[str] = None
        self.lookup_results: List[ReactEvidenceUnit] = []
        self.lookup_index = 0
        self.steps = 0
        self.answer: Optional[str] = None
        self._build_pages()

    @staticmethod
    def _block_id(doc, fallback):
        meta = doc.get("metadata") or {}
        return meta.get("source_node_id", meta.get("node_id", doc.get("id", fallback)))

    @staticmethod
    def _page_key(doc, fallback):
        meta = doc.get("metadata") or {}
        for key in ("hotpot_title", "section_id", "section", "title_path"):
            value = str(meta.get(key) or "").strip()
            if value:
                return value
        return f"evidence-{fallback}"

    def _build_pages(self):
        for index, doc in enumerate(self.docs):
            key = self._page_key(doc, index)
            if key not in self.pages:
                self.pages[key] = []
                self.page_keys.append(key)
            self.pages[key].append(
                ReactEvidenceUnit(
                    block_id=self._block_id(doc, index),
                    page_key=key,
                    content=str(doc.get("content") or ""),
                    metadata=dict(doc.get("metadata") or {}),
                    corpus_index=index,
                )
            )
```

- [ ] **Step 4: Run grouping tests and verify GREEN**

Run the command from Step 2.

Expected: two tests pass.

- [ ] **Step 5: Write failing tests for Search semantics**

Append tests covering exact page matching, normalized containment matching,
BM25 fallback, five-unit observations, and unknown searches:

```python
    def test_search_prefers_exact_page_and_returns_first_five_units(self):
        docs = [make_doc(i, f"Sentence {i}.", "Methods") for i in range(1, 7)]
        env = ReactLocalEnvironment(docs=docs, bm25=None, page_observation_units=5)

        result = env.step("Search[Methods]")

        self.assertTrue(result.valid_action)
        self.assertEqual(env.active_page, "Methods")
        self.assertEqual([item.block_id for item in result.evidence], [1, 2, 3, 4, 5])

    def test_search_uses_bm25_result_page_when_title_does_not_match(self):
        class FakeBM25:
            def search(self, query_text, top_k):
                return [make_doc(8, "Target evidence.", "Results", score=9.0)]

        docs = [
            make_doc(7, "Other evidence.", "Introduction"),
            make_doc(8, "Target evidence.", "Results"),
        ]
        env = ReactLocalEnvironment(docs=docs, bm25=FakeBM25(), page_observation_units=5)

        result = env.step("Search[target metric]")

        self.assertEqual(env.active_page, "Results")
        self.assertEqual([item.block_id for item in result.evidence], [8])

    def test_search_unknown_page_returns_deterministic_observation(self):
        env = ReactLocalEnvironment(
            docs=[make_doc(1, "Only evidence.", "Introduction")],
            bm25=None,
            page_observation_units=5,
        )

        result = env.step("Search[Missing]")

        self.assertFalse(result.valid_action)
        self.assertEqual(result.text, "Could not find Missing.")
```

- [ ] **Step 6: Run Search tests and verify RED**

Run:

```powershell
python -m unittest tests.test_react_env -v
```

Expected: Search tests fail because `step()` is missing.

- [ ] **Step 7: Implement Search and normalized matching**

Add:

```python
    @staticmethod
    def _normalize(text):
        return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))

    def _find_page(self, query):
        normalized = self._normalize(query)
        exact = {self._normalize(key): key for key in self.page_keys}
        if normalized in exact:
            return exact[normalized]
        containing = [
            key for key in self.page_keys
            if normalized and (
                normalized in self._normalize(key)
                or self._normalize(key) in normalized
            )
        ]
        if containing:
            return min(containing, key=lambda key: (len(key), self.page_keys.index(key)))
        if self.bm25 is not None:
            hits = self.bm25.search(query_text=query, top_k=self.search_topk)
            if hits:
                return self._page_key(hits[0], hits[0].get("id", 0))
        return None

    def _search(self, query):
        page_key = self._find_page(query)
        if page_key not in self.pages:
            return ReactObservation(
                text=f"Could not find {query}.",
                valid_action=False,
                action_type="search",
            )
        self.active_page = page_key
        self.lookup_keyword = None
        self.lookup_results = []
        self.lookup_index = 0
        evidence = self.pages[page_key][:self.page_observation_units]
        text = " ".join(item.content.strip() for item in evidence if item.content.strip())
        return ReactObservation(text=text, evidence=evidence, action_type="search")
```

Dispatch `Search[...]` from `step()`, preserving original capitalization
insensitivity only for the action name.

- [ ] **Step 8: Write failing tests for Lookup, Finish, and invalid actions**

Append:

```python
    def test_lookup_advances_and_resets_per_keyword(self):
        docs = [
            make_doc(1, "The model uses BERT. BERT improves accuracy.", "Results"),
            make_doc(2, "Accuracy reaches 91 percent.", "Results"),
        ]
        env = ReactLocalEnvironment(docs=docs, bm25=None, page_observation_units=5)
        env.step("Search[Results]")

        first = env.step("Lookup[BERT]")
        second = env.step("Lookup[BERT]")
        reset = env.step("Lookup[accuracy]")

        self.assertIn("(Result 1 / 2)", first.text)
        self.assertIn("(Result 2 / 2)", second.text)
        self.assertIn("(Result 1 / 2)", reset.text)

    def test_finish_preserves_empty_answer(self):
        env = ReactLocalEnvironment([], bm25=None, page_observation_units=5)

        result = env.step("Finish[]")

        self.assertTrue(result.done)
        self.assertEqual(result.answer, "")
        self.assertEqual(env.answer, "")

    def test_lookup_without_page_and_invalid_action_are_explicit(self):
        env = ReactLocalEnvironment([], bm25=None, page_observation_units=5)

        lookup = env.step("Lookup[key]")
        invalid = env.step("Open[url]")

        self.assertEqual(lookup.text, "No active page.")
        self.assertEqual(invalid.text, "Invalid action: Open[url]")
        self.assertFalse(invalid.valid_action)
```

- [ ] **Step 9: Run environment tests and verify RED**

Run:

```powershell
python -m unittest tests.test_react_env -v
```

Expected: Lookup and Finish tests fail.

- [ ] **Step 10: Implement Lookup, Finish, and action dispatch**

Implement sentence splitting with source preservation:

```python
    @staticmethod
    def _sentences(unit):
        parts = re.split(r"(?<=[.!?])\s+", unit.content.strip())
        return [
            ReactEvidenceUnit(
                block_id=unit.block_id,
                page_key=unit.page_key,
                content=part.strip(),
                metadata=unit.metadata,
                corpus_index=unit.corpus_index,
            )
            for part in parts if part.strip()
        ]

    def _lookup(self, keyword):
        if self.active_page is None:
            return ReactObservation(
                text="No active page.",
                valid_action=False,
                action_type="lookup",
            )
        if self.lookup_keyword != keyword:
            self.lookup_keyword = keyword
            self.lookup_results = [
                sentence
                for unit in self.pages[self.active_page]
                for sentence in self._sentences(unit)
                if keyword.lower() in sentence.content.lower()
            ]
            self.lookup_index = 0
        if self.lookup_index >= len(self.lookup_results):
            return ReactObservation(text="No more results.", action_type="lookup")
        item = self.lookup_results[self.lookup_index]
        self.lookup_index += 1
        return ReactObservation(
            text=f"(Result {self.lookup_index} / {len(self.lookup_results)}) {item.content}",
            evidence=[item],
            action_type="lookup",
        )

    def step(self, action):
        action = str(action or "").strip()
        self.steps += 1
        match = re.fullmatch(r"(?i:search)\[(.*)\]", action)
        if match:
            return self._search(match.group(1).strip())
        match = re.fullmatch(r"(?i:lookup)\[(.*)\]", action)
        if match:
            return self._lookup(match.group(1).strip())
        match = re.fullmatch(r"(?i:finish)\[(.*)\]", action, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            self.answer = answer
            return ReactObservation(
                text="Episode finished.",
                done=True,
                answer=answer,
                action_type="finish",
            )
        return ReactObservation(
            text=f"Invalid action: {action}",
            valid_action=False,
            action_type="invalid",
        )
```

- [ ] **Step 11: Run all environment tests**

Run:

```powershell
python -m unittest tests.test_react_env -v
```

Expected: all environment tests pass.

- [ ] **Step 12: Commit the environment**

```powershell
git add Core/rag/react_env.py tests/test_react_env.py
git commit -m "feat: add local ReAct page environment"
```

## Task 2: Add Fixed Few-Shot Prompts and Strict Action Parsing

**Files:**
- Create: `Core/rag/react_prompt.py`
- Create: `tests/test_react_prompt.py`

- [ ] **Step 1: Write failing parser tests**

Create tests for normal responses, extra text, malformed responses, and bracketed
answers:

```python
import unittest

from Core.rag.react_prompt import parse_react_response


class ReactPromptTests(unittest.TestCase):
    def test_parses_numbered_thought_and_action(self):
        parsed = parse_react_response(
            "Thought 2: I should inspect the results.\\nAction 2: Search[Results]",
            step=2,
        )
        self.assertEqual(parsed.thought, "I should inspect the results.")
        self.assertEqual(parsed.action, "Search[Results]")
        self.assertTrue(parsed.complete)

    def test_finish_allows_brackets_inside_answer(self):
        parsed = parse_react_response(
            "Thought 1: The answer is known.\\nAction 1: Finish[F1 [macro]]",
            step=1,
        )
        self.assertEqual(parsed.action, "Finish[F1 [macro]]")

    def test_malformed_response_keeps_first_thought_for_repair(self):
        parsed = parse_react_response("Need another section.", step=3)
        self.assertEqual(parsed.thought, "Need another section.")
        self.assertEqual(parsed.action, "")
        self.assertFalse(parsed.complete)
```

- [ ] **Step 2: Run parser tests and verify RED**

Run:

```powershell
python -m unittest tests.test_react_prompt -v
```

Expected: import failure because the prompt module does not exist.

- [ ] **Step 3: Implement the parser records**

Create:

```python
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ParsedReactResponse:
    thought: str
    action: str
    complete: bool


def parse_react_response(raw, step):
    text = str(raw or "").strip()
    thought_match = re.search(
        rf"(?:^|\\n)Thought\\s*{step}\\s*:\\s*(.*?)(?=\\nAction\\s*{step}\\s*:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    action_match = re.search(
        rf"(?:^|\\n)Action\\s*{step}\\s*:\\s*((?:Search|Lookup|Finish)\\[.*\\])",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    thought = (
        thought_match.group(1).strip()
        if thought_match
        else (text.splitlines()[0].strip() if text else "")
    )
    action = action_match.group(1).strip() if action_match else ""
    return ParsedReactResponse(thought=thought, action=action, complete=bool(action))
```

- [ ] **Step 4: Run parser tests and verify GREEN**

Run the command from Step 2.

Expected: parser tests pass.

- [ ] **Step 5: Write failing tests for prompt selection and leakage**

Add tests that require exactly six demonstrations, dataset aliases, and Qasper
train IDs:

```python
from Core.rag.react_prompt import QASPER_DEMO_IDS, build_react_prompt, get_react_examples

    def test_hotpot_and_qasper_each_have_six_fixed_examples(self):
        self.assertEqual(get_react_examples("hotpotqa").count("Question:"), 6)
        self.assertEqual(get_react_examples("qasper").count("Question:"), 6)

    def test_qasper_examples_are_from_fixed_train_ids(self):
        self.assertEqual(
            QASPER_DEMO_IDS,
            (
                "753990d0b621d390ed58f20c4d9e4f065f0dc672",
                "44c4bd6decc86f1091b5fc0728873d9324cdde4e",
                "003f884d3893532f8c302431c9f70be6f64d9be8",
                "938cf30c4f1d14fa182e82919e16072fdbcf2a82",
                "19c9cfbc4f29104200393e848b7b9be41913a7ac",
                "cd1034c183edf630018f47ff70b48d74d2bb1649",
            ),
        )

    def test_prompt_appends_question_and_existing_trajectory(self):
        prompt = build_react_prompt(
            dataset_name="qasper",
            question="What metric is reported?",
            trajectory="Thought 1: Inspect results.\\nAction 1: Search[Results]\\nObservation 1: F1 is reported.\\n",
            step=2,
        )
        self.assertIn("Question: What metric is reported?", prompt)
        self.assertTrue(prompt.endswith("Thought 2:"))
```

- [ ] **Step 6: Run prompt tests and verify RED**

Run:

```powershell
python -m unittest tests.test_react_prompt -v
```

Expected: prompt asset and builder tests fail.

- [ ] **Step 7: Vendor the HotpotQA six-shot asset**

Copy the exact `webthink_simple6` value from
`ysymyth/ReAct@6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9`,
file `prompts/prompts_naive.json`, into `HOTPOTQA_REACT_EXAMPLES`.

Retain the upstream action casing and trajectory text. Add this source comment:

```python
# ReAct ICLR 2023 official six-shot HotpotQA prompt, MIT licensed.
# Source commit: 6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9
```

- [ ] **Step 8: Add six Qasper train trajectories**

Define `QASPER_DEMO_IDS` with the IDs in Step 5 and add fixed trajectories
using these action plans:

```text
753990...: Search[Proposed Method ::: Discourse Relation-Based Event Pairs]
             -> Lookup[seed lexicon]
             -> Finish[a vocabulary of positive and negative predicates]

44c4bd...: Search[Experiments ::: Dataset ::: AL, CA, and CO]
             -> Lookup[Japanese Web corpus]
             -> Finish[7,000,000 event pairs]

003f88...: Search[Applying the typology to Reddit]
             -> Lookup[English]
             -> Finish[No]

938cf3...: Search[Community-level measures]
             -> Lookup[temporally dynamic]
             -> Finish[the average volatility of all utterances]

19c9cf...: Search[Experimental Studies ::: Dataset and Evaluation Metrics]
             -> Lookup[questions]
             -> Finish[2,714]

cd1034...: Search[Generative Model]
             -> Lookup[human]
             -> Finish[Yes]
```

Each observation is copied from the corresponding Qasper train evidence text,
not synthesized from validation data. Include the complete Question, numbered
Thought, Action, and Observation fields.

- [ ] **Step 9: Implement prompt construction**

Define the official instruction and builder:

```python
REACT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types:
(1) Search[query], which searches the local document collection and returns the first evidence units of the best matching page.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current page.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""


def get_react_examples(dataset_name, prompt_file=""):
    if prompt_file:
        from pathlib import Path
        return Path(prompt_file).read_text(encoding="utf-8")
    name = str(dataset_name or "").lower()
    if "hotpot" in name:
        return HOTPOTQA_REACT_EXAMPLES
    if "qasper" in name:
        return QASPER_REACT_EXAMPLES
    raise ValueError(f"Unsupported ReAct dataset: {dataset_name}")


def build_react_prompt(dataset_name, question, trajectory, step, prompt_file=""):
    prefix = (
        REACT_INSTRUCTION
        + get_react_examples(dataset_name, prompt_file=prompt_file).rstrip()
        + "\n"
    )
    current = f"Question: {question.strip()}\\n{trajectory}"
    return prefix + current + f"Thought {step}:"
```

- [ ] **Step 10: Run all prompt tests**

Run:

```powershell
python -m unittest tests.test_react_prompt -v
```

Expected: all prompt tests pass.

- [ ] **Step 11: Commit prompt assets and parser**

```powershell
git add Core/rag/react_prompt.py tests/test_react_prompt.py
git commit -m "feat: add paper-faithful ReAct prompts"
```

## Task 3: Implement the Seven-Step ReAct Runner

**Files:**
- Create: `Core/rag/react_runner.py`
- Create: `tests/test_react_runner.py`

- [ ] **Step 1: Write a failing Search-Lookup-Finish integration test**

Create a fake LLM whose outputs contain official-style numbered fields and a
two-page corpus. Assert trajectory ordering, final answer, counters, and
visible evidence:

```python
import unittest

from Core.rag.react_env import ReactLocalEnvironment
from Core.rag.react_runner import ReactRunner


def make_doc(node_id, content, section, score=1.0):
    return {
        "content": content,
        "score": score,
        "metadata": {
            "node_id": node_id,
            "source_node_id": node_id,
            "section_id": section,
            "section": section,
            "title_path": section,
            "qasper_evidence_text": content,
            "source": "qasper_paragraph",
        },
    }


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def get_completion(self, prompt, json_response=False):
        self.prompts.append(prompt)
        return self.responses.pop(0)


class ReactRunnerTests(unittest.TestCase):
    def test_runs_search_lookup_finish_trajectory(self):
        docs = [
            make_doc(1, "Ada was born in London. She was a mathematician.", "Ada"),
            make_doc(2, "London is in England.", "London"),
        ]
        llm = FakeLLM([
            " I need Ada's birthplace.\\nAction 1: Search[Ada]",
            " The page names London.\\nAction 2: Lookup[born]",
            " The answer is London.\\nAction 3: Finish[London]",
        ])
        runner = ReactRunner(
            llm=llm,
            environment=ReactLocalEnvironment(docs, bm25=None),
            dataset_name="hotpotqa",
            max_steps=7,
        )

        result = runner.run("Where was Ada born?")

        self.assertEqual(result.answer, "London")
        self.assertEqual([step.action_type for step in result.steps], ["search", "lookup", "finish"])
        self.assertEqual(result.num_calls, 3)
        self.assertEqual(result.num_bad_calls, 0)
        self.assertEqual([unit.block_id for unit in result.observed_evidence], [1])
        self.assertEqual(result.termination_reason, "finish")
```

- [ ] **Step 2: Run the runner test and verify RED**

Run:

```powershell
python -m unittest tests.test_react_runner.ReactRunnerTests.test_runs_search_lookup_finish_trajectory -v
```

Expected: import failure because `react_runner.py` does not exist.

- [ ] **Step 3: Implement runner records and the main loop**

Create immutable step/result records:

```python
from dataclasses import dataclass, field
from typing import Any, List

from Core.rag.react_prompt import build_react_prompt, parse_react_response


@dataclass
class ReactStep:
    step: int
    thought: str
    action: str
    action_type: str
    observation: str
    valid_action: bool
    evidence_ids: List[Any] = field(default_factory=list)


@dataclass
class ReactRunResult:
    answer: str
    steps: List[ReactStep]
    observed_evidence: list
    num_calls: int
    num_bad_calls: int
    search_count: int
    lookup_count: int
    termination_reason: str
```

`ReactRunner.__init__` accepts `llm`, `environment`, `dataset_name`,
`max_steps`, and `prompt_file=""`. It passes `prompt_file` to
`build_react_prompt`, leaving the fixed built-in six-shot assets as the
default.

Implement `ReactRunner.run(question)` so it:

- Builds the prompt from the complete accumulated trajectory.
- Calls `llm.get_completion(..., json_response=False)`.
- Prepends `Thought {i}:` before parsing because the official completion begins
  after that prefix.
- Executes the parsed action.
- Deduplicates evidence by block ID in first-observation order.
- Appends exact numbered trajectory text.
- Returns immediately on Finish.

- [ ] **Step 4: Run the integration test and verify GREEN**

Run the command from Step 2.

Expected: the test passes.

- [ ] **Step 5: Write failing tests for repair and step-limit termination**

Add:

```python
    def test_repairs_action_once_when_combined_response_is_malformed(self):
        docs = [make_doc(1, "Ada was born in London.", "Ada")]
        llm = FakeLLM([
            " I should search Ada.",
            "Search[Ada]",
            " The answer is London.\\nAction 2: Finish[London]",
        ])
        result = ReactRunner(
            llm=llm,
            environment=ReactLocalEnvironment(docs, bm25=None),
            dataset_name="hotpotqa",
            max_steps=7,
        ).run("Where was Ada born?")

        self.assertEqual(result.answer, "London")
        self.assertEqual(result.num_calls, 3)
        self.assertEqual(result.num_bad_calls, 1)
        self.assertEqual(result.steps[0].action, "Search[Ada]")

    def test_finishes_empty_after_seven_nonterminal_steps(self):
        docs = [make_doc(1, "Evidence.", "Page")]
        llm = FakeLLM([
            f" Keep searching.\\nAction {i}: Search[Page]"
            for i in range(1, 8)
        ])
        result = ReactRunner(
            llm=llm,
            environment=ReactLocalEnvironment(docs, bm25=None),
            dataset_name="hotpotqa",
            max_steps=7,
        ).run("Question?")

        self.assertEqual(result.answer, "")
        self.assertEqual(len(result.steps), 8)
        self.assertEqual(result.steps[-1].action, "Finish[]")
        self.assertEqual(result.termination_reason, "step_limit")
```

- [ ] **Step 6: Run runner tests and verify RED**

Run:

```powershell
python -m unittest tests.test_react_runner -v
```

Expected: repair and step-limit assertions fail.

- [ ] **Step 7: Implement the official-style Action repair call**

When the first parse is incomplete, make exactly one call with:

```python
repair_prompt = (
    prompt
    + parsed.thought
    + f"\nAction {step}:"
)
```

Normalize a bare repaired action to the numbered action form for parsing. If it
still fails, pass the raw repaired text to the environment so it becomes an
invalid action and consumes the step.

After seven nonterminal steps, append a synthetic step for `Finish[]` and set
`termination_reason="step_limit"`.

- [ ] **Step 8: Run all runner tests**

Run:

```powershell
python -m unittest tests.test_react_runner -v
```

Expected: all runner tests pass.

- [ ] **Step 9: Commit the runner**

```powershell
git add Core/rag/react_runner.py tests/test_react_runner.py
git commit -m "feat: implement ReAct interaction loop"
```

## Task 4: Wire ReAct into Vanilla RAG and Preserve Output Semantics

**Files:**
- Modify: `Core/configs/rag/vanilla_config.py`
- Modify: `Core/utils/resource_loader.py`
- Modify: `Core/rag/vanilla_rag.py`
- Create: `tests/test_react_wiring.py`

- [ ] **Step 1: Write failing configuration and resource-loader tests**

Test the defaults and verify ReAct loads `bm25_vdb/bm25_index.pkl`, matching
IRCoT:

```python
import unittest
from unittest.mock import patch

from Core.configs.rag.vanilla_config import VanillaConfig


class ReactWiringTests(unittest.TestCase):
    def test_config_accepts_react_with_official_defaults(self):
        cfg = VanillaConfig(retrieval_method="react")
        self.assertEqual(cfg.react_max_steps, 7)
        self.assertEqual(cfg.react_search_topk, 1)
        self.assertEqual(cfg.react_page_observation_units, 5)
        self.assertEqual(cfg.react_prompt_file, "")
        self.assertEqual(cfg.react_dataset_name, "qasper")
```

Add a loader test following the existing IRCoT loader test pattern and assert
that `dependencies["bm25"]` is populated.

- [ ] **Step 2: Run wiring tests and verify RED**

Run:

```powershell
python -m unittest tests.test_react_wiring -v
```

Expected: Pydantic rejects `retrieval_method="react"` or fields are missing.

- [ ] **Step 3: Add ReAct configuration**

Modify the Literal and fields:

```python
        "ircot",
        "react",
```

```python
    react_max_steps: int = Field(default=7, ge=1)
    react_search_topk: int = Field(default=1, ge=1)
    react_page_observation_units: int = Field(default=5, ge=1)
    react_prompt_file: str = Field(default="")
    react_dataset_name: Literal["qasper", "hotpotqa"] = "qasper"
```

- [ ] **Step 4: Load BM25 for ReAct**

Change:

```python
        elif retrieval_method == "ircot":
```

to:

```python
        elif retrieval_method in {"ircot", "react"}:
```

- [ ] **Step 5: Run configuration and loader tests**

Run the command from Step 2.

Expected: config and loader tests pass.

- [ ] **Step 6: Write a failing persistence integration test**

Create a deterministic fake LLM and BM25, configure
`react_dataset_name="qasper"`, and invoke `VanillaRAG.generation(...)` through
its unchanged public call shape, then assert:

```python
self.assertEqual(payload["strategy"], "react")
self.assertEqual(payload["react_termination_reason"], "finish")
self.assertEqual(payload["supporting_block_ids"], [1])
self.assertEqual(rag.last_answer_short, "London")
self.assertEqual(rag.last_retrieved_block_ids, [1])
self.assertTrue((output_dir / "evidence_chain.json").exists())
```

Also assert that `result.json` compatibility state can consume the answer and
IDs through `BaseRAG`'s existing fields.

- [ ] **Step 7: Run persistence test and verify RED**

Run:

```powershell
python -m unittest tests.test_react_wiring.ReactWiringTests.test_vanilla_rag_persists_react_outputs -v
```

Expected: generation does not route to ReAct.

- [ ] **Step 8: Route ReAct generation**

Add to `VanillaRAG.generation`:

```python
        if self.cfg.retrieval_method == "react":
            return self._react_generation(query, query_output_dir)
```

Pass the dataset name through `react_dataset_name` rather than changing the
public `generation()` signature. Pass `react_prompt_file` into the runner, and
pass `react_search_topk` plus `react_page_observation_units` into the
environment so every declared configuration field affects runtime behavior.

- [ ] **Step 9: Implement `_react_generation` and output persistence**

Build corpus docs directly from `self.bm25.original_docs` and
`self.bm25.metadatas`. Run the environment and runner, then write:

```python
payload = {
    "strategy": "react",
    "ranked_results": observed_items,
    "selected": observed_items,
    "supporting_evidence": observed_items,
    "retrieved_block_ids": block_ids,
    "supporting_block_ids": block_ids,
    "react_steps": step_payloads,
    "react_num_calls": run.num_calls,
    "react_num_bad_calls": run.num_bad_calls,
    "react_search_count": run.search_count,
    "react_lookup_count": run.lookup_count,
    "react_termination_reason": run.termination_reason,
}
```

Write `retrieval_res.json` and `evidence_chain.json` with
`make_json_safe(..., allow_nan=False)` following the IRCoT persistence style.

Set:

```python
self.last_answer_short = run.answer
self.last_answer_rationale = ""
self.last_retrieved_block_ids = block_ids
self.last_supporting_block_ids = block_ids
```

Return a JSON answer string containing `answer_short`,
`answer_rationale=""`, and `supporting_block_ids`, so the existing
`result.json` writer and official exporters use the same stable fields as other
baselines.

- [ ] **Step 10: Run all ReAct wiring tests**

Run:

```powershell
python -m unittest tests.test_react_wiring -v
```

Expected: all wiring tests pass.

- [ ] **Step 11: Run IRCoT and baseline regressions**

Run:

```powershell
python -m unittest tests.test_ircot_baseline tests.test_bm25_baseline tests.test_qasper_official_eval tests.test_hotpotqa_eval -v
```

Expected: all existing tests pass.

- [ ] **Step 12: Commit wiring**

```powershell
git add Core/configs/rag/vanilla_config.py Core/utils/resource_loader.py Core/rag/vanilla_rag.py tests/test_react_wiring.py
git commit -m "feat: wire ReAct baseline into vanilla RAG"
```

## Task 5: Add ReAct Diagnostics and Experiment Integrity Gates

**Files:**
- Create: `Scripts/eval/react_diagnostics.py`
- Create: `tests/test_react_diagnostics.py`

- [ ] **Step 1: Write a failing diagnostics aggregation test**

Use two temporary query directories with known counters:

```python
import json
import tempfile
import unittest
from pathlib import Path

from Scripts.eval.react_diagnostics import aggregate_react_diagnostics


class ReactDiagnosticsTests(unittest.TestCase):
    def test_aggregates_steps_calls_actions_and_termination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, payload in enumerate([
                {
                    "react_steps": [{"valid_action": True}, {"valid_action": True}],
                    "react_num_calls": 2,
                    "react_num_bad_calls": 0,
                    "react_search_count": 1,
                    "react_lookup_count": 0,
                    "react_termination_reason": "finish",
                },
                {
                    "react_steps": [{"valid_action": False}] * 8,
                    "react_num_calls": 8,
                    "react_num_bad_calls": 1,
                    "react_search_count": 0,
                    "react_lookup_count": 1,
                    "react_termination_reason": "step_limit",
                },
            ]):
                query_dir = root / f"query_{index:03d}"
                query_dir.mkdir()
                (query_dir / "retrieval_res.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )

            report = aggregate_react_diagnostics(root)

        self.assertEqual(report["queries"], 2)
        self.assertEqual(report["mean_steps"], 5.0)
        self.assertEqual(report["mean_calls"], 5.0)
        self.assertEqual(report["step_limit_rate"], 0.5)
        self.assertEqual(report["invalid_action_rate"], 0.8)
```

- [ ] **Step 2: Run diagnostics test and verify RED**

Run:

```powershell
python -m unittest tests.test_react_diagnostics -v
```

Expected: import failure because the diagnostics script does not exist.

- [ ] **Step 3: Implement diagnostics aggregation and CLI**

Implement:

```python
def aggregate_react_diagnostics(eval_dir: Path) -> dict:
    payloads = []
    for path in sorted(eval_dir.glob("query_*/retrieval_res.json")):
        with path.open("r", encoding="utf-8") as handle:
            payloads.append(json.load(handle))
    if not payloads:
        raise ValueError(f"No ReAct retrieval results found in {eval_dir}")
    steps = [step for payload in payloads for step in payload.get("react_steps", [])]
    invalid = sum(not bool(step.get("valid_action", True)) for step in steps)
    return {
        "queries": len(payloads),
        "mean_steps": sum(len(p.get("react_steps", [])) for p in payloads) / len(payloads),
        "mean_calls": sum(p.get("react_num_calls", 0) for p in payloads) / len(payloads),
        "mean_searches": sum(p.get("react_search_count", 0) for p in payloads) / len(payloads),
        "mean_lookups": sum(p.get("react_lookup_count", 0) for p in payloads) / len(payloads),
        "bad_call_rate": sum(p.get("react_num_bad_calls", 0) for p in payloads)
        / max(1, sum(p.get("react_num_calls", 0) for p in payloads)),
        "invalid_action_rate": invalid / max(1, len(steps)),
        "step_limit_rate": sum(
            p.get("react_termination_reason") == "step_limit" for p in payloads
        ) / len(payloads),
    }
```

The CLI accepts repeated `--eval-dir`, `--expected-queries`, and `--output`.
It fails when the total number of query results differs from the expected
count.

- [ ] **Step 4: Run diagnostics tests**

Run:

```powershell
python -m unittest tests.test_react_diagnostics -v
```

Expected: all diagnostics tests pass.

- [ ] **Step 5: Run the experiment validator regression**

Run the existing Qasper result validator tests together with:

```powershell
python -m unittest tests.test_qasper_run_validator tests.test_qasper_official_eval tests.test_hotpotqa_eval -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit diagnostics**

```powershell
git add Scripts/eval/react_diagnostics.py tests/test_react_diagnostics.py
git commit -m "feat: add ReAct experiment diagnostics"
```

## Task 6: Create Local Configs and Run Two-Dataset Smoke Tests

**Files:**
- Create locally: `runs/qasper_evibridge/config/react_qasper.yaml`
- Create locally: `runs/hotpotqa_evibridge/config/react_hotpotqa.yaml`
- Create locally: small smoke dataset configs under each run directory

- [ ] **Step 1: Copy the existing IRCoT local configs**

Use the existing files so private LLM settings stay local:

```powershell
Copy-Item -LiteralPath runs\qasper_evibridge\config\ircot_qasper.yaml -Destination runs\qasper_evibridge\config\react_qasper.yaml
Copy-Item -LiteralPath runs\hotpotqa_evibridge\config\ircot_hotpotqa.yaml -Destination runs\hotpotqa_evibridge\config\react_hotpotqa.yaml
```

- [ ] **Step 2: Change only ReAct fields**

Set Qasper:

```yaml
rag:
  strategy: vanilla
  retrieval_method: react
  bm25_vdb_dir_name: bm25_vdb
  react_dataset_name: qasper
  react_max_steps: 7
  react_search_topk: 1
  react_page_observation_units: 5
  react_prompt_file: ""
```

Set HotpotQA identically except:

```yaml
  react_dataset_name: hotpotqa
```

Preserve all already-configured LLM settings and never print API keys.

- [ ] **Step 3: Build one-document smoke dataset configs**

For Qasper, reuse:

```text
runs/qasper_evibridge/config/qasper_validation_bookrag_b1_1doc.yaml
```

For HotpotQA, copy the first row of
`runs/hotpotqa_evibridge/processed/hotpotqa_validation_200.json` into the
local ignored file:

```text
runs/hotpotqa_evibridge/processed/hotpotqa_validation_react_smoke_1q.json
```

Create
`runs/hotpotqa_evibridge/config/hotpotqa_validation_react_smoke.yaml` with:

```yaml
dataset_path: "runs\\hotpotqa_evibridge\\processed\\hotpotqa_validation_react_smoke_1q.json"
working_dir: "runs\\hotpotqa_evibridge\\work_validation_200"
dataset_name: hotpotqa
```

The one-row file retains every field from the source row, especially
`doc_uuid`, `hotpotqa_question_id`, `hotpot_supporting_facts`, and
`hotpot_node_facts`. It is a local smoke artifact and must not be committed.

- [ ] **Step 4: Run Qasper smoke**

```powershell
python main.py -c runs\qasper_evibridge\config\react_qasper.yaml -d runs\qasper_evibridge\config\qasper_validation_react_smoke.yaml --nsplit 1 --num 1 rag
```

Expected:

- `eval_qasper_react/final_results.json` covers every smoke query.
- Every query has `retrieval_res.json` and `evidence_chain.json`.
- No query exceeds seven model-selected actions.

- [ ] **Step 5: Run HotpotQA smoke**

```powershell
python main.py -c runs\hotpotqa_evibridge\config\react_hotpotqa.yaml -d runs\hotpotqa_evibridge\config\hotpotqa_validation_react_smoke.yaml --nsplit 1 --num 1 rag
```

Expected: one complete prediction with title/sentence evidence mapping.

- [ ] **Step 6: Run smoke diagnostics**

Run `react_diagnostics.py` against each smoke output with the exact expected
query count.

Expected: both commands exit 0, and JSON reports contain no missing query.

- [ ] **Step 7: Run the focused regression suite**

```powershell
python -m unittest tests.test_react_env tests.test_react_prompt tests.test_react_runner tests.test_react_wiring tests.test_react_diagnostics tests.test_ircot_baseline tests.test_bm25_baseline tests.test_qasper_official_eval tests.test_hotpotqa_eval -v
```

Expected: all tests pass.

## Task 7: Run Qasper Validation Full

**Files:**
- Generate locally under: `runs/qasper_evibridge/work_validation_full/*/eval_qasper_react/`
- Generate locally under: `runs/qasper_evibridge/work_validation_full/0_results/`

- [ ] **Step 1: Confirm no stale partial ReAct results**

Count existing `eval_qasper_react/final_results.json` and query directories.
Do not delete anything. If partial results exist, rely on
`rag_force_reprocess: false` to resume missing queries.

- [ ] **Step 2: Run shard 1**

```powershell
python main.py -c runs\qasper_evibridge\config\react_qasper.yaml -d runs\qasper_evibridge\config\qasper_validation_full.yaml --nsplit 4 --num 1 rag
```

- [ ] **Step 3: Run shard 2**

Use the same command with `--num 2`.

- [ ] **Step 4: Run shard 3**

Use the same command with `--num 3`.

- [ ] **Step 5: Run shard 4**

Use the same command with `--num 4`.

- [ ] **Step 6: Validate Qasper completeness**

Run diagnostics/result validation with:

```text
expected documents = 281
expected questions = 1005
```

Expected: 1,005/1,005 questions and zero partial document results.

- [ ] **Step 7: Export and evaluate official Qasper predictions**

```powershell
python Scripts\eval\qasper_official.py --dataset-config runs\qasper_evibridge\config\qasper_validation_full.yaml --method react --output-dir runs\qasper_evibridge\work_validation_full\0_results\qasper_official_react_dynamic_output --answer-source output --text-evidence-only --dynamic-evidence-topk
```

Expected:

- `predictions.jsonl`
- `gold_official.json`
- `official_eval.json`
- `Missing predictions = 0`

- [ ] **Step 8: Generate Qasper ReAct diagnostics**

Aggregate all 281 per-document `eval_qasper_react` directories and write:

```text
runs/qasper_evibridge/work_validation_full/0_results/qasper_react_diagnostics.json
```

## Task 8: Run Weighted HotpotQA 1,000

**Files:**
- Generate locally under the three existing HotpotQA working directories.
- Generate weighted output under the random600 `0_results` directory.

- [ ] **Step 1: Run first200 ReAct**

```powershell
python main.py -c runs\hotpotqa_evibridge\config\react_hotpotqa.yaml -d runs\hotpotqa_evibridge\config\hotpotqa_validation_200.yaml --nsplit 1 --num 1 rag
python Eval\evaluation.py -d runs\hotpotqa_evibridge\config\hotpotqa_validation_200.yaml --method react --max_workers 1
```

Expected: 200/200 predictions and `missing_predictions=0`.

- [ ] **Step 2: Run second200 ReAct**

Repeat Step 1 with:

```text
runs/hotpotqa_evibridge/config/hotpotqa_validation_200_split2.yaml
```

Expected: 200/200 predictions.

- [ ] **Step 3: Run random600 shard 1**

```powershell
python main.py -c runs\hotpotqa_evibridge\config\react_hotpotqa.yaml -d runs\hotpotqa_evibridge\config\hotpotqa_validation_random600_exclude400_bridge_diag_seed42.yaml --nsplit 3 --num 1 rag
```

- [ ] **Step 4: Run random600 shard 2**

Use the same command with `--num 2`.

- [ ] **Step 5: Run random600 shard 3**

Use the same command with `--num 3`.

- [ ] **Step 6: Evaluate random600**

```powershell
python Eval\evaluation.py -d runs\hotpotqa_evibridge\config\hotpotqa_validation_random600_exclude400_bridge_diag_seed42.yaml --method react --max_workers 1
```

Expected: 600/600 predictions and `missing_predictions=0`.

- [ ] **Step 7: Compute weighted official metrics**

Read the three `.score.json` files and calculate each metric with:

```text
metric_1000 = first200 * 0.2 + second200 * 0.2 + random600 * 0.6
```

Write:

```text
runs/hotpotqa_evibridge/work_validation_random600_exclude_bridge_diag_seed42/0_results/hotpotqa_1000_random_weighted_react.score.json
```

Include:

- `answer_em`
- `answer_f1`
- `sp_f1`
- `joint_f1`
- `missing_predictions`
- source file paths and weights

- [ ] **Step 8: Generate HotpotQA ReAct diagnostics**

Aggregate all three split outputs and write:

```text
runs/hotpotqa_evibridge/work_validation_random600_exclude_bridge_diag_seed42/0_results/hotpotqa_1000_react_diagnostics.json
```

## Task 9: Update Comparison Tables and Run Final Verification

**Files:**
- Create locally: `runs/qasper_evibridge/work_validation_full/0_results/qasper_validation_full_official_main_table_with_react.md`
- Create locally: matching JSON table.
- Create locally: `runs/hotpotqa_evibridge/work_validation_random600_exclude_bridge_diag_seed42/0_results/hotpotqa_1000_random_weighted_official_table_with_react.md`
- Create locally: matching JSON table.

- [ ] **Step 1: Add ReAct to the Qasper table**

Copy the current official main table containing IRCoT. Read `Answer F1` and
`Evidence F1` from
`qasper_official_react_dynamic_output/official_eval.json` and append a ReAct
row using those parsed numeric values:

```text
Method = ReAct
Answer F1 = official_eval["Answer F1"]
Evidence F1 = official_eval["Evidence F1"]
```

Values must come directly from
`qasper_official_react_dynamic_output/official_eval.json`.

- [ ] **Step 2: Add ReAct to the HotpotQA table**

Copy the current weighted official table containing IRCoT. Read the metrics
from `hotpotqa_1000_random_weighted_react.score.json` and append a ReAct row
using those parsed numeric values:

```text
Method = ReAct
Answer EM = score["answer_em"]
Answer F1 = score["answer_f1"]
Supporting Fact F1 = score["sp_f1"]
Joint F1 = score["joint_f1"]
```

Values must come directly from
`hotpotqa_1000_random_weighted_react.score.json`.

- [ ] **Step 3: Check table provenance**

Verify:

- Qasper table says 281 papers and 1,005 questions.
- HotpotQA table states 200/200/600 weights.
- Only official metrics appear.
- `Missing` is not a displayed table column.
- ReAct is described as local-corpus ReAct prompting, not the live-Wikipedia
  setting from the original paper.

- [ ] **Step 4: Run the complete test suite**

```powershell
python -m unittest discover tests -v
```

Expected: zero failures and zero errors.

- [ ] **Step 5: Run syntax and diff checks**

```powershell
python -m compileall Core Scripts tests
git diff --check
git status --short
```

Expected: compilation succeeds; no whitespace errors; only intentional source
and test changes are listed. `runs/` outputs remain ignored.

- [ ] **Step 6: Verify formal result completeness one last time**

Confirm:

```text
Qasper: 281/281 documents, 1005/1005 questions
HotpotQA first200: 200/200
HotpotQA second200: 200/200
HotpotQA random600: 600/600
Official missing predictions: 0
```

- [ ] **Step 7: Commit final source changes if any remain**

Do not commit `runs/` outputs or private configs. Commit only source, tests, and
public documentation:

```powershell
git add Core Scripts tests docs
git commit -m "feat: add ReAct-RAG baseline experiments"
```
