from __future__ import annotations

import json
from pathlib import Path


REACT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types:
(1) Search[entity], which searches the exact entity or section in the allowed local document collection and returns the first evidence units if it exists. If not, it returns the most similar local page.
(2) Lookup[keyword], which returns the next sentence or paragraph containing keyword in the current page.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""


def _official_hotpot_examples(source_root: str | Path) -> str:
    prompt_path = Path(source_root) / "prompts" / "prompts_naive.json"
    payload = json.loads(prompt_path.read_text(encoding="utf-8-sig"))
    examples = str(payload.get("webthink_simple6") or "").strip()
    if examples.count("Question:") != 6:
        raise ValueError("official webthink_simple6 prompt must contain six examples")
    return examples


def get_react_examples(dataset: str, source_root: str | Path) -> str:
    name = str(dataset or "").lower()
    if "hotpot" in name:
        return _official_hotpot_examples(source_root)
    if "qasper" in name:
        examples = (Path(__file__).with_name("qasper_train_examples.txt")).read_text(
            encoding="utf-8"
        ).strip()
        if examples.count("Question:") != 6:
            raise ValueError("Qasper ReAct prompt must contain six train examples")
        return examples
    raise ValueError(f"unsupported ReAct dataset: {dataset!r}")


def build_prompt_prefix(examples: str) -> str:
    return REACT_INSTRUCTION + str(examples).strip() + "\n"
