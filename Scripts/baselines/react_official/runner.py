from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol

from .environment import ObservationResult
from .prompts import build_prompt_prefix


class CompletionModel(Protocol):
    def complete(
        self,
        model: str,
        prompt: str,
        stop: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, dict[str, int]]: ...


class ReActEnvironment(Protocol):
    def execute(self, action: str, argument: str) -> ObservationResult: ...


@dataclass(frozen=True)
class ParsedAction:
    name: str
    argument: str
    raw: str


@dataclass
class ReActStep:
    step: int
    thought: str
    action: str
    action_name: str
    action_argument: str
    observation: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    raw_main_response: str = ""
    raw_repair_response: str = ""
    repaired: bool = False
    valid_action: bool = True


@dataclass
class ReActRunResult:
    question: str
    answer: str
    steps: list[ReActStep]
    trajectory: str
    termination_reason: str
    model_calls: int
    repair_calls: int
    invalid_actions: int
    search_calls: int
    lookup_calls: int
    usage: dict[str, int]
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [asdict(step) for step in self.steps]
        return payload


def parse_action(value: str) -> ParsedAction:
    raw = str(value or "").strip()
    match = re.fullmatch(
        r"(?is)(Search|Lookup|Finish)\[(.*)\]",
        raw,
    )
    if not match:
        raise ValueError(f"invalid ReAct action: {raw!r}")
    name = match.group(1).lower().capitalize()
    return ParsedAction(name=name, argument=match.group(2).strip(), raw=raw)


def _main_parts(raw: str, step: int) -> tuple[str, str] | None:
    text = str(raw or "").strip()
    marker = re.compile(rf"\nAction\s*{step}\s*:\s*", flags=re.IGNORECASE)
    split = marker.split(text, maxsplit=1)
    if len(split) != 2:
        return None
    thought = re.sub(
        rf"^Thought\s*{step}\s*:\s*",
        "",
        split[0].strip(),
        flags=re.IGNORECASE,
    )
    action_lines = split[1].strip().splitlines()
    if not action_lines:
        return None
    action = action_lines[0].strip()
    return thought, action


def _first_thought_line(raw: str, step: int) -> str:
    line = str(raw or "").strip().splitlines()[0] if str(raw or "").strip() else ""
    return re.sub(
        rf"^Thought\s*{step}\s*:\s*", "", line, flags=re.IGNORECASE
    ).strip()


def _add_usage(total: dict[str, int], usage: Mapping[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[key] += int(usage.get(key) or 0)


class ReActRunner:
    def __init__(
        self,
        completion_model: CompletionModel,
        *,
        model_name: str,
        max_steps: int = 7,
        temperature: float = 0.0,
        max_tokens: int = 100,
    ):
        self.completion_model = completion_model
        self.model_name = str(model_name)
        self.max_steps = int(max_steps)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def _complete(self, prompt: str, stop: str) -> tuple[str, dict[str, int]]:
        return self.completion_model.complete(
            self.model_name,
            prompt,
            stop,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def run(
        self,
        question: str,
        environment: ReActEnvironment,
        *,
        examples: str,
    ) -> ReActRunResult:
        started = time.perf_counter()
        prefix = build_prompt_prefix(examples)
        base = prefix + f"Question: {str(question).strip()}\n"
        trajectory = ""
        steps: list[ReActStep] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        model_calls = 0
        repair_calls = 0
        invalid_actions = 0
        search_calls = 0
        lookup_calls = 0
        answer = ""
        termination_reason = "step_limit"

        for step_number in range(1, self.max_steps + 1):
            main_prompt = base + trajectory + f"Thought {step_number}:"
            main_raw, main_usage = self._complete(
                main_prompt, f"\nObservation {step_number}:"
            )
            model_calls += 1
            _add_usage(usage, main_usage)
            parts = _main_parts(main_raw, step_number)
            repaired = parts is None
            repair_raw = ""
            if parts is None:
                thought = _first_thought_line(main_raw, step_number)
                repair_prompt = (
                    base
                    + trajectory
                    + f"Thought {step_number}: {thought}\nAction {step_number}:"
                )
                repair_raw, repair_usage = self._complete(repair_prompt, "\n")
                model_calls += 1
                repair_calls += 1
                _add_usage(usage, repair_usage)
                repair_lines = repair_raw.strip().splitlines()
                action_text = repair_lines[0].strip() if repair_lines else ""
            else:
                thought, action_text = parts

            valid_action = True
            try:
                action = parse_action(action_text)
            except ValueError:
                valid_action = False
                invalid_actions += 1
                action = ParsedAction("Invalid", action_text, action_text)

            if action.name == "Finish":
                answer = action.argument
                observation = ""
                evidence: list[dict[str, Any]] = []
                termination_reason = "finish"
            elif action.name in {"Search", "Lookup"}:
                if action.name == "Search":
                    search_calls += 1
                else:
                    lookup_calls += 1
                result = environment.execute(action.name, action.argument)
                observation = result.observation.replace("\\n", "")
                evidence = list(result.evidence)
            else:
                observation = "Invalid action. Use Search, Lookup, or Finish."
                evidence = []

            step_record = ReActStep(
                step=step_number,
                thought=thought,
                action=action_text,
                action_name=action.name,
                action_argument=action.argument,
                observation=observation,
                evidence=evidence,
                raw_main_response=main_raw,
                raw_repair_response=repair_raw,
                repaired=repaired,
                valid_action=valid_action,
            )
            steps.append(step_record)
            trajectory += (
                f"Thought {step_number}: {thought}\n"
                f"Action {step_number}: {action_text}\n"
                f"Observation {step_number}: {observation}\n"
            )
            if action.name == "Finish":
                break

        return ReActRunResult(
            question=str(question),
            answer=answer,
            steps=steps,
            trajectory=base + trajectory,
            termination_reason=termination_reason,
            model_calls=model_calls,
            repair_calls=repair_calls,
            invalid_actions=invalid_actions,
            search_calls=search_calls,
            lookup_calls=lookup_calls,
            usage=usage,
            elapsed_seconds=time.perf_counter() - started,
        )
