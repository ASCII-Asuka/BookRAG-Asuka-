from dataclasses import dataclass, field
import re
from typing import Any, List

from Core.rag.react_prompt import (
    ParsedReactResponse,
    build_react_prompt,
    parse_react_response,
)


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


class ReactRunner:
    def __init__(
        self,
        llm,
        environment,
        dataset_name,
        max_steps=7,
        prompt_file="",
    ):
        self.llm = llm
        self.environment = environment
        self.dataset_name = dataset_name
        self.max_steps = max(1, int(max_steps))
        self.prompt_file = prompt_file

    def run(self, question):
        trajectory = ""
        steps = []
        observed_evidence = []
        observed_ids = set()
        num_calls = 0
        num_bad_calls = 0
        search_count = 0
        lookup_count = 0

        for step_number in range(1, self.max_steps + 1):
            prompt = build_react_prompt(
                dataset_name=self.dataset_name,
                question=question,
                trajectory=trajectory,
                step=step_number,
                prompt_file=self.prompt_file,
            )
            raw = self.llm.get_completion(
                prompt,
                json_response=False,
            )
            num_calls += 1
            parsed = parse_react_response(
                f"Thought {step_number}:{raw}",
                step=step_number,
            )
            if not parsed.complete:
                num_bad_calls += 1
                repair_prompt = (
                    prompt
                    + parsed.thought
                    + f"\nAction {step_number}:"
                )
                repaired_raw = self.llm.get_completion(
                    repair_prompt,
                    json_response=False,
                )
                num_calls += 1
                repaired_text = str(repaired_raw or "").strip()
                if re.match(
                    rf"(?i)^Action\s*{step_number}\s*:",
                    repaired_text,
                ):
                    normalized_repair = (
                        f"Thought {step_number}: {parsed.thought}\n"
                        f"{repaired_text}"
                    )
                else:
                    normalized_repair = (
                        f"Thought {step_number}: {parsed.thought}\n"
                        f"Action {step_number}: {repaired_text}"
                    )
                repaired = parse_react_response(
                    normalized_repair,
                    step=step_number,
                )
                parsed = ParsedReactResponse(
                    thought=parsed.thought,
                    action=(
                        repaired.action
                        if repaired.complete
                        else repaired_text
                    ),
                    complete=repaired.complete,
                )
            observation = self.environment.step(parsed.action)
            if observation.action_type == "search":
                search_count += 1
            elif observation.action_type == "lookup":
                lookup_count += 1

            for evidence in observation.evidence:
                evidence_key = (type(evidence.block_id).__name__, str(evidence.block_id))
                if evidence_key in observed_ids:
                    continue
                observed_ids.add(evidence_key)
                observed_evidence.append(evidence)

            step_record = ReactStep(
                step=step_number,
                thought=parsed.thought,
                action=parsed.action,
                action_type=observation.action_type,
                observation=observation.text,
                valid_action=observation.valid_action,
                evidence_ids=[
                    evidence.block_id
                    for evidence in observation.evidence
                ],
            )
            steps.append(step_record)
            trajectory += (
                f"Thought {step_number}: {parsed.thought}\n"
                f"Action {step_number}: {parsed.action}\n"
                f"Observation {step_number}: {observation.text}\n"
            )
            if observation.done:
                return ReactRunResult(
                    answer=observation.answer,
                    steps=steps,
                    observed_evidence=observed_evidence,
                    num_calls=num_calls,
                    num_bad_calls=num_bad_calls,
                    search_count=search_count,
                    lookup_count=lookup_count,
                    termination_reason="finish",
                )

        finish_step = self.max_steps + 1
        finish_observation = self.environment.step("Finish[]")
        steps.append(
            ReactStep(
                step=finish_step,
                thought="",
                action="Finish[]",
                action_type=finish_observation.action_type,
                observation=finish_observation.text,
                valid_action=finish_observation.valid_action,
                evidence_ids=[],
            )
        )
        return ReactRunResult(
            answer=finish_observation.answer,
            steps=steps,
            observed_evidence=observed_evidence,
            num_calls=num_calls,
            num_bad_calls=num_bad_calls,
            search_count=search_count,
            lookup_count=lookup_count,
            termination_reason="step_limit",
        )
