import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


DEFAULT_ANSWER_TYPE_QUOTAS = {
    "extractive": 45,
    "abstractive": 25,
    "boolean": 15,
    "none": 15,
}
DEFAULT_MINIMUM_INTENT_COUNTS = {"comparison": 30, "multi-hop": 8}


def _question_id(row: Mapping[str, Any]) -> str:
    question_id = str(row.get("qasper_question_id") or "").strip()
    if not question_id:
        raise ValueError("Every Qasper row must have qasper_question_id")
    return question_id


def qasper_answer_type(row: Mapping[str, Any]) -> str:
    answers = list(row.get("answer", []) or [])
    answerable = [answer for answer in answers if not bool(answer.get("unanswerable"))]
    if not answerable:
        return "none"
    if any(answer.get("yes_no") is not None for answer in answerable):
        return "boolean"
    if any(answer.get("extractive_spans") for answer in answerable):
        return "extractive"
    return "abstractive"


def _shuffled_rows(
    rows: Sequence[Mapping[str, Any]], seed: int, salt: str
) -> List[Mapping[str, Any]]:
    ordered = sorted(rows, key=_question_id)
    digest = hashlib.sha256(f"{seed}:{salt}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    rng.shuffle(ordered)
    return ordered


def _validate_source_rows(
    rows: Sequence[Mapping[str, Any]], demand_intents: Mapping[str, str]
) -> None:
    question_ids = [_question_id(row) for row in rows]
    duplicates = [
        question_id
        for question_id, count in Counter(question_ids).items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate Qasper question IDs: {duplicates[:5]}")
    missing_intents = [
        question_id for question_id in question_ids if question_id not in demand_intents
    ]
    if missing_intents:
        raise ValueError(
            f"Missing frozen demand intents for {len(missing_intents)} questions: "
            f"{missing_intents[:5]}"
        )


def build_support_optimization_split(
    *,
    rows: Sequence[Mapping[str, Any]],
    demand_intents: Mapping[str, str],
    seed: int,
    target_size: int,
    answer_type_quotas: Mapping[str, int],
    minimum_intent_counts: Mapping[str, int],
) -> Dict[str, Any]:
    rows = list(rows)
    _validate_source_rows(rows, demand_intents)
    if sum(int(value) for value in answer_type_quotas.values()) != int(target_size):
        raise ValueError("Answer-type quotas must sum to target_size")
    if target_size > len(rows):
        raise ValueError("target_size exceeds source population")

    rows_by_id = {_question_id(row): row for row in rows}
    answer_type_by_id = {
        question_id: qasper_answer_type(row)
        for question_id, row in rows_by_id.items()
    }
    available_types = Counter(answer_type_by_id.values())
    for answer_type, quota in answer_type_quotas.items():
        if available_types[answer_type] < int(quota):
            raise ValueError(
                f"Answer-type quota {answer_type}={quota} exceeds available "
                f"{available_types[answer_type]}"
            )

    selected_ids = set()
    selected_type_counts = Counter()
    intent_order = sorted(
        minimum_intent_counts,
        key=lambda intent: (
            sum(value == intent for value in demand_intents.values()),
            intent,
        ),
    )
    for intent in intent_order:
        required = int(minimum_intent_counts[intent])
        available = [
            row
            for row in rows
            if demand_intents[_question_id(row)] == intent
        ]
        if len(available) < required:
            raise ValueError(
                f"Intent minimum {intent}={required} exceeds available {len(available)}"
            )
        shuffled = _shuffled_rows(available, seed, f"intent:{intent}")
        shuffled_index = {
            _question_id(row): index for index, row in enumerate(shuffled)
        }
        while sum(
            demand_intents[question_id] == intent for question_id in selected_ids
        ) < required:
            candidates = []
            for row in shuffled:
                question_id = _question_id(row)
                if question_id in selected_ids:
                    continue
                answer_type = answer_type_by_id[question_id]
                quota = int(answer_type_quotas.get(answer_type, 0))
                if selected_type_counts[answer_type] >= quota:
                    continue
                fill_ratio = selected_type_counts[answer_type] / max(quota, 1)
                candidates.append(
                    (fill_ratio, shuffled_index[question_id], question_id, answer_type)
                )
            if not candidates:
                raise ValueError(
                    f"Intent minimum {intent} cannot be satisfied within answer-type quotas"
                )
            _, _, question_id, answer_type = min(candidates)
            selected_ids.add(question_id)
            selected_type_counts[answer_type] += 1

    for answer_type, raw_quota in answer_type_quotas.items():
        quota = int(raw_quota)
        candidates = _shuffled_rows(
            [
                row
                for row in rows
                if answer_type_by_id[_question_id(row)] == answer_type
                and _question_id(row) not in selected_ids
            ],
            seed,
            f"answer-type:{answer_type}",
        )
        needed = quota - selected_type_counts[answer_type]
        if len(candidates) < needed:
            raise ValueError(
                f"Answer-type quota {answer_type} cannot be filled after intent sampling"
            )
        for row in candidates[:needed]:
            selected_ids.add(_question_id(row))
            selected_type_counts[answer_type] += 1

    tuning_rows = [row for row in rows if _question_id(row) in selected_ids]
    holdout_rows = [row for row in rows if _question_id(row) not in selected_ids]
    validate_support_split(
        source_rows=rows,
        tuning_rows=tuning_rows,
        holdout_rows=holdout_rows,
        demand_intents=demand_intents,
        answer_type_quotas=answer_type_quotas,
        minimum_intent_counts=minimum_intent_counts,
    )
    return {
        "tuning_rows": tuning_rows,
        "holdout_rows": holdout_rows,
        "selection_seed": int(seed),
        "answer_type_counts": dict(selected_type_counts),
        "intent_counts": dict(
            Counter(demand_intents[_question_id(row)] for row in tuning_rows)
        ),
    }


def validate_support_split(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    tuning_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    demand_intents: Mapping[str, str],
    answer_type_quotas: Mapping[str, int],
    minimum_intent_counts: Mapping[str, int],
) -> None:
    _validate_source_rows(source_rows, demand_intents)
    source_ids = {_question_id(row) for row in source_rows}
    tuning_ids = {_question_id(row) for row in tuning_rows}
    holdout_ids = {_question_id(row) for row in holdout_rows}
    if len(tuning_ids) != len(tuning_rows):
        raise ValueError("Tuning split contains duplicate question IDs")
    if len(holdout_ids) != len(holdout_rows):
        raise ValueError("Holdout split contains duplicate question IDs")
    overlap = tuning_ids & holdout_ids
    if overlap:
        raise ValueError(f"Tuning/holdout overlap: {sorted(overlap)[:5]}")
    if tuning_ids | holdout_ids != source_ids:
        missing = source_ids - (tuning_ids | holdout_ids)
        extra = (tuning_ids | holdout_ids) - source_ids
        raise ValueError(
            f"Split does not cover source exactly; missing={len(missing)} extra={len(extra)}"
        )

    type_counts = Counter(qasper_answer_type(row) for row in tuning_rows)
    expected_size = sum(int(value) for value in answer_type_quotas.values())
    if len(tuning_rows) != expected_size:
        raise ValueError(
            f"Tuning size {len(tuning_rows)} does not match quota total {expected_size}"
        )
    for answer_type, quota in answer_type_quotas.items():
        if type_counts[answer_type] != int(quota):
            raise ValueError(
                f"Answer-type quota mismatch for {answer_type}: "
                f"{type_counts[answer_type]} != {quota}"
            )

    intent_counts = Counter(
        demand_intents[_question_id(row)] for row in tuning_rows
    )
    for intent, minimum in minimum_intent_counts.items():
        if intent_counts[intent] < int(minimum):
            raise ValueError(
                f"Intent minimum mismatch for {intent}: "
                f"{intent_counts[intent]} < {minimum}"
            )


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_demand_intents(
    retrieval_root: Path, method: str
) -> Dict[str, str]:
    method_dir_name = f"eval_qasper_{method}"
    intents = {}
    for result_path in retrieval_root.rglob("result.json"):
        if result_path.parent.parent.name != method_dir_name:
            continue
        retrieval_path = result_path.with_name("retrieval_res.json")
        if not retrieval_path.exists():
            raise ValueError(f"Missing retrieval_res.json beside {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
        question_id = str(result.get("qasper_question_id") or "").strip()
        if not question_id:
            raise ValueError(f"Missing qasper_question_id in {result_path}")
        if question_id in intents:
            raise ValueError(f"Duplicate retrieval output for {question_id}")
        demand = retrieval.get("demand") or {}
        intent = str(demand.get("intent") or "fact").strip().lower()
        intents[question_id] = intent
    return intents


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_manifest_generation(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_path = Path(args.dataset).resolve()
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Qasper dataset must be a JSON list")
    if len(rows) != int(args.expected_source_count):
        raise ValueError(
            f"Expected {args.expected_source_count} source questions, got {len(rows)}"
        )
    demand_intents = load_frozen_demand_intents(
        Path(args.retrieval_root).resolve(), args.method
    )
    split = build_support_optimization_split(
        rows=rows,
        demand_intents=demand_intents,
        seed=args.seed,
        target_size=100,
        answer_type_quotas=DEFAULT_ANSWER_TYPE_QUOTAS,
        minimum_intent_counts=DEFAULT_MINIMUM_INTENT_COUNTS,
    )
    tuning_rows = split["tuning_rows"]
    holdout_rows = split["holdout_rows"]
    _write_json(Path(args.tuning_output), tuning_rows)
    _write_json(Path(args.holdout_output), holdout_rows)
    manifest = {
        "source_dataset": str(dataset_path),
        "source_sha256": dataset_sha256(dataset_path),
        "source_question_count": len(rows),
        "selection_seed": int(args.seed),
        "tuning_question_count": len(tuning_rows),
        "holdout_question_count": len(holdout_rows),
        "answer_type_quotas": DEFAULT_ANSWER_TYPE_QUOTAS,
        "minimum_intent_counts": DEFAULT_MINIMUM_INTENT_COUNTS,
        "answer_type_counts": split["answer_type_counts"],
        "intent_counts": split["intent_counts"],
        "tuning_questions": [
            {
                "question_id": _question_id(row),
                "doc_uuid": str(row.get("doc_uuid") or ""),
                "answer_type": qasper_answer_type(row),
                "demand_intent": demand_intents[_question_id(row)],
            }
            for row in tuning_rows
        ],
        "holdout_question_ids": [_question_id(row) for row in holdout_rows],
    }
    _write_json(Path(args.manifest_output), manifest)
    return manifest


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the locked Qasper support-pruning tuning/holdout split."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--retrieval-root", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--tuning-output", required=True)
    parser.add_argument("--holdout-output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-source-count", type=int, default=764)
    return parser


def main() -> None:
    manifest = run_manifest_generation(create_parser().parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
