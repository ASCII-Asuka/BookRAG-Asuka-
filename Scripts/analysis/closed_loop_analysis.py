"""Reconstruct and audit CoSE-RAG closed-loop retrieval behavior.

The analysis is deliberately post-hoc: it reads completed inference logs and
never uses gold evidence until after each round's selected context is fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ATOMIC_BLOCK_TYPES = {"paragraph"}


def _normal_id(value: Any) -> Any:
    if isinstance(value, int):
        return value
    text = str(value)
    try:
        return int(text)
    except ValueError:
        return text


def _hashable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_hashable(item) for item in value)
    return value


def evidence_prf(prediction: Sequence[Any], gold: Sequence[Any]) -> tuple[float, float, float]:
    """Set-based evidence precision, recall and F1 using official empty rules."""
    pred_set = {_hashable(item) for item in prediction}
    gold_set = {_hashable(item) for item in gold}
    if not pred_set and not gold_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not gold_set:
        return 0.0, 0.0, 0.0
    common = len(pred_set & gold_set)
    precision = common / len(pred_set)
    recall = common / len(gold_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _default_token_counter(text: str) -> int:
    # Keep the exact token-cost helper used by the evidence selector while
    # importing it lazily so this analysis remains importable in lean tests.
    from Core.rag.evibridge_selector import _token_cost

    return _token_cost(text)


def _atomic_block_ids(block_ids: Iterable[Any], blocks: Mapping[Any, Mapping[str, Any]]) -> list[Any]:
    output: list[Any] = []
    seen = set()
    for raw_id in block_ids:
        block_id = _normal_id(raw_id)
        block = blocks.get(block_id) or blocks.get(str(block_id))
        if not block or str(block.get("block_type", "")).lower() not in ATOMIC_BLOCK_TYPES:
            continue
        if block_id not in seen:
            seen.add(block_id)
            output.append(block_id)
    return output


def _block_view(block_id: Any, blocks: Mapping[Any, Mapping[str, Any]]) -> dict[str, Any]:
    block = blocks.get(block_id) or blocks.get(str(block_id)) or {}
    metadata = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
    return {
        "block_id": block_id,
        "block_type": block.get("block_type"),
        "source_node_id": block.get("source_node_id"),
        "doc_name": block.get("doc_name"),
        "doc_path": block.get("doc_path"),
        "page": block.get("page"),
        "section_id": block.get("section_id"),
        "title_path": block.get("title_path"),
        "parent_id": block.get("parent_id"),
        "text": str(block.get("text") or ""),
        "metadata": metadata,
    }


def _selected_ids(iteration: Mapping[str, Any], blocks: Mapping[Any, Mapping[str, Any]]) -> list[Any]:
    output = []
    seen = set()
    for raw_id in iteration.get("selected_block_ids") or []:
        block_id = _normal_id(raw_id)
        if not (blocks.get(block_id) or blocks.get(str(block_id))):
            raise ValueError(f"selected block_id {block_id} has no source mapping")
        if block_id in seen:
            continue
        seen.add(block_id)
        output.append(block_id)
    return output


def _qasper_gold_groups(result: Mapping[str, Any]) -> list[list[Any]]:
    groups = []
    for answer in result.get("answer") or []:
        ids = answer.get("evidence_paragraph_ids") or []
        groups.append([_normal_id(item) for item in ids])
    return groups or [[]]


def _hotpot_gold(result: Mapping[str, Any]) -> list[list[Any]]:
    return [[str(item[0]), int(item[1])] for item in (result.get("hotpot_supporting_facts") or [])]


def _hotpot_prediction(block_ids: Sequence[Any], result: Mapping[str, Any], blocks: Mapping[Any, Mapping[str, Any]]) -> list[list[Any]]:
    node_facts = result.get("hotpot_node_facts") or {}
    facts: list[list[Any]] = []
    seen = set()
    for block_id in block_ids:
        fact = node_facts.get(str(block_id)) or node_facts.get(block_id)
        if not fact:
            block = blocks.get(block_id) or blocks.get(str(block_id)) or {}
            metadata = block.get("metadata") or {}
            title = metadata.get("hotpot_title") or block.get("hotpot_title")
            sent_id = metadata.get("hotpot_sent_id", block.get("hotpot_sent_id"))
            if title is not None and sent_id is not None:
                fact = [title, sent_id]
        if not fact or len(fact) < 2:
            continue
        normalized = [str(fact[0]), int(fact[1])]
        key = tuple(normalized)
        if key not in seen:
            seen.add(key)
            facts.append(normalized)
    return facts


def _round_evidence(
    dataset: str,
    atomic_ids: Sequence[Any],
    result: Mapping[str, Any],
    blocks: Mapping[Any, Mapping[str, Any]],
) -> tuple[list[Any], float, float, float]:
    if dataset == "qasper":
        prediction = list(atomic_ids)
        candidates = [(*evidence_prf(prediction, group), group) for group in _qasper_gold_groups(result)]
        precision, recall, f1, _ = max(candidates, key=lambda item: item[2])
        return prediction, precision, recall, f1
    if dataset == "hotpotqa":
        prediction = _hotpot_prediction(atomic_ids, result, blocks)
        precision, recall, f1 = evidence_prf(prediction, _hotpot_gold(result))
        return prediction, precision, recall, f1
    raise ValueError(f"unsupported dataset: {dataset}")


def _verification(iteration: Mapping[str, Any]) -> Mapping[str, Any]:
    value = iteration.get("verification")
    return value if isinstance(value, dict) else {}


def build_query_log(
    dataset: str,
    result: Mapping[str, Any],
    retrieval: Mapping[str, Any],
    blocks: Mapping[Any, Mapping[str, Any]],
    token_counter: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Build one auditable record from a completed result and retrieval log."""
    token_counter = token_counter or _default_token_counter
    rounds = []
    for raw_iteration in retrieval.get("iterations") or []:
        selected_ids = _selected_ids(raw_iteration, blocks)
        atomic_ids = _atomic_block_ids(selected_ids, blocks)
        auxiliary_ids = [item for item in selected_ids if item not in set(atomic_ids)]
        prediction, precision, recall, f1 = _round_evidence(dataset, atomic_ids, result, blocks)
        verification = _verification(raw_iteration)
        candidate_ids = sorted(
            {_normal_id(item) for item in (raw_iteration.get("candidate_scores") or {}).keys()},
            key=str,
        )
        rounds.append(
            {
                "iteration": int(raw_iteration.get("iteration") or len(rounds) + 1),
                "selected_block_ids": selected_ids,
                "selected_atomic_ids": atomic_ids,
                "selected_auxiliary_ids": auxiliary_ids,
                "selected_atomic_blocks": [_block_view(item, blocks) for item in atomic_ids],
                "atomic_block_count": len(atomic_ids),
                "auxiliary_node_count": len(auxiliary_ids),
                "context_tokens": sum(
                    token_counter(str((blocks.get(item) or blocks.get(str(item)) or {}).get("text") or ""))
                    for item in atomic_ids
                ),
                "candidate_ids": candidate_ids,
                "sufficient": bool(verification.get("sufficient", False)),
                "missing": list(verification.get("missing") or []),
                "missing_types": list(verification.get("missing_types") or []),
                "missing_bridge_types": list(verification.get("missing_bridge_types") or []),
                "next_action": verification.get("next_action") or raw_iteration.get("next_action"),
                "predicted_evidence": prediction,
                "evidence_precision": precision,
                "evidence_recall": recall,
                "evidence_f1": f1,
            }
        )
    if not rounds:
        raise ValueError("retrieval log contains no iterations")

    first_ids = rounds[0]["selected_atomic_ids"]
    final_ids = rounds[-1]["selected_atomic_ids"]
    first_set = set(first_ids)
    added_ids = [item for item in final_ids if item not in first_set]
    removed_ids = [item for item in first_ids if item not in set(final_ids)]
    added_tokens = sum(
        token_counter(str((blocks.get(item) or blocks.get(str(item)) or {}).get("text") or ""))
        for item in added_ids
    )
    first_iteration = (retrieval.get("iterations") or [{}])[0]
    refinement = first_iteration.get("refinement") if isinstance(first_iteration.get("refinement"), dict) else {}
    qid = (
        result.get("qasper_question_id")
        if dataset == "qasper"
        else result.get("hotpotqa_question_id") or result.get("question_id") or result.get("_id") or result.get("doc_uuid")
    )
    second_raw = (retrieval.get("iterations") or [{}, {}])[1] if len(retrieval.get("iterations") or []) >= 2 else {}
    relevant_paths = []
    for path in second_raw.get("connector_paths") or []:
        path_nodes = {_normal_id(item) for item in (path.get("nodes") or [])}
        if path_nodes & set(added_ids):
            relevant_paths.append(path)
    relevant_bridges = []
    for edge in retrieval.get("selected_bridges") or []:
        endpoints = {_normal_id(edge.get("source_id")), _normal_id(edge.get("target_id"))}
        if endpoints & set(added_ids):
            relevant_bridges.append(edge)
    return {
        "dataset": dataset,
        "question_id": str(qid or ""),
        "doc_uuid": str(result.get("doc_uuid") or ""),
        "question": str(result.get("question") or ""),
        "question_type": result.get("hotpot_type") if dataset == "hotpotqa" else result.get("answer_format"),
        "round_count": len(rounds),
        "y_1": bool(rounds[0]["sufficient"]),
        "y_2": bool(rounds[1]["sufficient"]) if len(rounds) >= 2 else None,
        "y_final": bool(rounds[-1]["sufficient"]),
        "first_round_accepted": bool(rounds[0]["sufficient"]),
        "second_round_triggered": len(rounds) >= 2,
        "final_sufficient": bool(rounds[-1]["sufficient"]),
        "stopping_reason": retrieval.get("stopping_reason"),
        "refinement_query": refinement.get("query"),
        "missing_demand": list(rounds[0].get("missing") or rounds[0].get("missing_types") or []),
        "suggested_bridge_types": list(rounds[0].get("missing_bridge_types") or []),
        "controlled_action": rounds[0].get("next_action"),
        "first_atomic_block_count": rounds[0]["atomic_block_count"],
        "final_atomic_block_count": rounds[-1]["atomic_block_count"],
        "first_auxiliary_node_count": rounds[0]["auxiliary_node_count"],
        "final_auxiliary_node_count": rounds[-1]["auxiliary_node_count"],
        "first_context_tokens": rounds[0]["context_tokens"],
        "final_context_tokens": rounds[-1]["context_tokens"],
        "first_evidence_precision": rounds[0]["evidence_precision"],
        "first_evidence_recall": rounds[0]["evidence_recall"],
        "first_evidence_f1": rounds[0]["evidence_f1"],
        "final_evidence_precision": rounds[-1]["evidence_precision"],
        "final_evidence_recall": rounds[-1]["evidence_recall"],
        "final_evidence_f1": rounds[-1]["evidence_f1"],
        "added_atomic_ids": added_ids,
        "removed_atomic_ids": removed_ids,
        "added_atomic_block_count": len(added_ids),
        "added_context_tokens": added_tokens,
        "evidence_gain": rounds[-1]["evidence_f1"] - rounds[0]["evidence_f1"],
        "added_bridge_paths": relevant_paths,
        "added_selected_bridges": relevant_bridges,
        "rounds": rounds,
    }


def validate_query_logs(logs: Sequence[Mapping[str, Any]], expected_total: int) -> None:
    if len(logs) != expected_total:
        raise ValueError(f"expected {expected_total} query logs, found {len(logs)}")
    question_ids = [str(log.get("question_id") or "") for log in logs]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("duplicate question_id in query logs")
    if any(not item for item in question_ids):
        raise ValueError("empty question_id in query logs")
    for log in logs:
        rounds = log.get("rounds") or []
        if not 1 <= int(log.get("round_count") or 0) <= 2:
            raise ValueError(f"round_count exceeds R_max for {log.get('question_id')}")
        if bool(log.get("second_round_triggered")) != (len(rounds) >= 2):
            raise ValueError(f"round trigger mismatch for {log.get('question_id')}")
        if log.get("first_round_accepted") and log.get("second_round_triggered"):
            raise ValueError(f"accepted first round unexpectedly triggered expansion for {log.get('question_id')}")
        if len(rounds) >= 2:
            candidates = set(rounds[1].get("candidate_ids") or [])
            added = set(log.get("added_atomic_ids") or [])
            if not added.issubset(candidates):
                raise ValueError(
                    f"added evidence is not in second-round candidates for {log.get('question_id')}"
                )


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def aggregate_dataset(logs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not logs:
        raise ValueError("cannot aggregate empty query logs")
    triggered = [log for log in logs if log.get("second_round_triggered")]
    expanded_accept = [log for log in triggered if log.get("final_sufficient")]
    termination_counts = Counter(str(log.get("stopping_reason") or "unspecified") for log in logs)
    final_insufficient_termination_counts = Counter(
        str(log.get("stopping_reason") or "unspecified")
        for log in logs
        if not log.get("final_sufficient")
    )
    unaccepted_without_second = [
        log for log in logs if not log.get("first_round_accepted") and not log.get("second_round_triggered")
    ]
    return {
        "dataset": logs[0].get("dataset"),
        "total_questions": len(logs),
        "first_round_accept_count": sum(bool(log.get("first_round_accepted")) for log in logs),
        "first_round_accept_rate": _mean([float(bool(log.get("first_round_accepted"))) for log in logs]),
        "second_round_trigger_count": len(triggered),
        "second_round_trigger_rate": len(triggered) / len(logs),
        "unaccepted_without_second_round_count": len(unaccepted_without_second),
        "expanded_accept_count": len(expanded_accept),
        "expanded_accept_denominator": len(triggered),
        "expanded_accept_rate": len(expanded_accept) / len(triggered) if triggered else 0.0,
        "final_insufficient_count": sum(not bool(log.get("final_sufficient")) for log in logs),
        "final_insufficient_rate": _mean([float(not bool(log.get("final_sufficient"))) for log in logs]),
        "mean_round_count": _mean([float(log.get("round_count") or 0.0) for log in logs]),
        "triggered_mean_added_atomic_blocks": _mean(
            [float(log.get("added_atomic_block_count") or 0.0) for log in triggered]
        ),
        "triggered_first_evidence_metric": _mean(
            [float((log.get("rounds") or [{}])[0].get("evidence_f1") or 0.0) for log in triggered]
        ),
        "triggered_final_evidence_metric": _mean(
            [float((log.get("rounds") or [{}])[-1].get("evidence_f1") or 0.0) for log in triggered]
        ),
        "triggered_mean_evidence_gain": _mean([float(log.get("evidence_gain") or 0.0) for log in triggered]),
        "triggered_mean_added_context_tokens": _mean(
            [float(log.get("added_context_tokens") or 0.0) for log in triggered]
        ),
        "termination_counts": dict(sorted(termination_counts.items())),
        "final_insufficient_termination_counts": dict(sorted(final_insufficient_termination_counts.items())),
        "consistency_checks_passed": True,
    }


def choose_median_gain_case(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("no eligible trajectory cases")
    ordered = sorted(candidates, key=lambda item: (float(item.get("evidence_gain") or 0.0), str(item.get("question_id") or "")))
    median_gain = statistics.median(float(item.get("evidence_gain") or 0.0) for item in ordered)
    selected = min(
        ordered,
        key=lambda item: (abs(float(item.get("evidence_gain") or 0.0) - median_gain), str(item.get("question_id") or "")),
    )
    return dict(selected)


def choose_successful_qasper_case(
    strict_candidates: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not strict_candidates:
        raise ValueError("no strict Qasper trajectory cases")
    reference_median = float(
        statistics.median(float(item.get("evidence_gain") or 0.0) for item in strict_candidates)
    )
    successful = [
        item
        for item in strict_candidates
        if float(item.get("answer_f1") or 0.0) >= 0.8
        and bool(item.get("added_gold_support_ids"))
    ]
    if not successful:
        raise ValueError("no successful-and-cited Qasper trajectory cases")
    selected = min(
        successful,
        key=lambda item: (
            abs(float(item.get("evidence_gain") or 0.0) - reference_median),
            str(item.get("question_id") or ""),
        ),
    )
    return dict(selected), {
        "strict_candidate_count": len(strict_candidates),
        "successful_candidate_count": len(successful),
        "candidate_median_evidence_gain": reference_median,
        "selected_answer_f1": float(selected.get("answer_f1") or 0.0),
        "selected_added_gold_support_ids": list(selected.get("added_gold_support_ids") or []),
    }


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _index_blocks(doc_dir: Path) -> dict[Any, dict[str, Any]]:
    payload = _load_json(doc_dir / "evibridge_index.json")
    blocks = payload.get("blocks") or []
    output = {}
    for block in blocks:
        block_id = _normal_id(block.get("block_id"))
        if block_id in output:
            raise ValueError(f"duplicate block_id {block_id} in {doc_dir}")
        output[block_id] = block
    return output


def load_experiment_logs(
    dataset: str,
    root: Path,
    method_dir_name: str,
    expected_total: int,
) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    for doc_dir in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name):
        method_dir = doc_dir / method_dir_name
        if not method_dir.is_dir():
            continue
        blocks = _index_blocks(doc_dir)
        query_dirs = sorted(
            (item for item in method_dir.iterdir() if item.is_dir() and item.name.startswith("query_")),
            key=lambda item: item.name,
        )
        for query_dir in query_dirs:
            result_path = query_dir / "result.json"
            retrieval_path = query_dir / "retrieval_res.json"
            if not result_path.is_file() or not retrieval_path.is_file():
                raise ValueError(f"incomplete query output: {query_dir}")
            result = _load_json(result_path)
            retrieval = _load_json(retrieval_path)
            log = build_query_log(dataset, result, retrieval, blocks)
            log["source_query_dir"] = str(query_dir.resolve())
            log["source_index"] = str((doc_dir / "evibridge_index.json").resolve())
            logs.append(log)
    validate_query_logs(logs, expected_total=expected_total)
    return logs


def _qasper_gold_answers(result: Mapping[str, Any]) -> list[str]:
    answers = []
    for annotation in result.get("answer") or []:
        if annotation.get("unanswerable"):
            value = "Unanswerable"
        elif annotation.get("yes_no") is not None:
            value = "Yes" if bool(annotation.get("yes_no")) else "No"
        elif annotation.get("extractive_spans"):
            value = "; ".join(str(item) for item in annotation.get("extractive_spans") or [])
        else:
            value = str(annotation.get("free_form_answer") or "").strip()
        if value and value not in answers:
            answers.append(value)
    return answers


def qasper_answer_f1(result: Mapping[str, Any]) -> float:
    from Scripts.eval.qasper_official import token_f1_score

    prediction = str(result.get("answer_short") or result.get("output") or "")
    references = _qasper_gold_answers(result)
    return max((token_f1_score(prediction, reference) for reference in references), default=0.0)


def _case_gold_answer(dataset: str, result: Mapping[str, Any]) -> Any:
    if dataset == "qasper":
        return _qasper_gold_answers(result)
    return result.get("hotpot_answer", result.get("answer"))


def _baseline_record(
    dataset: str,
    query_dir: Path,
    blocks: Mapping[Any, Mapping[str, Any]],
    main_result: Mapping[str, Any],
) -> dict[str, Any]:
    result = _load_json(query_dir / "result.json")
    retrieval = _load_json(query_dir / "retrieval_res.json")
    evidence = retrieval.get("supporting_evidence") or []
    evidence_views = []
    ids = []
    for item in evidence:
        raw_id = item.get("id", item.get("block_id", item.get("source_node_id")))
        block_id = _normal_id(raw_id)
        ids.append(block_id)
        evidence_views.append(
            {
                "block_id": block_id,
                "source_node_id": item.get("source_node_id", item.get("node_id")),
                "page": item.get("page"),
                "section_id": item.get("section_id") or item.get("section") or item.get("hotpot_title"),
                "hotpot_title": item.get("hotpot_title"),
                "hotpot_sent_id": item.get("hotpot_sent_id"),
                "text": str(item.get("content") or item.get("text") or item.get("qasper_evidence_text") or ""),
                "rank": item.get("supporting_rank", item.get("rank")),
            }
        )
    atomic_ids = _atomic_block_ids(ids, blocks)
    prediction, precision, recall, f1 = _round_evidence(dataset, atomic_ids, main_result, blocks)
    return {
        "answer": result.get("answer_short") or result.get("output"),
        "supporting_evidence": evidence_views,
        "predicted_evidence": prediction,
        "evidence_precision": precision,
        "evidence_recall": recall,
        "evidence_f1": f1,
        "source_query_dir": str(query_dir.resolve()),
    }


def _gold_atomic_ids(dataset: str, result: Mapping[str, Any], blocks: Mapping[Any, Mapping[str, Any]]) -> set[Any]:
    if dataset == "qasper":
        return {item for group in _qasper_gold_groups(result) for item in group}
    gold = {tuple(item) for item in _hotpot_gold(result)}
    matching = set()
    for item in blocks:
        predicted = _hotpot_prediction([item], result, blocks)
        if predicted and tuple(predicted[0]) in gold:
            matching.add(item)
    return matching


def _case_candidate(
    dataset: str,
    log: Mapping[str, Any],
    result: Mapping[str, Any],
    blocks: Mapping[Any, Mapping[str, Any]],
) -> bool:
    if not log.get("second_round_triggered") or float(log.get("evidence_gain") or 0.0) <= 0:
        return False
    gold_ids = _gold_atomic_ids(dataset, result, blocks)
    if not set(log.get("added_atomic_ids") or []) & gold_ids:
        return False
    if dataset == "qasper":
        return max((len(group) for group in _qasper_gold_groups(result)), default=0) >= 2
    return str(result.get("hotpot_type") or "").lower() == "bridge"


def _make_case(
    dataset: str,
    log: Mapping[str, Any],
    method_dir_name: str,
    baseline_dir_name: str,
) -> dict[str, Any]:
    query_dir = Path(str(log["source_query_dir"]))
    doc_dir = query_dir.parent.parent
    result = _load_json(query_dir / "result.json")
    retrieval = _load_json(query_dir / "retrieval_res.json")
    blocks = _index_blocks(doc_dir)
    baseline_dir = doc_dir / baseline_dir_name / query_dir.name
    if not baseline_dir.is_dir():
        raise ValueError(f"missing BM25+Reranker case output: {baseline_dir}")
    final_ids = set((log.get("rounds") or [{}])[-1].get("selected_atomic_ids") or [])
    supporting_ids = [_normal_id(item) for item in (result.get("supporting_block_ids") or [])]
    invalid_support = [
        item
        for item in supporting_ids
        if item not in final_ids
        or str((blocks.get(item) or {}).get("block_type") or "").lower() not in ATOMIC_BLOCK_TYPES
    ]
    if invalid_support:
        raise ValueError(
            f"final supporting evidence is outside generated atomic context for {log.get('question_id')}: {invalid_support}"
        )
    gold_ids = _gold_atomic_ids(dataset, result, blocks)
    added_gold_matching = [item for item in log.get("added_atomic_ids") or [] if item in gold_ids]
    added_gold_support = [item for item in added_gold_matching if item in supporting_ids]
    return {
        "dataset": dataset,
        "question_id": log.get("question_id"),
        "doc_uuid": log.get("doc_uuid"),
        "question_type": log.get("question_type"),
        "question": result.get("question"),
        "gold_answer": _case_gold_answer(dataset, result),
        "gold_evidence": (
            _qasper_gold_groups(result) if dataset == "qasper" else _hotpot_gold(result)
        ),
        "bm25_reranker": _baseline_record(dataset, baseline_dir, blocks, result),
        "round_1": (log.get("rounds") or [{}])[0],
        "diagnosis": {
            "missing_demand": log.get("missing_demand"),
            "suggested_bridge_types": log.get("suggested_bridge_types"),
            "controlled_action": log.get("controlled_action"),
            "refinement_query": log.get("refinement_query"),
        },
        "round_2_added_evidence": [_block_view(item, blocks) for item in log.get("added_atomic_ids") or []],
        "round_2_added_gold_matching_ids": added_gold_matching,
        "round_2_added_gold_support_ids": added_gold_support,
        "round_2_bridge_paths": log.get("added_bridge_paths") or [],
        "round_2_selected_bridges": log.get("added_selected_bridges") or [],
        "final_output": {
            "answer": result.get("answer_short") or result.get("output"),
            "answer_f1": qasper_answer_f1(result) if dataset == "qasper" else None,
            "supporting_block_ids": supporting_ids,
            "supporting_blocks": [_block_view(item, blocks) for item in supporting_ids],
            "evidence_precision": (log.get("rounds") or [{}])[-1].get("evidence_precision"),
            "evidence_recall": (log.get("rounds") or [{}])[-1].get("evidence_recall"),
            "evidence_f1": (log.get("rounds") or [{}])[-1].get("evidence_f1"),
        },
        "evidence_gain": log.get("evidence_gain"),
        "source_query_dir": str(query_dir.resolve()),
        "source_index": str((doc_dir / "evibridge_index.json").resolve()),
        "method_directory": method_dir_name,
    }


def select_cases(
    logs_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    method_dirs: Mapping[str, str],
    baseline_dirs: Mapping[str, str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "selection_protocol": (
            "Filter actual second-round runs with positive evidence gain and a newly added gold-matching "
            "atomic block; require multi-block gold evidence for Qasper and bridge type for HotpotQA. "
            "For Qasper, additionally require final Answer F1 >= 0.8 and require an added gold-matching "
            "block to appear in final supporting evidence, then select nearest the original strict-candidate "
            "evidence-gain median. For HotpotQA, select nearest its strict-candidate median. Break ties by question_id."
        ),
        "gold_usage": "Gold answers/evidence are used only after inference for evaluation and case filtering.",
        "cases": {},
    }
    for dataset, logs in logs_by_dataset.items():
        eligible = []
        for log in logs:
            query_dir = Path(str(log["source_query_dir"]))
            result = _load_json(query_dir / "result.json")
            blocks = _index_blocks(query_dir.parent.parent)
            if _case_candidate(dataset, log, result, blocks):
                enriched = dict(log)
                if dataset == "qasper":
                    gold_ids = _gold_atomic_ids(dataset, result, blocks)
                    supporting_ids = {
                        _normal_id(item) for item in (result.get("supporting_block_ids") or [])
                    }
                    enriched["answer_f1"] = qasper_answer_f1(result)
                    enriched["added_gold_support_ids"] = [
                        item
                        for item in log.get("added_atomic_ids") or []
                        if item in gold_ids and item in supporting_ids
                    ]
                eligible.append(enriched)
        rule = "strict"
        if not eligible:
            eligible = [
                log
                for log in logs
                if log.get("second_round_triggered") and float(log.get("evidence_gain") or 0.0) > 0
            ]
            rule = "fallback: triggered second round with positive evidence gain"
        if dataset == "qasper" and rule == "strict":
            selected, selection_audit = choose_successful_qasper_case(eligible)
            rule = "successful-and-cited candidate nearest the original strict-candidate median"
        else:
            selected = choose_median_gain_case(eligible)
            selection_audit = {
                "strict_candidate_count": len(eligible),
                "successful_candidate_count": None,
                "candidate_median_evidence_gain": statistics.median(
                    float(item.get("evidence_gain") or 0.0) for item in eligible
                ),
            }
        case = _make_case(dataset, selected, method_dirs[dataset], baseline_dirs[dataset])
        case["selection"] = {
            "rule": rule,
            "candidate_count": selection_audit["strict_candidate_count"],
            **selection_audit,
            "selected_evidence_gain": float(selected.get("evidence_gain") or 0.0),
        }
        payload["cases"][dataset] = case
    return payload


def write_outputs(
    output_dir: Path,
    logs_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    cases: Mapping[str, Any],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_path = output_dir / "closed_loop_query_logs.jsonl"
    with logs_path.open("w", encoding="utf-8") as handle:
        for dataset in ("qasper", "hotpotqa"):
            for log in logs_by_dataset[dataset]:
                handle.write(json.dumps(log, ensure_ascii=False) + "\n")
    summaries = [aggregate_dataset(logs_by_dataset[dataset]) for dataset in ("qasper", "hotpotqa")]
    summary_path = output_dir / "closed_loop_behavior_summary.csv"
    fieldnames = list(summaries[0].keys())
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            serializable = {
                key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, dict) else value
                for key, value in row.items()
            }
            writer.writerow(serializable)
    cases_path = output_dir / "retrieval_trajectory_cases.json"
    cases_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return logs_path, summary_path, cases_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qasper-root", required=True)
    parser.add_argument("--hotpot-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    roots = {"qasper": Path(args.qasper_root), "hotpotqa": Path(args.hotpot_root)}
    method_dirs = {
        "qasper": "eval_qasper_evibridge_support_pruned",
        "hotpotqa": "eval_hotpotqa_evibridge",
    }
    baseline_dirs = {
        "qasper": "eval_qasper_bm25_rerank",
        "hotpotqa": "eval_hotpotqa_bm25_rerank",
    }
    logs_by_dataset = {
        "qasper": load_experiment_logs("qasper", roots["qasper"], method_dirs["qasper"], 1005),
        "hotpotqa": load_experiment_logs("hotpotqa", roots["hotpotqa"], method_dirs["hotpotqa"], 1000),
    }
    cases = select_cases(logs_by_dataset, method_dirs, baseline_dirs)
    paths = write_outputs(Path(args.output_dir), logs_by_dataset, cases)
    print(json.dumps({"outputs": [str(path.resolve()) for path in paths]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
