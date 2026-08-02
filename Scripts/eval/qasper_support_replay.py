import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Collection, Dict, Mapping, Sequence
from urllib.parse import urlsplit


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Core.rag.evibridge_support_pruner import prune_supporting_evidence
from Eval.utils.paired_bootstrap import paired_bootstrap_ci
from Scripts.eval.qasper_official import (
    _gold_answers_and_evidence,
    evaluate_qasper_official,
    evaluate_qasper_official_details,
)


DEFAULT_SUPPORT_TYPES = {"paragraph", "table", "caption", "figure"}
_SENSITIVE_KEY_PARTS = ("api_key", "token", "secret", "password", "authorization")


def _int_key_mapping(values: Mapping[Any, Any]) -> Dict[int, Any]:
    return {int(key): value for key, value in (values or {}).items()}


def reconstruct_support_candidates(
    *,
    retrieval: Mapping[str, Any],
    index_payload: Mapping[str, Any],
    topk: int = 20,
    allowed_types: Collection[str] = DEFAULT_SUPPORT_TYPES,
) -> list[Dict[str, Any]]:
    allowed = set(allowed_types)
    selected = list(
        retrieval.get("selected_payload") or retrieval.get("selected") or []
    )
    selected_by_id = {
        int(item["block_id"]): dict(item)
        for item in selected
        if item.get("block_id") is not None
        and item.get("block_type") in allowed
        and item.get("evidence_role") != "bridge_auxiliary"
    }
    blocks_by_id = {
        int(item["block_id"]): item
        for item in index_payload.get("blocks", []) or []
        if item.get("block_id") is not None
    }
    raw_scores = _int_key_mapping(retrieval.get("typed_ppr_scores") or {})
    raw_parts = _int_key_mapping(retrieval.get("typed_ppr_score_parts") or {})
    ranked_ids = []
    for block_id, raw_score in raw_scores.items():
        block = blocks_by_id.get(block_id)
        if block is None or block.get("block_type") not in allowed:
            continue
        parts = raw_parts.get(block_id) or {}
        rerank_rank = parts.get("rerank_rank")
        ranked_ids.append(
            (
                block_id,
                int(rerank_rank) if rerank_rank is not None else 10**9,
                -float(raw_score),
            )
        )
    ranked_ids.sort(key=lambda item: (item[1], item[2], item[0]))
    ordered_ids = list(selected_by_id)
    ordered_ids.extend(
        block_id for block_id, _, _ in ranked_ids if block_id not in selected_by_id
    )
    if int(topk) > 0:
        ordered_ids = ordered_ids[: int(topk)]

    candidates = []
    for block_id in ordered_ids:
        if block_id in selected_by_id:
            item = dict(selected_by_id[block_id])
            item["score_parts"] = dict(item.get("score_parts") or {})
        else:
            block = blocks_by_id[block_id]
            item = {
                "block_id": block_id,
                "block_type": block.get("block_type"),
                "page": block.get("page"),
                "section_id": block.get("section_id"),
                "metadata": dict(block.get("metadata") or {}),
                "text": str(block.get("text") or ""),
                "score": float(raw_scores.get(block_id, 0.0)),
                "score_parts": dict(raw_parts.get(block_id) or {}),
                "bridge_types": [],
                "evidence_role": "answer_evidence",
                "selection_rank": None,
            }
        candidates.append(item)
    return candidates


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _query_outputs_by_question_id(
    retrieval_root: Path, method: str
) -> Dict[str, Dict[str, Path]]:
    method_dir_name = f"eval_qasper_{method}"
    outputs = {}
    for result_path in retrieval_root.rglob("result.json"):
        if result_path.parent.parent.name != method_dir_name:
            continue
        result = _load_json(result_path)
        question_id = str(result.get("qasper_question_id") or "").strip()
        if not question_id:
            raise ValueError(f"Missing qasper_question_id in {result_path}")
        if question_id in outputs:
            raise ValueError(f"Duplicate query output for {question_id}")
        retrieval_path = result_path.with_name("retrieval_res.json")
        index_path = result_path.parents[2] / "evibridge_index.json"
        if not retrieval_path.exists():
            raise ValueError(f"Missing retrieval output for {question_id}")
        if not index_path.exists():
            raise ValueError(f"Missing EviBridge index for {question_id}")
        outputs[question_id] = {
            "result": result_path,
            "retrieval": retrieval_path,
            "index": index_path,
        }
    return outputs


def _is_unanswerable(answer: Any) -> bool:
    normalized = " ".join(
        "".join(
            character if character.isalpha() else " "
            for character in str(answer or "").lower()
        ).split()
    )
    return normalized in {
        "unanswerable",
        "not answerable",
        "not enough information",
        "cannot be answered",
        "no answer",
        "unknown",
    }


def _verifier_state_contradiction(verification: Mapping[str, Any]) -> bool:
    if not verification:
        return False
    sufficient = bool(verification.get("sufficient"))
    action = str(verification.get("next_action") or "").strip().lower()
    missing = any(
        verification.get(key)
        for key in ("missing", "missing_types", "missing_bridge_types")
    )
    if sufficient:
        return action != "accept" or missing
    return action == "accept"


def build_replay_cases(
    *,
    dataset_rows: Sequence[Mapping[str, Any]],
    retrieval_root: Path,
    method: str,
    topk: int = 20,
    allowed_types: Collection[str] = DEFAULT_SUPPORT_TYPES,
) -> list[Dict[str, Any]]:
    outputs = _query_outputs_by_question_id(Path(retrieval_root), method)
    dataset_ids = [
        str(row.get("qasper_question_id") or "").strip() for row in dataset_rows
    ]
    if not all(dataset_ids) or len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("Replay dataset must have unique non-empty question IDs")
    missing = [question_id for question_id in dataset_ids if question_id not in outputs]
    if missing:
        raise ValueError(
            f"Missing completed outputs for {len(missing)} replay questions: {missing[:5]}"
        )

    index_cache = {}
    cases = []
    for row, question_id in zip(dataset_rows, dataset_ids):
        paths = outputs[question_id]
        result = _load_json(paths["result"])
        retrieval = _load_json(paths["retrieval"])
        index_path = paths["index"]
        if index_path not in index_cache:
            index_cache[index_path] = _load_json(index_path)
        index_payload = index_cache[index_path]
        blocks_by_id = {
            int(item["block_id"]): item
            for item in index_payload.get("blocks", []) or []
            if item.get("block_id") is not None
        }
        candidates = reconstruct_support_candidates(
            retrieval=retrieval,
            index_payload=index_payload,
            topk=topk,
            allowed_types=allowed_types,
        )
        text_by_id = {
            int(item["block_id"]): str(item.get("text") or "")
            for item in candidates
        }
        controller = retrieval.get("support_controller") or {}
        citation_validation = retrieval.get("citation_validation") or {}
        anchors = list(
            controller.get("anchored_block_ids")
            or citation_validation.get("valid_ids")
            or []
        )
        current_final_ids = list(
            retrieval.get("supporting_block_ids")
            or result.get("supporting_block_ids")
            or []
        )
        for block_id in current_final_ids:
            block_id = int(block_id)
            if block_id not in text_by_id and block_id in blocks_by_id:
                text_by_id[block_id] = str(blocks_by_id[block_id].get("text") or "")
        missing_text_ids = [
            int(block_id)
            for block_id in current_final_ids
            if int(block_id) not in text_by_id
        ]
        if missing_text_ids:
            raise ValueError(
                f"Missing evidence text for {question_id}: {missing_text_ids}"
            )
        demand = retrieval.get("demand") or {}
        intent = str(demand.get("intent") or "fact").strip().lower()
        draft_answer = str(
            result.get("answer_short") or result.get("output") or ""
        ).strip()
        cases.append(
            {
                "question_id": question_id,
                "doc_uuid": str(row.get("doc_uuid") or ""),
                "question": str(result.get("question") or row.get("question") or ""),
                "draft_answer": draft_answer,
                "intent": intent,
                "subqueries": list(demand.get("subqueries") or []),
                "candidates": candidates,
                "anchor_ids": [int(value) for value in anchors],
                "current_final_ids": [int(value) for value in current_final_ids],
                "text_by_id": text_by_id,
                "max_items": 3 if intent in {"comparison", "multi-hop"} else 2,
                "allowed_types": list(allowed_types),
                "unanswerable": _is_unanswerable(draft_answer),
                "triggered": bool(controller.get("triggered")),
                "trigger_reasons": list(controller.get("trigger_reasons") or []),
                "verifier_state_contradiction": _verifier_state_contradiction(
                    retrieval.get("verification") or {}
                ),
                "source_result_path": str(paths["result"]),
                "source_retrieval_path": str(paths["retrieval"]),
                "source_index_path": str(paths["index"]),
            }
        )
    return cases


def _safe_provider_identity(provider_identity: Mapping[str, Any]) -> Dict[str, Any]:
    api_base = str(provider_identity.get("api_base") or "")
    parsed = urlsplit(api_base)
    if parsed.scheme and parsed.hostname:
        host = parsed.hostname
        if parsed.port:
            host = f"{host}:{parsed.port}"
        endpoint = f"{parsed.scheme}://{host}{parsed.path.rstrip('/')}"
    else:
        endpoint = api_base.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    return {
        "model_name": str(provider_identity.get("model_name") or ""),
        "api_base": endpoint,
        "backend": str(provider_identity.get("backend") or ""),
    }


def _safe_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        str(key): value
        for key, value in config.items()
        if not any(part in str(key).lower() for part in _SENSITIVE_KEY_PARTS)
    }


def _text_sha256(text: Any) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def support_rerank_fingerprint(
    *,
    question: str,
    draft_answer: str,
    candidate_payload: Sequence[Mapping[str, Any]],
    provider_identity: Mapping[str, Any],
    controller_config: Mapping[str, Any],
) -> str:
    payload = {
        "version": 1,
        "question": str(question),
        "draft_answer": str(draft_answer),
        "candidates": [
            {
                "block_id": int(item["block_id"]),
                "block_type": str(item.get("block_type") or ""),
                "text_sha256": _text_sha256(item.get("text")),
            }
            for item in candidate_payload
        ],
        "provider": _safe_provider_identity(provider_identity),
        "controller_config": _safe_config(controller_config),
    }
    return _canonical_sha256(payload)


def _rerank_query(replay_case: Mapping[str, Any]) -> str:
    return (
        f"Question: {replay_case.get('question', '')}\n"
        f"Draft answer: {replay_case.get('draft_answer', '')}\n"
        f"Evidence intent: {replay_case.get('intent', 'fact')}"
    )


def _rerank_documents(candidates: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        f"section={item.get('section_id') or ''}\n"
        f"type={item.get('block_type', 'unknown')}\n{item.get('text', '')}"
        for item in candidates
    ]


def build_score_cache_entry(
    *,
    reranker: Any,
    replay_case: Mapping[str, Any],
    provider_identity: Mapping[str, Any],
    controller_config: Mapping[str, Any],
) -> Dict[str, Any]:
    candidates = list(replay_case.get("candidates") or [])
    fingerprint = support_rerank_fingerprint(
        question=str(replay_case.get("question") or ""),
        draft_answer=str(replay_case.get("draft_answer") or ""),
        candidate_payload=candidates,
        provider_identity=provider_identity,
        controller_config=controller_config,
    )
    scores = reranker.rerank(
        query=_rerank_query(replay_case),
        documents=_rerank_documents(candidates),
        batch_size=max(int(controller_config.get("rerank_batch_size", 50)), 1),
    )
    if len(scores) != len(candidates):
        raise ValueError(
            f"Reranker score count mismatch: {len(scores)} != {len(candidates)}"
        )
    numeric_scores = [float(score) for score in scores]
    if not all(math.isfinite(score) for score in numeric_scores):
        raise ValueError("Reranker returned a non-finite score")
    order = sorted(
        range(len(candidates)),
        key=lambda index: (-numeric_scores[index], int(candidates[index]["block_id"])),
    )
    rank_by_position = {position: rank for rank, position in enumerate(order, 1)}
    return {
        "version": 1,
        "question_id": str(replay_case.get("question_id") or ""),
        "fingerprint": fingerprint,
        "provider": _safe_provider_identity(provider_identity),
        "controller_config": _safe_config(controller_config),
        "scores": [
            {
                "block_id": int(item["block_id"]),
                "score": numeric_scores[position],
                "rank": rank_by_position[position],
            }
            for position, item in enumerate(candidates)
        ],
    }


def load_validated_score_cache(
    path: Path, expected_fingerprint: str
) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fingerprint") != expected_fingerprint:
        raise ValueError(
            f"Score-cache fingerprint mismatch for {path}: "
            f"{payload.get('fingerprint')} != {expected_fingerprint}"
        )
    scores = list(payload.get("scores") or [])
    block_ids = [int(item["block_id"]) for item in scores]
    if len(block_ids) != len(set(block_ids)):
        raise ValueError(f"Duplicate score IDs in {path}")
    if not all(math.isfinite(float(item["score"])) for item in scores):
        raise ValueError(f"Non-finite score in {path}")
    return payload


def get_or_build_score_cache(
    *,
    cache_path: Path,
    reranker: Any,
    replay_case: Mapping[str, Any],
    provider_identity: Mapping[str, Any],
    controller_config: Mapping[str, Any],
) -> tuple[Dict[str, Any], bool]:
    expected_fingerprint = support_rerank_fingerprint(
        question=str(replay_case.get("question") or ""),
        draft_answer=str(replay_case.get("draft_answer") or ""),
        candidate_payload=list(replay_case.get("candidates") or []),
        provider_identity=provider_identity,
        controller_config=controller_config,
    )
    if cache_path.exists():
        return load_validated_score_cache(cache_path, expected_fingerprint), True
    payload = build_score_cache_entry(
        reranker=reranker,
        replay_case=replay_case,
        provider_identity=provider_identity,
        controller_config=controller_config,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return payload, False


def replay_controller(
    *,
    replay_case: Mapping[str, Any],
    score_entry: Mapping[str, Any],
    policy: str,
    min_normalized_relevance: float,
    redundancy_overlap_threshold: float,
) -> Dict[str, Any]:
    candidates = [dict(item) for item in replay_case.get("candidates") or []]
    candidate_ids = [int(item["block_id"]) for item in candidates]
    score_items = list(score_entry.get("scores") or [])
    score_ids = [int(item["block_id"]) for item in score_items]
    if set(candidate_ids) != set(score_ids) or len(candidate_ids) != len(score_ids):
        raise ValueError(
            f"Candidate/score IDs mismatch; candidate IDs={candidate_ids} score IDs={score_ids}"
        )
    score_by_id = {int(item["block_id"]): item for item in score_items}
    for candidate in candidates:
        block_id = int(candidate["block_id"])
        score = score_by_id[block_id]
        score_parts = dict(candidate.get("score_parts") or {})
        score_parts["answer_conditioned_rerank_score"] = float(score["score"])
        score_parts["answer_conditioned_rerank_rank"] = int(score["rank"])
        candidate["score_parts"] = score_parts
    ranked = sorted(
        candidates,
        key=lambda item: (
            int(item["score_parts"]["answer_conditioned_rerank_rank"]),
            int(item["block_id"]),
        ),
    )
    allowed_types = set(replay_case.get("allowed_types") or DEFAULT_SUPPORT_TYPES)
    candidate_id_set = {
        int(item["block_id"])
        for item in candidates
        if item.get("block_type") in allowed_types
        and item.get("evidence_role") != "bridge_auxiliary"
    }
    anchors = []
    for raw_id in replay_case.get("anchor_ids") or []:
        block_id = int(raw_id)
        if block_id in candidate_id_set and block_id not in anchors:
            anchors.append(block_id)

    if replay_case.get("unanswerable"):
        final_ids = []
        diagnostics = []
        stopping_reason = "unanswerable"
    elif policy == "fill_budget":
        final_ids = list(anchors)
        target = max(int(replay_case.get("max_items") or 0), len(final_ids))
        for item in ranked:
            if len(final_ids) >= target:
                break
            block_id = int(item["block_id"])
            if block_id in candidate_id_set and block_id not in final_ids:
                final_ids.append(block_id)
        diagnostics = []
        stopping_reason = "budget_reached" if len(final_ids) >= target else "candidates_exhausted"
    elif policy == "coverage_prune":
        pruning = prune_supporting_evidence(
            question=str(replay_case.get("question") or ""),
            draft_answer=str(replay_case.get("draft_answer") or ""),
            intent=str(replay_case.get("intent") or "fact"),
            subqueries=list(replay_case.get("subqueries") or []),
            candidates=ranked,
            anchor_ids=anchors,
            allowed_types=allowed_types,
            max_items=int(replay_case.get("max_items") or 0),
            min_normalized_relevance=float(min_normalized_relevance),
            redundancy_overlap_threshold=float(redundancy_overlap_threshold),
            unanswerable=False,
        )
        final_ids = list(pruning["final_ids"])
        diagnostics = pruning["candidate_diagnostics"]
        stopping_reason = pruning["stopping_reason"]
    else:
        raise ValueError(f"Unsupported replay policy: {policy}")

    if any(block_id not in candidate_id_set for block_id in final_ids):
        raise ValueError("Replay produced an ID outside the eligible candidate whitelist")
    score_fingerprint = str(score_entry.get("fingerprint") or _canonical_sha256(score_items))
    return {
        "question_id": str(replay_case.get("question_id") or ""),
        "answer": str(replay_case.get("draft_answer") or ""),
        "final_ids": final_ids,
        "candidate_diagnostics": diagnostics,
        "stopping_reason": stopping_reason,
        "score_fingerprint": score_fingerprint,
    }


def _prediction_from_ids(
    replay_case: Mapping[str, Any], final_ids: Sequence[int]
) -> Dict[str, Any]:
    text_by_id = {
        int(key): str(value)
        for key, value in (replay_case.get("text_by_id") or {}).items()
    }
    ranked_evidence = [
        str(item.get("text") or "")
        for item in replay_case.get("candidates") or []
        if str(item.get("text") or "")
    ]
    return {
        "answer": str(replay_case.get("draft_answer") or ""),
        "evidence": [text_by_id[block_id] for block_id in final_ids],
        "ranked_evidence": ranked_evidence,
    }


def _details_by_metric(
    details: Sequence[Mapping[str, Any]], metric: str
) -> Dict[str, float]:
    return {
        str(item["question_id"]): float(item[metric])
        for item in details
    }


def evaluate_replay_grid(
    *,
    replay_cases: Sequence[Mapping[str, Any]],
    gold: Mapping[str, Sequence[Mapping[str, Any]]],
    score_entries: Mapping[str, Mapping[str, Any]],
    min_relevance_grid: Sequence[float] = (0.35, 0.5, 0.65),
    redundancy_grid: Sequence[float] = (0.65, 0.75, 0.85),
    bootstrap_resamples: int = 10000,
) -> Dict[str, Any]:
    question_ids = [str(case.get("question_id") or "") for case in replay_cases]
    if not all(question_ids) or len(question_ids) != len(set(question_ids)):
        raise ValueError("Replay cases must have unique non-empty question IDs")
    if set(question_ids) != set(gold):
        raise ValueError("Replay cases and gold question IDs must match exactly")

    baseline_predictions = {}
    for replay_case in replay_cases:
        question_id = str(replay_case["question_id"])
        current_ids = [int(value) for value in replay_case.get("current_final_ids") or []]
        baseline_predictions[question_id] = _prediction_from_ids(
            replay_case, current_ids
        )
    baseline_scores = evaluate_qasper_official(dict(gold), baseline_predictions)
    baseline_details = evaluate_qasper_official_details(dict(gold), baseline_predictions)
    baseline_evidence_f1 = _details_by_metric(baseline_details, "evidence_f1")

    configurations = []
    for min_relevance, redundancy_threshold in product(
        min_relevance_grid, redundancy_grid
    ):
        predictions = {}
        final_counts = Counter()
        stopping_reasons = Counter()
        illegal_final_ids = 0
        verifier_contradictions = 0
        for replay_case in replay_cases:
            question_id = str(replay_case["question_id"])
            if replay_case.get("triggered"):
                if question_id not in score_entries:
                    raise ValueError(f"Missing score entry for triggered question {question_id}")
                replayed = replay_controller(
                    replay_case=replay_case,
                    score_entry=score_entries[question_id],
                    policy="coverage_prune",
                    min_normalized_relevance=float(min_relevance),
                    redundancy_overlap_threshold=float(redundancy_threshold),
                )
                final_ids = list(replayed["final_ids"])
                stopping_reasons[replayed["stopping_reason"]] += 1
            else:
                final_ids = [
                    int(value)
                    for value in replay_case.get("current_final_ids") or []
                ]
                stopping_reasons["not_triggered"] += 1
            eligible_ids = {
                int(item["block_id"])
                for item in replay_case.get("candidates") or []
                if item.get("block_type")
                in set(replay_case.get("allowed_types") or DEFAULT_SUPPORT_TYPES)
                and item.get("evidence_role") != "bridge_auxiliary"
            }
            illegal_final_ids += sum(
                block_id not in eligible_ids for block_id in final_ids
            )
            verifier_contradictions += int(
                replay_case.get("verifier_state_contradiction", False)
            )
            final_counts[len(final_ids)] += 1
            predictions[question_id] = _prediction_from_ids(
                replay_case, final_ids
            )

        scores = evaluate_qasper_official(dict(gold), predictions)
        details = evaluate_qasper_official_details(dict(gold), predictions)
        deltas = {
            metric: round(float(scores[metric]) - float(baseline_scores[metric]), 6)
            for metric in (
                "Answer F1",
                "Evidence Precision",
                "Evidence Recall",
                "Evidence F1",
            )
        }
        bootstrap = paired_bootstrap_ci(
            baseline=baseline_evidence_f1,
            candidate=_details_by_metric(details, "evidence_f1"),
            n_resamples=int(bootstrap_resamples),
            seed=42,
        )
        gates = {
            "evidence_f1_delta": deltas["Evidence F1"] >= 0.020,
            "evidence_precision_delta": deltas["Evidence Precision"] >= 0.030,
            "evidence_recall_delta": deltas["Evidence Recall"] >= -0.015,
            "answer_f1_unchanged": abs(deltas["Answer F1"]) <= 1e-12,
            "zero_illegal_final_ids": illegal_final_ids == 0,
            "zero_verifier_contradictions": verifier_contradictions == 0,
            "no_additional_llm_calls": True,
            "token_cost_not_increased": True,
        }
        configurations.append(
            {
                **scores,
                "thresholds": {
                    "min_relevance": float(min_relevance),
                    "redundancy_overlap": float(redundancy_threshold),
                },
                "delta": deltas,
                "paired_bootstrap_evidence_f1": bootstrap,
                "final_evidence_count_distribution": {
                    str(key): value for key, value in sorted(final_counts.items())
                },
                "stopping_reasons": dict(sorted(stopping_reasons.items())),
                "illegal_final_ids": illegal_final_ids,
                "verifier_state_contradictions": verifier_contradictions,
                "gates": gates,
                "passes_all_gates": all(gates.values()),
            }
        )

    passing = [item for item in configurations if item["passes_all_gates"]]
    selected = None
    if passing:
        selected = min(
            passing,
            key=lambda item: (
                -float(item["Evidence F1"]),
                -float(item["Evidence Precision"]),
                item["thresholds"]["min_relevance"],
                item["thresholds"]["redundancy_overlap"],
            ),
        )
    return {
        "question_count": len(replay_cases),
        "baseline": baseline_scores,
        "configurations": configurations,
        "selected": selected,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def run_replay_experiment(
    *,
    dataset_rows: Sequence[Mapping[str, Any]],
    retrieval_root: Path,
    method: str,
    cache_dir: Path,
    output_dir: Path,
    reranker: Any,
    provider_identity: Mapping[str, Any],
    controller_config: Mapping[str, Any],
    min_relevance_grid: Sequence[float] = (0.35, 0.5, 0.65),
    redundancy_grid: Sequence[float] = (0.65, 0.75, 0.85),
    bootstrap_resamples: int = 10000,
    decision_provenance: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    cases = build_replay_cases(
        dataset_rows=dataset_rows,
        retrieval_root=Path(retrieval_root),
        method=method,
        topk=int(controller_config.get("answer_conditioned_support_topk", 20)),
    )
    triggered = [case for case in cases if case.get("triggered")]
    score_entries = {}
    cache_hits = 0
    cache_misses = 0
    for index, replay_case in enumerate(triggered, 1):
        question_id = str(replay_case["question_id"])
        cache_path = Path(cache_dir) / f"{question_id}.json"
        if not cache_path.exists() and reranker is None:
            raise ValueError(
                f"Score cache is missing for {question_id}, but no reranker is available"
            )
        score_entry, cache_hit = get_or_build_score_cache(
            cache_path=cache_path,
            reranker=reranker,
            replay_case=replay_case,
            provider_identity=provider_identity,
            controller_config=controller_config,
        )
        score_entries[question_id] = score_entry
        cache_hits += int(cache_hit)
        cache_misses += int(not cache_hit)
        print(
            f"[replay-cache] {index}/{len(triggered)} {question_id} "
            f"{'hit' if cache_hit else 'miss'}",
            flush=True,
        )

    gold = _gold_answers_and_evidence(
        list(dataset_rows), text_evidence_only=False
    )
    report = evaluate_replay_grid(
        replay_cases=cases,
        gold=gold,
        score_entries=score_entries,
        min_relevance_grid=min_relevance_grid,
        redundancy_grid=redundancy_grid,
        bootstrap_resamples=bootstrap_resamples,
    )
    report.update(
        {
            "method": method,
            "triggered_questions": len(triggered),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "llm_calls": 0,
            "provider": _safe_provider_identity(provider_identity),
            "score_cache_sha256": _canonical_sha256(
                {
                    question_id: {
                        "fingerprint": entry.get("fingerprint"),
                        "scores": entry.get("scores"),
                    }
                    for question_id, entry in sorted(score_entries.items())
                }
            ),
        }
    )
    output_dir = Path(output_dir)
    _write_json(output_dir / "replay_cases.json", cases)
    _write_json(output_dir / "replay_report.json", report)
    if report.get("selected") is not None:
        decision = {
            "question_count": report["question_count"],
            "triggered_questions": len(triggered),
            "thresholds": report["selected"]["thresholds"],
            "metrics": {
                key: report["selected"][key]
                for key in (
                    "Answer F1",
                    "Evidence Precision",
                    "Evidence Recall",
                    "Evidence F1",
                )
            },
            "delta": report["selected"]["delta"],
            "gates": report["selected"]["gates"],
            "passes_all_gates": report["selected"]["passes_all_gates"],
            "provenance": {
                **dict(decision_provenance or {}),
                "score_cache_sha256": report["score_cache_sha256"],
            },
        }
        _write_json(output_dir / "replay_decision.json", decision)
    return report


def _parse_float_grid(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Threshold grid cannot be empty")
    if any(item < 0.0 or item > 1.0 for item in values):
        raise argparse.ArgumentTypeError("Thresholds must be within [0, 1]")
    return values


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay the Qasper weak-support controller without LLM calls."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--retrieval-root", required=True)
    parser.add_argument("--method", default="evibridge_support_controller")
    parser.add_argument("--system-config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--min-relevance-grid",
        type=_parse_float_grid,
        default=(0.35, 0.5, 0.65),
    )
    parser.add_argument(
        "--redundancy-grid",
        type=_parse_float_grid,
        default=(0.65, 0.75, 0.85),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    return parser


def _create_reranker_from_system_config(system_config_path: str):
    from Core.configs.system_config import load_system_config
    from Core.provider.rerank import TextRerankerProvider

    cfg = load_system_config(system_config_path)
    rag = cfg.rag.strategy_config
    reranker_cfg = rag.reranker_config
    reranker = TextRerankerProvider(
        model_name=reranker_cfg.model_name,
        device=reranker_cfg.device,
        max_length=reranker_cfg.max_length,
        backend=reranker_cfg.backend,
        api_base=reranker_cfg.api_base,
        api_key=reranker_cfg.api_key,
        max_retries=getattr(reranker_cfg, "max_retries", 3),
        retry_backoff=getattr(reranker_cfg, "retry_backoff", 0.5),
        request_timeout=getattr(reranker_cfg, "request_timeout", 60.0),
    )
    provider_identity = {
        "model_name": reranker_cfg.model_name,
        "api_base": reranker_cfg.api_base,
        "backend": reranker_cfg.backend,
    }
    controller_config = {
        "answer_conditioned_support_topk": int(
            rag.answer_conditioned_support_topk
        ),
        "rerank_batch_size": int(rag.rerank_batch_size),
    }
    return reranker, provider_identity, controller_config


def main() -> None:
    args = create_parser().parse_args()
    rows = _load_json(Path(args.dataset))
    if not isinstance(rows, list):
        raise ValueError("Replay dataset must be a JSON list")
    reranker, provider_identity, controller_config = (
        _create_reranker_from_system_config(args.system_config)
    )
    manifest_path = Path(args.manifest) if args.manifest else None
    if manifest_path is not None and not manifest_path.is_file():
        raise FileNotFoundError(f"Replay manifest does not exist: {manifest_path}")
    decision_provenance = {
        "source_commit": _source_commit(),
        "config_sha256": _file_sha256(Path(args.system_config)),
        "dataset_sha256": _file_sha256(Path(args.dataset)),
        "manifest_sha256": (
            _file_sha256(manifest_path) if manifest_path is not None else None
        ),
    }
    try:
        report = run_replay_experiment(
            dataset_rows=rows,
            retrieval_root=Path(args.retrieval_root),
            method=args.method,
            cache_dir=Path(args.cache_dir),
            output_dir=Path(args.output_dir),
            reranker=reranker,
            provider_identity=provider_identity,
            controller_config=controller_config,
            min_relevance_grid=args.min_relevance_grid,
            redundancy_grid=args.redundancy_grid,
            bootstrap_resamples=args.bootstrap_resamples,
            decision_provenance=decision_provenance,
        )
    finally:
        reranker.close()
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if report.get("selected") is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
