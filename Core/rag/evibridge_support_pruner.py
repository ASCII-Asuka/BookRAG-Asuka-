import math
import re
from typing import Any, Collection, Dict, List, Mapping, Sequence, Set


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "they",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
_SCORE_KEY = "answer_conditioned_rerank_score"
_RANK_KEY = "answer_conditioned_rerank_rank"
_EPSILON = 1e-12
_RELEVANCE_WEIGHT = 0.85
_COVERAGE_WEIGHT = 1.0 - _RELEVANCE_WEIGHT


def tokenize_support_text(text: str) -> Set[str]:
    return {
        token
        for token in _TOKEN_PATTERN.findall(str(text or "").lower())
        if token not in _STOP_WORDS
    }


def support_overlap(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _raw_relevance(item: Mapping[str, Any]) -> float:
    score_parts = item.get("score_parts") or {}
    value = score_parts.get(_SCORE_KEY, item.get(_SCORE_KEY, item.get("score", 0.0)))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _rerank_rank(item: Mapping[str, Any], position: int) -> int:
    score_parts = item.get("score_parts") or {}
    value = score_parts.get(_RANK_KEY, item.get(_RANK_KEY, position + 1))
    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return position + 1


def normalize_candidate_relevance(
    candidates: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    normalized = []
    finite_positions = []
    for position, original in enumerate(candidates):
        item = dict(original)
        item["score_parts"] = dict(item.get("score_parts") or {})
        raw_score = _raw_relevance(item)
        item["_candidate_position"] = position
        item["_raw_relevance"] = raw_score
        item["_rerank_rank"] = _rerank_rank(item, position)
        item["_normalized_relevance"] = None
        if math.isfinite(raw_score):
            finite_positions.append(position)
        normalized.append(item)

    if not finite_positions:
        return normalized

    finite_scores = [normalized[position]["_raw_relevance"] for position in finite_positions]
    low = min(finite_scores)
    high = max(finite_scores)
    if high - low > _EPSILON:
        for position in finite_positions:
            score = normalized[position]["_raw_relevance"]
            normalized[position]["_normalized_relevance"] = (score - low) / (high - low)
        return normalized

    ranked_positions = sorted(
        finite_positions,
        key=lambda position: (
            normalized[position]["_rerank_rank"],
            normalized[position]["_candidate_position"],
            int(normalized[position].get("block_id", 0)),
        ),
    )
    denominator = max(len(ranked_positions) - 1, 1)
    for rank, position in enumerate(ranked_positions):
        normalized[position]["_normalized_relevance"] = 1.0 - rank / denominator
    return normalized


def support_budget_for_intent(intent: str) -> int:
    return 3 if str(intent or "").strip().lower() in {"comparison", "multi-hop"} else 2


def _is_canonical_boolean_answer(answer: str) -> bool:
    return _TOKEN_PATTERN.findall(str(answer or "").strip().lower()) in (["yes"], ["no"])


def _is_overview_section(section_id: Any) -> bool:
    value = str(section_id or "").strip().lower()
    return any(
        re.search(rf"\b{label}\b", value)
        for label in ("abstract", "introduction", "overview")
    )


def _facet_payload(
    question: str, draft_answer: str, subqueries: Sequence[str]
) -> List[Dict[str, Any]]:
    raw_facets = [("question", question), ("draft_answer", draft_answer)]
    raw_facets.extend(
        (f"subquery_{index}", value) for index, value in enumerate(subqueries, 1)
    )
    return [
        {"name": name, "tokens": tokens}
        for name, value in raw_facets
        if (tokens := tokenize_support_text(value))
    ]


def _coverage_for_tokens(
    tokens: Set[str], facets: Sequence[Mapping[str, Any]]
) -> List[float]:
    return [
        len(tokens & facet["tokens"]) / len(facet["tokens"])
        for facet in facets
    ]


def _maximum_overlap(
    tokens: Set[str],
    selected_ids: Sequence[int],
    tokens_by_id: Mapping[int, Set[str]],
) -> tuple[float, Any]:
    maximum = 0.0
    overlap_block_id = None
    for block_id in selected_ids:
        overlap = support_overlap(tokens, tokens_by_id.get(block_id, set()))
        if overlap > maximum:
            maximum = overlap
            overlap_block_id = block_id
    return maximum, overlap_block_id


def prune_supporting_evidence(
    *,
    question: str,
    draft_answer: str,
    intent: str,
    subqueries: Sequence[str],
    candidates: Sequence[Mapping[str, Any]],
    anchor_ids: Sequence[int],
    allowed_types: Collection[str],
    max_items: int,
    min_normalized_relevance: float,
    redundancy_overlap_threshold: float,
    unanswerable: bool = False,
) -> Dict[str, Any]:
    normalized = normalize_candidate_relevance(candidates)
    allowed = set(allowed_types)
    candidate_by_id = {
        int(item["block_id"]): item
        for item in normalized
        if item.get("block_id") is not None
    }
    diagnostics_by_id: Dict[int, Dict[str, Any]] = {}
    tokens_by_id = {
        block_id: tokenize_support_text(item.get("text", ""))
        for block_id, item in candidate_by_id.items()
    }
    for item in normalized:
        block_id = int(item["block_id"])
        raw_relevance = item["_raw_relevance"]
        diagnostics_by_id[block_id] = {
            "block_id": block_id,
            "raw_relevance": raw_relevance if math.isfinite(raw_relevance) else None,
            "normalized_relevance": item["_normalized_relevance"],
            "rerank_rank": item["_rerank_rank"],
            "marginal_coverage": 0.0,
            "facet_coverage": {},
            "utility": None,
            "max_overlap": 0.0,
            "overlap_block_id": None,
            "decision": "rejected",
            "reason": "not_evaluated",
        }

    if unanswerable:
        for diagnostic in diagnostics_by_id.values():
            diagnostic["reason"] = "unanswerable"
        return {
            "final_ids": [],
            "added_ids": [],
            "candidate_diagnostics": [
                diagnostics_by_id[int(item["block_id"])] for item in normalized
            ],
            "stopping_reason": "unanswerable",
        }

    selected_ids = []
    seen_anchor_ids = set()
    for raw_id in anchor_ids:
        block_id = int(raw_id)
        if block_id in seen_anchor_ids:
            continue
        seen_anchor_ids.add(block_id)
        item = candidate_by_id.get(block_id)
        if item is None or item.get("block_type") not in allowed:
            continue
        selected_ids.append(block_id)
        diagnostic = diagnostics_by_id[block_id]
        diagnostic["decision"] = "accepted"
        diagnostic["reason"] = "anchor"

    anchor_set = set(selected_ids)
    facets = _facet_payload(question, draft_answer, subqueries)
    current_coverage = [0.0 for _ in facets]
    for block_id in selected_ids:
        coverage = _coverage_for_tokens(tokens_by_id.get(block_id, set()), facets)
        current_coverage = [
            max(current, candidate)
            for current, candidate in zip(current_coverage, coverage)
        ]

    remaining = []
    for item in normalized:
        block_id = int(item["block_id"])
        diagnostic = diagnostics_by_id[block_id]
        if block_id in anchor_set:
            continue
        if item.get("block_type") not in allowed:
            diagnostic["reason"] = "ineligible_type"
            continue
        if not math.isfinite(item["_raw_relevance"]):
            diagnostic["reason"] = "non_finite_score"
            continue
        remaining.append(item)

    cap = max(int(max_items), 0)
    if anchor_set:
        cap = min(cap, len(anchor_set) + 1)
    stopping_reason = "no_eligible_candidate"
    if len(selected_ids) >= cap:
        stopping_reason = "budget_reached"
        for item in remaining:
            diagnostics_by_id[int(item["block_id"])]["reason"] = "budget_reached"
        remaining = []

    if selected_ids and remaining and not _is_canonical_boolean_answer(draft_answer):
        top_eligible = min(
            [*([candidate_by_id[block_id] for block_id in selected_ids]), *remaining],
            key=lambda item: (
                item["_rerank_rank"],
                item["_candidate_position"],
                int(item["block_id"]),
            ),
        )
        top_block_id = int(top_eligible["block_id"])
        if (
            top_block_id in anchor_set
            and not _is_overview_section(top_eligible.get("section_id"))
        ):
            stopping_reason = "answer_complete_anchor"
            for item in remaining:
                diagnostics_by_id[int(item["block_id"])]["reason"] = (
                    "answer_complete_anchor"
                )
            remaining = []
        elif top_block_id not in anchor_set:
            normalized_relevance = float(
                top_eligible["_normalized_relevance"] or 0.0
            )
            tokens = tokens_by_id.get(top_block_id, set())
            overlap, overlap_block_id = _maximum_overlap(
                tokens, selected_ids, tokens_by_id
            )
            if (
                normalized_relevance >= float(min_normalized_relevance)
                and overlap < float(redundancy_overlap_threshold)
            ):
                coverage = _coverage_for_tokens(tokens, facets)
                marginal_parts = [
                    max(0.0, candidate - current)
                    for current, candidate in zip(current_coverage, coverage)
                ]
                marginal = (
                    sum(marginal_parts) / len(marginal_parts)
                    if marginal_parts
                    else 0.0
                )
                diagnostic = diagnostics_by_id[top_block_id]
                diagnostic["facet_coverage"] = {
                    facet["name"]: round(value, 8)
                    for facet, value in zip(facets, coverage)
                }
                diagnostic["marginal_coverage"] = round(marginal, 8)
                diagnostic["max_overlap"] = round(overlap, 8)
                diagnostic["overlap_block_id"] = overlap_block_id
                diagnostic["utility"] = round(
                    _RELEVANCE_WEIGHT * normalized_relevance
                    + _COVERAGE_WEIGHT * marginal,
                    8,
                )
                diagnostic["decision"] = "accepted"
                diagnostic["reason"] = "top_relevance_supplement"
                selected_ids.append(top_block_id)
                current_coverage = [
                    max(current, candidate)
                    for current, candidate in zip(current_coverage, coverage)
                ]
                remaining.remove(top_eligible)

    if len(selected_ids) >= cap and remaining:
        stopping_reason = "budget_reached"
        for item in remaining:
            diagnostics_by_id[int(item["block_id"])]["reason"] = "budget_reached"
        remaining = []

    if not selected_ids and remaining and cap > 0:
        fallback = min(
            remaining,
            key=lambda item: (
                -float(item["_raw_relevance"]),
                item["_rerank_rank"],
                item["_candidate_position"],
                int(item["block_id"]),
            ),
        )
        block_id = int(fallback["block_id"])
        selected_ids.append(block_id)
        diagnostic = diagnostics_by_id[block_id]
        diagnostic["decision"] = "accepted"
        diagnostic["reason"] = "safe_fallback"
        coverage = _coverage_for_tokens(tokens_by_id.get(block_id, set()), facets)
        diagnostic["facet_coverage"] = {
            facet["name"]: round(value, 8) for facet, value in zip(facets, coverage)
        }
        diagnostic["marginal_coverage"] = round(
            sum(coverage) / len(coverage) if coverage else 0.0, 8
        )
        diagnostic["utility"] = round(
            _RELEVANCE_WEIGHT * float(fallback["_normalized_relevance"] or 0.0)
            + _COVERAGE_WEIGHT * diagnostic["marginal_coverage"],
            8,
        )
        current_coverage = [
            max(current, candidate)
            for current, candidate in zip(current_coverage, coverage)
        ]
        remaining.remove(fallback)

    while remaining and len(selected_ids) < cap:
        qualified = []
        rejection_reasons = []
        for item in remaining:
            block_id = int(item["block_id"])
            diagnostic = diagnostics_by_id[block_id]
            tokens = tokens_by_id.get(block_id, set())
            coverage = _coverage_for_tokens(tokens, facets)
            marginal_parts = [
                max(0.0, candidate - current)
                for current, candidate in zip(current_coverage, coverage)
            ]
            marginal = sum(marginal_parts) / len(marginal_parts) if marginal_parts else 0.0
            overlap, overlap_block_id = _maximum_overlap(
                tokens, selected_ids, tokens_by_id
            )
            diagnostic["facet_coverage"] = {
                facet["name"]: round(value, 8)
                for facet, value in zip(facets, coverage)
            }
            diagnostic["marginal_coverage"] = round(marginal, 8)
            diagnostic["max_overlap"] = round(overlap, 8)
            diagnostic["overlap_block_id"] = overlap_block_id

            if overlap >= float(redundancy_overlap_threshold):
                diagnostic["reason"] = "redundant"
                rejection_reasons.append("all_redundant")
                continue
            if marginal <= _EPSILON:
                diagnostic["reason"] = "no_marginal_coverage"
                rejection_reasons.append("no_marginal_coverage")
                continue
            normalized_relevance = float(item["_normalized_relevance"] or 0.0)
            if normalized_relevance < float(min_normalized_relevance):
                diagnostic["reason"] = "below_relevance"
                rejection_reasons.append("below_relevance")
                continue
            utility = (
                _RELEVANCE_WEIGHT * normalized_relevance
                + _COVERAGE_WEIGHT * marginal
            )
            diagnostic["utility"] = round(utility, 8)
            qualified.append((utility, item, coverage))

        if not qualified:
            supports_relevance_fallback = (
                str(intent or "").strip().lower() in {"comparison", "multi-hop"}
                and not _is_canonical_boolean_answer(draft_answer)
                and not any(block_id not in anchor_set for block_id in selected_ids)
            )
            fallback_candidates = [
                item
                for item in remaining
                if diagnostics_by_id[int(item["block_id"])]["reason"]
                == "no_marginal_coverage"
                and float(item["_normalized_relevance"] or 0.0)
                >= float(min_normalized_relevance)
            ]
            if supports_relevance_fallback and fallback_candidates:
                chosen = min(
                    fallback_candidates,
                    key=lambda item: (
                        -float(item["_normalized_relevance"] or 0.0),
                        item["_rerank_rank"],
                        item["_candidate_position"],
                        int(item["block_id"]),
                    ),
                )
                block_id = int(chosen["block_id"])
                selected_ids.append(block_id)
                diagnostic = diagnostics_by_id[block_id]
                diagnostic["decision"] = "accepted"
                diagnostic["reason"] = "high_relevance_fallback"
                diagnostic["utility"] = round(
                    _RELEVANCE_WEIGHT
                    * float(chosen["_normalized_relevance"] or 0.0),
                    8,
                )
                remaining.remove(chosen)
                stopping_reason = "high_relevance_fallback"
                break
            if "no_marginal_coverage" in rejection_reasons:
                stopping_reason = "no_marginal_coverage"
            elif "all_redundant" in rejection_reasons:
                stopping_reason = "all_redundant"
            elif "below_relevance" in rejection_reasons:
                stopping_reason = "below_relevance"
            else:
                stopping_reason = "no_eligible_candidate"
            break

        _, chosen, coverage = min(
            qualified,
            key=lambda value: (
                -value[0],
                value[1]["_rerank_rank"],
                value[1]["_candidate_position"],
                int(value[1]["block_id"]),
            ),
        )
        block_id = int(chosen["block_id"])
        selected_ids.append(block_id)
        diagnostic = diagnostics_by_id[block_id]
        diagnostic["decision"] = "accepted"
        diagnostic["reason"] = "coverage_gain"
        current_coverage = [
            max(current, candidate)
            for current, candidate in zip(current_coverage, coverage)
        ]
        remaining.remove(chosen)

    if len(selected_ids) >= cap and remaining:
        stopping_reason = "budget_reached"
        for item in remaining:
            diagnostic = diagnostics_by_id[int(item["block_id"])]
            if diagnostic["decision"] != "accepted":
                diagnostic["reason"] = "budget_reached"
    elif not remaining and stopping_reason == "no_eligible_candidate":
        stopping_reason = "candidates_exhausted"

    added_ids = [block_id for block_id in selected_ids if block_id not in anchor_set]
    return {
        "final_ids": selected_ids,
        "added_ids": added_ids,
        "candidate_diagnostics": [
            diagnostics_by_id[int(item["block_id"])] for item in normalized
        ],
        "stopping_reason": stopping_reason,
    }
