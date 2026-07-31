import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from Scripts.preprocess.hotpotqa_evibridge import (
    _as_list,
    _clean_title,
    _get_index,
    _normalize_title,
    _row_id,
    _sentences_from_value,
    _supporting_facts,
)


ALGORITHM_VERSION = "hotpotqa-fixed1000-v1"
Stratum = Tuple[str, str]


def stable_rank(question_id: str, seed: int) -> str:
    value = f"{ALGORITHM_VERSION}\0seed={seed}\0{question_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def audit_supporting_facts(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    context = row.get("context") or {}
    titles = [_clean_title(value) for value in _as_list(context.get("title", []))]
    sentence_groups = _as_list(context.get("sentences", []))
    issues: List[Dict[str, Any]] = []
    for title, sent_id in _supporting_facts(row):
        exact_indices = [
            index for index, candidate in enumerate(titles)
            if candidate == _clean_title(title)
        ]
        candidate_indices = exact_indices
        if not candidate_indices:
            normalized = _normalize_title(title)
            candidate_indices = [
                index for index, candidate in enumerate(titles)
                if _normalize_title(candidate) == normalized
            ]
        if not candidate_indices:
            issues.append(
                {
                    "reason": "missing_title",
                    "title": title,
                    "sent_id": int(sent_id),
                }
            )
            continue
        if len(candidate_indices) > 1:
            issues.append(
                {
                    "reason": "ambiguous_title",
                    "title": title,
                    "sent_id": int(sent_id),
                }
            )
            continue
        sentences = _sentences_from_value(
            _get_index(sentence_groups, candidate_indices[0], [])
        )
        if int(sent_id) < 0 or int(sent_id) >= len(sentences):
            issues.append(
                {
                    "reason": "sentence_id_out_of_range",
                    "title": title,
                    "sent_id": int(sent_id),
                    "sentence_count": len(sentences),
                }
            )
    return issues


def allocate_hamilton(
    counts: Dict[Stratum, int],
    sample_size: int,
) -> Dict[Stratum, int]:
    total = sum(counts.values())
    if total <= 0 or sample_size <= 0:
        return {key: 0 for key in counts}
    target = min(int(sample_size), total)
    exact = {
        key: count * target / total
        for key, count in counts.items()
    }
    quotas = {
        key: math.floor(value)
        for key, value in exact.items()
    }
    remaining = target - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda key: (-(exact[key] - quotas[key]), key),
    )
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def select_fixed_rows(
    rows: List[Dict[str, Any]],
    sample_size: int = 1000,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    question_ids = [_row_id(row) for row in rows]
    if any(not question_id for question_id in question_ids):
        raise ValueError("HotpotQA source contains an empty question ID")
    duplicates = sorted(
        question_id
        for question_id, count in Counter(question_ids).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate HotpotQA question IDs: {duplicates[:10]}")

    eligible: List[Tuple[int, Dict[str, Any], Stratum]] = []
    excluded: List[Dict[str, Any]] = []
    eligible_counts: Counter[Stratum] = Counter()
    source_counts: Counter[Stratum] = Counter()
    for source_index, row in enumerate(rows):
        stratum = (str(row.get("type") or ""), str(row.get("level") or ""))
        source_counts[stratum] += 1
        issues = audit_supporting_facts(row)
        if issues:
            excluded.append(
                {
                    "question_id": _row_id(row),
                    **issues[0],
                    "issues": issues,
                }
            )
            continue
        eligible.append((source_index, row, stratum))
        eligible_counts[stratum] += 1

    quotas = allocate_hamilton(dict(eligible_counts), sample_size)
    by_stratum: Dict[Stratum, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for source_index, row, stratum in eligible:
        by_stratum[stratum].append((source_index, row))

    selected_ids = set()
    for stratum, candidates in by_stratum.items():
        ranked = sorted(
            candidates,
            key=lambda item: stable_rank(_row_id(item[1]), seed),
        )
        selected_ids.update(
            _row_id(row)
            for _, row in ranked[: quotas.get(stratum, 0)]
        )
    selected = [
        row
        for row in rows
        if _row_id(row) in selected_ids
    ]

    def serialize_counts(values: Dict[Stratum, int]) -> Dict[str, int]:
        return {
            f"{question_type}|{level}": int(values[(question_type, level)])
            for question_type, level in sorted(values)
        }

    audit = {
        "algorithm_version": ALGORITHM_VERSION,
        "seed": int(seed),
        "source_count": len(rows),
        "eligible_count": len(eligible),
        "question_count": len(selected),
        "source_strata": serialize_counts(source_counts),
        "eligible_strata": serialize_counts(eligible_counts),
        "selected_strata": serialize_counts(quotas),
        "excluded": excluded,
        "selected_question_ids": [_row_id(row) for row in selected],
    }
    return selected, audit
