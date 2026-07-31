import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from Scripts.preprocess.hotpotqa_evibridge import (
    _as_list,
    _clean_title,
    _get_index,
    _normalize_title,
    _row_id,
    _sentences_from_value,
    _supporting_facts,
    _write_unified_rows,
    hotpotqa_row_to_tree,
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


def load_parquet_rows(path: str | Path) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq

    return [
        dict(row)
        for row in pq.read_table(path).to_pylist()
    ]


def prepare_fixed_sample(
    parquet_path: str | Path,
    output_root: str | Path,
    working_dir: str | Path,
    sample_size: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    source_path = Path(parquet_path).resolve()
    output_dir = Path(output_root).resolve()
    work_dir = Path(working_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    rows = load_parquet_rows(source_path)
    selected, audit = select_fixed_rows(
        rows,
        sample_size=sample_size,
        seed=seed,
    )
    base_name = (
        f"hotpotqa_distractor_validation_fixed{len(selected)}_seed{seed}"
    )
    selected_raw_path = output_dir / f"{base_name}.raw.json"
    unified_path = output_dir / f"{base_name}.json"
    manifest_path = output_dir / f"{base_name}.manifest.json"
    dataset_config_path = output_dir / f"{base_name}.yaml"

    selected_raw_path.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    unified_rows = _write_unified_rows(
        raw_rows=selected,
        output_path=str(unified_path),
        split="validation",
        subset="distractor",
    )

    for row in selected:
        question_id = _row_id(row)
        tree, _ = hotpotqa_row_to_tree(
            row=row,
            save_dir=str(work_dir / question_id),
            doc_path=f"hotpotqa://distractor/validation/{question_id}",
        )
        tree.save_to_file()

    gold_fact_count = sum(
        len(row.get("hotpot_supporting_facts") or [])
        for row in unified_rows
    )
    mapped_fact_count = sum(
        len(row.get("evidence_block_ids") or [])
        for row in unified_rows
    )
    if mapped_fact_count != gold_fact_count:
        raise ValueError(
            "HotpotQA supporting-fact mapping is incomplete: "
            f"mapped={mapped_fact_count}, gold={gold_fact_count}"
        )
    mapping_coverage = (
        mapped_fact_count / gold_fact_count
        if gold_fact_count
        else 1.0
    )

    manifest = {
        "schema_version": 1,
        **audit,
        "source_path": source_path.as_posix(),
        "source_size": source_path.stat().st_size,
        "source_sha256": _sha256_file(source_path),
        "selected_raw_path": selected_raw_path.as_posix(),
        "selected_raw_sha256": _sha256_file(selected_raw_path),
        "unified_path": unified_path.as_posix(),
        "unified_sha256": _sha256_file(unified_path),
        "working_dir": work_dir.as_posix(),
        "gold_fact_count": gold_fact_count,
        "mapped_fact_count": mapped_fact_count,
        "mapping_coverage": mapping_coverage,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    dataset_config_path.write_text(
        f"dataset_path: {json.dumps(unified_path.as_posix())}\n"
        f"working_dir: {json.dumps(work_dir.as_posix())}\n"
        "dataset_name: hotpotqa\n"
        f"manifest_path: {json.dumps(manifest_path.as_posix())}\n",
        encoding="utf-8",
    )
    return {
        **audit,
        "mapping_coverage": mapping_coverage,
        "manifest_path": str(manifest_path),
        "dataset_path": str(unified_path),
        "selected_raw_path": str(selected_raw_path),
        "dataset_config_path": str(dataset_config_path),
        "working_dir": str(work_dir),
    }


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a deterministic HotpotQA fixed validation sample."
    )
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--working-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = prepare_fixed_sample(
        parquet_path=args.parquet,
        output_root=args.output_root,
        working_dir=args.working_dir,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
