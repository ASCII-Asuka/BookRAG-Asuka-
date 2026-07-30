import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Mapping


def paired_bootstrap_ci(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    baseline_ids = set(baseline)
    candidate_ids = set(candidate)
    if baseline_ids != candidate_ids:
        missing_candidate = sorted(baseline_ids - candidate_ids)
        missing_baseline = sorted(candidate_ids - baseline_ids)
        raise ValueError(
            "Paired bootstrap requires identical question IDs; "
            f"missing from candidate={missing_candidate[:10]}, "
            f"missing from baseline={missing_baseline[:10]}"
        )
    if not baseline_ids:
        raise ValueError("Paired bootstrap requires at least one question")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    question_ids = sorted(baseline_ids)
    differences = [
        float(candidate[qid]) - float(baseline[qid])
        for qid in question_ids
    ]
    observed = sum(differences) / len(differences)
    rng = random.Random(seed)
    bootstrap_deltas = []
    for _ in range(n_resamples):
        sampled = [
            differences[rng.randrange(len(differences))]
            for _ in differences
        ]
        bootstrap_deltas.append(sum(sampled) / len(sampled))
    bootstrap_deltas.sort()
    lower = _percentile(bootstrap_deltas, 0.025)
    upper = _percentile(bootstrap_deltas, 0.975)
    non_positive = (sum(delta <= 0.0 for delta in bootstrap_deltas) + 1) / (
        n_resamples + 1
    )
    non_negative = (sum(delta >= 0.0 for delta in bootstrap_deltas) + 1) / (
        n_resamples + 1
    )
    return {
        "question_count": len(question_ids),
        "candidate_minus_baseline": observed,
        "ci_95": [lower, upper],
        "two_sided_p_value": min(1.0, 2.0 * min(non_positive, non_negative)),
        "n_resamples": n_resamples,
        "seed": seed,
    }


def load_metric_by_question(path: str | Path, metric: str) -> Dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of per-question rows in {path}")
    scores: Dict[str, float] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        question_id = _question_id(row)
        if not question_id:
            raise ValueError(f"Missing question ID in {path}: {row}")
        if question_id in scores:
            raise ValueError(f"Duplicate question ID in {path}: {question_id}")
        if metric not in row:
            raise ValueError(f"Metric {metric!r} missing for question {question_id}")
        scores[question_id] = float(row[metric])
    return scores


def compare_detail_files(
    baseline_path: str | Path,
    candidate_path: str | Path,
    metric: str,
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    result = paired_bootstrap_ci(
        baseline=load_metric_by_question(baseline_path, metric),
        candidate=load_metric_by_question(candidate_path, metric),
        n_resamples=n_resamples,
        seed=seed,
    )
    return {
        "metric": metric,
        "baseline_path": str(Path(baseline_path)),
        "candidate_path": str(Path(candidate_path)),
        **result,
    }


def _question_id(row: Dict[str, Any]) -> str:
    for key in (
        "qasper_question_id",
        "hotpotqa_question_id",
        "question_id",
        "_id",
        "id",
        "doc_uuid",
    ):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute a paired bootstrap 95% CI from per-question evaluation files."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = compare_detail_files(
        baseline_path=args.baseline,
        candidate_path=args.candidate,
        metric=args.metric,
        n_resamples=args.resamples,
        seed=args.seed,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
