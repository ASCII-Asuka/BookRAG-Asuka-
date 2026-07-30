import argparse
import json
from pathlib import Path


def aggregate_react_diagnostics(eval_dirs) -> dict:
    directories = _as_directories(eval_dirs)
    payloads = []
    elapsed_seconds = 0.0
    token_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for eval_dir in directories:
        for path in sorted(eval_dir.glob("query_*/retrieval_res.json")):
            with path.open("r", encoding="utf-8") as handle:
                payloads.append(json.load(handle))
        cost_path = eval_dir / "token_cost.json"
        if cost_path.exists():
            with cost_path.open("r", encoding="utf-8") as handle:
                cost = json.load(handle)
            elapsed_seconds += float(cost.get("time") or 0.0)
            rag_cost = cost.get("rag_cost") or {}
            for key in token_totals:
                token_totals[key] += int(rag_cost.get(key) or 0)

    if not payloads:
        raise ValueError(
            "No ReAct retrieval results found in "
            + ", ".join(str(path) for path in directories)
        )
    steps = [
        step
        for payload in payloads
        for step in payload.get("react_steps", [])
    ]
    invalid = sum(
        not bool(step.get("valid_action", True))
        for step in steps
    )
    calls = sum(
        int(payload.get("react_num_calls", 0) or 0)
        for payload in payloads
    )
    report = {
        "documents": len(directories),
        "queries": len(payloads),
        "mean_steps": (
            sum(len(payload.get("react_steps", [])) for payload in payloads)
            / len(payloads)
        ),
        "mean_calls": calls / len(payloads),
        "mean_searches": (
            sum(
                int(payload.get("react_search_count", 0) or 0)
                for payload in payloads
            )
            / len(payloads)
        ),
        "mean_lookups": (
            sum(
                int(payload.get("react_lookup_count", 0) or 0)
                for payload in payloads
            )
            / len(payloads)
        ),
        "bad_call_rate": (
            sum(
                int(payload.get("react_num_bad_calls", 0) or 0)
                for payload in payloads
            )
            / max(1, calls)
        ),
        "invalid_action_rate": invalid / max(1, len(steps)),
        "step_limit_rate": (
            sum(
                payload.get("react_termination_reason") == "step_limit"
                for payload in payloads
            )
            / len(payloads)
        ),
        "elapsed_seconds": elapsed_seconds,
        "token_totals": token_totals,
    }
    return report


def validate_expected_queries(report: dict, expected_queries: int) -> dict:
    observed = int(report.get("queries") or 0)
    expected = int(expected_queries)
    if observed != expected:
        raise ValueError(
            f"ReAct result count {observed} != expected queries {expected}"
        )
    return report


def _as_directories(eval_dirs) -> list[Path]:
    if isinstance(eval_dirs, (str, Path)):
        return [Path(eval_dirs)]
    return [Path(path) for path in eval_dirs]


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate ReAct trajectory and cost diagnostics.",
    )
    parser.add_argument(
        "--eval-dir",
        action="append",
        required=True,
        help="One eval_*_react directory. Repeat for multiple documents.",
    )
    parser.add_argument(
        "--expected-queries",
        type=int,
        required=True,
        help="Expected total query count; mismatch fails the command.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON report path.",
    )
    args = parser.parse_args()

    report = aggregate_react_diagnostics(args.eval_dir)
    validate_expected_queries(report, args.expected_queries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
