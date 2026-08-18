from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import load_json, write_json


def _mean(total: float, count: int) -> float:
    return total / count if count else 0.0


def summarize_trajectories(
    trajectories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    query_count = len(trajectories)
    total_steps = sum(len(item.get("steps") or []) for item in trajectories)
    model_calls = sum(int(item.get("model_calls") or 0) for item in trajectories)
    repair_calls = sum(int(item.get("repair_calls") or 0) for item in trajectories)
    invalid_actions = sum(
        int(item.get("invalid_actions") or 0) for item in trajectories
    )
    search_calls = sum(int(item.get("search_calls") or 0) for item in trajectories)
    lookup_calls = sum(int(item.get("lookup_calls") or 0) for item in trajectories)
    prompt_tokens = sum(
        int(dict(item.get("usage") or {}).get("prompt_tokens") or 0)
        for item in trajectories
    )
    completion_tokens = sum(
        int(dict(item.get("usage") or {}).get("completion_tokens") or 0)
        for item in trajectories
    )
    elapsed = sum(float(item.get("elapsed_seconds") or 0.0) for item in trajectories)
    return {
        "query_count": query_count,
        "mean_steps": _mean(total_steps, query_count),
        "mean_model_calls": _mean(model_calls, query_count),
        "mean_search_calls": _mean(search_calls, query_count),
        "mean_lookup_calls": _mean(lookup_calls, query_count),
        "repair_call_rate": _mean(repair_calls, model_calls),
        "repair_query_rate": _mean(
            sum(1 for item in trajectories if int(item.get("repair_calls") or 0) > 0),
            query_count,
        ),
        "invalid_action_rate": _mean(invalid_actions, total_steps),
        "invalid_action_query_rate": _mean(
            sum(
                1
                for item in trajectories
                if int(item.get("invalid_actions") or 0) > 0
            ),
            query_count,
        ),
        "step_limit_rate": _mean(
            sum(
                1
                for item in trajectories
                if str(item.get("termination_reason") or "") == "step_limit"
            ),
            query_count,
        ),
        "mean_prompt_tokens": _mean(prompt_tokens, query_count),
        "mean_completion_tokens": _mean(completion_tokens, query_count),
        "mean_total_tokens": _mean(prompt_tokens + completion_tokens, query_count),
        "mean_seconds": _mean(elapsed, query_count),
        "totals": {
            "steps": total_steps,
            "model_calls": model_calls,
            "repair_calls": repair_calls,
            "invalid_actions": invalid_actions,
            "search_calls": search_calls,
            "lookup_calls": lookup_calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "seconds": elapsed,
        },
    }


def summarize_manifest(
    manifest_path: str | Path, expected_queries: int | None = None
) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    manifest = load_json(path)
    if manifest.get("status") != "complete":
        raise RuntimeError("ReAct raw manifest is not complete")
    run_directory = Path(manifest["run_directory"]).resolve()
    trajectories: list[dict[str, Any]] = []
    question_ids: set[str] = set()
    for entry in manifest.get("scopes") or []:
        scope = load_json(run_directory / str(entry["relative_path"]))
        if scope.get("status") != "complete":
            raise RuntimeError(f"incomplete raw scope: {entry.get('scope_id')!r}")
        for query in scope.get("queries") or []:
            question_id = str(query.get("question_id") or "")
            if not question_id or question_id in question_ids:
                raise ValueError(f"duplicate or empty question ID: {question_id!r}")
            question_ids.add(question_id)
            trajectories.append(dict(query))
    expected = (
        int(expected_queries)
        if expected_queries is not None
        else int(manifest.get("query_count") or 0)
    )
    if len(trajectories) != expected:
        raise RuntimeError(
            f"diagnostic coverage mismatch: expected {expected}, got {len(trajectories)}"
        )
    return {
        "dataset": manifest.get("dataset"),
        "run_id": manifest.get("run_id"),
        "expected_queries": expected,
        "complete": True,
        **summarize_trajectories(trajectories),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize official ReAct trajectories")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-queries", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = summarize_manifest(args.manifest, args.expected_queries)
    output = Path(args.output) if args.output else Path(args.manifest).with_name(
        "diagnostics.json"
    )
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

