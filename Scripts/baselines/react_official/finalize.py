from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .common import load_json, sha256_file, sha256_json, write_json


def _evidence_key(item: Mapping[str, Any], dataset: str) -> tuple[Any, ...] | None:
    if dataset == "hotpotqa":
        title = str(item.get("hotpot_title") or item.get("title") or "")
        sent_id = item.get("hotpot_sent_id", item.get("sent_id"))
        try:
            return (title, int(sent_id)) if title else None
        except (TypeError, ValueError):
            return None
    paragraph_id = item.get("paragraph_id")
    if paragraph_id is None:
        paragraph_id = item.get("source_id")
    return (str(paragraph_id),) if paragraph_id is not None else None


def rank_observed_evidence(
    steps: Sequence[Mapping[str, Any]], *, dataset: str
) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for action_name in ("Lookup", "Search"):
        for step in steps:
            if str(step.get("action_name") or "").lower() != action_name.lower():
                continue
            for raw_item in step.get("evidence") or []:
                if not isinstance(raw_item, Mapping):
                    continue
                item = dict(raw_item)
                key = _evidence_key(item, dataset)
                if key is None or key in seen:
                    continue
                seen.add(key)
                item["observation_action"] = action_name
                item["supporting_rank"] = len(ordered) + 1
                item["rank"] = len(ordered) + 1
                ordered.append(item)
    return ordered


def _prepared_query(scope: Mapping[str, Any], question_id: str) -> dict[str, Any]:
    matches = [
        dict(query)
        for query in scope.get("queries") or []
        if str(query.get("question_id") or "") == question_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one prepared query for {question_id!r}, got {len(matches)}"
        )
    return matches[0]


def _legal_evidence(
    scope: Mapping[str, Any], dataset: str
) -> dict[tuple[Any, ...], dict[str, Any]]:
    legal: dict[tuple[Any, ...], dict[str, Any]] = {}
    for chunk in scope.get("chunks") or []:
        metadata = dict(chunk.get("metadata") or {})
        source_id = str(chunk.get("source_id") or "")
        if dataset == "hotpotqa":
            title = str(metadata.get("hotpot_title") or metadata.get("title") or "")
            for sentence in metadata.get("hotpot_sentences") or []:
                if not isinstance(sentence, Mapping) or not title:
                    continue
                atom = {
                    "source_id": source_id,
                    "hotpot_title": title,
                    "hotpot_sent_id": int(sentence.get("sent_id")),
                    "content": str(sentence.get("text") or ""),
                }
                legal[(title, atom["hotpot_sent_id"])] = atom
        else:
            paragraph_id = metadata.get("paragraph_id")
            if paragraph_id is None:
                paragraph_id = metadata.get("qasper_paragraph_id", source_id)
            atom = {
                "source_id": source_id,
                "paragraph_id": paragraph_id,
                "content": str(
                    metadata.get("qasper_evidence_text")
                    or metadata.get("original_text")
                    or chunk.get("content")
                    or ""
                ),
                "qasper_evidence_text": str(
                    metadata.get("qasper_evidence_text")
                    or metadata.get("original_text")
                    or chunk.get("content")
                    or ""
                ),
                "section": str(metadata.get("section") or ""),
                "block_type": "paragraph",
                "source": "qasper_paragraph",
            }
            legal[(str(paragraph_id),)] = atom
    return legal


def _validate_ranked(
    ranked: Sequence[Mapping[str, Any]],
    scope: Mapping[str, Any],
    dataset: str,
) -> list[dict[str, Any]]:
    legal = _legal_evidence(scope, dataset)
    validated: list[dict[str, Any]] = []
    for raw in ranked:
        key = _evidence_key(raw, dataset)
        atom = legal.get(key) if key is not None else None
        if atom is None:
            raise ValueError(f"trajectory contains out-of-scope evidence: {key!r}")
        validated.append(
            {
                **atom,
                "observation_action": raw.get("observation_action"),
                "supporting_rank": len(validated) + 1,
                "rank": len(validated) + 1,
            }
        )
    return validated


def _query_cost(raw_query: Mapping[str, Any], identity: str) -> dict[str, Any]:
    usage = dict(raw_query.get("usage") or {})
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or prompt + completion)
    elapsed = float(raw_query.get("elapsed_seconds") or 0.0)
    return {
        "question_id": str(raw_query.get("question_id") or ""),
        "finalization_identity": identity,
        "rag_cost": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        },
        "time": elapsed,
        "stages": {
            "react": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
                "time": elapsed,
            }
        },
    }


def _sum_costs(costs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prompt = sum(int(item["rag_cost"]["prompt_tokens"]) for item in costs)
    completion = sum(int(item["rag_cost"]["completion_tokens"]) for item in costs)
    return {
        "rag_cost": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
        "time": sum(float(item.get("time") or 0.0) for item in costs),
        "queries": list(costs),
    }


def _load_cached(
    query_directory: Path, question_id: str, identity: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    paths = [
        query_directory / "result.json",
        query_directory / "retrieval_res.json",
        query_directory / "evidence_chain.json",
        query_directory / "token_cost.json",
    ]
    if not all(path.exists() for path in paths):
        return None
    result, retrieval, _chain, cost = [load_json(path) for path in paths]
    if (
        str(result.get("question_id") or result.get("qasper_question_id") or "")
        != question_id
        or result.get("finalization_identity") != identity
        or retrieval.get("finalization_identity") != identity
        or cost.get("finalization_identity") != identity
    ):
        return None
    return dict(result), dict(cost)


def finalize_runs(
    raw_manifest_path: str | Path, config_path: str | Path
) -> dict[str, Any]:
    manifest_path = Path(raw_manifest_path).resolve()
    raw_manifest = load_json(manifest_path)
    if raw_manifest.get("status") != "complete":
        raise RuntimeError("ReAct raw manifest is not complete")
    run_directory = Path(raw_manifest["run_directory"]).resolve()
    prepared_manifest_path = Path(raw_manifest["prepared_manifest_path"]).resolve()
    prepared_root = prepared_manifest_path.parent
    prepared_manifest = load_json(prepared_manifest_path)
    prepared_entries = {
        str(entry.get("scope_id") or ""): dict(entry)
        for entry in prepared_manifest.get("scopes") or []
    }
    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    method_suffix = str(config.get("method_suffix") or "react_official_repro")
    max_support = int(
        dict(config.get("evidence") or {}).get("max_supporting_evidence") or 4
    )
    dataset = str(raw_manifest.get("dataset") or prepared_manifest.get("dataset") or "").lower()
    working_dir = Path(prepared_manifest["working_dir"]).resolve()
    completed_ids: set[str] = set()
    scope_count = 0

    for raw_entry in raw_manifest.get("scopes") or []:
        scope_id = str(raw_entry.get("scope_id") or "")
        prepared_entry = prepared_entries.get(scope_id)
        if prepared_entry is None:
            raise ValueError(f"raw manifest contains unknown scope: {scope_id!r}")
        scope = load_json(prepared_root / prepared_entry["relative_path"])
        raw_scope = load_json(run_directory / raw_entry["relative_path"])
        if raw_scope.get("status") != "complete":
            raise RuntimeError(f"raw scope is not complete: {scope_id!r}")
        identity = sha256_json(
            {
                "raw_cache_identity": raw_scope.get("cache_identity"),
                "config_sha256": sha256_file(config_file),
                "method_suffix": method_suffix,
            }
        )
        result_directory = working_dir / scope_id / f"eval_{dataset}_{method_suffix}"
        final_results: list[dict[str, Any]] = []
        query_costs: list[dict[str, Any]] = []

        for index, raw_query in enumerate(raw_scope.get("queries") or [], start=1):
            question_id = str(raw_query.get("question_id") or "")
            if not question_id or question_id in completed_ids:
                raise ValueError(f"duplicate or empty finalized question ID: {question_id!r}")
            completed_ids.add(question_id)
            prepared_query = _prepared_query(scope, question_id)
            query_directory = result_directory / f"query_{index:03d}"
            cached = _load_cached(query_directory, question_id, identity)
            if cached is not None:
                result, cost = cached
                final_results.append(result)
                query_costs.append(cost)
                continue

            observed = rank_observed_evidence(
                raw_query.get("steps") or [], dataset=dataset
            )
            ranked = _validate_ranked(observed, scope, dataset)
            supporting = ranked[:max_support]
            if dataset == "hotpotqa":
                supporting_ids = [
                    [item["hotpot_title"], item["hotpot_sent_id"]]
                    for item in supporting
                ]
            else:
                supporting_ids = [item["paragraph_id"] for item in supporting]

            citation_validation = {
                "policy": "trace-observed-only",
                "gold_used": False,
                "fallback_used": False,
                "observed_count": len(ranked),
                "submitted_count": len(supporting),
            }
            retrieval_payload = {
                "method": method_suffix,
                "finalization_identity": identity,
                "ranked_results": ranked,
                "selected": supporting,
                "supporting_evidence": supporting,
                "supporting_block_ids": supporting_ids,
                "support_controller": {"enabled": True, "source": "react-trajectory"},
                "citation_validation": citation_validation,
                "trajectory": raw_query.get("trajectory") or "",
                "steps": raw_query.get("steps") or [],
                "termination_reason": raw_query.get("termination_reason"),
                "model_calls": int(raw_query.get("model_calls") or 0),
                "repair_calls": int(raw_query.get("repair_calls") or 0),
                "invalid_actions": int(raw_query.get("invalid_actions") or 0),
            }
            row = dict(prepared_query.get("dataset_row") or {})
            answer = str(raw_query.get("answer") or "").strip()
            result = {
                **row,
                "question_id": question_id,
                "question": str(prepared_query.get("question") or ""),
                "answer_short": answer,
                "output": answer,
                "pred": answer,
                "retrieved_block_ids": [item.get("source_id") for item in ranked],
                "retrieved_node_ids": [item.get("source_id") for item in ranked],
                "supporting_block_ids": supporting_ids,
                "supporting_facts": supporting_ids if dataset == "hotpotqa" else [],
                "finalization_identity": identity,
            }
            cost = _query_cost(raw_query, identity)
            chain = {
                "ordered_evidence": ranked,
                "supporting_evidence": supporting,
                "trajectory": raw_query.get("trajectory") or "",
                "steps": raw_query.get("steps") or [],
                "citation_validation": citation_validation,
            }
            write_json(query_directory / "result.json", result)
            write_json(query_directory / "retrieval_res.json", retrieval_payload)
            write_json(query_directory / "evidence_chain.json", chain)
            write_json(query_directory / "token_cost.json", cost)
            final_results.append(result)
            query_costs.append(cost)

        write_json(result_directory / "final_results.json", final_results)
        write_json(result_directory / "token_cost.json", _sum_costs(query_costs))
        scope_count += 1

    expected = int(raw_manifest.get("query_count") or len(completed_ids))
    if len(completed_ids) != expected:
        raise RuntimeError(
            f"final prediction coverage mismatch: expected {expected}, got {len(completed_ids)}"
        )
    return {
        "dataset": dataset,
        "method_suffix": method_suffix,
        "scope_count": scope_count,
        "question_count": len(completed_ids),
        "complete": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize official ReAct trajectories")
    parser.add_argument("--raw-manifest", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = finalize_runs(args.raw_manifest, args.config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

