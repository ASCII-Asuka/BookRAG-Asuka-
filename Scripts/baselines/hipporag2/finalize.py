from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import yaml

from Core.configs.llm_config import LLMConfig
from Core.provider.TokenTracker import TokenTracker
from Core.provider.llm import LLM
from Core.utils.utils import num_tokens, try_parse_json_object

from .common import (
    apply_generation_budget,
    combine_online_cost,
    load_json,
    sha256_file,
    sha256_json,
    validate_hotpot_citations,
    validate_qasper_citations,
    write_json,
)


Generator = Callable[[str, str, Sequence[Mapping[str, Any]]], Dict[str, Any]]


def _usage_delta(after: Mapping[str, int], before: Mapping[str, int]) -> Dict[str, int]:
    prompt = max(0, int(after.get("prompt_tokens", 0)) - int(before.get("prompt_tokens", 0)))
    completion = max(
        0,
        int(after.get("completion_tokens", 0)) - int(before.get("completion_tokens", 0)),
    )
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


class UnifiedGenerator:
    def __init__(self, config: Mapping[str, Any]):
        generation = dict(config.get("generation") or {})
        api_base = os.environ.get("HIPPORAG2_GENERATION_BASE_URL") or os.environ.get(
            "HIPPORAG2_LLM_BASE_URL"
        )
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_base or not api_key:
            raise RuntimeError(
                "HIPPORAG2_GENERATION_BASE_URL (or HIPPORAG2_LLM_BASE_URL) and OPENAI_API_KEY are required"
            )
        self.llm = LLM(
            LLMConfig(
                model_name=str(
                    generation.get("model_name") or "Qwen/Qwen3-VL-8B-Instruct"
                ),
                api_base=api_base,
                api_key=api_key,
                temperature=float(generation.get("temperature", 0.1)),
                max_tokens=int(generation.get("model_context_tokens", 32768)),
                max_output_tokens=int(generation.get("max_output_tokens", 512)),
                backend="openai",
                max_workers=1,
            )
        )

    @staticmethod
    def _prompt(dataset: str, question: str, selected: Sequence[Mapping[str, Any]]) -> str:
        evidence_lines = []
        for item in selected:
            source_id = str(item.get("source_id") or "")
            evidence_lines.append(f"[Evidence {source_id}]\n{item.get('content', '')}")
        evidence = "\n\n".join(evidence_lines)
        if dataset == "hotpotqa":
            output_contract = (
                '{"answer_short":"...","answer_rationale":"...",'
                '"supporting_facts":[["exact title",0]]}'
            )
            citation_rule = "Supporting facts must use only exact titles and sentence IDs shown in the evidence."
        else:
            output_contract = (
                '{"answer_short":"...","answer_rationale":"...",'
                '"supporting_block_ids":["evidence source id"]}'
            )
            citation_rule = "Supporting block IDs must use only the Evidence IDs shown above."
        return (
            "Answer the question using only the supplied evidence. Return one JSON object and no extra text.\n"
            f"{citation_rule}\nOutput schema: {output_contract}\n\n"
            f"Question: {question}\n\nEvidence:\n{evidence}"
        )

    def __call__(
        self, dataset: str, question: str, selected: Sequence[Mapping[str, Any]]
    ) -> Dict[str, Any]:
        tracker = TokenTracker.get_instance()
        before = tracker.get_usage()
        started = time.perf_counter()
        response = self.llm.get_completion(self._prompt(dataset, question, selected), json_response=True)
        elapsed = time.perf_counter() - started
        after = tracker.get_usage()
        _, payload = try_parse_json_object(response)
        if not isinstance(payload, dict) or not str(
            payload.get("answer_short") or payload.get("answer") or ""
        ).strip():
            raise RuntimeError("final generator returned no valid answer JSON")
        payload["usage"] = _usage_delta(after, before)
        payload["time"] = elapsed
        return payload


def _generation_answer(payload: Mapping[str, Any]) -> tuple[str, str]:
    answer = str(payload.get("answer_short") or payload.get("answer") or "").strip()
    if not answer:
        raise RuntimeError("generator returned an empty answer")
    rationale = str(payload.get("answer_rationale") or payload.get("rationale") or "").strip()
    return answer, rationale


def _sum_costs(costs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    prompt = sum(int(cost.get("rag_cost", {}).get("prompt_tokens", 0)) for cost in costs)
    completion = sum(int(cost.get("rag_cost", {}).get("completion_tokens", 0)) for cost in costs)
    return {
        "rag_cost": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
        "time": sum(float(cost.get("time", 0.0)) for cost in costs),
        "queries": list(costs),
    }


def _matching_query(scope: Mapping[str, Any], question_id: str) -> Mapping[str, Any]:
    matches = [
        query
        for query in scope.get("queries") or []
        if str(query.get("question_id")) == question_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one prepared query for {question_id!r}, got {len(matches)}")
    return matches[0]


def _load_completed_query(
    query_dir: Path, question_id: str, finalization_identity: str
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]] | None:
    paths = [
        query_dir / "result.json",
        query_dir / "retrieval_res.json",
        query_dir / "token_cost.json",
    ]
    if not all(path.exists() for path in paths):
        return None
    result, retrieval, cost = [load_json(path) for path in paths]
    if str(result.get("question_id") or "") != question_id:
        return None
    if result.get("finalization_identity") != finalization_identity:
        return None
    if retrieval.get("finalization_identity") != finalization_identity:
        return None
    if cost.get("finalization_identity") != finalization_identity:
        return None
    if not str(result.get("answer_short") or "").strip():
        return None
    if not isinstance(retrieval.get("supporting_evidence"), list):
        return None
    return result, retrieval, cost


def finalize_runs(
    retrieval_manifest_path: str,
    config_path: str,
    generator: Generator | None = None,
    token_counter: Callable[[str], int] = num_tokens,
) -> Dict[str, Any]:
    raw_manifest_path = Path(retrieval_manifest_path).resolve()
    raw_root = raw_manifest_path.parent
    raw_manifest = load_json(raw_manifest_path)
    prepared_manifest_path = Path(raw_manifest["prepared_manifest_path"]).resolve()
    prepared_root = prepared_manifest_path.parent
    prepared_manifest = load_json(prepared_manifest_path)
    prepared_by_scope = {
        str(entry["scope_id"]): entry for entry in prepared_manifest.get("scopes") or []
    }
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    config_sha256 = sha256_file(config_path)
    method_suffix = str(config.get("method_suffix") or "hipporag2_official")
    generation = generator
    max_blocks = int(config.get("generation_max_blocks", 10))
    max_tokens = int(config.get("generation_max_tokens", 4000))
    max_support = int(config.get("supporting_evidence_topk", 4))
    working_dir = Path(prepared_manifest["working_dir"]).resolve()
    dataset = str(raw_manifest.get("dataset") or prepared_manifest.get("dataset") or "").lower()
    completed_question_ids: set[str] = set()
    scope_count = 0

    for raw_entry in raw_manifest.get("scopes") or []:
        scope_id = str(raw_entry.get("scope_id") or "")
        if scope_id not in prepared_by_scope:
            raise ValueError(f"retrieval manifest contains unknown scope: {scope_id!r}")
        scope = load_json(prepared_root / prepared_by_scope[scope_id]["relative_path"])
        raw_scope = load_json(raw_root / raw_entry["relative_path"])
        if str(raw_scope.get("scope_id")) != scope_id:
            raise ValueError(f"scope ID mismatch for {scope_id!r}")
        if raw_scope.get("status") not in (None, "complete"):
            raise RuntimeError(f"retrieval scope {scope_id!r} is not complete")
        finalization_identity = sha256_json(
            {
                "retrieval_cache_identity": raw_scope.get("cache_identity"),
                "finalize_config_sha256": config_sha256,
                "method_suffix": method_suffix,
            }
        )
        result_dir = working_dir / scope_id / f"eval_{dataset}_{method_suffix}"
        final_results = []
        query_costs = []
        for query_index, raw_query in enumerate(raw_scope.get("queries") or [], start=1):
            question_id = str(raw_query.get("question_id") or "")
            if not question_id or question_id in completed_question_ids:
                raise ValueError(f"duplicate or empty finalized question ID: {question_id!r}")
            completed_question_ids.add(question_id)
            prepared_query = _matching_query(scope, question_id)
            query_dir = result_dir / f"query_{query_index:03d}"
            cached_final = _load_completed_query(
                query_dir, question_id, finalization_identity
            )
            if cached_final is not None:
                cached_result, _, cached_cost = cached_final
                final_results.append(cached_result)
                query_costs.append(cached_cost)
                continue
            generation_stage_started = time.perf_counter()
            ranked = list(raw_query.get("ranked_results") or [])
            selected = apply_generation_budget(
                ranked,
                max_blocks=max_blocks,
                max_tokens=max_tokens,
                token_counter=token_counter,
            )
            if not selected:
                raise RuntimeError(f"no evidence fits the generation budget for {question_id!r}")
            if generation is None:
                generation = UnifiedGenerator(config)
            generated = generation(
                dataset, str(prepared_query.get("question") or ""), selected
            )
            answer_short, answer_rationale = _generation_answer(generated)
            if dataset == "hotpotqa":
                supporting_evidence, citation_validation = validate_hotpot_citations(
                    generated.get("supporting_facts") or [],
                    selected,
                    max_support,
                    question=str(prepared_query.get("question") or ""),
                )
                supporting_block_ids = [
                    [item["hotpot_title"], item["hotpot_sent_id"]]
                    for item in supporting_evidence
                ]
            else:
                requested_ids = generated.get("supporting_block_ids") or generated.get(
                    "supporting_paragraph_ids"
                ) or []
                supporting_evidence, citation_validation = validate_qasper_citations(
                    requested_ids, selected, max_support
                )
                supporting_block_ids = [
                    item.get("paragraph_id") for item in supporting_evidence
                ]
            generation_seconds = max(
                float(generated.get("time") or 0.0),
                time.perf_counter() - generation_stage_started,
            )
            query_cost = combine_online_cost(
                raw_query.get("retrieval_token_cost") or {},
                generated.get("usage") or {},
                float(raw_query.get("retrieval_time") or 0.0),
                generation_seconds,
            )
            query_cost["question_id"] = question_id
            query_cost["finalization_identity"] = finalization_identity
            query_costs.append(query_cost)

            retrieval_payload = {
                "method": method_suffix,
                "finalization_identity": finalization_identity,
                "ranked_results": ranked,
                "selected_generation_context": selected,
                "supporting_evidence": supporting_evidence,
                "supporting_block_ids": supporting_block_ids,
                "graph_seeds": raw_query.get("graph_seeds") or [],
                "citation_validation": citation_validation,
                "retrieval_time": float(raw_query.get("retrieval_time") or 0.0),
                "ppr_time": float(raw_query.get("ppr_time") or 0.0),
                "recognition_memory_time": float(
                    raw_query.get("recognition_memory_time") or 0.0
                ),
                "retrieval_token_cost": raw_query.get("retrieval_token_cost") or {},
                "dense_fallback": bool(raw_query.get("dense_fallback", False)),
                "dense_fallback_reason": raw_query.get("dense_fallback_reason"),
            }
            row = dict(prepared_query.get("dataset_row") or {})
            result = {
                **row,
                "question_id": question_id,
                "question": str(prepared_query.get("question") or ""),
                "answer_short": answer_short,
                "answer_rationale": answer_rationale,
                "output": answer_short,
                "pred": answer_short,
                "retrieved_block_ids": [item.get("source_id") for item in ranked],
                "retrieved_node_ids": [item.get("source_id") for item in ranked],
                "supporting_block_ids": supporting_block_ids,
                "finalization_identity": finalization_identity,
            }
            write_json(query_dir / "result.json", result)
            write_json(query_dir / "retrieval_res.json", retrieval_payload)
            write_json(query_dir / "token_cost.json", query_cost)
            write_json(
                query_dir / "evidence_chain.json",
                {
                    "ordered_evidence": selected,
                    "supporting_evidence": supporting_evidence,
                    "graph_seeds": raw_query.get("graph_seeds") or [],
                    "citation_validation": citation_validation,
                },
            )
            final_results.append(result)

        write_json(result_dir / "final_results.json", final_results)
        write_json(result_dir / "token_cost.json", _sum_costs(query_costs))
        write_json(
            result_dir / "hipporag2_index_stats.json",
            raw_scope.get("index_stats") or {},
        )
        scope_count += 1

    expected_questions = int(raw_manifest.get("query_count") or len(completed_question_ids))
    if len(completed_question_ids) != expected_questions:
        raise RuntimeError(
            f"prediction coverage mismatch: expected {expected_questions}, got {len(completed_question_ids)}"
        )
    return {
        "dataset": dataset,
        "method_suffix": method_suffix,
        "scope_count": scope_count,
        "question_count": len(completed_question_ids),
        "complete": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate answers and finalize official HippoRAG 2 outputs")
    parser.add_argument("--retrieval-manifest", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = finalize_runs(args.retrieval_manifest, args.config)
    print(f"Finalized {summary['scope_count']} scopes and {summary['question_count']} questions")


if __name__ == "__main__":
    main()
