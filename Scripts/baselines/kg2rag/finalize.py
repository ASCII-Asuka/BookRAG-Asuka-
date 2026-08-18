from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from Core.utils.utils import num_tokens

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
from .model_clients import OpenAICompatibleClient


Generator = Callable[[str, str, Sequence[Mapping[str, Any]]], dict[str, Any]]


class UnifiedGenerator:
    def __init__(self, config: Mapping[str, Any]):
        generation = dict(config.get("generation") or {})
        llm_base = os.getenv("KG2RAG_GENERATION_BASE_URL") or os.getenv(
            "KG2RAG_LLM_BASE_URL", ""
        )
        embedding_base = os.getenv("KG2RAG_EMBEDDING_BASE_URL", llm_base)
        reranker_base = os.getenv("KG2RAG_RERANKER_BASE_URL", embedding_base)
        self.client = OpenAICompatibleClient(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            llm_base_url=llm_base,
            embedding_base_url=embedding_base,
            reranker_base_url=reranker_base,
        )
        self.model = str(
            generation.get("model_name") or "Qwen/Qwen3-VL-8B-Instruct"
        )
        self.temperature = float(generation.get("temperature", 0.1))
        self.max_output_tokens = int(generation.get("max_output_tokens", 512))

    @staticmethod
    def _messages(
        dataset: str, question: str, selected: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, str]]:
        evidence = "\n\n".join(
            f"[Evidence {item.get('source_id')}]\n{item.get('content', '')}"
            for item in selected
        )
        if dataset == "hotpotqa":
            schema = (
                '{"answer_short":"concise answer","answer_rationale":"brief support",'
                '"supporting_facts":[["exact title",0]]}'
            )
            citation_rule = (
                "Supporting facts must use only exact titles and sentence IDs shown "
                "inside the supplied evidence."
            )
        else:
            schema = (
                '{"answer_short":"concise answer","answer_rationale":"brief support",'
                '"supporting_block_ids":["evidence source id"]}'
            )
            citation_rule = (
                "Supporting block IDs must use only Evidence IDs shown below."
            )
        return [
            {
                "role": "system",
                "content": (
                    "Answer using only supplied evidence. Return exactly one JSON object. "
                    + citation_rule
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Output schema: {schema}\nQuestion: {question}\n\nEvidence:\n{evidence}"
                ),
            },
        ]

    def __call__(
        self, dataset: str, question: str, selected: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        started = time.perf_counter()
        payload, usage = self.client.chat_json(
            self.model,
            self._messages(dataset, question, selected),
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        if not str(payload.get("answer_short") or payload.get("answer") or "").strip():
            raise RuntimeError("final generator returned no answer")
        payload["usage"] = usage
        payload["time"] = time.perf_counter() - started
        return payload


def _matching_query(scope: Mapping[str, Any], question_id: str) -> Mapping[str, Any]:
    matches = [
        query
        for query in scope.get("queries") or []
        if str(query.get("question_id") or "") == question_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one prepared query for {question_id!r}, got {len(matches)}"
        )
    return matches[0]


def _answer(payload: Mapping[str, Any]) -> tuple[str, str]:
    answer = str(payload.get("answer_short") or payload.get("answer") or "").strip()
    if not answer:
        raise RuntimeError("generator returned an empty answer")
    rationale = str(
        payload.get("answer_rationale") or payload.get("rationale") or ""
    ).strip()
    return answer, rationale


def _sum_costs(costs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prompt = sum(int(cost.get("rag_cost", {}).get("prompt_tokens", 0)) for cost in costs)
    completion = sum(
        int(cost.get("rag_cost", {}).get("completion_tokens", 0)) for cost in costs
    )
    return {
        "rag_cost": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
        "time": sum(float(cost.get("time") or 0.0) for cost in costs),
        "queries": list(costs),
    }


def _load_completed_query(
    query_directory: Path, question_id: str, identity: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    result_path = query_directory / "result.json"
    retrieval_path = query_directory / "retrieval_res.json"
    cost_path = query_directory / "token_cost.json"
    if not (result_path.exists() and retrieval_path.exists() and cost_path.exists()):
        return None
    result = load_json(result_path)
    retrieval = load_json(retrieval_path)
    cost = load_json(cost_path)
    if (
        str(result.get("question_id") or "") != question_id
        or result.get("finalization_identity") != identity
        or retrieval.get("finalization_identity") != identity
        or cost.get("finalization_identity") != identity
        or not str(result.get("answer_short") or "").strip()
    ):
        return None
    return result, cost


def finalize_runs(
    retrieval_manifest_path: str | Path,
    config_path: str | Path,
    generator: Generator | None = None,
    token_counter: Callable[[str], int] = num_tokens,
) -> dict[str, Any]:
    manifest_path = Path(retrieval_manifest_path).resolve()
    raw_manifest = load_json(manifest_path)
    if raw_manifest.get("status") != "complete":
        raise RuntimeError("KG2RAG retrieval manifest is not complete")
    run_directory = Path(raw_manifest["run_directory"]).resolve()
    prepared_manifest_path = Path(raw_manifest["prepared_manifest_path"]).resolve()
    prepared_root = prepared_manifest_path.parent
    prepared_manifest = load_json(prepared_manifest_path)
    prepared_by_scope = {
        str(entry["scope_id"]): entry for entry in prepared_manifest.get("scopes") or []
    }
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    method_suffix = str(config.get("method_suffix") or "kg2rag_official")
    config_sha = sha256_file(config_path)
    max_blocks = int(config.get("generation_max_blocks") or 10)
    max_tokens = int(config.get("generation_max_tokens") or 4000)
    max_support = int(config.get("supporting_evidence_topk") or 4)
    working_dir = Path(prepared_manifest["working_dir"]).resolve()
    dataset = str(
        raw_manifest.get("dataset") or prepared_manifest.get("dataset") or ""
    ).lower()
    generation = generator
    completed_ids: set[str] = set()
    scope_count = 0

    for raw_entry in raw_manifest.get("scopes") or []:
        scope_id = str(raw_entry.get("scope_id") or "")
        prepared_entry = prepared_by_scope.get(scope_id)
        if prepared_entry is None:
            raise ValueError(f"retrieval manifest contains unknown scope: {scope_id!r}")
        scope = load_json(prepared_root / prepared_entry["relative_path"])
        raw_scope = load_json(run_directory / raw_entry["relative_path"])
        if str(raw_scope.get("scope_id") or "") != scope_id:
            raise ValueError(f"scope ID mismatch for {scope_id!r}")
        identity = sha256_json(
            {
                "retrieval_cache_identity": raw_scope.get("cache_identity"),
                "finalize_config_sha256": config_sha,
                "method_suffix": method_suffix,
            }
        )
        result_directory = working_dir / scope_id / f"eval_{dataset}_{method_suffix}"
        final_results: list[dict[str, Any]] = []
        query_costs: list[dict[str, Any]] = []

        for query_index, raw_query in enumerate(raw_scope.get("queries") or [], start=1):
            question_id = str(raw_query.get("question_id") or "")
            if not question_id or question_id in completed_ids:
                raise ValueError(f"duplicate or empty finalized question ID: {question_id!r}")
            completed_ids.add(question_id)
            prepared_query = _matching_query(scope, question_id)
            query_directory = result_directory / f"query_{query_index:03d}"
            cached = _load_completed_query(query_directory, question_id, identity)
            if cached is not None:
                cached_result, cached_cost = cached
                final_results.append(cached_result)
                query_costs.append(cached_cost)
                continue

            ranked = list(raw_query.get("ranked_results") or [])
            selected = apply_generation_budget(
                ranked,
                max_blocks=max_blocks,
                max_tokens=max_tokens,
                token_counter=token_counter,
            )
            if not selected:
                raise RuntimeError(
                    f"no evidence fits generation budget for {question_id!r}"
                )
            if generation is None:
                generation = UnifiedGenerator(config)
            generated = generation(
                dataset, str(prepared_query.get("question") or ""), selected
            )
            answer_short, answer_rationale = _answer(generated)

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

            generation_stage = dict(generated.get("usage") or {})
            generation_stage["time"] = float(generated.get("time") or 0.0)
            retrieval_stage = dict(raw_query.get("retrieval_token_cost") or {})
            retrieval_stage["time"] = float(
                retrieval_stage.get("time") or raw_query.get("retrieval_time") or 0.0
            )
            query_cost = combine_online_cost(retrieval_stage, generation_stage)
            query_cost["question_id"] = question_id
            query_cost["finalization_identity"] = identity
            query_costs.append(query_cost)

            retrieval_payload = {
                "method": method_suffix,
                "finalization_identity": identity,
                "ranked_results": ranked,
                "selected_generation_context": selected,
                "supporting_evidence": supporting_evidence,
                "supporting_block_ids": supporting_block_ids,
                "semantic_seeds": raw_query.get("semantic_seeds") or [],
                "expanded_source_ids": raw_query.get("expanded_source_ids") or [],
                "expansion_edges": raw_query.get("expansion_edges") or [],
                "components": raw_query.get("components") or [],
                "citation_validation": citation_validation,
                "retrieval_time": float(raw_query.get("retrieval_time") or 0.0),
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
                "supporting_facts": (
                    supporting_block_ids if dataset == "hotpotqa" else []
                ),
                "finalization_identity": identity,
            }
            write_json(query_directory / "result.json", result)
            write_json(query_directory / "retrieval_res.json", retrieval_payload)
            write_json(query_directory / "token_cost.json", query_cost)
            write_json(
                query_directory / "evidence_chain.json",
                {
                    "ordered_evidence": selected,
                    "supporting_evidence": supporting_evidence,
                    "semantic_seeds": raw_query.get("semantic_seeds") or [],
                    "expanded_source_ids": raw_query.get("expanded_source_ids") or [],
                    "expansion_edges": raw_query.get("expansion_edges") or [],
                    "components": raw_query.get("components") or [],
                    "citation_validation": citation_validation,
                },
            )
            final_results.append(result)

        write_json(result_directory / "final_results.json", final_results)
        write_json(result_directory / "token_cost.json", _sum_costs(query_costs))
        write_json(
            result_directory / "kg2rag_index_stats.json",
            raw_scope.get("index_stats") or {},
        )
        scope_count += 1

    expected_questions = int(raw_manifest.get("query_count") or len(completed_ids))
    if len(completed_ids) != expected_questions:
        raise RuntimeError(
            f"prediction coverage mismatch: expected {expected_questions}, got {len(completed_ids)}"
        )
    return {
        "dataset": dataset,
        "method_suffix": method_suffix,
        "scope_count": scope_count,
        "question_count": len(completed_ids),
        "complete": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize KG2RAG outputs")
    parser.add_argument("--retrieval-manifest", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = finalize_runs(args.retrieval_manifest, args.config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
