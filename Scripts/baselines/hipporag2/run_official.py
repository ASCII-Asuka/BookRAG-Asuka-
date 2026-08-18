from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np
import requests
import yaml

from .common import (
    build_cache_identity,
    load_json,
    normalize_ranked_results,
    safe_filename,
    sha256_file,
    sha256_json,
    write_json,
)


HippoFactory = Callable[[Mapping[str, Any], str], Any]
ChunkFactory = Callable[[str, str, Dict[str, Any]], Any]
OFFICIAL_REVISION = "c617143f01477243992a63b2e2151cc003dd3b21"
OFFICIAL_PACKAGE_VERSION = "2.0.0a4"
_INVALID_UNICODE_ESCAPE = re.compile(r"\\u(?![0-9a-fA-F]{4})")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


def sqlite_token_usage(root: str | Path) -> Dict[str, int]:
    prompt_tokens = 0
    completion_tokens = 0
    for database in Path(root).rglob("*.sqlite"):
        try:
            connection = sqlite3.connect(str(database))
            rows = connection.execute("SELECT metadata FROM cache").fetchall()
            connection.close()
        except (sqlite3.Error, OSError):
            continue
        for (metadata_text,) in rows:
            try:
                metadata = json.loads(metadata_text or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(metadata, Mapping):
                continue
            prompt_tokens += int(metadata.get("prompt_tokens") or 0)
            completion_tokens += int(metadata.get("completion_tokens") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


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


def _directory_size(path: str | Path) -> int:
    total = 0
    for candidate in Path(path).rglob("*"):
        if candidate.is_file():
            try:
                total += candidate.stat().st_size
            except OSError:
                pass
    return total


def _attach_vllm_bearer_auth(
    embedding_model: Any,
    api_key: str,
    *,
    max_attempts: int = 5,
    retry_backoff_seconds: float = 2.0,
) -> None:
    """Add authentication and bounded transient retries without editing upstream."""

    attempts = int(max_attempts)
    if attempts < 1:
        raise ValueError("embedding max_attempts must be at least 1")
    backoff = float(retry_backoff_seconds)
    if backoff < 0:
        raise ValueError("embedding retry_backoff_seconds cannot be negative")

    def call_model(model: Any, input_text: Any) -> np.ndarray:
        texts = [input_text] if isinstance(input_text, str) else input_text
        for attempt in range(attempts):
            try:
                response = requests.post(
                    model.base_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                    json={"model": model.model_id, "input": texts},
                    timeout=(15.0, 180.0),
                )
                response.raise_for_status()
                payload = response.json()
                return np.array([item["embedding"] for item in payload["data"]])
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if attempt + 1 >= attempts:
                    raise
                time.sleep(backoff * (2**attempt))
        raise RuntimeError("embedding request retry loop terminated unexpectedly")

    embedding_model.call_model = MethodType(call_model, embedding_model)


def _attach_openie_worker_limit(openie_module: Any, max_workers: int) -> None:
    """Cap upstream OpenIE thread pools without modifying official source."""

    worker_limit = int(max_workers)
    if worker_limit < 1:
        raise ValueError("HIPPORAG2_OPENIE_MAX_WORKERS must be at least 1")
    current_executor = openie_module.ThreadPoolExecutor
    base_executor = getattr(
        current_executor,
        "_hipporag2_base_executor",
        current_executor,
    )

    class LimitedThreadPoolExecutor(base_executor):
        _hipporag2_base_executor = base_executor

        def __init__(self, max_workers: int | None = None, *args: Any, **kwargs: Any) -> None:
            requested_workers = worker_limit if max_workers is None else int(max_workers)
            super().__init__(min(requested_workers, worker_limit), *args, **kwargs)

    openie_module.ThreadPoolExecutor = LimitedThreadPoolExecutor


def _repair_unterminated_triple_lines(response: str) -> str:
    """Repair unescaped inner quotes or a missing final quote on triple lines."""

    repaired_lines = []
    for line in response.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        newline = line[len(content) :]
        stripped = content.strip()
        if stripped.startswith("[") and (
            stripped.endswith("],") or stripped.endswith("]")
        ):
            repaired = []
            in_string = False
            escaped = False
            for index, character in enumerate(content):
                if escaped:
                    escaped = False
                    repaired.append(character)
                elif character == "\\" and in_string:
                    escaped = True
                    repaired.append(character)
                elif character != '"':
                    repaired.append(character)
                elif not in_string:
                    in_string = True
                    repaired.append(character)
                else:
                    suffix = content[index + 1 :].lstrip()
                    if suffix.startswith(",") or suffix.startswith("]"):
                        in_string = False
                        repaired.append(character)
                    else:
                        repaired.append(r'\"')
            content = "".join(repaired)
            if in_string:
                closing_bracket = content.rfind("]")
                content = content[:closing_bracket] + '"' + content[closing_bracket:]
        repaired_lines.append(content + newline)
    return "".join(repaired_lines)


def _triples_from_json_response(response: str) -> list[list[str]] | None:
    decoder = json.JSONDecoder()
    for offset, character in enumerate(response or ""):
        if character != "{":
            continue
        candidate = response[offset:]
        repairs = [candidate]
        unicode_repaired = _INVALID_UNICODE_ESCAPE.sub(lambda _: r"\\u", candidate)
        if unicode_repaired != candidate:
            repairs.append(unicode_repaired)
        line_repaired = _repair_unterminated_triple_lines(unicode_repaired)
        if line_repaired not in repairs:
            repairs.append(line_repaired)
        payload = None
        for repaired in repairs:
            try:
                payload, _ = decoder.raw_decode(repaired)
            except json.JSONDecodeError:
                continue
            break
        if payload is None:
            continue
        triples = payload.get("triples") if isinstance(payload, Mapping) else None
        if not isinstance(triples, list):
            continue
        normalized: list[list[str]] = []
        seen: set[tuple[str, str, str]] = set()
        for triple in triples:
            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                continue
            item = tuple(str(value) for value in triple)
            if item not in seen:
                seen.add(item)
                normalized.append(list(item))
        return normalized
    return None


def _attach_json_safe_openie(openie: Any) -> None:
    """Recover valid JSON rejected by upstream eval and fail on real parse loss."""

    original_triple_extraction = openie.triple_extraction
    original_batch_openie = openie.batch_openie

    def triple_extraction(
        model: Any,
        chunk_key: str,
        passage: str,
        named_entities: Sequence[str],
    ) -> Any:
        result = None
        for attempt in range(3):
            result = original_triple_extraction(
                chunk_key,
                passage + ("\n" * attempt),
                list(named_entities),
            )
            metadata = dict(getattr(result, "metadata", {}) or {})
            if not metadata.get("error"):
                return result
            recovered = _triples_from_json_response(str(getattr(result, "response", "") or ""))
            if recovered is not None:
                metadata.pop("error", None)
                metadata["adapter_json_recovered"] = True
                result.metadata = metadata
                result.triples = recovered
                return result
        return result

    def batch_openie(model: Any, chunks: Mapping[str, Any]) -> Any:
        ner_results, triple_results = original_batch_openie(chunks)
        failures = []
        for result in list(ner_results.values()) + list(triple_results.values()):
            metadata = dict(getattr(result, "metadata", {}) or {})
            if metadata.get("error"):
                failures.append(f"{getattr(result, 'chunk_id', 'unknown')}: {metadata['error']}")
        if failures:
            raise RuntimeError("OpenIE parsing failed after two retries: " + "; ".join(failures))
        return ner_results, triple_results

    openie.triple_extraction = MethodType(triple_extraction, openie)
    openie.batch_openie = MethodType(batch_openie, openie)


def _default_factories(settings: Mapping[str, Any], save_dir: str) -> tuple[Any, ChunkFactory]:
    try:
        from hipporag import HippoRAG
        from hipporag.information_extraction import openie_openai
        from hipporag.utils.config_utils import BaseConfig
        from hipporag.utils.misc_utils import Chunk
    except ImportError as exc:
        raise RuntimeError(
            "Official HippoRAG 2 is not importable. Run setup_environment.ps1 with Python 3.10 first."
        ) from exc

    configured_revision = str(settings.get("source_revision") or "")
    if configured_revision != OFFICIAL_REVISION:
        raise RuntimeError(
            f"official source revision mismatch: expected {OFFICIAL_REVISION}, got {configured_revision}"
        )
    installed_version = importlib.metadata.version("hipporag")
    if installed_version != OFFICIAL_PACKAGE_VERSION:
        raise RuntimeError(
            f"official package version mismatch: expected {OFFICIAL_PACKAGE_VERSION}, got {installed_version}"
        )

    hippo = dict(settings.get("hipporag") or {})
    llm_base_url = os.environ.get("HIPPORAG2_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    embedding_base_url = os.environ.get("HIPPORAG2_EMBEDDING_BASE_URL")
    if not llm_base_url:
        raise RuntimeError("HIPPORAG2_LLM_BASE_URL is required")
    if not embedding_base_url:
        raise RuntimeError("HIPPORAG2_EMBEDDING_BASE_URL is required")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required by the official OpenAI-compatible client")

    _attach_openie_worker_limit(
        openie_openai,
        int(os.environ.get("HIPPORAG2_OPENIE_MAX_WORKERS", "4")),
    )

    config = BaseConfig(
        save_dir=save_dir,
        llm_name=str(hippo.get("llm_model_name") or "Qwen/Qwen3-VL-8B-Instruct"),
        llm_base_url=llm_base_url,
        embedding_model_name=str(
            hippo.get("embedding_model_name") or "VLLM/Qwen/Qwen3-Embedding-0.6B"
        ),
        embedding_base_url=embedding_base_url,
        temperature=float(hippo.get("temperature", 0.0)),
        damping=float(hippo.get("damping", 0.5)),
        linking_top_k=int(hippo.get("linking_top_k", 5)),
        passage_node_weight=float(hippo.get("passage_node_weight", 0.05)),
        save_openie=bool(hippo.get("save_openie", True)),
        retrieval_top_k=int(settings.get("retrieval_topk", 20)),
        openie_mode="online",
        max_retry_attempts=2,
    )
    agent = HippoRAG(global_config=config)
    _attach_vllm_bearer_auth(agent.embedding_model, os.environ["OPENAI_API_KEY"])
    _attach_json_safe_openie(agent.openie)
    return agent, Chunk


def _index_with_retries(agent: Any, chunks: Sequence[Any], attempts: int = 3) -> None:
    error: Exception | None = None
    for _ in range(attempts):
        try:
            agent.index(list(chunks))
            return
        except Exception as exc:  # official OpenIE raises several backend-specific errors
            error = exc
    raise RuntimeError(f"HippoRAG 2 indexing failed after {attempts} attempts") from error


def _validate_cached_scope(payload: Mapping[str, Any], scope: Mapping[str, Any], cache_identity: str) -> bool:
    if payload.get("status") != "complete" or payload.get("cache_identity") != cache_identity:
        return False
    expected = [str(query.get("question_id")) for query in scope.get("queries") or []]
    actual = [str(query.get("question_id")) for query in payload.get("queries") or []]
    return expected == actual and len(actual) == len(set(actual))


def _valid_cached_query(
    payload: Mapping[str, Any], question_id: str, known_source_ids: set[str]
) -> bool:
    if str(payload.get("question_id") or "") != question_id:
        return False
    ranked = list(payload.get("ranked_results") or [])
    source_ids = [str(item.get("source_id") or "") for item in ranked]
    return bool(source_ids) and len(source_ids) == len(set(source_ids)) and set(source_ids) <= known_source_ids


def _index_stats(agent: Any, chunks: Sequence[Any], save_dir: Path) -> Dict[str, Any]:
    graph = getattr(agent, "graph", None)
    return {
        "passages": len(chunks),
        "graph_nodes": int(graph.vcount()) if graph is not None and hasattr(graph, "vcount") else None,
        "graph_edges": int(graph.ecount()) if graph is not None and hasattr(graph, "ecount") else None,
        "entities": len(getattr(agent, "ent_node_to_chunk_ids", {}) or {}),
        "facts": len(getattr(agent, "proc_triples_to_docs", {}) or {}),
        "index_bytes": _directory_size(save_dir),
    }


def run_official_scopes(
    prepared_root: str,
    config_path: str,
    output_root: str,
    hipporag_factory: HippoFactory | None = None,
    chunk_factory: ChunkFactory | None = None,
) -> Dict[str, Any]:
    prepared = Path(prepared_root).resolve()
    prepared_manifest = load_json(prepared / "manifest.json")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    config_sha256 = sha256_file(config_path)
    hippo = dict(config.get("hipporag") or {})
    source_revision = str(config.get("source_revision") or "")
    if not source_revision:
        raise ValueError("source_revision is required")
    llm_model = str(hippo.get("llm_model_name") or "Qwen/Qwen3-VL-8B-Instruct")
    embedding_model = str(
        hippo.get("embedding_model_name") or "VLLM/Qwen/Qwen3-Embedding-0.6B"
    )
    run_id = sha256_json(
        {
            "config_sha256": config_sha256,
            "source_revision": source_revision,
            "llm_model": llm_model,
            "embedding_model": embedding_model,
        }
    )[:16]
    output = Path(output_root).resolve()
    scope_entries = []
    total_queries = 0

    for entry in prepared_manifest.get("scopes") or []:
        scope = load_json(prepared / entry["relative_path"])
        cache_identity = build_cache_identity(
            str(scope.get("corpus_sha256") or ""),
            config_sha256,
            source_revision,
            llm_model,
            embedding_model,
        )
        scope_folder = (
            f"{Path(safe_filename(str(scope['scope_id']))).stem}-{cache_identity[:12]}"
        )
        result_path = output / "runs" / run_id / scope_folder / "scope_result.json"
        relative_path = result_path.relative_to(output).as_posix()
        cached: Dict[str, Any] = {}
        if result_path.exists():
            cached = load_json(result_path)
            if _validate_cached_scope(cached, scope, cache_identity):
                scope_entries.append(
                    {
                        "scope_id": scope["scope_id"],
                        "relative_path": relative_path,
                        "cache_identity": cache_identity,
                        "reused": True,
                    }
                )
                total_queries += len(cached.get("queries") or [])
                continue

        save_dir = result_path.parent / "index"
        save_dir.mkdir(parents=True, exist_ok=True)
        if hipporag_factory is None:
            agent, official_chunk_factory = _default_factories(config, str(save_dir))
        else:
            agent = hipporag_factory(config, str(save_dir))
            if chunk_factory is None:
                raise ValueError("chunk_factory is required with a custom hipporag_factory")
            official_chunk_factory = chunk_factory
        chunks = [
            official_chunk_factory(
                str(chunk.get("index_content") or chunk.get("content") or ""),
                str(chunk.get("source_id") or ""),
                dict(chunk.get("metadata") or {}),
            )
            for chunk in scope.get("chunks") or []
        ]
        offline_before = sqlite_token_usage(save_dir)
        index_started = time.perf_counter()
        try:
            _index_with_retries(agent, chunks, attempts=3)
        except Exception as exc:
            failure = {
                "status": "failed",
                "scope_id": scope["scope_id"],
                "cache_identity": cache_identity,
                "error": f"{type(exc).__name__}: {exc}",
            }
            write_json(result_path.parent / "failure.json", failure)
            raise
        index_seconds = time.perf_counter() - index_started
        offline_after = sqlite_token_usage(save_dir)
        stats = _index_stats(agent, chunks, save_dir)
        current_offline_cost = _usage_delta(offline_after, offline_before)
        previous_stats = (
            dict(cached.get("index_stats") or {})
            if cached.get("cache_identity") == cache_identity
            else {}
        )
        previous_offline_cost = dict(previous_stats.get("offline_token_cost") or {})
        combined_offline_cost = {
            key: int(previous_offline_cost.get(key) or 0)
            + int(current_offline_cost.get(key) or 0)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        stats.update(
            {
                "source_revision": source_revision,
                "corpus_sha256": scope.get("corpus_sha256"),
                "config_sha256": config_sha256,
                "index_time": float(previous_stats.get("index_time") or 0.0)
                + index_seconds,
                "offline_token_cost": combined_offline_cost,
            }
        )

        known_source_ids = {
            str(chunk.get("source_id") or "") for chunk in scope.get("chunks") or []
        }
        cached_queries = {
            str(item.get("question_id")): item
            for item in cached.get("queries") or []
            if isinstance(item, Mapping)
        } if cached.get("cache_identity") == cache_identity else {}
        query_outputs = []
        seen_questions: set[str] = set()
        checkpoint_queries = []
        for prepared_query in scope.get("queries") or []:
            prepared_question_id = str(prepared_query.get("question_id") or "")
            checkpoint = cached_queries.get(prepared_question_id)
            if checkpoint is not None and _valid_cached_query(
                checkpoint, prepared_question_id, known_source_ids
            ):
                checkpoint_queries.append(dict(checkpoint))
        write_json(
            result_path,
            {
                "status": "partial",
                "dataset": scope.get("dataset"),
                "scope_id": scope.get("scope_id"),
                "cache_identity": cache_identity,
                "source_revision": source_revision,
                "config_sha256": config_sha256,
                "index_stats": stats,
                "queries": checkpoint_queries,
            },
        )
        for query in scope.get("queries") or []:
            question_id = str(query.get("question_id") or "")
            if not question_id or question_id in seen_questions:
                raise ValueError(f"duplicate or empty question ID in scope result: {question_id!r}")
            seen_questions.add(question_id)
            cached_query = cached_queries.get(question_id)
            if cached_query is not None and _valid_cached_query(
                cached_query, question_id, known_source_ids
            ):
                query_outputs.append(dict(cached_query))
                continue
            before = sqlite_token_usage(save_dir)
            ppr_before = float(getattr(agent, "ppr_time", 0.0) or 0.0)
            rerank_before = float(getattr(agent, "rerank_time", 0.0) or 0.0)
            started = time.perf_counter()
            try:
                solutions = agent.retrieve(
                    [str(query.get("question") or "")],
                    num_to_retrieve=int(config.get("retrieval_topk", 20)),
                )
            except Exception as exc:
                write_json(
                    result_path.parent / "failure.json",
                    {
                        "status": "failed",
                        "scope_id": scope["scope_id"],
                        "question_id": question_id,
                        "cache_identity": cache_identity,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                raise
            retrieval_seconds = time.perf_counter() - started
            after = sqlite_token_usage(save_dir)
            ppr_seconds = max(
                0.0, float(getattr(agent, "ppr_time", ppr_before) or 0.0) - ppr_before
            )
            recognition_seconds = max(
                0.0,
                float(getattr(agent, "rerank_time", rerank_before) or 0.0)
                - rerank_before,
            )
            if len(solutions) != 1:
                raise RuntimeError(f"expected one HippoRAG solution, got {len(solutions)}")
            solution = solutions[0]
            docs = list(getattr(solution, "docs", []) or [])
            raw_scores = getattr(solution, "doc_scores", None)
            scores = _jsonable(raw_scores if raw_scores is not None else [])
            metadatas = list(getattr(solution, "doc_metadata", []) or [])
            graph_seeds = _jsonable(getattr(solution, "graph_seeds", []) or [])
            ranked = normalize_ranked_results(scope, docs, scores, metadatas, graph_seeds)
            dense_fallback = len(graph_seeds) == 0
            query_outputs.append(
                {
                    "question_id": question_id,
                    "question": str(query.get("question") or ""),
                    "ranked_results": ranked,
                    "graph_seeds": graph_seeds,
                    "retrieval_time": retrieval_seconds,
                    "ppr_time": ppr_seconds,
                    "recognition_memory_time": recognition_seconds,
                    "retrieval_token_cost": _usage_delta(after, before),
                    "dense_fallback": dense_fallback,
                    "dense_fallback_reason": "no_graph_facts" if dense_fallback else None,
                }
            )
            write_json(
                result_path,
                {
                    "status": "partial",
                    "dataset": scope.get("dataset"),
                    "scope_id": scope.get("scope_id"),
                    "cache_identity": cache_identity,
                    "source_revision": source_revision,
                    "config_sha256": config_sha256,
                    "index_stats": stats,
                    "queries": query_outputs,
                },
            )

        payload = {
            "status": "complete",
            "dataset": scope.get("dataset"),
            "scope_id": scope.get("scope_id"),
            "cache_identity": cache_identity,
            "source_revision": source_revision,
            "config_sha256": config_sha256,
            "index_stats": stats,
            "queries": query_outputs,
        }
        write_json(result_path, payload)
        write_json(result_path.parent / "hipporag2_index_stats.json", stats)
        scope_entries.append(
            {
                "scope_id": scope["scope_id"],
                "relative_path": relative_path,
                "cache_identity": cache_identity,
                "reused": False,
            }
        )
        total_queries += len(query_outputs)

    manifest = {
        "format_version": 1,
        "dataset": prepared_manifest.get("dataset"),
        "run_id": run_id,
        "source_revision": source_revision,
        "config_sha256": config_sha256,
        "prepared_manifest_path": str(prepared / "manifest.json"),
        "scope_count": len(scope_entries),
        "query_count": total_queries,
        "scopes": scope_entries,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the official HippoRAG 2 index and retriever")
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    manifest = run_official_scopes(args.prepared_root, args.config, args.output_root)
    print(f"Completed {manifest['scope_count']} scopes and {manifest['query_count']} queries")


if __name__ == "__main__":
    main()
