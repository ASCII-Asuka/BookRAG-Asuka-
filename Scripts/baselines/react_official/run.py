from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from .common import (
    adapter_code_sha256,
    build_run_identity,
    load_json,
    safe_filename,
    scope_corpus_sha256,
    sha256_file,
    sha256_json,
    write_json,
)
from .environment import HotpotEnvironment, QasperEnvironment
from .model_client import CompletionClient
from .prompts import get_react_examples
from .runner import ReActRunner


SOURCE_LOCK_PATH = Path(__file__).with_name("official-source-lock.json")


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("ReAct reproduction config must be a mapping")
    return dict(payload)


def _validate_official_source(source_root: Path) -> dict[str, Any]:
    lock = load_json(SOURCE_LOCK_PATH)
    entrypoint = source_root / str(lock["official_entrypoint"])
    if not entrypoint.exists():
        raise FileNotFoundError(f"locked official ReAct entrypoint is missing: {entrypoint}")
    actual = sha256_file(entrypoint).lower()
    expected = str(lock["official_entrypoint_sha256"]).lower()
    if actual != expected:
        raise RuntimeError("official ReAct entrypoint hash does not match source lock")
    return dict(lock)


def _make_environment(dataset: str, chunks: list[dict[str, Any]], limit: int):
    if dataset == "qasper":
        return QasperEnvironment.from_chunks(
            chunks, max_observation_characters=limit
        )
    if dataset == "hotpotqa":
        return HotpotEnvironment.from_chunks(
            chunks, max_observation_characters=limit
        )
    raise ValueError(f"unsupported ReAct dataset: {dataset!r}")


def _query_is_complete(
    payload: Mapping[str, Any], question_id: str, cache_identity: str
) -> bool:
    return (
        str(payload.get("question_id") or "") == question_id
        and payload.get("cache_identity") == cache_identity
        and "answer" in payload
        and bool(str(payload.get("trajectory") or "").strip())
        and isinstance(payload.get("steps"), list)
        and isinstance(payload.get("usage"), Mapping)
        and "elapsed_seconds" in payload
    )


def _scope_result_payload(
    dataset: str,
    scope_id: str,
    corpus_sha256: str,
    cache_identity: str,
    queries: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "dataset": dataset,
        "scope_id": scope_id,
        "corpus_sha256": corpus_sha256,
        "cache_identity": cache_identity,
        "query_count": len(queries),
        "queries": queries,
    }


def _default_client(config: Mapping[str, Any]) -> CompletionClient:
    react = dict(config.get("react") or {})
    base_url = (
        os.getenv("REACT_LLM_BASE_URL")
        or os.getenv("KG2RAG_LLM_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or ""
    )
    return CompletionClient(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=base_url,
        timeout=float(react.get("request_timeout_seconds") or 180),
        max_attempts=int(react.get("max_attempts") or 3),
        retry_backoff_seconds=float(react.get("retry_backoff_seconds") or 2),
    )


def run_prepared(
    prepared_root: str | Path,
    config_path: str | Path,
    output_root: str | Path,
    *,
    completion_model: Any | None = None,
    source_root: str | Path = "runs/third_party/ReAct",
    max_scopes: int | None = None,
) -> dict[str, Any]:
    prepared = Path(prepared_root).resolve()
    manifest_path = prepared / "manifest.json"
    prepared_manifest = load_json(manifest_path)
    config_file = Path(config_path).resolve()
    config = _load_config(config_file)
    official_source = Path(source_root).resolve()
    lock = _validate_official_source(official_source)
    dataset = str(prepared_manifest.get("dataset") or "").lower()
    if dataset not in {"qasper", "hotpotqa"}:
        raise ValueError(f"unsupported prepared dataset: {dataset!r}")
    react = dict(config.get("react") or {})
    model_name = str(react.get("model_name") or "Qwen/Qwen3-VL-8B-Instruct")
    examples = get_react_examples(dataset, official_source)
    adapter_root = Path(__file__).resolve().parent
    adapter_sha256 = adapter_code_sha256(
        [
            adapter_root / name
            for name in (
                "common.py",
                "environment.py",
                "model_client.py",
                "prompts.py",
                "qasper_train_examples.txt",
                "runner.py",
                "run.py",
            )
        ]
    )
    global_identity = build_run_identity(
        sha256_file(manifest_path),
        sha256_json(
            {
                "config_sha256": sha256_file(config_file),
                "examples_sha256": sha256_json(examples),
                "adapter_sha256": adapter_sha256,
            }
        ),
        str(lock["commit"]),
        model_name,
    )
    run_id = global_identity[:20]
    output = Path(output_root).resolve()
    run_directory = output / "runs" / run_id
    scopes_directory = run_directory / "scopes"
    scopes_directory.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    write_json(
        status_path,
        {
            "status": "running",
            "stage": "run_official",
            "dataset": dataset,
            "run_id": run_id,
        },
    )

    scope_entries = list(prepared_manifest.get("scopes") or [])
    if max_scopes is not None:
        scope_entries = scope_entries[: max(0, int(max_scopes))]
    loaded_scopes: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen_scope_ids: set[str] = set()
    seen_question_ids: set[str] = set()
    for entry in scope_entries:
        scope_id = str(entry.get("scope_id") or "")
        if not scope_id or scope_id in seen_scope_ids:
            raise ValueError(f"duplicate or empty prepared scope ID: {scope_id!r}")
        seen_scope_ids.add(scope_id)
        scope = load_json(prepared / str(entry["relative_path"]))
        if str(scope.get("scope_id") or "") != scope_id:
            raise ValueError(f"prepared scope ID mismatch for {scope_id!r}")
        actual_corpus_sha = scope_corpus_sha256(scope.get("chunks") or [])
        expected_corpus_sha = str(scope.get("corpus_sha256") or "")
        if actual_corpus_sha != expected_corpus_sha:
            raise ValueError(f"prepared corpus hash mismatch for scope {scope_id!r}")
        for query in scope.get("queries") or []:
            question_id = str(query.get("question_id") or "")
            if not question_id or question_id in seen_question_ids:
                raise ValueError(
                    f"duplicate or empty prepared question ID: {question_id!r}"
                )
            seen_question_ids.add(question_id)
        loaded_scopes.append((dict(entry), dict(scope)))

    runner = ReActRunner(
        completion_model or _default_client(config),
        model_name=model_name,
        max_steps=int(react.get("max_steps") or 7),
        temperature=float(react.get("temperature", 0.0)),
        max_tokens=int(react.get("max_output_tokens") or 100),
    )
    observation_limit = int(
        dict(config.get("environment") or {}).get("max_observation_characters")
        or 4000
    )
    manifest_scopes: list[dict[str, Any]] = []
    cached_question_count = 0
    completed_question_count = 0

    try:
        for entry, scope in loaded_scopes:
            scope_id = str(scope["scope_id"])
            corpus_sha = str(scope["corpus_sha256"])
            cache_identity = build_run_identity(
                corpus_sha,
                global_identity,
                str(lock["commit"]),
                model_name,
            )
            scope_result_path = scopes_directory / safe_filename(scope_id)
            cached_queries: dict[str, dict[str, Any]] = {}
            if scope_result_path.exists():
                cached_scope = load_json(scope_result_path)
                if cached_scope.get("cache_identity") == cache_identity:
                    for query_payload in cached_scope.get("queries") or []:
                        question_id = str(query_payload.get("question_id") or "")
                        if _query_is_complete(
                            query_payload, question_id, cache_identity
                        ):
                            cached_queries[question_id] = dict(query_payload)

            raw_queries: list[dict[str, Any]] = []
            for query in scope.get("queries") or []:
                question_id = str(query["question_id"])
                cached = cached_queries.get(question_id)
                if cached is not None:
                    raw_queries.append(cached)
                    cached_question_count += 1
                    completed_question_count += 1
                    continue
                environment = _make_environment(
                    dataset, list(scope.get("chunks") or []), observation_limit
                )
                result = runner.run(
                    str(query.get("question") or ""),
                    environment,
                    examples=examples,
                )
                raw_query = {
                    "question_id": question_id,
                    "question": str(query.get("question") or ""),
                    "cache_identity": cache_identity,
                    **result.to_dict(),
                }
                raw_queries.append(raw_query)
                completed_question_count += 1
                write_json(
                    scope_result_path,
                    _scope_result_payload(
                        dataset,
                        scope_id,
                        corpus_sha,
                        cache_identity,
                        raw_queries,
                        "running",
                    ),
                )

            write_json(
                scope_result_path,
                _scope_result_payload(
                    dataset,
                    scope_id,
                    corpus_sha,
                    cache_identity,
                    raw_queries,
                    "complete",
                ),
            )
            manifest_scopes.append(
                {
                    "scope_id": scope_id,
                    "relative_path": str(scope_result_path.relative_to(run_directory)),
                    "query_count": len(raw_queries),
                    "cache_identity": cache_identity,
                }
            )

        expected_query_count = sum(
            int(entry.get("query_count") or len(scope.get("queries") or []))
            for entry, scope in loaded_scopes
        )
        if completed_question_count != expected_query_count:
            raise RuntimeError(
                "prediction coverage mismatch: "
                f"expected {expected_query_count}, got {completed_question_count}"
            )
        raw_manifest_path = run_directory / "manifest.json"
        raw_manifest = {
            "status": "complete",
            "dataset": dataset,
            "run_id": run_id,
            "run_identity": global_identity,
            "method_suffix": str(
                config.get("method_suffix") or "react_official_repro"
            ),
            "source_revision": str(lock["commit"]),
            "adapter_sha256": adapter_sha256,
            "model_name": model_name,
            "prepared_manifest_path": str(manifest_path),
            "run_directory": str(run_directory),
            "scope_count": len(manifest_scopes),
            "query_count": completed_question_count,
            "scopes": manifest_scopes,
        }
        write_json(raw_manifest_path, raw_manifest)
        summary = {
            "status": "complete",
            "dataset": dataset,
            "run_id": run_id,
            "run_directory": str(run_directory),
            "manifest_path": str(raw_manifest_path),
            "scope_count": len(manifest_scopes),
            "question_count": completed_question_count,
            "cached_question_count": cached_question_count,
        }
        write_json(status_path, {**summary, "stage": "run_official"})
        return summary
    except Exception as error:
        write_json(
            status_path,
            {
                "status": "failed",
                "stage": "run_official",
                "dataset": dataset,
                "run_id": run_id,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official-control-flow ReAct")
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-root", default="runs/third_party/ReAct")
    parser.add_argument("--max-scopes", type=int)
    args = parser.parse_args()
    summary = run_prepared(
        args.prepared_root,
        args.config,
        args.output_root,
        source_root=args.source_root,
        max_scopes=args.max_scopes,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
