from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import yaml

from .common import (
    build_cache_identity,
    load_json,
    safe_filename,
    sha256_json,
    write_json,
)
from .index import index_scope
from .model_clients import OpenAICompatibleClient
from .retrieval import retrieve


def _load_config(
    config_path: str | Path, override: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    value = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("KG2RAG config must be a YAML object")
    config = dict(value)
    if override:
        config.update(dict(override))
    return config


def _status(
    output_root: Path,
    *,
    status: str,
    stage: str,
    completed_scopes: int,
    total_scopes: int,
    completed_queries: int,
    total_queries: int,
    last_error: str | None = None,
) -> None:
    write_json(
        output_root / "status.json",
        {
            "status": status,
            "stage": stage,
            "completed_scopes": completed_scopes,
            "total_scopes": total_scopes,
            "completed_queries": completed_queries,
            "total_queries": total_queries,
            "last_progress_at": time.time(),
            "last_error": last_error,
        },
    )


def _index_is_valid(
    payload: Mapping[str, Any],
    scope: Mapping[str, Any],
    cache_identity: str,
) -> bool:
    return (
        str(payload.get("scope_id") or "") == str(scope.get("scope_id") or "")
        and str(payload.get("corpus_sha256") or "")
        == str(scope.get("corpus_sha256") or "")
        and str(payload.get("cache_identity") or "") == cache_identity
        and [str(chunk.get("source_id") or "") for chunk in payload.get("chunks") or []]
        == [str(chunk.get("source_id") or "") for chunk in scope.get("chunks") or []]
    )


def _scope_result_is_complete(
    payload: Mapping[str, Any],
    scope: Mapping[str, Any],
    cache_identity: str,
) -> bool:
    expected = [str(query.get("question_id") or "") for query in scope.get("queries") or []]
    actual = [str(query.get("question_id") or "") for query in payload.get("queries") or []]
    return (
        str(payload.get("cache_identity") or "") == cache_identity
        and expected == actual
        and all(query.get("ranked_results") is not None for query in payload.get("queries") or [])
    )


def run_scopes(
    prepared_root: str | Path,
    config_path: str | Path,
    output_root: str | Path,
    *,
    clients: Any | None = None,
    config_override: Mapping[str, Any] | None = None,
    stage: str = "all",
    limit_scopes: int | None = None,
) -> dict[str, Any]:
    if stage not in {"index", "retrieve", "all"}:
        raise ValueError(f"unsupported KG2RAG stage: {stage!r}")
    prepared = Path(prepared_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared_manifest = load_json(prepared / "manifest.json")
    config = _load_config(config_path, config_override)
    config_sha = sha256_json(config)
    run_id = sha256_json(
        {
            "prepared_manifest": prepared_manifest,
            "config_sha256": config_sha,
            "source_revision": config.get("source_revision"),
        }
    )[:20]
    run_directory = output / "runs" / run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    scope_entries = list(prepared_manifest.get("scopes") or [])
    if limit_scopes is not None:
        scope_entries = scope_entries[: max(0, int(limit_scopes))]
    total_queries = sum(int(entry.get("query_count") or 0) for entry in scope_entries)
    if not total_queries:
        total_queries = sum(
            len(load_json(prepared / entry["relative_path"]).get("queries") or [])
            for entry in scope_entries
        )
    completed_scopes = 0
    completed_queries = 0
    results: list[dict[str, Any]] = []
    model_clients = clients or OpenAICompatibleClient.from_environment()
    _status(
        output,
        status="running",
        stage=stage,
        completed_scopes=0,
        total_scopes=len(scope_entries),
        completed_queries=0,
        total_queries=total_queries,
    )

    try:
        for entry in scope_entries:
            scope = load_json(prepared / entry["relative_path"])
            scope_id = str(scope.get("scope_id") or "")
            cache_identity = build_cache_identity(
                str(scope.get("corpus_sha256") or ""),
                config_sha,
                str(config.get("source_revision") or ""),
                str(dict(config.get("openie") or {}).get("model_name") or ""),
                str(dict(config.get("embedding") or {}).get("model_name") or ""),
                str(dict(config.get("reranker") or {}).get("model_name") or ""),
            )
            scope_folder = Path(safe_filename(scope_id).removesuffix(".json"))
            scope_directory = run_directory / "scopes" / scope_folder
            index_directory = scope_directory / "index"
            index_path = index_directory / "index.json"
            scope_result_path = scope_directory / "scope_result.json"
            reused = False

            if scope_result_path.exists() and stage != "index":
                cached_result = load_json(scope_result_path)
                if _scope_result_is_complete(cached_result, scope, cache_identity):
                    reused = True
                    completed_scopes += 1
                    completed_queries += len(cached_result.get("queries") or [])
                    results.append(
                        {
                            "scope_id": scope_id,
                            "relative_path": scope_result_path.relative_to(
                                run_directory
                            ).as_posix(),
                            "cache_identity": cache_identity,
                            "reused": True,
                            "query_count": len(cached_result.get("queries") or []),
                        }
                    )
                    _status(
                        output,
                        status="running",
                        stage=stage,
                        completed_scopes=completed_scopes,
                        total_scopes=len(scope_entries),
                        completed_queries=completed_queries,
                        total_queries=total_queries,
                    )
                    continue

            if index_path.exists():
                scope_index = load_json(index_path)
                if not _index_is_valid(scope_index, scope, cache_identity):
                    scope_index = index_scope(
                        scope, config, model_clients, index_directory
                    )
            elif stage == "retrieve":
                raise FileNotFoundError(
                    f"missing KG2RAG index for scope {scope_id!r}: {index_path}"
                )
            else:
                scope_index = index_scope(scope, config, model_clients, index_directory)

            scope_index["cache_identity"] = cache_identity
            write_json(index_path, scope_index)
            scope_index["embedding_path"] = str(
                index_directory
                / str(scope_index["chunk_embeddings"]["vectors_file"])
            )

            query_results: list[dict[str, Any]] = []
            if stage != "index":
                for query in scope.get("queries") or []:
                    retrieval = retrieve(
                        str(query.get("question") or ""),
                        scope_index,
                        model_clients,
                        embedding_model=str(
                            dict(config.get("embedding") or {}).get("model_name") or ""
                        ),
                        reranker_model=str(
                            dict(config.get("reranker") or {}).get("model_name") or ""
                        ),
                        seed_topk=int(config.get("seed_topk") or 10),
                        retrieval_topk=int(config.get("retrieval_topk") or 20),
                        expansion_hops=int(config.get("expansion_hops") or 1),
                    )
                    query_results.append({**dict(query), **retrieval})
                    completed_queries += 1
                    write_json(
                        scope_result_path,
                        {
                            "dataset": scope.get("dataset"),
                            "scope_id": scope_id,
                            "cache_identity": cache_identity,
                            "index_stats": scope_index.get("index_stats") or {},
                            "queries": query_results,
                        },
                    )

            if stage == "index":
                write_json(
                    scope_result_path,
                    {
                        "dataset": scope.get("dataset"),
                        "scope_id": scope_id,
                        "cache_identity": cache_identity,
                        "index_stats": scope_index.get("index_stats") or {},
                        "queries": [],
                        "index_only": True,
                    },
                )
            completed_scopes += 1
            results.append(
                {
                    "scope_id": scope_id,
                    "relative_path": scope_result_path.relative_to(
                        run_directory
                    ).as_posix(),
                    "cache_identity": cache_identity,
                    "reused": reused,
                    "query_count": len(query_results),
                }
            )
            _status(
                output,
                status="running",
                stage=stage,
                completed_scopes=completed_scopes,
                total_scopes=len(scope_entries),
                completed_queries=completed_queries,
                total_queries=total_queries,
            )

        manifest = {
            "status": "complete",
            "run_id": run_id,
            "dataset": prepared_manifest.get("dataset"),
            "working_dir": prepared_manifest.get("working_dir"),
            "prepared_manifest_path": str(prepared / "manifest.json"),
            "config_path": str(Path(config_path).resolve()),
            "config_sha256": config_sha,
            "run_directory": str(run_directory),
            "scope_count": len(results),
            "query_count": completed_queries,
            "scopes": results,
        }
        write_json(run_directory / "manifest.json", manifest)
        write_json(output / "manifest.json", manifest)
        _status(
            output,
            status="complete",
            stage=stage,
            completed_scopes=completed_scopes,
            total_scopes=len(scope_entries),
            completed_queries=completed_queries,
            total_queries=total_queries,
        )
        return manifest
    except Exception as error:
        _status(
            output,
            status="failed",
            stage=stage,
            completed_scopes=completed_scopes,
            total_scopes=len(scope_entries),
            completed_queries=completed_queries,
            total_queries=total_queries,
            last_error=f"{type(error).__name__}: {error}",
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Index and retrieve with KG2RAG")
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stage", choices=["index", "retrieve", "all"], default="all")
    parser.add_argument("--limit-scopes", type=int)
    args = parser.parse_args()
    manifest = run_scopes(
        args.prepared_root,
        args.config,
        args.output_root,
        stage=args.stage,
        limit_scopes=args.limit_scopes,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
