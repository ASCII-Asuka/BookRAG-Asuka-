from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .common import normalize_entity, write_json


OPENIE_INSTRUCTION = (
    "Extract informative factual triples stated directly in the supplied text. "
    "Return JSON only in the form "
    '{"triples": [["head", "relation", "tail"]]}. '
    "Do not infer facts not stated in the text and do not use outside knowledge."
)

OPENIE_RECOVERY_INSTRUCTION = (
    "Extract at most 12 informative factual triples stated directly in the "
    "supplied text. Do not copy LaTeX or mathematical notation. Each head, "
    "relation, and tail must be a concise plain-text string of at most 160 "
    "characters. Return JSON only in the form "
    '{"triples": [["head", "relation", "tail"]]}. '
    "Do not infer facts not stated in the text and do not use outside knowledge."
)


def _compact_math(text: str) -> str:
    patterns = (
        r"\$\$.*?\$\$",
        r"\\\[.*?\\\]",
        r"\\\(.*?\\\)",
        r"\$.*?\$",
    )
    compacted = text
    for pattern in patterns:
        compacted = re.sub(
            pattern,
            " [MATHEMATICAL NOTATION] ",
            compacted,
            flags=re.DOTALL,
        )
    return re.sub(r"[ \t]+", " ", compacted)


def openie_input(
    chunk: Mapping[str, Any], *, recovery: bool = False
) -> list[dict[str, str]]:
    content = str(chunk.get("content") or "")
    return [
        {
            "role": "system",
            "content": (
                OPENIE_RECOVERY_INSTRUCTION if recovery else OPENIE_INSTRUCTION
            ),
        },
        {
            "role": "user",
            "content": _compact_math(content) if recovery else content,
        },
    ]


def parse_triples(payload: Mapping[str, Any]) -> list[list[str]]:
    triples: list[list[str]] = []
    for raw in payload.get("triples") or []:
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            continue
        if not all(isinstance(value, str) and value.strip() for value in raw):
            continue
        head, relation, tail = (value.strip() for value in raw)
        if not normalize_entity(head) or not normalize_entity(tail):
            continue
        triples.append([head, relation, tail])
    return triples


def _add_usage(total: dict[str, int], usage: Mapping[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[key] += int(usage.get(key) or 0)


def _embed_chunks(
    chunks: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    clients: Any,
) -> tuple[np.ndarray, dict[str, int]]:
    embedding_config = dict(config.get("embedding") or {})
    model = str(embedding_config.get("model_name") or "")
    batch_size = max(1, int(embedding_config.get("batch_size") or 16))
    vectors: list[np.ndarray] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        matrix, usage = clients.embed(
            model, [str(chunk.get("content") or "") for chunk in batch]
        )
        if len(matrix) != len(batch):
            raise ValueError("embedding batch does not match chunk batch")
        vectors.append(np.asarray(matrix, dtype=np.float32))
        _add_usage(usage_total, usage)
    if not vectors:
        raise ValueError("cannot index a scope with no chunks")
    return np.vstack(vectors), usage_total


def index_scope(
    scope: Mapping[str, Any],
    config: Mapping[str, Any],
    clients: Any,
    output_dir: str | Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    chunks = list(scope.get("chunks") or [])
    source_ids = [str(chunk.get("source_id") or "") for chunk in chunks]
    if not source_ids or any(not source_id for source_id in source_ids):
        raise ValueError("scope has an empty source_id")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("scope has duplicate source_id values")

    openie_config = dict(config.get("openie") or {})
    model = str(openie_config.get("model_name") or "")
    temperature = float(openie_config.get("temperature") or 0.0)
    max_tokens = int(openie_config.get("max_output_tokens") or 1024)
    max_attempts = max(1, int(openie_config.get("max_attempts") or 3))
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    triples: list[dict[str, Any]] = []
    recovered_chunks = 0

    for chunk in chunks:
        source_id = str(chunk["source_id"])
        last_error: Exception | None = None
        parsed: list[list[str]] | None = None
        for attempt in range(max_attempts):
            try:
                payload, usage = clients.chat_json(
                    model,
                    openie_input(chunk, recovery=attempt > 0),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                _add_usage(usage_total, usage)
                parsed = parse_triples(payload)
                if attempt > 0:
                    recovered_chunks += 1
                break
            except (ValueError, RuntimeError) as error:
                last_error = error
        if parsed is None:
            raise RuntimeError(f"OpenIE failed for source_id {source_id!r}") from last_error
        for head, relation, tail in parsed:
            triples.append(
                {
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                    "head_key": normalize_entity(head),
                    "tail_key": normalize_entity(tail),
                    "source_id": source_id,
                }
            )

    embeddings, embedding_usage = _embed_chunks(chunks, config, clients)
    _add_usage(usage_total, embedding_usage)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    vectors_file = "chunk_embeddings.npy"
    np.save(output / vectors_file, embeddings, allow_pickle=False)

    entities = {
        entity
        for triple in triples
        for entity in (triple["head_key"], triple["tail_key"])
        if entity
    }
    result = {
        "dataset": str(scope.get("dataset") or ""),
        "scope_id": str(scope.get("scope_id") or ""),
        "corpus_sha256": str(scope.get("corpus_sha256") or ""),
        "chunks": chunks,
        "triples": triples,
        "chunk_embeddings": {
            "source_ids": source_ids,
            "vectors_file": vectors_file,
        },
        "index_stats": {
            "chunks": len(chunks),
            "entities": len(entities),
            "triples": len(triples),
            "offline_prompt_tokens": usage_total["prompt_tokens"],
            "offline_completion_tokens": usage_total["completion_tokens"],
            "offline_tokens": usage_total["total_tokens"],
            "offline_seconds": time.perf_counter() - started,
            "openie_recovered_chunks": recovered_chunks,
        },
    }
    write_json(output / "index.json", result)
    return result
