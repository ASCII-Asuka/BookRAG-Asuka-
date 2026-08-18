from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def safe_filename(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._") or "scope"
    suffix = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:10]
    return f"{stem[:80]}-{suffix}.json"


def scope_corpus_sha256(chunks: Sequence[Mapping[str, Any]]) -> str:
    material = [
        {
            "source_id": str(chunk.get("source_id", "")),
            "content": str(chunk.get("content", "")),
            "index_content": str(chunk.get("index_content") or chunk.get("content", "")),
            "metadata": dict(chunk.get("metadata") or {}),
        }
        for chunk in chunks
    ]
    return sha256_json(material)


def build_cache_identity(
    corpus_sha256: str,
    config_sha256: str,
    source_revision: str,
    llm_model_name: str,
    embedding_model_name: str,
) -> str:
    return sha256_json(
        {
            "corpus_sha256": corpus_sha256,
            "config_sha256": config_sha256,
            "source_revision": source_revision,
            "llm_model_name": llm_model_name,
            "embedding_model_name": embedding_model_name,
        }
    )


def normalize_ranked_results(
    scope: Mapping[str, Any],
    docs: Sequence[str],
    scores: Sequence[float],
    metadatas: Sequence[Mapping[str, Any]],
    graph_seeds: Sequence[Any] | None,
) -> List[Dict[str, Any]]:
    if not (len(docs) == len(scores) == len(metadatas)):
        raise ValueError("HippoRAG result lengths do not match")
    chunks = list(scope.get("chunks") or [])
    by_source: Dict[str, Mapping[str, Any]] = {}
    for chunk in chunks:
        source_id = str(chunk.get("source_id") or "")
        if not source_id or source_id in by_source:
            raise ValueError(f"duplicate or empty source_id in scope: {source_id!r}")
        by_source[source_id] = chunk

    ranked: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for rank, (returned_text, score, returned_metadata) in enumerate(
        zip(docs, scores, metadatas), start=1
    ):
        metadata = dict(returned_metadata or {})
        source_id = str(metadata.get("source_id") or "")
        if source_id not in by_source:
            raise ValueError(f"HippoRAG returned unknown source_id: {source_id!r}")
        if source_id in seen:
            raise ValueError(f"HippoRAG returned duplicate source_id: {source_id!r}")
        seen.add(source_id)
        canonical = by_source[source_id]
        canonical_metadata = dict(canonical.get("metadata") or {})
        canonical_metadata.update(metadata)
        canonical_metadata["source_id"] = source_id
        ranked.append(
            {
                "rank": rank,
                "score": float(score),
                "source_id": source_id,
                "content": str(canonical.get("content") or returned_text or ""),
                "original_text": str(
                    canonical_metadata.get("qasper_evidence_text")
                    or canonical.get("content")
                    or returned_text
                    or ""
                ),
                "qasper_evidence_text": canonical_metadata.get("qasper_evidence_text"),
                "paragraph_id": canonical_metadata.get("paragraph_id"),
                "hotpot_title": canonical_metadata.get("hotpot_title"),
                "metadata": canonical_metadata,
            }
        )
    return ranked


def apply_generation_budget(
    ranked_results: Sequence[Mapping[str, Any]],
    max_blocks: int,
    max_tokens: int,
    token_counter: Callable[[str], int],
) -> List[Dict[str, Any]]:
    if max_blocks <= 0 or max_tokens <= 0:
        return []
    selected: List[Dict[str, Any]] = []
    used_tokens = 0
    for item in ranked_results:
        if len(selected) >= max_blocks:
            break
        item_tokens = max(1, int(token_counter(str(item.get("content") or ""))))
        if used_tokens + item_tokens > max_tokens:
            break
        selected.append(dict(item))
        used_tokens += item_tokens
    return selected


def _qasper_evidence_item(item: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = dict(item.get("metadata") or {})
    paragraph_id = metadata.get("paragraph_id")
    if paragraph_id is None:
        paragraph_id = metadata.get("source_node_id")
    if paragraph_id is None:
        paragraph_id = item.get("source_id")
    text = str(metadata.get("qasper_evidence_text") or item.get("content") or "")
    return {
        "source_id": str(item.get("source_id") or ""),
        "paragraph_id": paragraph_id,
        "source_node_id": paragraph_id,
        "node_id": paragraph_id,
        "evidence_id": paragraph_id,
        "qasper_evidence_text": text,
        "content": text,
        "rank": item.get("rank"),
        "score": item.get("score"),
        "section": metadata.get("section") or "",
    }


def validate_qasper_citations(
    requested_ids: Iterable[Any],
    ranked_results: Sequence[Mapping[str, Any]],
    max_support: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    aliases: Dict[str, Mapping[str, Any]] = {}
    for item in ranked_results:
        metadata = dict(item.get("metadata") or {})
        values = [
            item.get("source_id"),
            metadata.get("paragraph_id"),
            metadata.get("source_node_id"),
        ]
        for value in values:
            if value is not None:
                aliases[str(value)] = item
    accepted: List[Mapping[str, Any]] = []
    rejected: List[str] = []
    seen_sources: set[str] = set()
    for raw_id in requested_ids or []:
        key = str(raw_id)
        item = aliases.get(key)
        source_id = str(item.get("source_id")) if item else ""
        if item is None or source_id in seen_sources:
            rejected.append(key)
            continue
        seen_sources.add(source_id)
        accepted.append(item)
        if len(accepted) >= max_support:
            break
    fallback_used = not accepted
    if fallback_used:
        accepted = list(ranked_results[:max_support])
    return (
        [_qasper_evidence_item(item) for item in accepted],
        {
            "requested": [str(value) for value in requested_ids or []],
            "accepted_source_ids": [str(item.get("source_id")) for item in accepted],
            "rejected": rejected,
            "fallback_used": fallback_used,
            "fallback_policy": "top-ranked-valid-paragraph" if fallback_used else None,
        },
    )


def _hotpot_legal_facts(
    ranked_results: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], List[Dict[str, Any]]]:
    by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    ordered: List[Dict[str, Any]] = []
    for item in ranked_results:
        metadata = dict(item.get("metadata") or {})
        title = str(metadata.get("hotpot_title") or metadata.get("title") or "")
        for sentence in metadata.get("hotpot_sentences") or []:
            if not isinstance(sentence, Mapping) or not title:
                continue
            try:
                sent_id = int(sentence.get("sent_id"))
            except (TypeError, ValueError):
                continue
            payload = {
                "hotpot_title": title,
                "hotpot_sent_id": sent_id,
                "text": str(sentence.get("text") or ""),
            }
            key = (title, sent_id)
            if key not in by_key:
                by_key[key] = payload
                ordered.append(payload)
    return by_key, ordered


def validate_hotpot_citations(
    requested_facts: Iterable[Any],
    ranked_results: Sequence[Mapping[str, Any]],
    max_support: int,
    question: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_key, ordered = _hotpot_legal_facts(ranked_results)
    if not ordered:
        raise ValueError("no legal HotpotQA title-sentence pair in retrieved passages")
    accepted: List[Dict[str, Any]] = []
    rejected: List[Any] = []
    seen: set[Tuple[str, int]] = set()
    normalized_requested: List[Dict[str, Any]] = []
    for fact in requested_facts or []:
        if isinstance(fact, Mapping):
            title = str(fact.get("title") or fact.get("hotpot_title") or "")
            raw_sent_id = fact.get("sent_id", fact.get("hotpot_sent_id"))
        elif isinstance(fact, (list, tuple)) and len(fact) >= 2:
            title, raw_sent_id = str(fact[0]), fact[1]
        else:
            rejected.append(fact)
            continue
        try:
            sent_id = int(raw_sent_id)
        except (TypeError, ValueError):
            rejected.append(fact)
            continue
        normalized_requested.append({"title": title, "sent_id": sent_id})
        key = (title, sent_id)
        if key not in by_key or key in seen:
            rejected.append({"title": title, "sent_id": sent_id})
            continue
        seen.add(key)
        accepted.append(by_key[key])
        if len(accepted) >= max_support:
            break
    fallback_used = not accepted
    if fallback_used:
        query_terms = set(re.findall(r"[A-Za-z0-9]+", question.lower()))
        if query_terms:
            position = {
                (item["hotpot_title"], item["hotpot_sent_id"]): index
                for index, item in enumerate(ordered)
            }
            accepted = sorted(
                ordered,
                key=lambda item: (
                    -len(
                        query_terms
                        & set(re.findall(r"[A-Za-z0-9]+", item["text"].lower()))
                    ),
                    position[(item["hotpot_title"], item["hotpot_sent_id"])],
                ),
            )[:max_support]
        else:
            accepted = ordered[:max_support]
    return (
        accepted,
        {
            "requested": normalized_requested,
            "accepted": [
                [item["hotpot_title"], item["hotpot_sent_id"]] for item in accepted
            ],
            "rejected": rejected,
            "fallback_used": fallback_used,
            "fallback_policy": "first-legal-sentence-from-top-ranked-passages" if fallback_used else None,
        },
    )


def _normalized_usage(usage: Mapping[str, Any] | None) -> Dict[str, int]:
    usage = dict(usage or {})
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def combine_online_cost(
    retrieval_usage: Mapping[str, Any] | None,
    generation_usage: Mapping[str, Any] | None,
    retrieval_seconds: float,
    generation_seconds: float,
) -> Dict[str, Any]:
    retrieval = _normalized_usage(retrieval_usage)
    generation = _normalized_usage(generation_usage)
    combined = {
        key: retrieval[key] + generation[key]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    return {
        "rag_cost": combined,
        "time": float(retrieval_seconds) + float(generation_seconds),
        "stages": {
            "retrieval": {**retrieval, "time": float(retrieval_seconds)},
            "generation": {**generation, "time": float(generation_seconds)},
        },
    }
