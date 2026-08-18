from __future__ import annotations

import hashlib
import json
import re
import string
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


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
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(output)


def safe_filename(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._") or "scope"
    suffix = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:10]
    return f"{stem[:80]}-{suffix}.json"


def normalize_entity(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = re.sub(r"[\u2010-\u2015]+", " ", text)
    text = text.strip(string.whitespace + string.punctuation)
    return re.sub(r"\s+", " ", text)


def scope_corpus_sha256(chunks: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json(
        [
            {
                "source_id": str(chunk.get("source_id") or ""),
                "content": str(chunk.get("content") or ""),
                "metadata": dict(chunk.get("metadata") or {}),
            }
            for chunk in chunks
        ]
    )


def build_cache_identity(
    corpus_sha256: str,
    config_sha256: str,
    source_revision: str,
    openie_model: str,
    embedding_model: str,
    reranker_model: str,
) -> str:
    return sha256_json(
        {
            "corpus_sha256": corpus_sha256,
            "config_sha256": config_sha256,
            "source_revision": source_revision,
            "openie_model": openie_model,
            "embedding_model": embedding_model,
            "reranker_model": reranker_model,
        }
    )


def apply_generation_budget(
    ranked_results: Sequence[Mapping[str, Any]],
    max_blocks: int,
    max_tokens: int,
    token_counter: Callable[[str], int],
) -> list[dict[str, Any]]:
    if max_blocks <= 0 or max_tokens <= 0:
        return []
    selected: list[dict[str, Any]] = []
    used_tokens = 0
    for item in ranked_results:
        if len(selected) >= max_blocks:
            break
        cost = max(1, int(token_counter(str(item.get("content") or ""))))
        if used_tokens + cost > max_tokens:
            break
        selected.append(dict(item))
        used_tokens += cost
    return selected


def _qasper_evidence(item: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(item.get("metadata") or {})
    paragraph_id = metadata.get("paragraph_id")
    if paragraph_id is None:
        paragraph_id = metadata.get("source_node_id", item.get("source_id"))
    text = str(metadata.get("qasper_evidence_text") or item.get("content") or "")
    return {
        "source_id": str(item.get("source_id") or ""),
        "paragraph_id": paragraph_id,
        "source_node_id": paragraph_id,
        "node_id": paragraph_id,
        "evidence_id": paragraph_id,
        "content": text,
        "qasper_evidence_text": text,
        "rank": item.get("rank"),
        "score": item.get("score"),
        "section": metadata.get("section") or "",
    }


def validate_qasper_citations(
    requested_ids: Iterable[Any],
    ranked_results: Sequence[Mapping[str, Any]],
    max_support: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aliases: dict[str, Mapping[str, Any]] = {}
    for item in ranked_results:
        metadata = dict(item.get("metadata") or {})
        for value in (
            item.get("source_id"),
            metadata.get("paragraph_id"),
            metadata.get("source_node_id"),
        ):
            if value is not None:
                aliases[str(value)] = item

    requested = [str(value) for value in requested_ids or []]
    accepted: list[Mapping[str, Any]] = []
    rejected: list[str] = []
    seen: set[str] = set()
    for key in requested:
        lookup_key = key
        if lookup_key not in aliases:
            label = re.fullmatch(
                r"(?i)\s*evidence\s*[:#-]?\s*(.+?)\s*", lookup_key
            )
            if label and label.group(1) in aliases:
                lookup_key = label.group(1)
        item = aliases.get(lookup_key)
        source_id = str(item.get("source_id") or "") if item else ""
        if item is None or source_id in seen:
            rejected.append(key)
            continue
        seen.add(source_id)
        accepted.append(item)
        if len(accepted) >= max_support:
            break

    fallback_used = not accepted
    if fallback_used:
        accepted = list(ranked_results[:max_support])
    evidence = [_qasper_evidence(item) for item in accepted]
    return evidence, {
        "requested": requested,
        "accepted_source_ids": [item["source_id"] for item in evidence],
        "rejected": rejected,
        "fallback_used": fallback_used,
        "fallback_policy": "top-ranked-valid-paragraph" if fallback_used else None,
    }


def _legal_hotpot_facts(
    ranked_results: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
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
            fact = {
                "hotpot_title": title,
                "hotpot_sent_id": sent_id,
                "text": str(sentence.get("text") or ""),
            }
            key = (title, sent_id)
            if key not in by_key:
                by_key[key] = fact
                ordered.append(fact)
    return by_key, ordered


def validate_hotpot_citations(
    requested_facts: Iterable[Any],
    ranked_results: Sequence[Mapping[str, Any]],
    max_support: int,
    question: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key, ordered = _legal_hotpot_facts(ranked_results)
    if not ordered:
        raise ValueError("no legal HotpotQA title-sentence pair in retrieved passages")

    requested: list[dict[str, Any]] = []
    rejected: list[Any] = []
    accepted: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for raw in requested_facts or []:
        if isinstance(raw, Mapping):
            title = str(raw.get("title") or raw.get("hotpot_title") or "")
            raw_sent_id = raw.get("sent_id", raw.get("hotpot_sent_id"))
        elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
            title, raw_sent_id = str(raw[0]), raw[1]
        else:
            rejected.append(raw)
            continue
        try:
            sent_id = int(raw_sent_id)
        except (TypeError, ValueError):
            rejected.append(raw)
            continue
        request = {"title": title, "sent_id": sent_id}
        requested.append(request)
        key = (title, sent_id)
        if key not in by_key or key in seen:
            rejected.append(request)
            continue
        seen.add(key)
        accepted.append(by_key[key])
        if len(accepted) >= max_support:
            break

    fallback_used = not accepted
    if fallback_used:
        query_terms = set(re.findall(r"[A-Za-z0-9]+", question.lower()))
        positions = {
            (fact["hotpot_title"], fact["hotpot_sent_id"]): index
            for index, fact in enumerate(ordered)
        }
        accepted = sorted(
            ordered,
            key=lambda fact: (
                -len(
                    query_terms
                    & set(re.findall(r"[A-Za-z0-9]+", fact["text"].lower()))
                ),
                positions[(fact["hotpot_title"], fact["hotpot_sent_id"])],
            ),
        )[:max_support]

    return accepted, {
        "requested": requested,
        "accepted": [
            [fact["hotpot_title"], fact["hotpot_sent_id"]] for fact in accepted
        ],
        "rejected": rejected,
        "fallback_used": fallback_used,
        "fallback_policy": (
            "first-legal-sentence-from-top-ranked-passages"
            if fallback_used
            else None
        ),
    }


def _usage(stage: Mapping[str, Any]) -> dict[str, int]:
    prompt = int(stage.get("prompt_tokens") or 0)
    completion = int(stage.get("completion_tokens") or 0)
    total = int(stage.get("total_tokens") or prompt + completion)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def combine_online_cost(
    retrieval: Mapping[str, Any], generation: Mapping[str, Any]
) -> dict[str, Any]:
    retrieval_usage = _usage(retrieval)
    generation_usage = _usage(generation)
    combined = {
        key: retrieval_usage[key] + generation_usage[key]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    retrieval_time = float(retrieval.get("time") or 0.0)
    generation_time = float(generation.get("time") or 0.0)
    return {
        "rag_cost": combined,
        "time": retrieval_time + generation_time,
        "stages": {
            "retrieval": {**retrieval_usage, "time": retrieval_time},
            "generation": {**generation_usage, "time": generation_time},
        },
    }
