from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping

from Core.Index.Tree import DocumentTree
from Core.configs.dataset_config import load_dataset_config

from .common import safe_filename, scope_corpus_sha256, sha256_file, write_json


def _node_type(node: Any) -> str:
    value = getattr(node, "type", "")
    return str(getattr(value, "value", value) or "").lower()


def _node_text(node: Any) -> str:
    metadata = getattr(node, "meta_info", None)
    return str(getattr(metadata, "content", "") or "").strip()


def _title_path(node: Any) -> list[str]:
    titles: list[str] = []
    parent = getattr(node, "parent", None)
    while parent is not None:
        if _node_type(parent) == "title" and _node_text(parent):
            titles.append(_node_text(parent))
        parent = getattr(parent, "parent", None)
    return list(reversed(titles))


def qasper_chunks_from_tree(tree: Any, doc_uuid: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in tree.get_nodes(hasRoot=False):
        if _node_type(node) != "text":
            continue
        text = _node_text(node)
        if not text:
            continue
        paragraph_id = getattr(node, "index_id", None)
        source_id = str(paragraph_id if paragraph_id is not None else "")
        if not source_id or source_id in seen:
            raise ValueError(
                f"duplicate or empty Qasper paragraph source_id: {source_id!r}"
            )
        seen.add(source_id)
        path = _title_path(node)
        section = path[-1] if path else ""
        content = f"[Section: {section}]\n{text}" if section else text
        chunks.append(
            {
                "source_id": source_id,
                "content": content,
                "metadata": {
                    "doc_uuid": doc_uuid,
                    "paragraph_id": paragraph_id,
                    "source_node_id": paragraph_id,
                    "qasper_paragraph_id": paragraph_id,
                    "qasper_evidence_text": text,
                    "section": section,
                    "title_path": " > ".join(path),
                    "source": "qasper_paragraph",
                },
            }
        )
    if not chunks:
        raise ValueError(f"Qasper scope {doc_uuid!r} contains no paragraph chunks")
    return chunks


def hotpot_chunks_from_tree(tree: Any, doc_uuid: str) -> list[dict[str, Any]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for node in tree.get_nodes(hasRoot=False):
        if _node_type(node) != "text":
            continue
        text = _node_text(node)
        if not text:
            continue
        metadata = getattr(node, "meta_info", None)
        source_metadata = dict(getattr(metadata, "pdf_para_block", None) or {})
        path = _title_path(node)
        title = str(source_metadata.get("hotpot_title") or (path[-1] if path else ""))
        if not title:
            raise ValueError(
                f"HotpotQA sentence {getattr(node, 'index_id', None)!r} has no title"
            )
        raw_sent_id = source_metadata.get("hotpot_sent_id")
        sent_id = int(raw_sent_id if raw_sent_id is not None else len(grouped.get(title, [])))
        grouped.setdefault(title, []).append(
            {
                "sent_id": sent_id,
                "text": text,
                "source_node_id": getattr(node, "index_id", None),
            }
        )

    chunks: list[dict[str, Any]] = []
    for title_index, (title, sentences) in enumerate(grouped.items()):
        sentence_ids = [int(sentence["sent_id"]) for sentence in sentences]
        if len(sentence_ids) != len(set(sentence_ids)):
            raise ValueError(f"duplicate sentence ID under HotpotQA title {title!r}")
        source_id = f"title-{title_index}"
        content = "\n".join(
            [f"Title: {title}"]
            + [f"[{sentence['sent_id']}] {sentence['text']}" for sentence in sentences]
        )
        chunks.append(
            {
                "source_id": source_id,
                "content": content,
                "metadata": {
                    "doc_uuid": doc_uuid,
                    "hotpot_title": title,
                    "title": title,
                    "hotpot_sentences": sentences,
                    "source": "hotpotqa_passage",
                },
            }
        )
    if not chunks:
        raise ValueError(f"HotpotQA scope {doc_uuid!r} contains no title passages")
    return chunks


def _question_id(row: Mapping[str, Any]) -> str:
    return str(
        row.get("hotpotqa_question_id")
        or row.get("qasper_question_id")
        or row.get("question_id")
        or row.get("doc_uuid")
        or ""
    )


def _group_rows(
    rows: Iterable[Mapping[str, Any]],
) -> OrderedDict[str, list[Mapping[str, Any]]]:
    grouped: OrderedDict[str, list[Mapping[str, Any]]] = OrderedDict()
    seen_questions: set[str] = set()
    for row in rows:
        doc_uuid = str(row.get("doc_uuid") or "")
        question_id = _question_id(row)
        if not doc_uuid:
            raise ValueError("dataset row has no doc_uuid")
        if not question_id or question_id in seen_questions:
            raise ValueError(f"duplicate or empty question ID: {question_id!r}")
        seen_questions.add(question_id)
        grouped.setdefault(doc_uuid, []).append(row)
    return grouped


def _query_record(row: Mapping[str, Any]) -> dict[str, Any]:
    question_id = _question_id(row)
    if not question_id:
        raise ValueError("dataset row has no stable question ID")
    return {
        "question_id": question_id,
        "question": str(row.get("question") or ""),
        "dataset_row": dict(row),
    }


def prepare_scopes(
    dataset_config_path: str,
    output_root: str,
    limit_scopes: int | None = None,
) -> dict[str, Any]:
    dataset_config = load_dataset_config(dataset_config_path)
    dataset_path = Path(dataset_config.dataset_path).resolve()
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("unified dataset must be a JSON list")
    dataset_name = str(dataset_config.dataset_name).lower()
    if "qasper" in dataset_name:
        dataset = "qasper"
    elif "hotpot" in dataset_name:
        dataset = "hotpotqa"
    else:
        raise ValueError(
            f"KG2RAG adapter supports only Qasper and HotpotQA, got {dataset_name!r}"
        )

    grouped = _group_rows(rows)
    if limit_scopes is not None:
        grouped = OrderedDict(list(grouped.items())[: max(0, int(limit_scopes))])

    output = Path(output_root).resolve()
    scope_directory = output / "scopes"
    scope_directory.mkdir(parents=True, exist_ok=True)
    scope_entries: list[dict[str, Any]] = []
    query_count = 0
    for scope_id, scope_rows in grouped.items():
        if dataset == "hotpotqa" and len(scope_rows) != 1:
            raise ValueError(
                f"HotpotQA requires one scope per question, got {len(scope_rows)} rows for {scope_id!r}"
            )
        tree_path = Path(dataset_config.working_dir).resolve() / scope_id / "tree.pkl"
        if not tree_path.exists():
            raise FileNotFoundError(
                f"missing DocumentTree for scope {scope_id!r}: {tree_path}"
            )
        tree = DocumentTree.load_from_file(str(tree_path))
        chunks = (
            qasper_chunks_from_tree(tree, scope_id)
            if dataset == "qasper"
            else hotpot_chunks_from_tree(tree, scope_id)
        )
        queries = [_query_record(row) for row in scope_rows]
        corpus_sha = scope_corpus_sha256(chunks)
        scope_payload = {
            "dataset": dataset,
            "scope_id": scope_id,
            "corpus_sha256": corpus_sha,
            "chunks": chunks,
            "queries": queries,
        }
        relative_path = Path("scopes") / safe_filename(scope_id)
        write_json(output / relative_path, scope_payload)
        scope_entries.append(
            {
                "scope_id": scope_id,
                "relative_path": relative_path.as_posix(),
                "corpus_sha256": corpus_sha,
                "chunk_count": len(chunks),
                "query_count": len(queries),
            }
        )
        query_count += len(queries)

    manifest = {
        "dataset": dataset,
        "dataset_config_path": str(Path(dataset_config_path).resolve()),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "working_dir": str(Path(dataset_config.working_dir).resolve()),
        "scope_count": len(scope_entries),
        "query_count": query_count,
        "scopes": scope_entries,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare closed-corpus KG2RAG scopes")
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit-scopes", type=int)
    args = parser.parse_args()
    manifest = prepare_scopes(
        args.dataset_config, args.output_root, limit_scopes=args.limit_scopes
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
