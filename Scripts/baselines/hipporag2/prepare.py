from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from Core.Index.Tree import DocumentTree
from Core.configs.dataset_config import load_dataset_config

from .common import safe_filename, scope_corpus_sha256, sha256_file, write_json


def _node_type(node: Any) -> str:
    value = getattr(node, "type", "")
    return str(getattr(value, "value", value) or "").lower()


def _node_text(node: Any) -> str:
    metadata = getattr(node, "meta_info", None)
    return str(getattr(metadata, "content", "") or "").strip()


def _title_path(node: Any) -> List[str]:
    titles: List[str] = []
    parent = getattr(node, "parent", None)
    while parent is not None:
        if _node_type(parent) == "title" and _node_text(parent):
            titles.append(_node_text(parent))
        parent = getattr(parent, "parent", None)
    return list(reversed(titles))


def qasper_chunks_from_tree(tree: Any, doc_uuid: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for node in tree.get_nodes(hasRoot=False):
        if _node_type(node) != "text":
            continue
        text = _node_text(node)
        if not text:
            continue
        source_id = str(getattr(node, "index_id", ""))
        if not source_id or source_id in seen:
            raise ValueError(f"duplicate or empty Qasper paragraph source_id: {source_id!r}")
        seen.add(source_id)
        path = _title_path(node)
        section = path[-1] if path else ""
        retrieval_text = f"[Section: {section}]\n{text}" if section else text
        paragraph_id: Any = getattr(node, "index_id", source_id)
        chunks.append(
            {
                "source_id": source_id,
                "content": retrieval_text,
                "index_content": f"{retrieval_text}\n[Source ID: {source_id}]",
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


def hotpot_chunks_from_tree(tree: Any, doc_uuid: str) -> List[Dict[str, Any]]:
    grouped: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for node in tree.get_nodes(hasRoot=False):
        if _node_type(node) != "text" or not _node_text(node):
            continue
        metadata = getattr(node, "meta_info", None)
        source_metadata = dict(getattr(metadata, "pdf_para_block", None) or {})
        path = _title_path(node)
        title = str(source_metadata.get("hotpot_title") or (path[-1] if path else ""))
        if not title:
            raise ValueError(f"HotpotQA sentence {getattr(node, 'index_id', None)!r} has no title")
        raw_sent_id = source_metadata.get("hotpot_sent_id")
        if raw_sent_id is None:
            raw_sent_id = len(grouped.get(title, []))
        sent_id = int(raw_sent_id)
        grouped.setdefault(title, []).append(
            {
                "sent_id": sent_id,
                "text": _node_text(node),
                "source_node_id": getattr(node, "index_id", None),
            }
        )
    chunks: List[Dict[str, Any]] = []
    for title_index, (title, sentences) in enumerate(grouped.items()):
        sent_ids = [int(sentence["sent_id"]) for sentence in sentences]
        if len(sent_ids) != len(set(sent_ids)):
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
                "index_content": f"{content}\n[Source ID: {source_id}]",
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


def _query_record(row: Mapping[str, Any]) -> Dict[str, Any]:
    question_id = _question_id(row)
    if not question_id:
        raise ValueError("dataset row has no stable question ID")
    return {
        "question_id": question_id,
        "question": str(row.get("question") or ""),
        "answer": row.get("answer"),
        "dataset_row": dict(row),
    }


def _group_rows(rows: Iterable[Mapping[str, Any]]) -> "OrderedDict[str, List[Mapping[str, Any]]]":
    grouped: "OrderedDict[str, List[Mapping[str, Any]]]" = OrderedDict()
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


def prepare_scopes(
    dataset_config_path: str,
    output_root: str,
    limit_scopes: int | None = None,
) -> Dict[str, Any]:
    dataset_config = load_dataset_config(dataset_config_path)
    dataset_path = Path(dataset_config.dataset_path).resolve()
    rows = __import__("json").loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("unified dataset must be a JSON list")
    dataset = str(dataset_config.dataset_name).lower()
    if "qasper" in dataset:
        dataset = "qasper"
    elif "hotpot" in dataset:
        dataset = "hotpotqa"
    else:
        raise ValueError(f"HippoRAG 2 adapter only supports Qasper and HotpotQA, got {dataset!r}")

    grouped = _group_rows(rows)
    if limit_scopes is not None:
        grouped = OrderedDict(list(grouped.items())[: max(0, int(limit_scopes))])
    output = Path(output_root).resolve()
    scope_dir = output / "scopes"
    scope_dir.mkdir(parents=True, exist_ok=True)
    scope_entries: List[Dict[str, Any]] = []
    query_count = 0
    for scope_id, scope_rows in grouped.items():
        if dataset == "hotpotqa" and len(scope_rows) != 1:
            raise ValueError(
                f"HotpotQA requires one graph per question, but scope {scope_id!r} has {len(scope_rows)} rows"
            )
        tree_path = Path(dataset_config.working_dir).resolve() / scope_id / "tree.pkl"
        if not tree_path.exists():
            raise FileNotFoundError(f"missing DocumentTree for scope {scope_id!r}: {tree_path}")
        tree = DocumentTree.load_from_file(str(tree_path))
        chunks = (
            qasper_chunks_from_tree(tree, scope_id)
            if dataset == "qasper"
            else hotpot_chunks_from_tree(tree, scope_id)
        )
        queries = [_query_record(row) for row in scope_rows]
        corpus_sha256 = scope_corpus_sha256(chunks)
        scope_payload = {
            "dataset": dataset,
            "scope_id": scope_id,
            "corpus_sha256": corpus_sha256,
            "chunks": chunks,
            "queries": queries,
        }
        scope_filename = safe_filename(scope_id)
        relative_path = (Path("scopes") / scope_filename).as_posix()
        write_json(scope_dir / scope_filename, scope_payload)
        scope_entries.append(
            {
                "scope_id": scope_id,
                "relative_path": relative_path,
                "corpus_sha256": corpus_sha256,
                "chunk_count": len(chunks),
                "query_count": len(queries),
            }
        )
        query_count += len(queries)

    manifest = {
        "format_version": 1,
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
    parser = argparse.ArgumentParser(description="Export Qasper/HotpotQA scopes for official HippoRAG 2")
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit-scopes", type=int)
    args = parser.parse_args()
    manifest = prepare_scopes(args.dataset_config, args.output_root, args.limit_scopes)
    print(f"Prepared {manifest['scope_count']} scopes and {manifest['query_count']} queries")


if __name__ == "__main__":
    main()
