import argparse
import hashlib
import html
import json
import os
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Core.Index.EvidenceBridgeIndex import EvidenceBridgeIndex
from Core.Index.Tree import DocumentTree, NodeType, TreeNode


def convert_hotpotqa_to_unified(
    raw_path: str,
    output_path: str,
    sample_size: int = 200,
    seed: int = 42,
    split: str = "validation",
    subset: str = "distractor",
) -> List[Dict[str, Any]]:
    raw_rows = _select_rows(_load_hotpotqa_rows(raw_path), sample_size=sample_size, seed=seed)
    return _write_unified_rows(
        raw_rows=raw_rows,
        output_path=output_path,
        split=split,
        subset=subset,
    )


def _write_unified_rows(
    raw_rows: List[Dict[str, Any]],
    output_path: str,
    split: str = "validation",
    subset: str = "distractor",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in raw_rows:
        doc_uuid = _row_id(row)
        _, node_lookup = hotpotqa_row_to_tree(
            row=row,
            save_dir=os.path.dirname(output_path) or ".",
            doc_path=_hotpotqa_doc_path(doc_uuid, split=split, subset=subset),
        )
        supporting_facts = _supporting_facts(row)
        evidence_ids = _evidence_node_ids(supporting_facts, node_lookup)
        rows.append(
            {
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "doc_uuid": doc_uuid,
                "doc_path": _hotpotqa_doc_path(doc_uuid, split=split, subset=subset),
                "question_id": doc_uuid,
                "hotpotqa_question_id": doc_uuid,
                "hotpot_answer": row.get("answer", ""),
                "hotpot_answer_type": _answer_type(row.get("answer", "")),
                "hotpot_type": row.get("type", ""),
                "hotpot_level": row.get("level", ""),
                "hotpot_supporting_facts": supporting_facts,
                "hotpot_node_facts": {
                    str(node_id): [title, sent_id]
                    for (title, sent_id), node_id in node_lookup.items()
                },
                "evidence_block_ids": evidence_ids,
                "evidence_paragraph_ids": evidence_ids,
                "evidence_texts": _evidence_texts(supporting_facts, row),
            }
        )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return rows


def build_hotpotqa_document_trees(
    raw_path: str,
    working_dir: str,
    sample_size: int = 200,
    seed: int = 42,
    split: str = "validation",
    subset: str = "distractor",
) -> Dict[str, str]:
    raw_rows = _select_rows(_load_hotpotqa_rows(raw_path), sample_size=sample_size, seed=seed)
    save_dirs: Dict[str, str] = {}
    for row in raw_rows:
        doc_uuid = _row_id(row)
        save_dir = os.path.join(working_dir, doc_uuid)
        tree, _ = hotpotqa_row_to_tree(
            row=row,
            save_dir=save_dir,
            doc_path=_hotpotqa_doc_path(doc_uuid, split=split, subset=subset),
        )
        tree.save_to_file()
        save_dirs[doc_uuid] = save_dir
    return save_dirs


def build_hotpotqa_evibridge_indexes(
    raw_path: str,
    working_dir: str,
    sample_size: int = 200,
    seed: int = 42,
    split: str = "validation",
    subset: str = "distractor",
) -> Dict[str, str]:
    raw_rows = _select_rows(_load_hotpotqa_rows(raw_path), sample_size=sample_size, seed=seed)
    save_dirs: Dict[str, str] = {}
    for row in raw_rows:
        doc_uuid = _row_id(row)
        save_dir = os.path.join(working_dir, doc_uuid)
        build_hotpotqa_evibridge_index_from_row(
            row=row,
            save_dir=save_dir,
            doc_path=_hotpotqa_doc_path(doc_uuid, split=split, subset=subset),
        )
        save_dirs[doc_uuid] = save_dir
    return save_dirs


def build_hotpotqa_evibridge_index_from_row(
    row: Dict[str, Any],
    save_dir: str,
    doc_path: str = "",
) -> EvidenceBridgeIndex:
    tree, _ = hotpotqa_row_to_tree(row=row, save_dir=save_dir, doc_path=doc_path)
    index = EvidenceBridgeIndex.from_tree(tree=tree, save_dir=save_dir)
    bm25 = index.build_bm25()
    index.save_bm25(bm25)
    index.save_to_dir()
    return index


def prepare_hotpotqa_sample(
    raw_path: str,
    output_path: str,
    working_dir: str,
    dataset_config_path: str = "",
    sample_size: int = 200,
    seed: int = 42,
    split: str = "validation",
    subset: str = "distractor",
) -> Dict[str, Any]:
    rows = convert_hotpotqa_to_unified(
        raw_path=raw_path,
        output_path=output_path,
        sample_size=sample_size,
        seed=seed,
        split=split,
        subset=subset,
    )
    save_dirs = build_hotpotqa_document_trees(
        raw_path=raw_path,
        working_dir=working_dir,
        sample_size=sample_size,
        seed=seed,
        split=split,
        subset=subset,
    )
    if dataset_config_path:
        _write_dataset_config(
            dataset_config_path=dataset_config_path,
            dataset_path=output_path,
            working_dir=working_dir,
            dataset_name="hotpotqa",
        )
    return {
        "document_count": len(save_dirs),
        "question_count": len(rows),
        "dataset_path": output_path,
        "working_dir": working_dir,
        "document_dirs": save_dirs,
    }


def prepare_hotpotqa_exclusive_sample(
    raw_path: str,
    output_path: str,
    working_dir: str,
    dataset_config_path: str = "",
    exclude_dataset_paths: Optional[List[str]] = None,
    sample_size: int = 600,
    seed: int = 42,
    split: str = "validation",
    subset: str = "distractor",
    report_path: str = "",
) -> Dict[str, Any]:
    raw_rows = _load_hotpotqa_rows(raw_path)
    excluded_ids = _load_question_ids_from_datasets(exclude_dataset_paths or [])
    candidates = [row for row in raw_rows if _row_id(row) and _row_id(row) not in excluded_ids]
    selected_rows = _select_rows(candidates, sample_size=sample_size, seed=seed)

    rows = _write_unified_rows(
        raw_rows=selected_rows,
        output_path=output_path,
        split=split,
        subset=subset,
    )
    save_dirs: Dict[str, str] = {}
    for row in selected_rows:
        doc_uuid = _row_id(row)
        save_dir = os.path.join(working_dir, doc_uuid)
        tree, _ = hotpotqa_row_to_tree(
            row=row,
            save_dir=save_dir,
            doc_path=_hotpotqa_doc_path(doc_uuid, split=split, subset=subset),
        )
        tree.save_to_file()
        save_dirs[doc_uuid] = save_dir
    if dataset_config_path:
        _write_dataset_config(
            dataset_config_path=dataset_config_path,
            dataset_path=output_path,
            working_dir=working_dir,
            dataset_name="hotpotqa",
        )
    summary = {
        "raw_count": len(raw_rows),
        "excluded_count": len(excluded_ids),
        "candidate_count": len(candidates),
        "question_count": len(rows),
        "document_count": len(save_dirs),
        "sample_size": sample_size,
        "seed": seed,
        "dataset_path": output_path,
        "working_dir": working_dir,
        "dataset_config_path": dataset_config_path,
        "exclude_dataset_paths": list(exclude_dataset_paths or []),
        "selected_question_ids": [_row_id(row) for row in selected_rows],
    }
    manifest_path = Path(output_path).with_suffix(".manifest.json")
    manifest = {
        "dataset": "hotpotqa",
        "split": split,
        "subset": subset,
        "seed": seed,
        "sample_size": sample_size,
        "raw_count": len(raw_rows),
        "excluded_count": len(excluded_ids),
        "candidate_count": len(candidates),
        "question_count": len(rows),
        "selected_question_ids": summary["selected_question_ids"],
        "dataset_sha256": hashlib.sha256(Path(output_path).read_bytes()).hexdigest(),
        "raw_sha256": hashlib.sha256(Path(raw_path).read_bytes()).hexdigest(),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["manifest_path"] = str(manifest_path)
    if report_path:
        _write_exclusive_sample_report(report_path, summary)
    return summary


def hotpotqa_row_to_tree(
    row: Dict[str, Any],
    save_dir: str,
    doc_path: str = "",
) -> Tuple[DocumentTree, Dict[Tuple[str, int], int]]:
    os.makedirs(save_dir, exist_ok=True)
    doc_uuid = _row_id(row)
    cfg = SimpleNamespace(save_path=save_dir)
    tree = DocumentTree(
        meta_dict={
            "file_name": f"{doc_uuid}.json",
            "file_path": doc_path or f"hotpotqa://{doc_uuid}",
        },
        cfg=cfg,
    )
    node_lookup: Dict[Tuple[str, int], int] = {}
    context = row.get("context") or {}
    titles = _as_list(context.get("title", []))
    sentence_groups = _as_list(context.get("sentences", []))
    for para_idx, title in enumerate(titles):
        clean_title = _clean_title(title) or f"Context {para_idx + 1}"
        title_node = _add_tree_node(
            tree=tree,
            parent=tree.root_node,
            node_type=NodeType.TITLE,
            content=clean_title,
            page_idx=para_idx,
        )
        sentences = _sentences_from_value(_get_index(sentence_groups, para_idx, []))
        for sent_idx, sentence in enumerate(sentences):
            sentence_text = str(sentence or "").strip()
            if not sentence_text:
                continue
            sentence_node = _add_tree_node(
                tree=tree,
                parent=title_node,
                node_type=NodeType.TEXT,
                content=sentence_text,
                page_idx=para_idx,
                source_metadata={
                    "hotpot_title": clean_title,
                    "hotpot_sent_id": sent_idx,
                    "source": "hotpotqa_sentence",
                },
            )
            node_lookup[(clean_title, sent_idx)] = sentence_node.index_id
    return tree, node_lookup


def _add_tree_node(
    tree: DocumentTree,
    parent: TreeNode,
    node_type: NodeType,
    content: str,
    page_idx: int = 0,
    source_metadata: Optional[Dict[str, Any]] = None,
) -> TreeNode:
    node = TreeNode(
        {
            "content": content,
            "page_idx": page_idx,
            "pdf_id": len(tree.nodes),
            "pdf_para_block": dict(source_metadata or {}),
        }
    )
    node.type = node_type
    node.outline_node = node_type == NodeType.TITLE
    tree.add_node(node)
    parent.add_child(node)
    return node


def _load_hotpotqa_rows(raw_path: str) -> List[Dict[str, Any]]:
    payload = _load_json(raw_path)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("validation", "train", "test", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        if all(isinstance(value, dict) for value in payload.values()):
            return [dict(value, id=str(key)) for key, value in payload.items()]
    raise ValueError(f"Unsupported HotpotQA raw JSON shape: {raw_path}")


def _select_rows(rows: List[Dict[str, Any]], sample_size: int, seed: int) -> List[Dict[str, Any]]:
    if sample_size <= 0 or sample_size >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(rows)), sample_size))
    return [rows[index] for index in selected_indices]


def _load_question_ids_from_datasets(dataset_paths: Iterable[str]) -> set[str]:
    ids: set[str] = set()
    for dataset_path in dataset_paths:
        if not dataset_path:
            continue
        payload = _load_json(dataset_path)
        rows = payload if isinstance(payload, list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            question_id = str(
                row.get("hotpotqa_question_id")
                or row.get("question_id")
                or row.get("doc_uuid")
                or row.get("id")
                or row.get("_id")
                or ""
            ).strip()
            if question_id:
                ids.add(question_id)
    return ids


def _write_exclusive_sample_report(report_path: str, summary: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    selected_ids = summary.get("selected_question_ids", [])
    preview = "\n".join(f"- {question_id}" for question_id in selected_ids[:20])
    if len(selected_ids) > 20:
        preview += f"\n- ... ({len(selected_ids) - 20} more)"
    content = (
        "# HotpotQA Random Exclusive Sample Report\n\n"
        "This subset is randomly sampled from HotpotQA distractor validation after excluding completed samples.\n\n"
        f"- Raw validation questions: {summary['raw_count']}\n"
        f"- Excluded completed questions: {summary['excluded_count']}\n"
        f"- Remaining candidate pool: {summary['candidate_count']}\n"
        f"- Selected questions: {summary['question_count']}\n"
        f"- Random seed: {summary['seed']}\n"
        f"- Dataset path: `{summary['dataset_path']}`\n"
        f"- Working dir: `{summary['working_dir']}`\n\n"
        "## Selected Question IDs Preview\n\n"
        f"{preview}\n"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)


def _supporting_facts(row: Dict[str, Any]) -> List[List[Any]]:
    facts = row.get("supporting_facts") or {}
    if isinstance(facts, dict):
        titles = _as_list(facts.get("title", []))
        sent_ids = _as_list(facts.get("sent_id", []))
        pairs = []
        for idx, title in enumerate(titles):
            sent_id = _get_index(sent_ids, idx, None)
            if sent_id is None:
                continue
            pairs.append([_clean_title(title), int(sent_id)])
        return pairs
    pairs = []
    for item in facts if isinstance(facts, list) else []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append([_clean_title(item[0]), int(item[1])])
    return pairs


def _evidence_node_ids(
    supporting_facts: Iterable[List[Any]],
    node_lookup: Dict[Tuple[str, int], int],
) -> List[int]:
    ids = []
    normalized_lookup = {
        (_normalize_title(title), sent_id): node_id
        for (title, sent_id), node_id in node_lookup.items()
    }
    for title, sent_id in supporting_facts:
        node_id = normalized_lookup.get((_normalize_title(title), int(sent_id)))
        if node_id is not None and node_id not in ids:
            ids.append(node_id)
    return ids


def _evidence_texts(supporting_facts: Iterable[List[Any]], row: Dict[str, Any]) -> List[str]:
    text_lookup = {}
    context = row.get("context") or {}
    titles = _as_list(context.get("title", []))
    sentence_groups = _as_list(context.get("sentences", []))
    for idx, title in enumerate(titles):
        clean_title = _clean_title(title)
        for sent_id, sentence in enumerate(_sentences_from_value(_get_index(sentence_groups, idx, []))):
            text_lookup[(_normalize_title(clean_title), sent_id)] = str(sentence).strip()
    texts = []
    for title, sent_id in supporting_facts:
        text = text_lookup.get((_normalize_title(title), int(sent_id)))
        if text:
            texts.append(text)
    return texts


def _hotpotqa_doc_path(doc_uuid: str, split: str, subset: str) -> str:
    return f"hotpotqa://{subset}/{split}/{doc_uuid}"


def _row_id(row: Dict[str, Any]) -> str:
    return str(row.get("id") or row.get("_id") or row.get("doc_uuid") or "").strip()


def _answer_type(answer: Any) -> str:
    normalized = str(answer or "").strip().lower()
    if normalized in {"yes", "no"}:
        return "yes_no"
    if normalized in {"", "noanswer", "not answerable"}:
        return "unanswerable"
    return "span"


def _write_dataset_config(
    dataset_config_path: str,
    dataset_path: str,
    working_dir: str,
    dataset_name: str,
) -> None:
    os.makedirs(os.path.dirname(dataset_config_path) or ".", exist_ok=True)
    content = (
        f"dataset_path: {json.dumps(str(Path(dataset_path)), ensure_ascii=False)}\n"
        f"working_dir: {json.dumps(str(Path(working_dir)), ensure_ascii=False)}\n"
        f"dataset_name: {dataset_name}\n"
    )
    with open(dataset_config_path, "w", encoding="utf-8") as f:
        f.write(content)


def _clean_title(value: Any) -> str:
    return html.unescape(str(value or "")).strip()


def _normalize_title(value: Any) -> str:
    return re.sub(r"\s+", " ", _clean_title(value).lower())


def _sentences_from_value(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value)]


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _get_index(values: List[Any], idx: int, default: Any = None) -> Any:
    return values[idx] if 0 <= idx < len(values) else default


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HotpotQA raw JSON for EviBridge-RAG experiments.")
    parser.add_argument("--raw", required=True, help="Path to raw HotpotQA JSON.")
    parser.add_argument("--output", required=True, help="Path to unified dataset JSON.")
    parser.add_argument("--tree-dir", default="", help="Optional directory for pseudo DocumentTree files.")
    parser.add_argument("--working-dir", default="", help="Optional directory for direct EviBridge indexes.")
    parser.add_argument("--dataset-config-output", default="", help="Optional output path for a dataset YAML config.")
    parser.add_argument("--sample-size", type=int, default=200, help="Number of questions to sample. <=0 means all.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for question sampling.")
    parser.add_argument("--split", default="validation", help="HotpotQA split name used in doc_path.")
    parser.add_argument("--subset", default="distractor", help="HotpotQA subset/config name used in doc_path.")
    args = parser.parse_args()

    convert_hotpotqa_to_unified(
        raw_path=args.raw,
        output_path=args.output,
        sample_size=args.sample_size,
        seed=args.seed,
        split=args.split,
        subset=args.subset,
    )
    if args.tree_dir:
        build_hotpotqa_document_trees(
            raw_path=args.raw,
            working_dir=args.tree_dir,
            sample_size=args.sample_size,
            seed=args.seed,
            split=args.split,
            subset=args.subset,
        )
    if args.working_dir:
        build_hotpotqa_evibridge_indexes(
            raw_path=args.raw,
            working_dir=args.working_dir,
            sample_size=args.sample_size,
            seed=args.seed,
            split=args.split,
            subset=args.subset,
        )
    if args.dataset_config_output:
        _write_dataset_config(
            dataset_config_path=args.dataset_config_output,
            dataset_path=args.output,
            working_dir=args.tree_dir or args.working_dir,
            dataset_name="hotpotqa",
        )


if __name__ == "__main__":
    main()
