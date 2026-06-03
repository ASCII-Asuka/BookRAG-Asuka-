import argparse
import json
import os
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


def convert_qasper_to_unified(
    raw_path: str,
    output_path: str,
    pdf_dir: str = "",
    paper_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    raw_data = _load_json(raw_path)
    rows: List[Dict[str, Any]] = []
    for paper_id, paper in _limited_papers(raw_data, paper_limit):
        _, paragraph_lookup = qasper_paper_to_tree(
            paper_id=paper_id,
            paper=paper,
            save_dir=os.path.dirname(output_path) or ".",
            doc_path=_paper_doc_path(paper_id, paper, pdf_dir),
        )
        for qa in _iter_qasper_qas(paper.get("qas", [])):
            answer_records = _answer_records(qa.get("answers", []), paragraph_lookup)
            evidence_ids = _unique_ids(
                block_id
                for answer in answer_records
                for block_id in answer.get("evidence_block_ids", [])
            )
            rows.append(
                {
                    "question": qa.get("question", ""),
                    "answer": answer_records,
                    "doc_uuid": paper_id,
                    "doc_path": _paper_doc_path(paper_id, paper, pdf_dir),
                    "qasper_question_id": qa.get("question_id", ""),
                    "title": paper.get("title", ""),
                    "evidence_block_ids": evidence_ids,
                    "evidence_paragraph_ids": evidence_ids,
                }
            )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return rows


def build_qasper_evibridge_index_from_paper(
    paper_id: str,
    paper: Dict[str, Any],
    save_dir: str,
    doc_path: str = "",
) -> EvidenceBridgeIndex:
    tree, _ = qasper_paper_to_tree(
        paper_id=paper_id,
        paper=paper,
        save_dir=save_dir,
        doc_path=doc_path,
    )
    index = EvidenceBridgeIndex.from_tree(tree=tree, save_dir=save_dir)
    bm25 = index.build_bm25()
    index.save_bm25(bm25)
    index.save_to_dir()
    return index


def build_qasper_evibridge_indexes(
    raw_path: str,
    working_dir: str,
    pdf_dir: str = "",
    paper_limit: Optional[int] = None,
) -> Dict[str, str]:
    raw_data = _load_json(raw_path)
    save_dirs: Dict[str, str] = {}
    for paper_id, paper in _limited_papers(raw_data, paper_limit):
        save_dir = os.path.join(working_dir, paper_id)
        os.makedirs(save_dir, exist_ok=True)
        build_qasper_evibridge_index_from_paper(
            paper_id=paper_id,
            paper=paper,
            save_dir=save_dir,
            doc_path=_paper_doc_path(paper_id, paper, pdf_dir),
        )
        save_dirs[paper_id] = save_dir
    return save_dirs


def build_qasper_document_trees(
    raw_path: str,
    working_dir: str,
    pdf_dir: str = "",
    paper_limit: Optional[int] = None,
) -> Dict[str, str]:
    raw_data = _load_json(raw_path)
    save_dirs: Dict[str, str] = {}
    for paper_id, paper in _limited_papers(raw_data, paper_limit):
        save_dir = os.path.join(working_dir, paper_id)
        tree, _ = qasper_paper_to_tree(
            paper_id=paper_id,
            paper=paper,
            save_dir=save_dir,
            doc_path=_paper_doc_path(paper_id, paper, pdf_dir),
        )
        tree.save_to_file()
        save_dirs[paper_id] = save_dir
    return save_dirs


def prepare_qasper_sample(
    raw_path: str,
    output_path: str,
    working_dir: str,
    sample_docs: int = 3,
    dataset_config_path: str = "",
    system_config_path: str = "",
    pdf_dir: str = "",
    dataset_name: str = "qasper",
    system_config_template: str = "",
) -> Dict[str, Any]:
    rows = convert_qasper_to_unified(
        raw_path=raw_path,
        output_path=output_path,
        pdf_dir=pdf_dir,
        paper_limit=sample_docs,
    )
    save_dirs = build_qasper_document_trees(
        raw_path=raw_path,
        working_dir=working_dir,
        pdf_dir=pdf_dir,
        paper_limit=sample_docs,
    )
    if dataset_config_path:
        _write_dataset_config(
            dataset_config_path=dataset_config_path,
            dataset_path=output_path,
            working_dir=working_dir,
            dataset_name=dataset_name,
        )
    if system_config_path:
        _write_system_config(
            system_config_path=system_config_path,
            system_config_template=system_config_template,
        )
    return {
        "paper_count": len(save_dirs),
        "question_count": len(rows),
        "dataset_path": output_path,
        "working_dir": working_dir,
        "paper_dirs": save_dirs,
    }


def qasper_paper_to_tree(
    paper_id: str,
    paper: Dict[str, Any],
    save_dir: str,
    doc_path: str = "",
) -> Tuple[DocumentTree, Dict[str, int]]:
    os.makedirs(save_dir, exist_ok=True)
    cfg = SimpleNamespace(save_path=save_dir)
    tree = DocumentTree(
        meta_dict={
            "file_name": f"{paper_id}.pdf",
            "file_path": doc_path or f"{paper_id}.pdf",
        },
        cfg=cfg,
    )
    paragraph_lookup: Dict[str, int] = {}

    for section_idx, section in enumerate(_paper_sections(paper)):
        section_name = section.get("section_name") or section.get("heading") or f"Section {section_idx + 1}"
        section_node = _add_tree_node(
            tree=tree,
            parent=tree.root_node,
            node_type=NodeType.TITLE,
            content=section_name,
            page_idx=section_idx,
        )
        for paragraph in section.get("paragraphs", []):
            paragraph_text = str(paragraph or "").strip()
            if not paragraph_text:
                continue
            paragraph_node = _add_tree_node(
                tree=tree,
                parent=section_node,
                node_type=NodeType.TEXT,
                content=paragraph_text,
                page_idx=section_idx,
            )
            paragraph_lookup[_normalize_evidence(paragraph_text)] = paragraph_node.index_id
    return tree, paragraph_lookup


def _paper_sections(paper: Dict[str, Any]) -> List[Dict[str, Any]]:
    sections = []
    abstract = paper.get("abstract")
    if abstract:
        sections.append(
            {
                "section_name": "Abstract",
                "paragraphs": _paragraphs_from_value(abstract),
            }
        )
    full_text = paper.get("full_text", []) or []
    if isinstance(full_text, dict):
        section_names = _as_list(full_text.get("section_name", []))
        paragraph_groups = _as_list(full_text.get("paragraphs", []))
        for idx, paragraphs in enumerate(paragraph_groups):
            section_name = _get_index(section_names, idx, "") or f"Section {idx + 1}"
            sections.append(
                {
                    "section_name": section_name,
                    "paragraphs": _paragraphs_from_value(paragraphs),
                }
            )
    else:
        for section in full_text:
            if isinstance(section, dict):
                paragraphs = section.get("paragraphs", [])
                sections.append(
                    {
                        "section_name": section.get("section_name")
                        or section.get("heading")
                        or "",
                        "paragraphs": _paragraphs_from_value(paragraphs),
                    }
                )
    return sections


def _answer_records(
    answers: Iterable[Dict[str, Any]],
    paragraph_lookup: Dict[str, int],
) -> List[Dict[str, Any]]:
    records = []
    for answer in _iter_answer_payloads(answers):
        evidence = [
            text
            for text in answer.get("evidence", []) or []
            if isinstance(text, str) and "FLOAT SELECTED" not in text
        ]
        evidence_block_ids = _match_evidence_ids(evidence, paragraph_lookup)
        records.append(
            {
                "unanswerable": bool(answer.get("unanswerable", False)),
                "extractive_spans": answer.get("extractive_spans", []) or [],
                "free_form_answer": answer.get("free_form_answer", "") or "",
                "yes_no": answer.get("yes_no", None),
                "evidence": evidence,
                "evidence_block_ids": evidence_block_ids,
                "evidence_paragraph_ids": evidence_block_ids,
            }
        )
    return records


def _iter_answer_payloads(answers: Any):
    if isinstance(answers, dict):
        answers = [answers]
    for answer_item in answers or []:
        if not isinstance(answer_item, dict):
            continue
        answer = answer_item.get("answer", answer_item)
        if isinstance(answer, list):
            for nested_answer in answer:
                if isinstance(nested_answer, dict):
                    yield nested_answer
        elif isinstance(answer, dict):
            yield answer


def _match_evidence_ids(evidence: Iterable[str], paragraph_lookup: Dict[str, int]) -> List[int]:
    matched_ids = []
    normalized_items = list(paragraph_lookup.items())
    for evidence_text in evidence:
        normalized = _normalize_evidence(evidence_text)
        if normalized in paragraph_lookup:
            matched_ids.append(paragraph_lookup[normalized])
            continue
        for paragraph_text, block_id in normalized_items:
            if normalized and (normalized in paragraph_text or paragraph_text in normalized):
                matched_ids.append(block_id)
                break
    return list(dict.fromkeys(matched_ids))


def _add_tree_node(
    tree: DocumentTree,
    parent: TreeNode,
    node_type: NodeType,
    content: str,
    page_idx: int = 0,
) -> TreeNode:
    node = TreeNode(
        {
            "content": content,
            "page_idx": page_idx,
            "pdf_id": len(tree.nodes),
        }
    )
    node.type = node_type
    node.outline_node = node_type == NodeType.TITLE
    tree.add_node(node)
    parent.add_child(node)
    return node


def _paper_doc_path(paper_id: str, paper: Dict[str, Any], pdf_dir: str) -> str:
    if paper.get("pdf_path"):
        return str(paper["pdf_path"])
    if pdf_dir:
        return os.path.join(pdf_dir, f"{paper_id}.pdf")
    return f"qasper://{paper_id}"


def _iter_qasper_papers(raw_data: Any):
    if isinstance(raw_data, dict):
        for paper_id, paper in raw_data.items():
            if isinstance(paper, dict):
                yield str(paper_id), paper
    elif isinstance(raw_data, list):
        for paper in raw_data:
            if isinstance(paper, dict):
                paper_id = str(paper.get("paper_id") or paper.get("id") or paper.get("doc_uuid"))
                yield paper_id, paper


def _limited_papers(raw_data: Any, paper_limit: Optional[int] = None):
    for idx, item in enumerate(_iter_qasper_papers(raw_data)):
        if paper_limit is not None and idx >= paper_limit:
            break
        yield item


def _iter_qasper_qas(qas: Any):
    if isinstance(qas, list):
        for qa in qas:
            if isinstance(qa, dict):
                yield qa
        return
    if not isinstance(qas, dict):
        return
    questions = _as_list(qas.get("question", []))
    question_ids = _as_list(qas.get("question_id", []))
    answer_groups = _as_list(qas.get("answers", []))
    for idx, question in enumerate(questions):
        yield {
            "question": question,
            "question_id": _get_index(question_ids, idx, ""),
            "answers": _get_index(answer_groups, idx, []),
        }


def _paragraphs_from_value(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        paragraphs = []
        for item in value:
            if isinstance(item, list):
                paragraphs.extend(str(text) for text in item if str(text).strip())
            elif str(item).strip():
                paragraphs.append(str(item))
        return paragraphs
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


def _unique_ids(values: Iterable[Any]) -> List[int]:
    ids = []
    for value in values:
        if value is None:
            continue
        int_value = int(value)
        if int_value not in ids:
            ids.append(int_value)
    return ids


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


def _write_system_config(system_config_path: str, system_config_template: str = "") -> None:
    os.makedirs(os.path.dirname(system_config_path) or ".", exist_ok=True)
    template_path = Path(system_config_template) if system_config_template else REPO_ROOT / "config" / "evibridge.yaml"
    if template_path.exists():
        config_text = template_path.read_text(encoding="utf-8")
    else:
        config_text = _minimal_evibridge_config()
    if "enable_vector_recall:" not in config_text:
        config_text += "\nrag:\n  strategy: evibridge\n  enable_vector_recall: false\n"
    Path(system_config_path).write_text(config_text, encoding="utf-8")


def _minimal_evibridge_config() -> str:
    return """# EviBridge-RAG Qasper template.
pdf_path: TODO
save_path: TODO
index_type: evibridge
rag:
  strategy: evibridge
  enable_vector_recall: false
"""


def _normalize_evidence(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Qasper raw JSON for EviBridge-RAG experiments.")
    parser.add_argument("--raw", required=True, help="Path to raw Qasper JSON.")
    parser.add_argument("--output", required=True, help="Path to unified dataset JSON.")
    parser.add_argument("--pdf-dir", default="", help="Directory containing Qasper PDFs.")
    parser.add_argument("--working-dir", default="", help="Optional directory for direct EviBridge indexes.")
    parser.add_argument("--tree-dir", default="", help="Optional directory for pseudo DocumentTree files.")
    parser.add_argument("--sample-docs", type=int, default=0, help="Limit conversion to the first N papers.")
    parser.add_argument("--dataset-config-output", default="", help="Optional output path for a dataset YAML config.")
    parser.add_argument("--system-config-output", default="", help="Optional output path for an EviBridge system YAML config.")
    parser.add_argument("--system-config-template", default="", help="Template config copied for --system-config-output.")
    args = parser.parse_args()

    paper_limit = args.sample_docs if args.sample_docs > 0 else None
    convert_qasper_to_unified(args.raw, args.output, args.pdf_dir, paper_limit=paper_limit)
    if args.tree_dir:
        build_qasper_document_trees(args.raw, args.tree_dir, args.pdf_dir, paper_limit=paper_limit)
    if args.working_dir:
        build_qasper_evibridge_indexes(args.raw, args.working_dir, args.pdf_dir, paper_limit=paper_limit)
    if args.dataset_config_output:
        _write_dataset_config(
            dataset_config_path=args.dataset_config_output,
            dataset_path=args.output,
            working_dir=args.tree_dir or args.working_dir,
            dataset_name="qasper",
        )
    if args.system_config_output:
        _write_system_config(args.system_config_output, args.system_config_template)


if __name__ == "__main__":
    main()
