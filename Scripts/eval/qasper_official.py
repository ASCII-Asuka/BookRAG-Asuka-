import argparse
import hashlib
import json
import os
import re
import subprocess
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Core.configs.dataset_config import load_dataset_config
from Scripts.eval.qasper_run_validator import validate_qasper_run

OFFICIAL_EVALUATOR_URL = (
    "https://huggingface.co/datasets/albertgong1/qasper/resolve/"
    "a8de10174c66470ee25cd1a5af9f34a494b60ab6/"
    "original_data/qasper-test-and-evaluator-v0.3/qasper_evaluator.py"
)


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(str(text or "").lower())))


def canonicalize_qasper_answer(text: Any) -> str:
    answer = str(text or "").strip()
    normalized = normalize_answer(answer)
    if normalized in {
        "not answerable",
        "not enough information",
        "cannot be answered",
        "no answer",
        "unknown",
    }:
        return "Unanswerable"
    return answer


def token_f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens) if prediction_tokens else 0.0
    recall = num_same / len(ground_truth_tokens) if ground_truth_tokens else 0.0
    return 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)


def paragraph_f1_score(prediction: List[str], ground_truth: List[str]) -> float:
    if not ground_truth and not prediction:
        return 1.0
    if not prediction or not ground_truth:
        return 0.0
    num_same = len(set(ground_truth).intersection(set(prediction)))
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    return 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)


def export_predictions(
    dataset_path: str,
    working_dir: str,
    dataset_name: str,
    method: str,
    output_path: str,
    answer_source: str = "output",
    paragraph_evidence_only: bool = True,
    top_k_evidence: int = 0,
    dynamic_evidence_topk: bool = False,
) -> List[Dict[str, Any]]:
    rows = _load_json(dataset_path)
    validate_qasper_run(
        dataset_path=dataset_path,
        working_dir=working_dir,
        dataset_name=dataset_name,
        method=method,
        require_query_outputs=False,
    )
    eval_answer_by_qid = _load_eval_answers(
        working_dir=working_dir,
        dataset_name=dataset_name,
        method=method,
    )
    predictions: List[Dict[str, Any]] = []

    for doc_uuid, doc_rows in _group_rows_by_doc(rows).items():
        result_dir = Path(working_dir) / doc_uuid / f"eval_{dataset_name}_{method}"
        final_results = _load_json(result_dir / "final_results.json")
        for query_idx, result in enumerate(final_results, start=1):
            question_id = str(
                result.get("qasper_question_id")
                or _get_index(doc_rows, query_idx - 1, {}).get("qasper_question_id")
                or ""
            )
            if not question_id:
                continue
            predicted_answer = canonicalize_qasper_answer(_prediction_answer(
                result=result,
                question_id=question_id,
                eval_answer_by_qid=eval_answer_by_qid,
                answer_source=answer_source,
            ))
            prediction = {
                "question_id": question_id,
                "predicted_answer": predicted_answer,
                "predicted_evidence": _prediction_evidence(
                    query_dir=result_dir / f"query_{query_idx:03d}",
                    paragraph_evidence_only=paragraph_evidence_only,
                    top_k_evidence=top_k_evidence,
                    answer_text=predicted_answer,
                    dynamic_evidence_topk=dynamic_evidence_topk,
                ),
            }
            predictions.append(prediction)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    _write_export_summary(
        output_path=output.parent / "export_summary.json",
        predictions=predictions,
        answer_source=answer_source,
        paragraph_evidence_only=paragraph_evidence_only,
        top_k_evidence=top_k_evidence,
        dynamic_evidence_topk=dynamic_evidence_topk,
    )
    coverage = validate_qasper_run(
        dataset_path=dataset_path,
        working_dir=working_dir,
        dataset_name=dataset_name,
        method=method,
        predictions_path=str(output),
        require_query_outputs=False,
    )
    coverage.update(
        {
            "complete": True,
            "dataset_sha256": _sha256_file(dataset_path),
            "predictions_sha256": _sha256_file(output),
        }
    )
    (output.parent / "coverage_manifest.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return predictions


def _write_export_summary(
    output_path: Path,
    predictions: List[Dict[str, Any]],
    answer_source: str,
    paragraph_evidence_only: bool,
    top_k_evidence: int,
    dynamic_evidence_topk: bool,
) -> None:
    evidence_counts = [
        len(prediction.get("predicted_evidence") or [])
        for prediction in predictions
    ]
    payload = {
        "num_predictions": len(predictions),
        "answer_source": answer_source,
        "paragraph_evidence_only": paragraph_evidence_only,
        "top_k_evidence": int(top_k_evidence or 0),
        "dynamic_evidence_topk": bool(dynamic_evidence_topk),
        "evidence_counts": evidence_counts,
        "evidence_count_distribution": dict(Counter(evidence_counts)),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_predictions_file(
    dataset_path: str,
    predictions_path: str,
    output_path: str = "",
    text_evidence_only: bool = False,
) -> Dict[str, Any]:
    rows = _load_json(dataset_path)
    gold = _gold_answers_and_evidence(rows, text_evidence_only=text_evidence_only)
    predicted = _load_predictions(predictions_path)
    scores = evaluate_qasper_official(gold, predicted)
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    return scores


def export_gold_for_official_evaluator(
    dataset_path: str,
    output_path: str,
) -> Dict[str, Any]:
    rows = _load_json(dataset_path)
    papers: Dict[str, Any] = {}
    for row in rows:
        doc_uuid = _doc_uuid_to_dir(row.get("doc_uuid"))
        paper = papers.setdefault(
            doc_uuid,
            {
                "title": row.get("title", ""),
                "abstract": "",
                "full_text": [],
                "figures_and_tables": [],
                "qas": [],
            },
        )
        paper["qas"].append(
            {
                "question_id": str(row.get("qasper_question_id") or row.get("question_id") or ""),
                "question": row.get("question", ""),
                "answers": [
                    {
                        "answer": answer,
                    }
                    for answer in row.get("answer", []) or []
                ],
            }
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(papers, ensure_ascii=False, indent=2), encoding="utf-8")
    return papers


def run_external_official_evaluator(
    evaluator_path: str,
    predictions_path: str,
    gold_path: str,
    output_path: str,
    text_evidence_only: bool = False,
) -> Dict[str, Any]:
    command = [
        sys.executable,
        evaluator_path,
        "--predictions",
        predictions_path,
        "--gold",
        gold_path,
    ]
    if text_evidence_only:
        command.append("--text_evidence_only")
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Official Qasper evaluator failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    scores = json.loads(result.stdout)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    return scores


def evaluate_qasper_official(
    gold: Dict[str, List[Dict[str, Any]]],
    predicted: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    max_answer_f1s = []
    max_evidence_f1s = []
    max_answer_f1s_by_type = {
        "extractive": [],
        "abstractive": [],
        "boolean": [],
        "none": [],
    }
    num_missing_predictions = 0

    for question_id, references in gold.items():
        if question_id not in predicted:
            num_missing_predictions += 1
            max_answer_f1s.append(0.0)
            max_evidence_f1s.append(0.0)
            continue

        prediction = predicted[question_id]
        answer_f1s_and_types = [
            (
                token_f1_score(prediction.get("answer", ""), reference.get("answer", "")),
                reference.get("type", "none"),
            )
            for reference in references
        ]
        max_answer_f1, answer_type = sorted(
            answer_f1s_and_types,
            key=lambda item: item[0],
            reverse=True,
        )[0]
        max_answer_f1s.append(max_answer_f1)
        max_answer_f1s_by_type.setdefault(answer_type, []).append(max_answer_f1)
        evidence_f1s = [
            paragraph_f1_score(
                prediction.get("evidence", []),
                reference.get("evidence", []),
            )
            for reference in references
        ]
        max_evidence_f1s.append(max(evidence_f1s) if evidence_f1s else 0.0)

    def mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "Answer F1": round(mean(max_answer_f1s), 6),
        "Answer F1 by type": {
            key: round(mean(value), 6)
            for key, value in max_answer_f1s_by_type.items()
        },
        "Evidence F1": round(mean(max_evidence_f1s), 6),
        "Missing predictions": num_missing_predictions,
    }


def run_export_and_eval(
    dataset_config_path: str,
    method: str,
    output_dir: str = "",
    answer_source: str = "output",
    text_evidence_only: bool = False,
    include_nonparagraph_evidence: bool = False,
    top_k_evidence: int = 0,
    dynamic_evidence_topk: bool = True,
) -> Dict[str, Any]:
    data_cfg = load_dataset_config(dataset_config_path)
    official_dir = Path(output_dir) if output_dir else Path(data_cfg.working_dir) / "0_results" / f"qasper_official_{method}"
    predictions_path = official_dir / "predictions.jsonl"
    scores_path = official_dir / "official_eval.json"
    export_predictions(
        dataset_path=data_cfg.dataset_path,
        working_dir=data_cfg.working_dir,
        dataset_name=data_cfg.dataset_name,
        method=method,
        output_path=str(predictions_path),
        answer_source=answer_source,
        paragraph_evidence_only=not include_nonparagraph_evidence,
        top_k_evidence=top_k_evidence,
        dynamic_evidence_topk=dynamic_evidence_topk,
    )
    scores = evaluate_predictions_file(
        dataset_path=data_cfg.dataset_path,
        predictions_path=str(predictions_path),
        output_path=str(scores_path),
        text_evidence_only=text_evidence_only,
    )
    print(json.dumps(scores, ensure_ascii=False, indent=2))
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved official evaluation to {scores_path}")
    return scores


def run_export_and_external_official_eval(
    dataset_config_path: str,
    method: str,
    output_dir: str = "",
    answer_source: str = "output",
    text_evidence_only: bool = False,
    include_nonparagraph_evidence: bool = False,
    top_k_evidence: int = 0,
    dynamic_evidence_topk: bool = True,
    official_evaluator_path: str = "",
) -> Dict[str, Any]:
    data_cfg = load_dataset_config(dataset_config_path)
    official_dir = Path(output_dir) if output_dir else Path(data_cfg.working_dir) / "0_results" / f"qasper_official_external_{method}"
    predictions_path = official_dir / "predictions.jsonl"
    gold_path = official_dir / "gold_official.json"
    evaluator_path = Path(official_evaluator_path) if official_evaluator_path else official_dir / "qasper_evaluator.py"
    scores_path = official_dir / "official_eval.json"

    export_predictions(
        dataset_path=data_cfg.dataset_path,
        working_dir=data_cfg.working_dir,
        dataset_name=data_cfg.dataset_name,
        method=method,
        output_path=str(predictions_path),
        answer_source=answer_source,
        paragraph_evidence_only=not include_nonparagraph_evidence,
        top_k_evidence=top_k_evidence,
        dynamic_evidence_topk=dynamic_evidence_topk,
    )
    export_gold_for_official_evaluator(
        dataset_path=data_cfg.dataset_path,
        output_path=str(gold_path),
    )
    if not evaluator_path.exists():
        _download_official_evaluator(str(evaluator_path))
    scores = run_external_official_evaluator(
        evaluator_path=str(evaluator_path),
        predictions_path=str(predictions_path),
        gold_path=str(gold_path),
        output_path=str(scores_path),
        text_evidence_only=text_evidence_only,
    )
    print(json.dumps(scores, ensure_ascii=False, indent=2))
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved official gold to {gold_path}")
    print(f"Used official evaluator at {evaluator_path}")
    print(f"Saved official evaluation to {scores_path}")
    return scores


def _gold_answers_and_evidence(
    rows: List[Dict[str, Any]],
    text_evidence_only: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    gold: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        question_id = str(row.get("qasper_question_id") or row.get("question_id") or "")
        if not question_id:
            continue
        gold[question_id] = [
            _reference_from_answer(answer, text_evidence_only=text_evidence_only)
            for answer in row.get("answer", []) or []
        ]
    return gold


def _reference_from_answer(answer_info: Dict[str, Any], text_evidence_only: bool) -> Dict[str, Any]:
    if answer_info.get("unanswerable"):
        return {"answer": "Unanswerable", "evidence": [], "type": "none"}
    if answer_info.get("extractive_spans"):
        answer = ", ".join(str(item) for item in answer_info.get("extractive_spans") or [])
        answer_type = "extractive"
    elif answer_info.get("free_form_answer"):
        answer = str(answer_info.get("free_form_answer"))
        answer_type = "abstractive"
    elif answer_info.get("yes_no") is True:
        answer = "Yes"
        answer_type = "boolean"
    elif answer_info.get("yes_no") is not None:
        answer = "No"
        answer_type = "boolean"
    else:
        answer = ""
        answer_type = "none"

    evidence = list(answer_info.get("evidence") or [])
    if text_evidence_only:
        evidence = [text for text in evidence if "FLOAT SELECTED" not in str(text)]
    return {"answer": answer, "evidence": evidence, "type": answer_type}


def _load_predictions(predictions_path: str) -> Dict[str, Dict[str, Any]]:
    predictions: Dict[str, Dict[str, Any]] = {}
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            predictions[str(data["question_id"])] = {
                "answer": canonicalize_qasper_answer(data.get("predicted_answer", "")),
                "evidence": data.get("predicted_evidence", []) or [],
            }
    return predictions


def _load_eval_answers(working_dir: str, dataset_name: str, method: str) -> Dict[str, str]:
    final_eval_path = Path(working_dir) / "0_results" / f"final_eval_{dataset_name}_{method}.json"
    if not final_eval_path.exists():
        return {}
    rows = _load_json(final_eval_path)
    answers = {}
    for row in rows:
        qid = row.get("qasper_question_id")
        pred = row.get("pred")
        if qid and pred and str(pred).lower() != "failed":
            answers[str(qid)] = str(pred)
    return answers


def _prediction_answer(
    result: Dict[str, Any],
    question_id: str,
    eval_answer_by_qid: Dict[str, str],
    answer_source: str,
) -> str:
    if answer_source == "pred":
        return eval_answer_by_qid.get(question_id, str(result.get("output", "")))
    if answer_source == "auto" and question_id in eval_answer_by_qid:
        return eval_answer_by_qid[question_id]
    if answer_source == "output" and result.get("answer_short"):
        return str(result.get("answer_short", ""))
    return str(result.get("output", ""))


def _prediction_evidence(
    query_dir: Path,
    paragraph_evidence_only: bool,
    top_k_evidence: int,
    answer_text: str = "",
    dynamic_evidence_topk: bool = False,
) -> List[str]:
    payload = _load_json_if_exists(query_dir / "retrieval_res.json")
    values = _evidence_values(payload)
    if not values:
        payload = _load_json_if_exists(query_dir / "evidence_chain.json")
        values = _evidence_values(payload)
    values = sorted(
        values,
        key=lambda item: item.get("supporting_rank", item.get("selection_rank", item.get("rank", 10**9)))
        if isinstance(item, dict)
        else 10**9,
    )
    evidence_limit = _evidence_limit(
        payload=payload,
        answer_text=answer_text,
        top_k_evidence=top_k_evidence,
        dynamic_evidence_topk=dynamic_evidence_topk,
    )
    if evidence_limit == 0:
        return []

    evidence = []
    tree_text_cache: Optional[Dict[int, str]] = None
    for item in values:
        if not isinstance(item, dict):
            continue
        expanded_texts: List[str] = []
        if _is_raptor_summary(item):
            if tree_text_cache is None:
                tree_text_cache = _load_tree_paragraph_texts(query_dir)
            expanded_texts = _raptor_child_evidence_texts(item, tree_text_cache)
        block_type = _item_block_type(item)
        if not expanded_texts:
            if paragraph_evidence_only and block_type not in (None, "", "paragraph"):
                continue
            text = (
                item.get("qasper_evidence_text")
                or item.get("text")
                or item.get("content")
                or item.get("evidence")
            )
            if text:
                expanded_texts = [str(text)]
        for text in expanded_texts:
            if text and text not in evidence:
                evidence.append(str(text))
            if evidence_limit > 0 and len(evidence) >= evidence_limit:
                break
        if evidence_limit > 0 and len(evidence) >= evidence_limit:
            break
    return evidence


def _item_block_type(item: Dict[str, Any]) -> Optional[str]:
    explicit = item.get("block_type")
    if explicit is not None:
        return str(explicit).strip().lower()
    node_type = str(item.get("node_type") or "").strip().lower()
    source = str(item.get("source") or "").strip().lower()
    if source == "raptor_summary" or node_type == "raptor_summary":
        return "summary"
    if source == "qasper_paragraph" or node_type in {"text", "paragraph"}:
        return "paragraph"
    return None


def _is_raptor_summary(item: Dict[str, Any]) -> bool:
    return (
        _item_block_type(item) == "summary"
        and bool(item.get("child_source_node_ids") or item.get("child_qasper_evidence_texts"))
    )


def _raptor_child_evidence_texts(
    item: Dict[str, Any],
    tree_text_cache: Dict[int, str],
) -> List[str]:
    inline_texts = _parse_inline_child_evidence_texts(item.get("child_qasper_evidence_texts"))
    if inline_texts:
        return inline_texts
    texts: List[str] = []
    for node_id in _parse_int_list(item.get("child_source_node_ids")):
        text = tree_text_cache.get(node_id)
        if text and text not in texts:
            texts.append(text)
    return texts


def _parse_inline_child_evidence_texts(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    raw = str(value)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    except Exception:
        pass
    return [item.strip() for item in raw.split("\n") if item.strip()]


def _parse_int_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, list):
        values = value
    else:
        values = re.findall(r"-?\d+", str(value))
    ids: List[int] = []
    for item in values:
        try:
            ids.append(int(item))
        except (TypeError, ValueError):
            continue
    return ids


def _load_tree_paragraph_texts(query_dir: Path) -> Dict[int, str]:
    doc_dir = query_dir.parent.parent
    try:
        from Core.Index.Tree import DocumentTree, NodeType

        tree_path = DocumentTree.get_save_path(str(doc_dir))
        if not Path(tree_path).exists():
            return {}
        tree = DocumentTree.load_from_file(tree_path)
    except Exception:
        return {}

    texts: Dict[int, str] = {}
    for node in tree.get_nodes(hasRoot=False):
        if getattr(node, "type", None) != NodeType.TEXT:
            continue
        text = str(getattr(node.meta_info, "content", "") or "").strip()
        if text:
            texts[int(node.index_id)] = text
    return texts


def _evidence_limit(
    payload: Any,
    answer_text: str,
    top_k_evidence: int,
    dynamic_evidence_topk: bool,
) -> int:
    if not dynamic_evidence_topk:
        return int(top_k_evidence) if top_k_evidence > 0 else -1
    dynamic_limit = _dynamic_evidence_topk(payload=payload, answer_text=answer_text)
    if top_k_evidence > 0:
        return min(int(top_k_evidence), dynamic_limit)
    return dynamic_limit


def _dynamic_evidence_topk(payload: Any, answer_text: str) -> int:
    normalized = normalize_answer(answer_text)
    if normalized in {"unanswerable", "not answerable", "not enough information"}:
        return 0
    demand = payload.get("demand", {}) if isinstance(payload, dict) else {}
    intent = str(demand.get("intent", "") if isinstance(demand, dict) else "").strip().lower()
    if normalized in {"yes", "no"} or intent == "boolean":
        return 3
    if intent in {"global-summary", "aggregation"}:
        return 2
    if len(normalized.split()) > 12:
        return 2
    return 3


def _evidence_values(payload: Any) -> List[Any]:
    if isinstance(payload, dict):
        return (
            payload.get("supporting_evidence")
            or payload.get("supporting")
            or payload.get("official_evidence")
            or payload.get("supporting_blocks")
            or payload.get("selected")
            or payload.get("evidence_chain")
            or payload.get("ranked_results")
            or payload.get("retrieval_results")
            or payload.get("nodes")
            or []
        )
    if isinstance(payload, list):
        return payload
    return []


def _group_rows_by_doc(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        doc_uuid = _doc_uuid_to_dir(row.get("doc_uuid"))
        groups.setdefault(doc_uuid, []).append(row)
    return groups


def _doc_uuid_to_dir(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _get_index(values: List[Any], idx: int, default: Any = None) -> Any:
    return values[idx] if 0 <= idx < len(values) else default


def _load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_json_if_exists(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    return _load_json(path)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_official_evaluator(output_path: str) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Downloading the official evaluator requires huggingface_hub. "
            "Install it with: pip install huggingface_hub"
        ) from exc
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id="albertgong1/qasper",
        repo_type="dataset",
        revision="a8de10174c66470ee25cd1a5af9f34a494b60ab6",
        filename="original_data/qasper-test-and-evaluator-v0.3/qasper_evaluator.py",
    )
    output.write_text(Path(downloaded).read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export BookRAG/EviBridge Qasper outputs to official Qasper format and evaluate them."
    )
    parser.add_argument("--dataset-config", required=True, help="Dataset YAML config used by the run.")
    parser.add_argument("--method", default="evibridge", help="Method suffix, e.g. evibridge.")
    parser.add_argument("--output-dir", default="", help="Directory for predictions.jsonl and official_eval.json.")
    parser.add_argument(
        "--answer-source",
        choices=["output", "pred", "auto"],
        default="output",
        help="Use raw RAG output, project-eval extracted pred, or pred when available.",
    )
    parser.add_argument(
        "--text-evidence-only",
        action="store_true",
        help="Ignore non-text gold evidence in official Evidence F1.",
    )
    parser.add_argument(
        "--include-nonparagraph-evidence",
        action="store_true",
        help="Export summary/patch/entity evidence too. Default exports paragraph evidence only.",
    )
    parser.add_argument(
        "--top-k-evidence",
        type=int,
        default=0,
        help="Keep only the first K exported evidence paragraphs. 0 means keep all.",
    )
    parser.add_argument(
        "--dynamic-evidence-topk",
        dest="dynamic_evidence_topk",
        action="store_true",
        default=True,
        help="Use a non-gold dynamic evidence count based on predicted answer and retrieval demand.",
    )
    parser.add_argument(
        "--fixed-evidence-topk",
        dest="dynamic_evidence_topk",
        action="store_false",
        help="Disable the shared dynamic evidence policy for diagnostic runs only.",
    )
    parser.add_argument(
        "--use-external-official-evaluator",
        action="store_true",
        help="Call the official qasper_evaluator.py script instead of the in-repo metric copy.",
    )
    parser.add_argument(
        "--official-evaluator-path",
        default="",
        help="Path to qasper_evaluator.py. If omitted with --use-external-official-evaluator, it is downloaded into output dir.",
    )
    args = parser.parse_args()

    if args.use_external_official_evaluator:
        run_export_and_external_official_eval(
            dataset_config_path=args.dataset_config,
            method=args.method,
            output_dir=args.output_dir,
            answer_source=args.answer_source,
            text_evidence_only=args.text_evidence_only,
            include_nonparagraph_evidence=args.include_nonparagraph_evidence,
            top_k_evidence=args.top_k_evidence,
            dynamic_evidence_topk=args.dynamic_evidence_topk,
            official_evaluator_path=args.official_evaluator_path,
        )
    else:
        run_export_and_eval(
            dataset_config_path=args.dataset_config,
            method=args.method,
            output_dir=args.output_dir,
            answer_source=args.answer_source,
            text_evidence_only=args.text_evidence_only,
            include_nonparagraph_evidence=args.include_nonparagraph_evidence,
            top_k_evidence=args.top_k_evidence,
            dynamic_evidence_topk=args.dynamic_evidence_topk,
        )


if __name__ == "__main__":
    main()
