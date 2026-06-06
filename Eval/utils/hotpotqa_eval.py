import json
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from Core.configs.dataset_config import DatasetConfig


YES_NO_NOANSWER = {"yes", "no", "noanswer"}


def normalize_answer(text: Any) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(str(text or "").lower())))


def answer_f1_score(prediction: Any, ground_truth: Any) -> Tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    if normalized_prediction in YES_NO_NOANSWER and normalized_prediction != normalized_ground_truth:
        return 0.0, 0.0, 0.0
    if normalized_ground_truth in YES_NO_NOANSWER and normalized_prediction != normalized_ground_truth:
        return 0.0, 0.0, 0.0
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction: Any, ground_truth: Any) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def supporting_fact_scores(
    predicted: Iterable[Iterable[Any]],
    gold: Iterable[Iterable[Any]],
) -> Tuple[float, float, float, float]:
    pred_set = {_fact_tuple(item) for item in predicted if _valid_fact(item)}
    gold_set = {_fact_tuple(item) for item in gold if _valid_fact(item)}
    em = float(pred_set == gold_set)
    if not pred_set and not gold_set:
        return em, 1.0, 1.0, 1.0
    if not pred_set or not gold_set:
        return em, 0.0, 0.0, 0.0
    true_positive = len(pred_set & gold_set)
    precision = true_positive / len(pred_set)
    recall = true_positive / len(gold_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return em, f1, precision, recall


def evaluate_hotpotqa_predictions(
    gold_rows: List[Dict[str, Any]],
    predictions: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    answer_em = []
    answer_f1 = []
    answer_precision = []
    answer_recall = []
    sp_em = []
    sp_f1 = []
    sp_precision = []
    sp_recall = []
    joint_em = []
    joint_f1 = []
    joint_precision = []
    joint_recall = []
    missing = 0

    for row in gold_rows:
        qid = _question_id(row)
        pred = predictions.get(qid)
        if pred is None:
            pred = {"answer": "", "sp": []}
            missing += 1
        pred_answer = pred.get("answer", "")
        gold_answer = row.get("hotpot_answer", row.get("answer", ""))
        ans_em = exact_match_score(pred_answer, gold_answer)
        ans_f1, ans_prec, ans_rec = answer_f1_score(pred_answer, gold_answer)
        gold_sp = row.get("hotpot_supporting_facts") or []
        pred_sp = pred.get("sp") or []
        fact_em, fact_f1, fact_prec, fact_rec = supporting_fact_scores(pred_sp, gold_sp)

        j_em = ans_em * fact_em
        j_prec = ans_prec * fact_prec
        j_rec = ans_rec * fact_rec
        j_f1 = 0.0 if j_prec + j_rec == 0 else 2 * j_prec * j_rec / (j_prec + j_rec)

        answer_em.append(ans_em)
        answer_f1.append(ans_f1)
        answer_precision.append(ans_prec)
        answer_recall.append(ans_rec)
        sp_em.append(fact_em)
        sp_f1.append(fact_f1)
        sp_precision.append(fact_prec)
        sp_recall.append(fact_rec)
        joint_em.append(j_em)
        joint_f1.append(j_f1)
        joint_precision.append(j_prec)
        joint_recall.append(j_rec)

    return {
        "answer_em": _mean(answer_em),
        "answer_f1": _mean(answer_f1),
        "answer_precision": _mean(answer_precision),
        "answer_recall": _mean(answer_recall),
        "sp_em": _mean(sp_em),
        "sp_f1": _mean(sp_f1),
        "sp_precision": _mean(sp_precision),
        "sp_recall": _mean(sp_recall),
        "joint_em": _mean(joint_em),
        "joint_f1": _mean(joint_f1),
        "joint_precision": _mean(joint_precision),
        "joint_recall": _mean(joint_recall),
        "missing_predictions": missing,
        "total_samples": len(gold_rows),
    }


def eval_hotpotqa(
    data_df: pd.DataFrame | List[Dict[str, Any]],
    data_cfg: DatasetConfig,
    method: str,
    max_workers: int = 1,
) -> Dict[str, Any]:
    rows = data_df.to_dict(orient="records") if hasattr(data_df, "to_dict") else list(data_df)
    predictions: Dict[str, Dict[str, Any]] = {}
    detailed_rows: List[Dict[str, Any]] = []

    for row in tqdm(rows, desc="Evaluating HotpotQA"):
        doc_uuid = str(row.get("doc_uuid"))
        qid = _question_id(row)
        result_dir = Path(data_cfg.working_dir) / doc_uuid / f"eval_{data_cfg.dataset_name}_{method}"
        final_result = _load_matching_final_result(result_dir / "final_results.json", qid)
        pred_answer = _prediction_answer(final_result)
        pred_sp = _prediction_supporting_facts(result_dir, final_result, row)
        predictions[qid] = {"answer": pred_answer, "sp": pred_sp}
        detail = {
            **row,
            "pred": pred_answer,
            "pred_supporting_facts": pred_sp,
        }
        detailed_rows.append(detail)

    scores = evaluate_hotpotqa_predictions(rows, predictions)
    for detail in detailed_rows:
        qid = _question_id(detail)
        single_scores = evaluate_hotpotqa_predictions([detail], {qid: predictions[qid]})
        detail.update(single_scores)

    save_dir = Path(data_cfg.working_dir) / "0_results"
    save_dir.mkdir(parents=True, exist_ok=True)
    detail_path = save_dir / f"final_eval_{data_cfg.dataset_name}_{method}.json"
    score_path = save_dir / f"final_eval_{data_cfg.dataset_name}_{method}.score.json"
    detail_path.write_text(json.dumps(detailed_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    score_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(scores, ensure_ascii=False, indent=2))
    print(f"Saved detailed results to {detail_path}")
    return scores


def _prediction_answer(result: Dict[str, Any]) -> str:
    for key in ("answer_short", "pred", "output"):
        value = result.get(key)
        if value:
            return str(value)
    return ""


def _prediction_supporting_facts(
    result_dir: Path,
    result: Dict[str, Any],
    gold_row: Dict[str, Any],
) -> List[List[Any]]:
    query_idx = int(result.get("_query_index", 1))
    retrieval_payload = _load_json_if_exists(result_dir / f"query_{query_idx:03d}" / "retrieval_res.json")
    for payload in (retrieval_payload, result):
        facts = _facts_from_payload(payload)
        if facts:
            return facts

    node_fact_map = {
        str(key): value
        for key, value in (gold_row.get("hotpot_node_facts") or {}).items()
    }
    predicted_ids = []
    for key in ("supporting_block_ids", "retrieved_block_ids", "retrieved_node_ids"):
        predicted_ids.extend(str(item) for item in result.get(key) or [])
    facts = []
    for block_id in predicted_ids:
        fact = node_fact_map.get(str(block_id))
        if fact and fact not in facts:
            facts.append(fact)
    return facts


def _facts_from_payload(payload: Any) -> List[List[Any]]:
    if not isinstance(payload, dict):
        return []
    values = (
        payload.get("supporting_evidence")
        or payload.get("selected")
        or payload.get("ranked_results")
        or []
    )
    facts = []
    for item in values:
        if not isinstance(item, dict):
            continue
        title = item.get("title") or item.get("hotpot_title") or item.get("section_id")
        sent_id = item.get("sent_id")
        if sent_id is None:
            sent_id = item.get("hotpot_sent_id")
        if title is not None and sent_id is not None:
            fact = [str(title), int(sent_id)]
            if fact not in facts:
                facts.append(fact)
    return facts


def _load_matching_final_result(final_path: Path, qid: str) -> Dict[str, Any]:
    final_results = _load_json_if_exists(final_path)
    if not isinstance(final_results, list):
        return {}
    for index, result in enumerate(final_results, start=1):
        if not isinstance(result, dict):
            continue
        if _question_id(result) == qid:
            result = dict(result)
            result["_query_index"] = index
            return result
    if len(final_results) == 1 and isinstance(final_results[0], dict):
        result = dict(final_results[0])
        result["_query_index"] = 1
        return result
    return {}


def _question_id(row: Dict[str, Any]) -> str:
    return str(row.get("hotpotqa_question_id") or row.get("question_id") or row.get("id") or "").strip()


def _valid_fact(item: Iterable[Any]) -> bool:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return False
    try:
        int(item[1])
    except (TypeError, ValueError):
        return False
    return bool(str(item[0]).strip())


def _fact_tuple(item: Iterable[Any]) -> Tuple[str, int]:
    values = list(item)
    return str(values[0]).strip(), int(values[1])


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(float(np.mean(values)), 6)


def _load_json_if_exists(path: str | Path) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
