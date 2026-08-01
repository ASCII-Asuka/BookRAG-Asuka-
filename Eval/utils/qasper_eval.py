"""
Official script for evaluating models built for the Qasper dataset. The script
outputs Answer F1 and Evidence F1 reported in the paper.
"""

from collections import Counter
import string
import re
import json

import os
from typing import Any
import pandas as pd
import numpy as np
from tqdm import tqdm

from Core.configs.dataset_config import DatasetConfig
from Eval.utils.extract_answer import AnswerExtractor, load_prompt
from Eval.utils.utils import get_all_cost

from concurrent.futures import ThreadPoolExecutor
from itertools import repeat  # Helper to pass constant arguments to map


def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def token_f1_score(prediction, ground_truth):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def paragraph_f1_score(prediction, ground_truth):
    if not ground_truth and not prediction:
        # The question is unanswerable and the prediction is empty.
        return 1.0
    num_same = len(set(ground_truth).intersection(set(prediction)))
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_answers_and_evidence(qa_info: list[dict[Any]], text_evidence_only: bool):
    references = []
    for answer_info in qa_info:
        if answer_info["unanswerable"]:
            references.append(
                {
                    "answer": "Not answerable",
                    "evidence": [],
                    "type": "none",
                    "answer_raw": "Not answerable",
                }
            )
        else:
            if answer_info["extractive_spans"]:
                answer = ", ".join(answer_info["extractive_spans"])
                answer_type = "extractive"
                answer_raw = answer_info["extractive_spans"]
            elif answer_info["free_form_answer"]:
                answer = answer_info["free_form_answer"]
                answer_type = "abstractive"
                answer_raw = answer_info["free_form_answer"]
            elif answer_info["yes_no"]:
                answer = "Yes"
                answer_type = "boolean"
                answer_raw = "Yes"
            elif answer_info["yes_no"] is not None:
                answer = "No"
                answer_type = "boolean"
                answer_raw = "No"

            if text_evidence_only:
                evidence = [
                    text
                    for text in answer_info["evidence"]
                    if "FLOAT SELECTED" not in text
                ]
            else:
                evidence = answer_info["evidence"]
            references.append(
                {
                    "answer": answer,
                    "evidence": evidence,
                    "type": answer_type,
                    "answer_raw": answer_raw,
                    "evidence_block_ids": answer_info.get("evidence_block_ids", []),
                    "evidence_paragraph_ids": answer_info.get("evidence_paragraph_ids", []),
                    "evidence_ids": answer_info.get("evidence_ids", []),
                }
            )

    return references


def get_accuracy(prediction, ground_truth: list[str]):
    for ground_truth in ground_truth:
        norm_pred = normalize_answer(prediction)
        norm_ans = normalize_answer(ground_truth)
        if norm_ans in norm_pred:
            return 1
    return 0


def eval_single_res(pred, gold_answer: list):
    # return accuracy, f1

    accuracy_score = 0.0
    f1_score = 0.0
    for gold in gold_answer:
        answer_raw = gold.get("answer_raw", "")
        if isinstance(answer_raw, str):
            answer_raw = [answer_raw]
        if isinstance(answer_raw, int):
            answer_raw = [str(answer_raw)]
        acc = get_accuracy(pred, answer_raw)
        accuracy_score = max(accuracy_score, acc)

        f1 = token_f1_score(pred, gold.get("answer", ""))
        f1_score = max(f1_score, f1)

    return accuracy_score, f1_score


def _normalize_evidence_text(text: Any) -> str:
    return normalize_answer(str(text or ""))


def _load_json_if_exists(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _evidence_texts_from_payload(payload: Any) -> list[str]:
    if payload is None:
        return []
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = (
            payload.get("evidence_chain")
            or payload.get("selected")
            or payload.get("ranked_results")
            or payload.get("retrieval_results")
            or payload.get("nodes")
            or []
        )
    else:
        values = []
    texts = []
    for item in values:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            text = (
                item.get("qasper_evidence_text")
                or item.get("text")
                or item.get("content")
                or item.get("evidence")
            )
            if text:
                texts.append(str(text))
    return texts


def _evidence_ids_from_payload(payload: Any) -> list[str]:
    if payload is None:
        return []
    ids = []
    values = []
    if isinstance(payload, dict):
        for key in ("retrieved_block_ids", "retrieved_node_ids"):
            for item in payload.get(key) or []:
                ids.append(str(item))
        values = (
            payload.get("evidence_chain")
            or payload.get("selected")
            or payload.get("ranked_results")
            or payload.get("retrieval_results")
            or payload.get("nodes")
            or []
        )
    elif isinstance(payload, list):
        values = payload
    for item in values:
        if not isinstance(item, dict):
            continue
        for key in ("block_id", "node_id", "paragraph_id", "evidence_id"):
            if item.get(key) is not None:
                ids.append(str(item[key]))
    return list(dict.fromkeys(ids))


def _bridge_types_from_payload(payload: Any) -> set[str]:
    if payload is None:
        return set()
    values = []
    if isinstance(payload, dict):
        values = payload.get("evidence_chain") or payload.get("selected") or []
    elif isinstance(payload, list):
        values = payload
    bridge_types = set()
    for item in values:
        if not isinstance(item, dict):
            continue
        for bridge_type in item.get("bridge_types") or []:
            bridge_types.add(str(bridge_type))
    return bridge_types


def _verification_from_payload(payload: Any) -> dict:
    if isinstance(payload, dict) and isinstance(payload.get("verification"), dict):
        return payload["verification"]
    return {}


def _gold_evidence_ids(gold_answers: list[dict]) -> list[str]:
    ids = []
    for answer in gold_answers:
        for key in ("evidence_block_ids", "evidence_paragraph_ids", "evidence_ids"):
            for item in answer.get(key) or []:
                ids.append(str(item))
    return list(dict.fromkeys(ids))


def _id_prf_score(
    predicted_ids: list[str], gold_ids: list[str]
) -> tuple[float, float, float]:
    pred_set = set(predicted_ids)
    gold_set = set(gold_ids)
    if not pred_set and not gold_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not gold_set:
        return 0.0, 0.0, 0.0
    matched = pred_set & gold_set
    precision = len(matched) / len(pred_set)
    recall = len(matched) / len(gold_set)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    return round(precision, 6), round(recall, 6), round(f1, 6)


def _evidence_scores(
    predicted_texts: list[str],
    gold_answers: list[dict],
    predicted_ids: list[str] | None = None,
) -> tuple[float, float, float]:
    gold_id_sets = [_gold_evidence_ids([answer]) for answer in gold_answers]
    if predicted_ids is not None and any(gold_id_sets):
        scores = [_id_prf_score(predicted_ids, gold_ids) for gold_ids in gold_id_sets]
        return max(enumerate(scores), key=lambda item: (item[1][2], -item[0]))[1]

    pred_norm = [_normalize_evidence_text(text) for text in predicted_texts if _normalize_evidence_text(text)]
    scores = []
    for answer in gold_answers or [{}]:
        gold_norm = [
            _normalize_evidence_text(text)
            for text in answer.get("evidence") or []
            if _normalize_evidence_text(text)
        ]
        if not gold_norm and not pred_norm:
            scores.append((1.0, 1.0, 1.0))
            continue
        if not gold_norm or not pred_norm:
            scores.append((0.0, 0.0, 0.0))
            continue
        matched_gold = set()
        matched_pred = set()
        for pred_idx, pred in enumerate(pred_norm):
            for gold_idx, gold in enumerate(gold_norm):
                if gold in pred or pred in gold:
                    matched_gold.add(gold_idx)
                    matched_pred.add(pred_idx)
        precision = len(matched_pred) / len(pred_norm)
        recall = len(matched_gold) / len(gold_norm)
        f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
        scores.append((round(precision, 6), round(recall, 6), round(f1, 6)))
    return max(enumerate(scores), key=lambda item: (item[1][2], -item[0]))[1]


def _load_evibridge_metrics(res_path: str, query_idx: int, gold_answers: list[dict]) -> dict:
    query_dir = os.path.join(res_path, f"query_{query_idx + 1:03d}")
    chain_payload = _load_json_if_exists(os.path.join(query_dir, "evidence_chain.json"))
    retrieval_payload = _load_json_if_exists(os.path.join(query_dir, "retrieval_res.json"))
    predicted_texts = _evidence_texts_from_payload(chain_payload)
    if not predicted_texts:
        predicted_texts = _evidence_texts_from_payload(retrieval_payload)
    predicted_ids = _evidence_ids_from_payload(chain_payload)
    if not predicted_ids:
        predicted_ids = _evidence_ids_from_payload(retrieval_payload)
    evidence_precision, evidence_recall, evidence_f1 = _evidence_scores(
        predicted_texts, gold_answers, predicted_ids
    )
    verification = _verification_from_payload(chain_payload) or _verification_from_payload(retrieval_payload)
    bridge_types = _bridge_types_from_payload(chain_payload) | _bridge_types_from_payload(retrieval_payload)
    iterations = retrieval_payload.get("iterations", []) if isinstance(retrieval_payload, dict) else []
    return {
        "evidence_f1": evidence_f1,
        "evidence_precision": evidence_precision,
        "evidence_recall": evidence_recall,
        "path_connectivity": round(float(verification.get("connectivity", 0.0)), 6)
        if verification
        else None,
        "noise_ratio": round(float(verification.get("noise", 0.0)), 6)
        if verification
        else None,
        "bridge_coverage": round(len(bridge_types & {"context", "semantic", "hierarchy"}) / 3, 6),
        "verifier_iterations": len(iterations) if iterations else (1 if verification else 0),
    }


def _doc_result_dir(data_cfg: DatasetConfig, doc_uuid: Any, method: str) -> str:
    dir_name = f"eval_{data_cfg.dataset_name}_{method}"
    return os.path.join(data_cfg.working_dir, str(doc_uuid), dir_name)


def eval_single_file(res_path: str, extractor: AnswerExtractor):
    res_file = os.path.join(res_path, "final_results.json")
    with open(res_file, "r", encoding="utf-8") as f:
        res_data = json.load(f)

    for query_idx, item in enumerate(res_data):
        question = item["question"]
        output = item["output"]
        gold_answers = get_answers_and_evidence(item["answer"], text_evidence_only=True)
        correct_answer = str(gold_answers[0].get("answer", ""))
        item['gold_answers'] = gold_answers
        extracted_res, pred_ans, pred_format, llm_score = extractor.extract(
            question, output, correct_answer
        )
        item["extracted_res"] = extracted_res
        item['pred'] = pred_ans
        item["pred_format"] = pred_format
        item["llm_score"] = llm_score
        
        acc, f1 = eval_single_res(pred_ans, gold_answers)
        item["acc"] = acc
        item["f1"] = f1
        item.update(_load_evibridge_metrics(res_path, query_idx, gold_answers))

    # Save results to output_dir
    save_path = os.path.join(res_path, "eval.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(res_data, f, ensure_ascii=False, indent=2)

    return res_data


def eval_qasper(
    data_df: pd.DataFrame,
    data_cfg: DatasetConfig,
    method: str,
    max_workers=4,
    api_config_path: str | None = None,
):
    document_groups = data_df.groupby(["doc_uuid", "doc_path"])

    extractor = AnswerExtractor(api_config_path=api_config_path)
    result = []

    if max_workers > 1:
        # Step 1: Prepare the arguments for all the function calls. This is very fast.
        # We create a list of the 'doc_res_dir' paths that will be processed.
        doc_res_dirs = []
        for (doc_uuid, doc_path), group in document_groups:
            doc_res_dir = _doc_result_dir(data_cfg, doc_uuid, method)
            doc_res_dirs.append(doc_res_dir)

        # Step 2: Execute `eval_single_file` in parallel using ThreadPoolExecutor.map
        # .map handles running the function on each item in the `doc_res_dirs` list.
        # `repeat(extractor)` and `repeat(prompt)` pass the same extractor and prompt
        # object to every function call.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # The `map` function returns results in the same order as the input iterable.
            # We wrap the iterator with tqdm for a progress bar.
            results_iterator = executor.map(
                eval_single_file,
                doc_res_dirs,  # The iterable of first arguments
                repeat(extractor),  # The constant second argument
            )

            # Step 3: Combine the results. Because .map preserves order, we can
            # simply loop through and extend our final list.
            for doc_res in tqdm(
                results_iterator, total=len(doc_res_dirs), desc="Processing Documents"
            ):
                result.extend(doc_res)
    else:
        for (doc_uuid, doc_path), group in tqdm(document_groups):
            doc_res_dir = _doc_result_dir(data_cfg, doc_uuid, method)

            doc_res = eval_single_file(doc_res_dir, extractor)
            result.extend(doc_res)

    average_acc = np.mean([item["acc"] for item in result])
    average_f1 = np.mean([item["f1"] for item in result])
    average_acc = round(average_acc, 6)
    average_f1 = round(average_f1, 6)
    avg_llm_score = np.mean(
        [item["llm_score"] for item in result if "llm_score" in item]
    )
    avg_llm_score = round(avg_llm_score, 6)
    print("--------------------------------------")
    print(f"total samples: {len(result)}")
    print(f"Avg acc: {average_acc:.6f}")
    print(f"Avg f1: {average_f1:.6f}")
    print(f"Avg llm_score: {avg_llm_score:.6f}")
    score_dict = {
        "Avg acc": average_acc,
        "Avg f1": average_f1,
        "Avg llm_score": avg_llm_score,
        "Total samples": len(result),
    }
    evibridge_metric_keys = [
        "evidence_f1",
        "evidence_precision",
        "evidence_recall",
        "path_connectivity",
        "noise_ratio",
        "bridge_coverage",
        "verifier_iterations",
    ]
    for key in evibridge_metric_keys:
        values = [item[key] for item in result if item.get(key) is not None]
        if values:
            score_dict[f"Avg {key}"] = round(float(np.mean(values)), 6)
    
    # answerable average score
    answerable_acc = []
    answerable_f1 = []
    answerable_llm_score = []
    for item in result:
        gold_answers = item.get('gold_answers', [])
        if gold_answers and gold_answers[0].get("answer", "") != "Not answerable":
            answerable_acc.append(item['acc'])
            answerable_f1.append(item['f1'])
            answerable_llm_score.append(item['llm_score'])
    acc_2 = np.mean(answerable_acc) if len(answerable_acc) > 0 else 0.0
    f1_2 = np.mean(answerable_f1) if len(answerable_f1) > 0 else 0.0
    avg_llm_score_2 = np.mean(answerable_llm_score) if len(answerable_llm_score)>0 else 0.0
    avg_llm_score_2 = round(avg_llm_score_2, 6)
    print("------- Answerable answer result --------")
    print(f"total answerable samples: {len(answerable_acc)}")
    print(f"Avg acc: {acc_2:.6f}")
    print(f"Avg f1: {f1_2:.6f}")
    print(f"Avg llm_score: {avg_llm_score_2:.6f}")
    score_dict["Answerable Avg acc"] = acc_2
    score_dict["Answerable Avg f1"] = f1_2
    score_dict["Answerable Avg llm_score"] = avg_llm_score_2

    cost_dict = get_all_cost(data_df, data_cfg, method)
    for k, v in cost_dict.items():
        if k not in score_dict:
            score_dict[k] = v

    save_dir = os.path.join(data_cfg.working_dir, "0_results")
    os.makedirs(save_dir, exist_ok=True)

    priority_keys = [
        "question",
        "answer",
        "pred",
        "acc",
        "f1",
        "llm_score",
        "evidence_f1",
        "evidence_precision",
        "evidence_recall",
        "path_connectivity",
        "noise_ratio",
        "bridge_coverage",
        "verifier_iterations",
        "extracted_res",
        "output",
    ]

    # 重新排序每个字典，将优先字段放在前面
    sorted_result = []
    for item in result:
        sorted_item = {k: item[k] for k in priority_keys if k in item}
        sorted_item.update({k: v for k, v in item.items() if k not in priority_keys})
        sorted_result.append(sorted_item)

    save_path = os.path.join(
        save_dir, f"final_eval_{data_cfg.dataset_name}_{method}.json"
    )
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(sorted_result, f, ensure_ascii=False, indent=2)

    score_save_path = os.path.join(
        save_dir, f"final_eval_{data_cfg.dataset_name}_{method}.score.json"
    )
    with open(score_save_path, "w", encoding="utf-8") as f:
        json.dump(score_dict, f, ensure_ascii=False, indent=2)
    print(f"Saved detailed results to {save_path}")
