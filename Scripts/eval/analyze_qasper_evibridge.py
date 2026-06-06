import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.eval.qasper_official import (
    evaluate_qasper_official,
    paragraph_f1_score,
    token_f1_score,
)
from Scripts.eval.qasper_run_validator import validate_prediction_coverage


def analyze_qasper_runs(
    gold_path: str,
    bm25_predictions_path: str,
    evibridge_predictions_path: str,
    baseline_predictions: Dict[str, str] | None = None,
    topk_values: Iterable[int] = (1, 2, 3, 4, 5, 10),
    output_path: str = "",
) -> Dict[str, Any]:
    gold_data = _load_json(gold_path)
    gold = _gold_by_question(gold_data)
    bm25 = _load_predictions(bm25_predictions_path)
    evibridge = _load_predictions(evibridge_predictions_path)
    predictions_by_method: Dict[str, Dict[str, Dict[str, Any]]] = {"bm25": bm25}
    for method, path in (baseline_predictions or {}).items():
        method_name = str(method).strip()
        if method_name and method_name not in predictions_by_method:
            predictions_by_method[method_name] = _load_predictions(path)
    predictions_by_method["evibridge"] = evibridge
    for method, predictions_path in {
        "bm25": bm25_predictions_path,
        **(baseline_predictions or {}),
        "evibridge": evibridge_predictions_path,
    }.items():
        validate_prediction_coverage(
            gold_question_ids=gold.keys(),
            predictions_path=predictions_path,
            label=str(method),
        )
    comparison = _comparison(gold, bm25, evibridge)
    for method, predictions in predictions_by_method.items():
        if method == "evibridge":
            continue
        comparison[f"{method}_vs_evibridge"] = _comparison_named(
            gold=gold,
            baseline_name=method,
            baseline=predictions,
            evibridge=evibridge,
        )
    report = {
        "overall": {
            "num_questions": len(gold),
            **{
                method: evaluate_qasper_official(gold, predictions)
                for method, predictions in predictions_by_method.items()
            },
        },
        "by_answer_type": _by_answer_type(gold, bm25, evibridge),
        "by_answer_type_all": _by_answer_type_all(gold, predictions_by_method),
        "comparison": comparison,
        "topk_curve": {
            method: _topk_curve(gold, predictions, topk_values)
            for method, predictions in predictions_by_method.items()
        },
        "loss_cases": _loss_cases(gold, bm25, evibridge),
    }
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _load_predictions(path: str) -> Dict[str, Dict[str, Any]]:
    predictions: Dict[str, Dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        predictions[str(item["question_id"])] = {
            "answer": str(item.get("predicted_answer", "")),
            "evidence": list(item.get("predicted_evidence") or []),
        }
    return predictions


def _gold_by_question(gold_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    gold: Dict[str, List[Dict[str, Any]]] = {}
    for paper in gold_data.values():
        for qa in paper.get("qas", []):
            references = []
            for annotation in qa.get("answers", []):
                answer = annotation.get("answer", {})
                if answer.get("unanswerable"):
                    references.append({"answer": "Unanswerable", "evidence": [], "type": "none"})
                    continue
                if answer.get("extractive_spans"):
                    answer_text = ", ".join(answer["extractive_spans"])
                    answer_type = "extractive"
                elif answer.get("free_form_answer"):
                    answer_text = answer["free_form_answer"]
                    answer_type = "abstractive"
                elif answer.get("yes_no") is True:
                    answer_text = "Yes"
                    answer_type = "boolean"
                elif answer.get("yes_no") is False:
                    answer_text = "No"
                    answer_type = "boolean"
                else:
                    answer_text = ""
                    answer_type = "none"
                references.append(
                    {
                        "answer": answer_text,
                        "evidence": [
                            item for item in (answer.get("evidence") or []) if "FLOAT SELECTED" not in item
                        ],
                        "type": answer_type,
                    }
                )
            gold[str(qa.get("question_id", ""))] = references
    return gold


def _score_one(prediction: Dict[str, Any], references: List[Dict[str, Any]]) -> Dict[str, Any]:
    answer_scores = [
        (token_f1_score(prediction.get("answer", ""), reference["answer"]), reference["type"])
        for reference in references
    ]
    evidence_scores = [
        paragraph_f1_score(prediction.get("evidence", []), reference["evidence"])
        for reference in references
    ]
    answer_f1, answer_type = max(answer_scores, key=lambda item: item[0]) if answer_scores else (0.0, "none")
    evidence_f1 = max(evidence_scores) if evidence_scores else 0.0
    return {"answer_f1": answer_f1, "evidence_f1": evidence_f1, "type": answer_type}


def _comparison(
    gold: Dict[str, List[Dict[str, Any]]],
    bm25: Dict[str, Dict[str, Any]],
    evibridge: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    answer_wins = {"bm25": 0, "evibridge": 0, "tie": 0}
    evidence_wins = {"bm25": 0, "evibridge": 0, "tie": 0}
    for qid, references in gold.items():
        bm25_score = _score_one(bm25.get(qid, {"answer": "", "evidence": []}), references)
        evibridge_score = _score_one(evibridge.get(qid, {"answer": "", "evidence": []}), references)
        _count_win(answer_wins, bm25_score["answer_f1"], evibridge_score["answer_f1"])
        _count_win(evidence_wins, bm25_score["evidence_f1"], evibridge_score["evidence_f1"])
    return {"answer_wins": answer_wins, "evidence_wins": evidence_wins}


def _comparison_named(
    gold: Dict[str, List[Dict[str, Any]]],
    baseline_name: str,
    baseline: Dict[str, Dict[str, Any]],
    evibridge: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    answer_wins = {baseline_name: 0, "evibridge": 0, "tie": 0}
    evidence_wins = {baseline_name: 0, "evibridge": 0, "tie": 0}
    for qid, references in gold.items():
        baseline_score = _score_one(baseline.get(qid, {"answer": "", "evidence": []}), references)
        evibridge_score = _score_one(evibridge.get(qid, {"answer": "", "evidence": []}), references)
        _count_win_named(answer_wins, baseline_name, baseline_score["answer_f1"], evibridge_score["answer_f1"])
        _count_win_named(
            evidence_wins,
            baseline_name,
            baseline_score["evidence_f1"],
            evibridge_score["evidence_f1"],
        )
    return {"answer_wins": answer_wins, "evidence_wins": evidence_wins}


def _by_answer_type(
    gold: Dict[str, List[Dict[str, Any]]],
    bm25: Dict[str, Dict[str, Any]],
    evibridge: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for qid, references in gold.items():
        bm25_score = _score_one(bm25.get(qid, {"answer": "", "evidence": []}), references)
        evibridge_score = _score_one(evibridge.get(qid, {"answer": "", "evidence": []}), references)
        answer_type = evibridge_score["type"]
        groups[answer_type].append({"bm25": bm25_score, "evibridge": evibridge_score})
    report = {}
    for answer_type, rows in groups.items():
        report[answer_type] = {
            "count": len(rows),
            "bm25_answer_f1": _mean(row["bm25"]["answer_f1"] for row in rows),
            "evibridge_answer_f1": _mean(row["evibridge"]["answer_f1"] for row in rows),
            "bm25_evidence_f1": _mean(row["bm25"]["evidence_f1"] for row in rows),
            "evibridge_evidence_f1": _mean(row["evibridge"]["evidence_f1"] for row in rows),
        }
    return report


def _by_answer_type_all(
    gold: Dict[str, List[Dict[str, Any]]],
    predictions_by_method: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for qid, references in gold.items():
        method_scores = {
            method: _score_one(predictions.get(qid, {"answer": "", "evidence": []}), references)
            for method, predictions in predictions_by_method.items()
        }
        answer_type = method_scores.get("evibridge", next(iter(method_scores.values())))["type"]
        groups[answer_type].append(method_scores)
    report: Dict[str, Any] = {}
    for answer_type, rows in groups.items():
        type_report: Dict[str, Any] = {"count": len(rows)}
        for method in predictions_by_method:
            type_report[f"{method}_answer_f1"] = _mean(row[method]["answer_f1"] for row in rows)
            type_report[f"{method}_evidence_f1"] = _mean(row[method]["evidence_f1"] for row in rows)
        report[answer_type] = type_report
    return report


def _topk_curve(
    gold: Dict[str, List[Dict[str, Any]]],
    predictions: Dict[str, Dict[str, Any]],
    topk_values: Iterable[int],
) -> Dict[str, Any]:
    curve = {}
    for topk in topk_values:
        trimmed = {
            qid: {
                "answer": prediction.get("answer", ""),
                "evidence": list(prediction.get("evidence") or [])[: int(topk)],
            }
            for qid, prediction in predictions.items()
        }
        curve[str(topk)] = evaluate_qasper_official(gold, trimmed)
    return curve


def _loss_cases(
    gold: Dict[str, List[Dict[str, Any]]],
    bm25: Dict[str, Dict[str, Any]],
    evibridge: Dict[str, Dict[str, Any]],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    cases = []
    for qid, references in gold.items():
        bm25_score = _score_one(bm25.get(qid, {"answer": "", "evidence": []}), references)
        evibridge_score = _score_one(evibridge.get(qid, {"answer": "", "evidence": []}), references)
        delta = (
            evibridge_score["answer_f1"]
            + evibridge_score["evidence_f1"]
            - bm25_score["answer_f1"]
            - bm25_score["evidence_f1"]
        )
        if delta < 0:
            cases.append(
                {
                    "question_id": qid,
                    "answer_type": evibridge_score["type"],
                    "delta": delta,
                    "bm25": bm25_score,
                    "evibridge": evibridge_score,
                }
            )
    return sorted(cases, key=lambda item: item["delta"])[:limit]


def _count_win(counter: Dict[str, int], bm25_score: float, evibridge_score: float) -> None:
    if evibridge_score > bm25_score:
        counter["evibridge"] += 1
    elif bm25_score > evibridge_score:
        counter["bm25"] += 1
    else:
        counter["tie"] += 1


def _count_win_named(
    counter: Dict[str, int],
    baseline_name: str,
    baseline_score: float,
    evibridge_score: float,
) -> None:
    if evibridge_score > baseline_score:
        counter["evibridge"] += 1
    elif baseline_score > evibridge_score:
        counter[baseline_name] += 1
    else:
        counter["tie"] += 1


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Qasper official-format baseline outputs.")
    parser.add_argument("--gold", required=True, help="Official gold JSON path.")
    parser.add_argument("--bm25-predictions", required=True, help="BM25 predictions.jsonl path.")
    parser.add_argument("--evibridge-predictions", required=True, help="EviBridge predictions.jsonl path.")
    parser.add_argument(
        "--baseline-predictions",
        action="append",
        default=[],
        help="Additional baseline spec in METHOD=path/to/predictions.jsonl form, e.g. raptor=...",
    )
    parser.add_argument("--output", default="", help="Optional JSON report output path.")
    parser.add_argument("--topk", default="1,2,3,4,5,10", help="Comma-separated evidence top-k values.")
    args = parser.parse_args()
    topk_values = [int(item) for item in args.topk.split(",") if item.strip()]
    extra_baselines = {}
    for spec in args.baseline_predictions:
        if "=" not in spec:
            raise SystemExit("--baseline-predictions must use METHOD=PATH.")
        method, path = spec.split("=", 1)
        extra_baselines[method.strip()] = path.strip()
    report = analyze_qasper_runs(
        gold_path=args.gold,
        bm25_predictions_path=args.bm25_predictions,
        evibridge_predictions_path=args.evibridge_predictions,
        baseline_predictions=extra_baselines,
        topk_values=topk_values,
        output_path=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
