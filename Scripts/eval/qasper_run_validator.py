import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Core.configs.dataset_config import load_dataset_config


class QasperRunValidationError(ValueError):
    """Raised when a Qasper run is incomplete or cannot be safely evaluated."""


def validate_qasper_run(
    dataset_path: str,
    working_dir: str,
    dataset_name: str,
    method: str,
    predictions_path: str = "",
    gold_path: str = "",
    require_query_outputs: bool = True,
) -> Dict[str, Any]:
    rows = _load_json(dataset_path)
    if not isinstance(rows, list):
        raise QasperRunValidationError(f"Dataset must be a list: {dataset_path}")

    expected_qids = _question_ids_from_rows(rows)
    groups = _group_rows_by_doc(rows)
    issues: List[str] = []
    final_qids: List[str] = []

    for doc_uuid, doc_rows in groups.items():
        result_dir = Path(working_dir) / doc_uuid / f"eval_{dataset_name}_{method}"
        final_path = result_dir / "final_results.json"
        if not final_path.exists():
            issues.append(f"{doc_uuid}: missing final_results.json at {final_path}")
            continue
        try:
            final_results = _load_json(final_path)
        except Exception as exc:
            issues.append(f"{doc_uuid}: cannot load final_results.json: {exc}")
            continue
        if not isinstance(final_results, list):
            issues.append(f"{doc_uuid}: final_results.json is not a list")
            continue
        if len(final_results) != len(doc_rows):
            issues.append(
                f"{doc_uuid}: final_results count {len(final_results)} != dataset count {len(doc_rows)}"
            )
        for index, row in enumerate(doc_rows, start=1):
            if require_query_outputs:
                result_path = result_dir / f"query_{index:03d}" / "result.json"
                if not result_path.exists():
                    issues.append(f"{doc_uuid}: missing query result {result_path}")
            result = final_results[index - 1] if index - 1 < len(final_results) else {}
            qid = _question_id(result) or _question_id(row)
            if qid:
                final_qids.append(qid)

    prediction_qids: List[str] = []
    if predictions_path:
        prediction_qids = _prediction_question_ids(Path(predictions_path), issues)
        _compare_id_sets(
            expected=set(expected_qids),
            observed=set(prediction_qids),
            observed_name="prediction",
            issues=issues,
        )
        _append_duplicate_issues(prediction_qids, "prediction", issues)

    gold_qids: List[str] = []
    if gold_path:
        gold_qids = _gold_question_ids(Path(gold_path), issues)
        _compare_id_sets(
            expected=set(expected_qids),
            observed=set(gold_qids),
            observed_name="gold",
            issues=issues,
        )
        _append_duplicate_issues(gold_qids, "gold", issues)

    _compare_id_sets(
        expected=set(expected_qids),
        observed=set(final_qids),
        observed_name="final_results",
        issues=issues,
    )
    _append_duplicate_issues(final_qids, "final_results", issues)
    _append_duplicate_issues(expected_qids, "dataset", issues)

    summary = {
        "dataset_path": str(dataset_path),
        "working_dir": str(working_dir),
        "dataset_name": dataset_name,
        "method": method,
        "expected_questions": len(expected_qids),
        "final_results_questions": len(final_qids),
        "predictions_questions": len(prediction_qids) if predictions_path else None,
        "gold_questions": len(gold_qids) if gold_path else None,
        "documents": len(groups),
        "issues": issues,
    }
    if issues:
        raise QasperRunValidationError(_format_issues(issues))
    return summary


def validate_prediction_coverage(
    gold_question_ids: Iterable[str],
    predictions_path: str,
    label: str = "prediction",
) -> Dict[str, Any]:
    issues: List[str] = []
    expected = set(str(item) for item in gold_question_ids if str(item))
    observed_qids = _prediction_question_ids(Path(predictions_path), issues)
    _compare_id_sets(expected, set(observed_qids), label, issues)
    _append_duplicate_issues(observed_qids, label, issues)
    summary = {
        "expected_questions": len(expected),
        "predictions_questions": len(observed_qids),
        "issues": issues,
    }
    if issues:
        raise QasperRunValidationError(_format_issues(issues))
    return summary


def _question_ids_from_rows(rows: Sequence[Dict[str, Any]]) -> List[str]:
    qids = []
    for row in rows:
        qid = _question_id(row)
        if qid:
            qids.append(qid)
    return qids


def _group_rows_by_doc(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        doc_uuid = _doc_uuid_to_dir(row.get("doc_uuid"))
        groups.setdefault(doc_uuid, []).append(row)
    return groups


def _doc_uuid_to_dir(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _question_id(item: Dict[str, Any]) -> str:
    return str(item.get("qasper_question_id") or item.get("question_id") or "").strip()


def _prediction_question_ids(path: Path, issues: List[str]) -> List[str]:
    if not path.exists():
        issues.append(f"missing predictions file: {path}")
        return []
    qids = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(f"{path}:{line_no}: invalid JSONL row: {exc}")
            continue
        qid = str(item.get("question_id") or "").strip()
        if not qid:
            issues.append(f"{path}:{line_no}: missing question_id")
            continue
        qids.append(qid)
    return qids


def _gold_question_ids(path: Path, issues: List[str]) -> List[str]:
    if not path.exists():
        issues.append(f"missing gold file: {path}")
        return []
    try:
        payload = _load_json(path)
    except Exception as exc:
        issues.append(f"cannot load gold file {path}: {exc}")
        return []
    qids = []
    if isinstance(payload, dict):
        for paper in payload.values():
            if not isinstance(paper, dict):
                continue
            for qa in paper.get("qas", []) or []:
                qid = str(qa.get("question_id") or "").strip()
                if qid:
                    qids.append(qid)
    return qids


def _compare_id_sets(
    expected: Set[str],
    observed: Set[str],
    observed_name: str,
    issues: List[str],
) -> None:
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing:
        issues.append(f"missing {observed_name} question ids: {missing[:10]}")
    if extra:
        issues.append(f"extra {observed_name} question ids: {extra[:10]}")


def _append_duplicate_issues(qids: Sequence[str], label: str, issues: List[str]) -> None:
    seen = set()
    duplicates = []
    for qid in qids:
        if qid in seen and qid not in duplicates:
            duplicates.append(qid)
        seen.add(qid)
    if duplicates:
        issues.append(f"duplicate {label} question ids: {duplicates[:10]}")


def _format_issues(issues: Sequence[str]) -> str:
    preview = "\n".join(f"- {item}" for item in issues[:20])
    suffix = "" if len(issues) <= 20 else f"\n... and {len(issues) - 20} more issues"
    return f"Qasper run validation failed with {len(issues)} issue(s):\n{preview}{suffix}"


def _load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate that a Qasper run is complete before evaluation.")
    parser.add_argument("--dataset-config", required=True, help="Dataset YAML config used by the run.")
    parser.add_argument("--method", required=True, help="Method suffix, e.g. evibridge.")
    parser.add_argument("--predictions", default="", help="Optional predictions.jsonl to validate.")
    parser.add_argument("--gold", default="", help="Optional official gold JSON to validate.")
    parser.add_argument(
        "--allow-missing-query-outputs",
        action="store_true",
        help="Only validate final_results.json, not per-query result.json files.",
    )
    args = parser.parse_args()
    data_cfg = load_dataset_config(args.dataset_config)
    summary = validate_qasper_run(
        dataset_path=data_cfg.dataset_path,
        working_dir=data_cfg.working_dir,
        dataset_name=data_cfg.dataset_name,
        method=args.method,
        predictions_path=args.predictions,
        gold_path=args.gold,
        require_query_outputs=not args.allow_missing_query_outputs,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
