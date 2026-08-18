from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "answer_em",
    "answer_f1",
    "sp_precision",
    "sp_recall",
    "sp_f1",
    "joint_f1",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fact(item: dict[str, Any]) -> tuple[str, int]:
    title = item.get("hotpot_title", item.get("title"))
    sent_id = item.get("hotpot_sent_id", item.get("sent_id"))
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"missing HotpotQA title in retrieved item {item.get('id')!r}")
    try:
        parsed_sent_id = int(sent_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid HotpotQA sentence ID in item {item.get('id')!r}") from exc
    return title, parsed_sent_id


def _node_id(item: dict[str, Any]) -> int:
    try:
        return int(item["id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("retrieved item has no valid integer node ID") from exc


def build_summary(
    *,
    work_root: Path,
    run_root: Path,
    manifest_path: Path,
    config_path: Path,
    expected_questions: int = 1000,
    bootstrap_path: Path | None = None,
) -> dict[str, Any]:
    work_root = Path(work_root)
    run_root = Path(run_root)
    score_path = work_root / "0_results" / "final_eval_hotpotqa_lead_only.score.json"
    audit_path = run_root / "audit_lead_only.json"
    score = _load(score_path)
    audit = _load(audit_path)
    manifest = _load(Path(manifest_path))

    if not audit.get("complete"):
        raise ValueError("Lead-only coverage audit is incomplete")
    if int(audit.get("completed_questions", -1)) != expected_questions:
        raise ValueError("Lead-only audit question count does not match the fixed dataset")
    if int(audit.get("completed_documents", -1)) != expected_questions:
        raise ValueError("Lead-only audit document count does not match the fixed dataset")
    if int(score.get("total_samples", -1)) != expected_questions:
        raise ValueError("official score sample count does not match the fixed dataset")
    if int(score.get("missing_predictions", -1)) != 0:
        raise ValueError("official score reports missing Lead-only predictions")

    document_dirs = [
        path
        for path in work_root.iterdir()
        if path.is_dir() and path.name not in {"0_results", "_logs"}
    ]
    eval_roots = [
        path / "eval_hotpotqa_lead_only"
        for path in document_dirs
        if (path / "eval_hotpotqa_lead_only").is_dir()
    ]
    if len(eval_roots) != expected_questions:
        raise ValueError(
            f"expected {expected_questions} Lead-only document outputs, found {len(eval_roots)}"
        )

    total_tokens = 0.0
    total_seconds = 0.0
    ranked_count = 0
    supporting_count = 0
    for eval_root in eval_roots:
        result_path = eval_root / "query_001" / "result.json"
        retrieval_path = eval_root / "query_001" / "retrieval_res.json"
        token_path = eval_root / "token_cost.json"
        for required in (result_path, retrieval_path, token_path):
            if not required.exists():
                raise FileNotFoundError(required)

        result = _load(result_path)
        retrieval = _load(retrieval_path)
        node_facts = result.get("hotpot_node_facts") or {}
        ranked = retrieval.get("ranked_results") or []
        supporting = retrieval.get("supporting_evidence") or []
        if not ranked:
            raise ValueError(f"Lead-only produced empty ranked results: {result_path}")

        ranked_ids: set[int] = set()
        for item in ranked:
            node_id = _node_id(item)
            title, sent_id = _fact(item)
            if sent_id != 0:
                raise ValueError(f"non-lead sentence retrieved for node {node_id}")
            mapped = node_facts.get(str(node_id), node_facts.get(node_id))
            if not isinstance(mapped, list) or len(mapped) < 2:
                raise ValueError(f"node {node_id} has no original HotpotQA fact mapping")
            if str(mapped[0]) != title or int(mapped[1]) != sent_id:
                raise ValueError(f"node {node_id} title--sentence mapping mismatch")
            ranked_ids.add(node_id)
            ranked_count += 1

        for item in supporting:
            node_id = _node_id(item)
            _, sent_id = _fact(item)
            if node_id not in ranked_ids or sent_id != 0:
                raise ValueError(f"supporting fact {node_id} is outside Lead-only candidates")
            supporting_count += 1

        retrieved_ids = {
            int(item)
            for item in (result.get("retrieved_node_ids") or result.get("retrieved_block_ids") or [])
        }
        submitted_ids = {int(item) for item in (result.get("supporting_block_ids") or [])}
        if not retrieved_ids or not retrieved_ids.issubset(ranked_ids):
            raise ValueError(f"result retrieved IDs do not match Lead-only candidates: {result_path}")
        if not submitted_ids or not submitted_ids.issubset(ranked_ids):
            raise ValueError(f"result supporting IDs do not match Lead-only candidates: {result_path}")

        token_cost = _load(token_path)
        total_tokens += float((token_cost.get("rag_cost") or {}).get("total_tokens") or 0.0)
        total_seconds += float(token_cost.get("time") or 0.0)

    metrics = {key: float(score[key]) for key in METRIC_KEYS}
    summary = {
        "dataset": "HotpotQA distractor validation fixed-1000",
        "method": "Lead-only",
        "method_suffix": "lead_only",
        "dataset_sha256": manifest.get("unified_sha256"),
        "coverage": {
            "questions": int(audit["completed_questions"]),
            "documents": int(audit["completed_documents"]),
            "missing_predictions": int(score["missing_predictions"]),
        },
        "metrics": metrics,
        "efficiency": {
            "tokens_per_question": total_tokens / expected_questions,
            "seconds_per_question": total_seconds / expected_questions,
            "question_count": expected_questions,
        },
        "provenance": {
            "validated_ranked_leads": ranked_count,
            "validated_supporting_facts": supporting_count,
            "all_ranked_sentence_ids": 0,
        },
        "sources": {
            "config": str(Path(config_path).resolve()),
            "manifest": str(Path(manifest_path).resolve()),
            "score": str(score_path.resolve()),
            "audit": str(audit_path.resolve()),
            "bootstrap": str(Path(bootstrap_path).resolve()) if bootstrap_path else None,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    efficiency = summary["efficiency"]
    coverage = summary["coverage"]
    return "\n".join(
        [
            "# HotpotQA Lead-only fixed-1000 Results",
            "",
            "| Answer EM | Answer F1 | SP P | SP R | SP F1 | Joint F1 |",
            "|---:|---:|---:|---:|---:|---:|",
            "| "
            + " | ".join(f"{100.0 * metrics[key]:.2f}" for key in METRIC_KEYS)
            + " |",
            "",
            "| Coverage | Tokens/Question | Seconds/Question |",
            "|---:|---:|---:|",
            f"| {coverage['questions']}/1000 | {efficiency['tokens_per_question']:.1f} | "
            f"{efficiency['seconds_per_question']:.2f} |",
            "",
            f"Dataset SHA-256: `{summary['dataset_sha256']}`",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--bootstrap")
    parser.add_argument("--expected-questions", type=int, default=1000)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    summary = build_summary(
        work_root=Path(args.work_root),
        run_root=Path(args.run_root),
        manifest_path=Path(args.manifest),
        config_path=Path(args.config),
        expected_questions=args.expected_questions,
        bootstrap_path=Path(args.bootstrap) if args.bootstrap else None,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_md.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps({"state": "complete", "questions": args.expected_questions}))


if __name__ == "__main__":
    main()
