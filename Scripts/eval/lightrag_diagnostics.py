import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Core.configs.dataset_config import load_dataset_config


class LightRAGDiagnosticError(ValueError):
    """Raised when a LightRAG graph-enhanced run is not valid as a graph baseline."""


def collect_lightrag_diagnostics(result_dirs: Iterable[str | Path]) -> Dict[str, Any]:
    query_summaries: List[Dict[str, Any]] = []
    final_results = 0
    documents = 0
    for result_dir in result_dirs:
        path = Path(result_dir)
        if not path.exists():
            continue
        documents += 1
        final_path = path / "final_results.json"
        if final_path.exists():
            try:
                payload = _load_json(final_path)
                if isinstance(payload, list):
                    final_results += len(payload)
            except Exception:
                pass
        for query_dir in sorted(path.glob("query_*")):
            if not query_dir.is_dir():
                continue
            retrieval_path = query_dir / "retrieval_res.json"
            if not retrieval_path.exists():
                query_summaries.append(
                    {
                        "query_dir": str(query_dir),
                        "missing_retrieval_res": True,
                    }
                )
                continue
            try:
                retrieval = _load_json(retrieval_path)
            except Exception as exc:
                query_summaries.append(
                    {
                        "query_dir": str(query_dir),
                        "missing_retrieval_res": True,
                        "error": str(exc),
                    }
                )
                continue
            graph = retrieval.get("graph_diagnostics") if isinstance(retrieval, dict) else {}
            graph = graph if isinstance(graph, dict) else {}
            supporting = retrieval.get("supporting_evidence") if isinstance(retrieval, dict) else []
            ranked = retrieval.get("ranked_results") if isinstance(retrieval, dict) else []
            query_summaries.append(
                {
                    "query_dir": str(query_dir),
                    "mode": str(retrieval.get("mode") or ""),
                    "effective_mode": str(retrieval.get("effective_mode") or ""),
                    "fallback_used": bool(retrieval.get("fallback_used")),
                    "ranked_results": len(ranked) if isinstance(ranked, list) else 0,
                    "supporting_evidence": len(supporting) if isinstance(supporting, list) else 0,
                    "chunks": _int(graph.get("chunks")),
                    "entities": _int(graph.get("entities")),
                    "relationships": _int(graph.get("relationships")),
                    "graph_nodes": _int(graph.get("graph_nodes")),
                    "graph_ready": graph.get("graph_ready") is True,
                }
            )

    total = len(query_summaries)
    mode_counts = Counter(item.get("effective_mode", "") for item in query_summaries if item.get("effective_mode"))
    summary = {
        "documents": documents,
        "total_queries": total,
        "final_results_questions": final_results,
        "graph_ready_queries": sum(1 for item in query_summaries if item.get("graph_ready")),
        "fallback_used_queries": sum(1 for item in query_summaries if item.get("fallback_used")),
        "missing_retrieval_res_queries": sum(1 for item in query_summaries if item.get("missing_retrieval_res")),
        "mode_counts": dict(mode_counts),
        "avg_chunks": _avg(item.get("chunks", 0) for item in query_summaries),
        "avg_entities": _avg(item.get("entities", 0) for item in query_summaries),
        "avg_relationships": _avg(item.get("relationships", 0) for item in query_summaries),
        "avg_graph_nodes": _avg(item.get("graph_nodes", 0) for item in query_summaries),
        "avg_ranked_results": _avg(item.get("ranked_results", 0) for item in query_summaries),
        "avg_supporting_evidence": _avg(item.get("supporting_evidence", 0) for item in query_summaries),
        "queries": query_summaries,
    }
    return summary


def validate_lightrag_graph_ready(summary: Dict[str, Any]) -> Dict[str, Any]:
    total = int(summary.get("total_queries") or 0)
    graph_ready = int(summary.get("graph_ready_queries") or 0)
    fallback = int(summary.get("fallback_used_queries") or 0)
    missing = int(summary.get("missing_retrieval_res_queries") or 0)
    issues = []
    if total <= 0:
        issues.append("no LightRAG queries were found")
    if graph_ready != total:
        issues.append(f"graph-ready queries {graph_ready} != total queries {total}")
    if fallback:
        issues.append(f"fallback used in {fallback} query/query results")
    if missing:
        issues.append(f"missing retrieval_res.json in {missing} query/query results")
    if issues:
        raise LightRAGDiagnosticError("LightRAG graph diagnostic failed: " + "; ".join(issues))
    return summary


def result_dirs_from_dataset_config(dataset_config: str, method: str) -> List[Path]:
    cfg = load_dataset_config(dataset_config)
    rows = _load_json(cfg.dataset_path)
    if not isinstance(rows, list):
        return []
    doc_ids = []
    seen = set()
    for row in rows:
        doc_uuid = _doc_uuid_to_dir(row.get("doc_uuid"))
        if doc_uuid and doc_uuid not in seen:
            seen.add(doc_uuid)
            doc_ids.append(doc_uuid)
    return [Path(cfg.working_dir) / doc_uuid / f"eval_{cfg.dataset_name}_{method}" for doc_uuid in doc_ids]


def format_markdown(summary: Dict[str, Any]) -> str:
    total = max(int(summary.get("total_queries") or 0), 1)
    graph_rate = float(summary.get("graph_ready_queries") or 0) / total
    fallback_rate = float(summary.get("fallback_used_queries") or 0) / total
    lines = [
        "| Metric | Value |",
        "|---|---:|",
        f"| Documents | {summary.get('documents', 0)} |",
        f"| Queries | {summary.get('total_queries', 0)} |",
        f"| Final results questions | {summary.get('final_results_questions', 0)} |",
        f"| Graph-ready queries | {summary.get('graph_ready_queries', 0)} ({graph_rate:.2%}) |",
        f"| Fallback-used queries | {summary.get('fallback_used_queries', 0)} ({fallback_rate:.2%}) |",
        f"| Missing retrieval_res queries | {summary.get('missing_retrieval_res_queries', 0)} |",
        f"| Avg chunks | {summary.get('avg_chunks', 0):.2f} |",
        f"| Avg entities | {summary.get('avg_entities', 0):.2f} |",
        f"| Avg relationships | {summary.get('avg_relationships', 0):.2f} |",
        f"| Avg graph nodes | {summary.get('avg_graph_nodes', 0):.2f} |",
        f"| Avg ranked results | {summary.get('avg_ranked_results', 0):.2f} |",
        f"| Avg supporting evidence | {summary.get('avg_supporting_evidence', 0):.2f} |",
    ]
    mode_counts = summary.get("mode_counts") or {}
    if mode_counts:
        modes = ", ".join(f"{mode}: {count}" for mode, count in sorted(mode_counts.items()))
        lines.append(f"| Effective modes | {modes} |")
    return "\n".join(lines)


def _avg(values: Iterable[Any]) -> float:
    numbers = [float(value or 0) for value in values]
    return sum(numbers) / len(numbers) if numbers else 0.0


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _doc_uuid_to_dir(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LightRAG graph-enhanced run diagnostics.")
    parser.add_argument("--dataset-config", default="", help="Dataset YAML config used by the run.")
    parser.add_argument("--method", default="lightrag", help="Method suffix, usually lightrag.")
    parser.add_argument(
        "--result-dir",
        action="append",
        default=[],
        help="One eval_*_lightrag directory. Can be provided multiple times.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--markdown-output", type=Path, help="Optional markdown output path.")
    parser.add_argument(
        "--require-graph-ready",
        action="store_true",
        help="Fail unless every query has graph_ready=true and fallback_used=false.",
    )
    args = parser.parse_args()

    result_dirs: Sequence[str | Path] = args.result_dir
    if args.dataset_config:
        result_dirs = [*result_dirs, *result_dirs_from_dataset_config(args.dataset_config, args.method)]
    if not result_dirs:
        raise SystemExit("Provide --dataset-config or at least one --result-dir.")

    summary = collect_lightrag_diagnostics(result_dirs)
    if args.require_graph_ready:
        validate_lightrag_graph_ready(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(format_markdown(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
