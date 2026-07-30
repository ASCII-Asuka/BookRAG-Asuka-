import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_metric(payload: Dict[str, Any], *names: str) -> Optional[float]:
    lowered = {str(key).lower().replace(" ", "_"): value for key, value in payload.items()}
    for name in names:
        key = name.lower().replace(" ", "_")
        if key in lowered:
            try:
                return float(lowered[key])
            except (TypeError, ValueError):
                return None
    return None


def _read_missing(payload: Dict[str, Any]) -> Optional[int]:
    value = _read_metric(payload, "Missing predictions", "missing_predictions", "missing")
    return int(value) if value is not None else None


def _method_from_path(path: Path) -> str:
    name = path.parent.name
    for prefix in ["qasper_official_", "official_"]:
        if name.startswith(prefix):
            name = name[len(prefix) :]
    for suffix in ["_output", "_pred"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name or path.parent.name


def parse_eval_spec(spec: str) -> Tuple[Optional[str], Path]:
    if "=" in spec:
        method, path = spec.split("=", 1)
        return method.strip() or None, Path(path.strip())
    return None, Path(spec)


def collect_rows(
    eval_specs: Iterable[str],
    allow_missing: bool = False,
    require_manifests: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    expected_policy: Optional[Dict[str, Any]] = None
    for spec in eval_specs:
        method, path = parse_eval_spec(spec)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        coverage_path = path.parent / "coverage_manifest.json"
        export_summary_path = path.parent / "export_summary.json"
        if require_manifests and (
            not coverage_path.exists() or not export_summary_path.exists()
        ):
            raise ValueError(
                f"{path} is missing coverage_manifest.json or export_summary.json. "
                "Refusing to build a paper table without reproducibility manifests."
            )
        policy = None
        if coverage_path.exists():
            coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
            if not coverage.get("complete"):
                raise ValueError(f"{coverage_path} does not mark the run complete.")
            expected = coverage.get("expected_questions")
            predicted = coverage.get("predictions_questions")
            if expected is not None and predicted is not None and int(expected) != int(predicted):
                raise ValueError(
                    f"{coverage_path} reports expected_questions={expected}, "
                    f"predictions_questions={predicted}."
                )
        if export_summary_path.exists():
            export_summary = json.loads(
                export_summary_path.read_text(encoding="utf-8")
            )
            policy = {
                key: export_summary.get(key)
                for key in (
                    "answer_source",
                    "paragraph_evidence_only",
                    "top_k_evidence",
                    "dynamic_evidence_topk",
                )
            }
            if expected_policy is None:
                expected_policy = policy
            elif policy != expected_policy:
                raise ValueError(
                    "All methods in a paper table must use a uniform export policy; "
                    f"expected {expected_policy}, got {policy} for {path}."
                )
        missing = _read_missing(payload)
        if not allow_missing and missing not in (None, 0):
            raise ValueError(
                f"{path} reports Missing predictions={missing}. "
                "Refusing to generate a paper table from a partial run."
            )
        rows.append(
            {
                "Method": method or _method_from_path(path),
                "Answer F1": _read_metric(payload, "Answer F1", "answer_f1"),
                "Evidence F1": _read_metric(payload, "Evidence F1", "evidence_f1"),
                "Missing": missing,
                "Path": str(path),
                "Export Policy": policy,
            }
        )
    if (
        require_manifests
        and expected_policy is not None
        and expected_policy.get("dynamic_evidence_topk") is not True
    ):
        raise ValueError(
            "Paper-table export policy must set dynamic_evidence_topk=true "
            "for every method."
        )
    return rows


def collect_from_root(
    root: Path,
    allow_missing: bool = False,
    require_manifests: bool = True,
) -> List[Dict[str, Any]]:
    specs = [str(path) for path in sorted(root.rglob("official_eval.json"))]
    return collect_rows(
        specs,
        allow_missing=allow_missing,
        require_manifests=require_manifests,
    )


def format_markdown_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| Method | Answer F1 | Evidence F1 | Missing |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        answer = "" if row["Answer F1"] is None else f"{row['Answer F1']:.4f}"
        evidence = "" if row["Evidence F1"] is None else f"{row['Evidence F1']:.4f}"
        missing = "" if row["Missing"] is None else str(row["Missing"])
        lines.append(f"| {row['Method']} | {answer} | {evidence} | {missing} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Qasper official_eval.json files into a baseline table."
    )
    parser.add_argument(
        "--eval",
        action="append",
        default=[],
        help="Evaluation spec. Use either path/to/official_eval.json or Method=path.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="Directory to recursively scan for official_eval.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional markdown output path.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional JSON output path for the parsed rows.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow official_eval.json files with Missing predictions > 0.",
    )
    parser.add_argument(
        "--allow-legacy-manifests",
        action="store_true",
        help="Allow legacy result directories without coverage/export manifests.",
    )
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    if args.root:
        rows.extend(
            collect_from_root(
                args.root,
                allow_missing=args.allow_missing,
                require_manifests=not args.allow_legacy_manifests,
            )
        )
    if args.eval:
        rows.extend(
            collect_rows(
                args.eval,
                allow_missing=args.allow_missing,
                require_manifests=not args.allow_legacy_manifests,
            )
        )
    if not rows:
        raise SystemExit("Provide at least one --eval or --root.")

    table = format_markdown_table(rows)
    print(table)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table + "\n", encoding="utf-8")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
