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


def collect_rows(eval_specs: Iterable[str], allow_missing: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in eval_specs:
        method, path = parse_eval_spec(spec)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
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
            }
        )
    return rows


def collect_from_root(root: Path, allow_missing: bool = False) -> List[Dict[str, Any]]:
    specs = [str(path) for path in sorted(root.rglob("official_eval.json"))]
    return collect_rows(specs, allow_missing=allow_missing)


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
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    if args.root:
        rows.extend(collect_from_root(args.root, allow_missing=args.allow_missing))
    if args.eval:
        rows.extend(collect_rows(args.eval, allow_missing=args.allow_missing))
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
