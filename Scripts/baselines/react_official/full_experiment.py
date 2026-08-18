from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .common import load_json, write_json
from .diagnostics import summarize_manifest
from .finalize import finalize_runs
from .model_client import CompletionClient
from .run import run_prepared


@dataclass(frozen=True)
class PrivateCredentials:
    model_name: str
    api_key: str
    completion_url: str


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _load_private_yaml(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    visited = set(seen or set())
    if resolved in visited:
        raise ValueError("circular private configuration inheritance")
    visited.add(resolved)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("private configuration must be a YAML mapping")
    current = dict(payload)
    parent = current.pop("extends", None)
    if not parent:
        return current
    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    return _deep_merge(_load_private_yaml(parent_path, visited), current)


def _chat_completion_url(base_url: str) -> str:
    base = str(base_url or "").rstrip("/")
    if not base:
        raise ValueError("private LLM API base is empty")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/completions"):
        return base[: -len("/completions")] + "/chat/completions"
    return base + "/chat/completions"


def load_private_credentials(path: str | Path) -> PrivateCredentials:
    config = _load_private_yaml(Path(path))
    llm = dict(config.get("llm") or {})
    credentials = PrivateCredentials(
        model_name=str(llm.get("model_name") or ""),
        api_key=str(llm.get("api_key") or ""),
        completion_url=_chat_completion_url(str(llm.get("api_base") or "")),
    )
    if not credentials.model_name or not credentials.api_key:
        raise ValueError("private LLM model name and API key are required")
    return credentials


def _key_value_pairs(values: list[str], label: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use dataset=path syntax")
        dataset, path = value.split("=", 1)
        name = dataset.strip().lower()
        if name not in {"qasper", "hotpotqa"} or name in parsed:
            raise ValueError(f"invalid or duplicate {label} dataset: {name!r}")
        parsed[name] = Path(path).resolve()
    if set(parsed) != {"qasper", "hotpotqa"}:
        raise ValueError(f"{label} requires qasper and hotpotqa entries")
    return parsed


def _health_check(client: CompletionClient, model_name: str) -> None:
    text, _usage = client.complete(
        model_name,
        "Return one short ReAct action: Finish[ok]",
        "\n",
        temperature=0.0,
        max_tokens=20,
    )
    if not text.strip():
        raise RuntimeError("model service health check returned empty text")


def _run_evaluator(
    command: list[str], output_directory: Path, label: str
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / f"{label}.stdout.log").open(
        "w", encoding="utf-8"
    ) as stdout, (output_directory / f"{label}.stderr.log").open(
        "w", encoding="utf-8"
    ) as stderr:
        subprocess.run(command, check=True, stdout=stdout, stderr=stderr, text=True)


def run_full_experiment(
    *,
    stage: str,
    prepared_roots: Mapping[str, Path],
    private_configs: Mapping[str, Path],
    public_config: Path,
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    status_path = output_root / "full_experiment_status.json"
    output_root.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {"status": "running", "stage": "starting", "datasets": {}}
    write_json(status_path, status)
    max_scopes = {"qasper": 2, "hotpotqa": 4} if stage == "smoke" else {}
    try:
        for dataset in ("qasper", "hotpotqa"):
            status["stage"] = f"{dataset}_health_check"
            write_json(status_path, status)
            credentials = load_private_credentials(private_configs[dataset])
            client = CompletionClient(
                api_key=credentials.api_key,
                base_url=credentials.completion_url,
                timeout=180,
                max_attempts=3,
                retry_backoff_seconds=2,
            )
            _health_check(client, credentials.model_name)
            status["stage"] = f"{dataset}_run"
            write_json(status_path, status)
            run_summary = run_prepared(
                prepared_roots[dataset],
                public_config,
                output_root / stage / dataset,
                completion_model=client,
                source_root=source_root,
                max_scopes=max_scopes.get(dataset),
            )
            status["datasets"][dataset] = {"run": run_summary}
            status["stage"] = f"{dataset}_finalize"
            write_json(status_path, status)
            finalize_summary = finalize_runs(run_summary["manifest_path"], public_config)
            diagnostic = summarize_manifest(run_summary["manifest_path"])
            write_json(
                Path(run_summary["run_directory"]) / "diagnostics.json", diagnostic
            )
            status["datasets"][dataset].update(
                {"finalize": finalize_summary, "diagnostics": diagnostic}
            )
            write_json(status_path, status)

            if stage == "full":
                prepared_manifest = load_json(
                    prepared_roots[dataset] / "manifest.json"
                )
                dataset_config = str(prepared_manifest["dataset_config_path"])
                status["stage"] = f"{dataset}_evaluate"
                write_json(status_path, status)
                if dataset == "qasper":
                    command = [
                        sys.executable,
                        "Scripts/eval/qasper_official.py",
                        "--dataset-config",
                        dataset_config,
                        "--method",
                        "react_official_repro",
                        "--answer-source",
                        "output",
                        "--dynamic-evidence-topk",
                    ]
                else:
                    command = [
                        sys.executable,
                        "Eval/evaluation.py",
                        "-d",
                        dataset_config,
                        "--method",
                        "react_official_repro",
                        "--max_workers",
                        "1",
                    ]
                _run_evaluator(command, output_root / stage / dataset, "evaluation")
                status["datasets"][dataset]["evaluation_complete"] = True
                write_json(status_path, status)

        status["status"] = "complete"
        status["stage"] = "complete"
        write_json(status_path, status)
        return status
    except Exception as error:
        status.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        write_json(status_path, status)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dual-dataset ReAct reproduction")
    parser.add_argument("--stage", choices=["smoke", "full"], required=True)
    parser.add_argument("--prepared-root", action="append", default=[], required=True)
    parser.add_argument("--private-config", action="append", default=[], required=True)
    parser.add_argument("--public-config", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    status = run_full_experiment(
        stage=args.stage,
        prepared_roots=_key_value_pairs(args.prepared_root, "prepared root"),
        private_configs=_key_value_pairs(args.private_config, "private config"),
        public_config=Path(args.public_config).resolve(),
        source_root=Path(args.source_root).resolve(),
        output_root=Path(args.output_root).resolve(),
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

