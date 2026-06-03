import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_DATASET_NAME = "allenai/qasper"
DEFAULT_SPLITS = ("train", "validation", "test")
PARQUET_REVISION = "refs/convert/parquet"


def download_qasper(
    output_root: str,
    dataset_name: str = DEFAULT_DATASET_NAME,
    cache_dir: str = "",
) -> Dict[str, Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'datasets'. Install it with: "
            "pip install datasets huggingface_hub pyarrow"
        ) from exc

    root = Path(output_root)
    hf_cache = Path(cache_dir) if cache_dir else root / "hf_cache"
    try:
        dataset = load_dataset(dataset_name, cache_dir=str(hf_cache))
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
        dataset = _download_parquet_splits(
            dataset_name=dataset_name,
            cache_dir=str(hf_cache),
        )
    return write_qasper_splits(dataset, output_root=output_root)


def write_qasper_splits(
    dataset: Dict[str, Any],
    output_root: str,
    splits: Iterable[str] = DEFAULT_SPLITS,
) -> Dict[str, Dict[str, Any]]:
    root = Path(output_root)
    raw_dir = root / "raw"
    processed_dir = root / "processed"
    hf_cache_dir = root / "hf_cache"
    for path in (raw_dir, processed_dir, hf_cache_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Dict[str, Any]] = {}
    available_splits = set(dataset.keys()) if hasattr(dataset, "keys") else set()
    for split in splits:
        if split not in available_splits:
            continue
        rows = _split_to_rows(dataset[split])
        split_path = raw_dir / f"qasper_{split}.json"
        split_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary = summarize_qasper_rows(rows)
        summary["path"] = str(split_path)
        manifest[split] = summary

    manifest_path = raw_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def summarize_qasper_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "paper_count": len(rows),
        "question_count": sum(_question_count(row.get("qas", [])) for row in rows),
        "has_answers": any(_has_answers(row.get("qas", [])) for row in rows),
        "has_evidence": any(_has_evidence(row.get("qas", [])) for row in rows),
    }


def _download_parquet_splits(
    dataset_name: str,
    cache_dir: str,
    splits: Iterable[str] = DEFAULT_SPLITS,
) -> Dict[str, List[Dict[str, Any]]]:
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Parquet fallback requires pandas and huggingface_hub. Install with: "
            "pip install pandas huggingface_hub pyarrow"
        ) from exc

    dataset: Dict[str, List[Dict[str, Any]]] = {}
    for split in splits:
        parquet_file = hf_hub_download(
            repo_id=dataset_name,
            repo_type="dataset",
            revision=PARQUET_REVISION,
            filename=f"qasper/{split}/0000.parquet",
            cache_dir=cache_dir,
        )
        rows = pd.read_parquet(parquet_file).to_dict(orient="records")
        dataset[split] = [_json_safe(row) for row in rows]
    return dataset


def _split_to_rows(split_data: Any) -> List[Dict[str, Any]]:
    if hasattr(split_data, "to_list"):
        rows = split_data.to_list()
    else:
        rows = list(split_data)
    return [_json_safe(dict(row)) for row in rows]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return value


def _question_count(qas: Any) -> int:
    if isinstance(qas, dict):
        questions = qas.get("question", [])
        return len(questions) if isinstance(questions, list) else int(bool(questions))
    if isinstance(qas, list):
        return len(qas)
    return 0


def _has_answers(qas: Any) -> bool:
    for answer in _iter_answers(qas):
        if answer:
            return True
    return False


def _has_evidence(qas: Any) -> bool:
    for answer in _iter_answers(qas):
        evidence = answer.get("evidence", []) if isinstance(answer, dict) else []
        if evidence:
            return True
    return False


def _iter_answers(qas: Any):
    answer_groups = []
    if isinstance(qas, dict):
        answer_groups = qas.get("answers", []) or []
    elif isinstance(qas, list):
        answer_groups = [qa.get("answers", []) for qa in qas if isinstance(qa, dict)]

    for group in answer_groups:
        if isinstance(group, dict):
            payload = group.get("answer", group)
            if isinstance(payload, list):
                for answer in payload:
                    if isinstance(answer, dict):
                        yield answer
            elif isinstance(payload, dict):
                yield payload
        elif isinstance(group, list):
            for item in group:
                if isinstance(item, dict):
                    payload = item.get("answer", item)
                    if isinstance(payload, dict):
                        yield payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Qasper from Hugging Face.")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Local Qasper root, e.g. D:\\Else\\Study\\datasets\\public\\qasper",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Hugging Face dataset name. Default: allenai/qasper",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Optional Hugging Face cache directory. Defaults to <output-root>\\hf_cache.",
    )
    args = parser.parse_args()

    manifest = download_qasper(
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
