import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_DATASET_NAME = "hotpotqa/hotpot_qa"
DEFAULT_CONFIG_NAME = "distractor"
DEFAULT_SPLITS = ("train", "validation")


def download_hotpotqa(
    output_root: str,
    dataset_name: str = DEFAULT_DATASET_NAME,
    config_name: str = DEFAULT_CONFIG_NAME,
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
    dataset = load_dataset(dataset_name, config_name, cache_dir=str(hf_cache))
    return write_hotpotqa_splits(
        dataset=dataset,
        output_root=output_root,
        config_name=config_name,
    )


def write_hotpotqa_splits(
    dataset: Dict[str, Any],
    output_root: str,
    config_name: str = DEFAULT_CONFIG_NAME,
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
        split_path = raw_dir / f"hotpotqa_{config_name}_{split}.json"
        split_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary = summarize_hotpotqa_rows(rows)
        summary["path"] = str(split_path)
        manifest[split] = summary

    manifest_path = raw_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def summarize_hotpotqa_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "question_count": len(rows),
        "has_answers": any(bool(row.get("answer")) for row in rows),
        "has_supporting_facts": any(_has_supporting_facts(row) for row in rows),
    }


def _has_supporting_facts(row: Dict[str, Any]) -> bool:
    facts = row.get("supporting_facts") or {}
    if isinstance(facts, dict):
        return bool(facts.get("title")) and bool(facts.get("sent_id"))
    return bool(facts)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HotpotQA from Hugging Face.")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Local HotpotQA root, e.g. D:\\Else\\Study\\datasets\\public\\hotpotqa",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Hugging Face dataset name. Default: hotpotqa/hotpot_qa",
    )
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="HotpotQA subset/config. Default: distractor",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Optional Hugging Face cache directory. Defaults to <output-root>\\hf_cache.",
    )
    args = parser.parse_args()

    manifest = download_hotpotqa(
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        config_name=args.config_name,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
