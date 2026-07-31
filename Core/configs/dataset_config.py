import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field
import yaml


class DatasetConfig(BaseModel):
    dataset_path: str = Field(..., description="Path to the JSON dataset file.")
    working_dir: str = Field(..., description="The working directory for the project.")
    dataset_name: str
    manifest_path: str = ""


def load_dataset_config(path: str) -> DatasetConfig:
    # ... standard YAML loading logic ...
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    data_cfg = DatasetConfig(**data)
    _validate_manifest(data_cfg)
    return data_cfg


def _validate_manifest(data_cfg: DatasetConfig) -> None:
    if not data_cfg.manifest_path:
        return
    manifest = json.loads(
        Path(data_cfg.manifest_path).read_text(encoding="utf-8")
    )
    dataset_path = Path(data_cfg.dataset_path)
    dataset_bytes = dataset_path.read_bytes()
    actual_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
    if actual_sha256 != str(manifest.get("unified_sha256") or ""):
        raise ValueError(
            "dataset SHA256 mismatch: "
            f"expected={manifest.get('unified_sha256')}, actual={actual_sha256}"
        )
    rows = json.loads(dataset_bytes.decode("utf-8"))
    actual_ids = [
        str(
            row.get("hotpotqa_question_id")
            or row.get("question_id")
            or row.get("doc_uuid")
            or ""
        )
        for row in rows
    ]
    expected_ids = [
        str(value)
        for value in manifest.get("selected_question_ids", [])
    ]
    if actual_ids != expected_ids:
        raise ValueError("selected question ID order mismatch")
    if len(rows) != int(manifest.get("question_count", -1)):
        raise ValueError("manifest question count mismatch")
