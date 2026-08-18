from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapter_code_sha256(paths: Sequence[str | Path]) -> str:
    files = [Path(path).resolve() for path in paths]
    return sha256_json(
        [
            {"name": path.name, "sha256": sha256_file(path)}
            for path in sorted(files, key=lambda item: str(item).lower())
        ]
    )


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(output)


def safe_filename(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._") or "scope"
    suffix = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:10]
    return f"{stem[:80]}-{suffix}.json"


def build_run_identity(
    corpus_sha256: str,
    config_sha256: str,
    source_revision: str,
    model_name: str,
) -> str:
    return sha256_json(
        {
            "corpus_sha256": str(corpus_sha256),
            "config_sha256": str(config_sha256),
            "source_revision": str(source_revision),
            "model_name": str(model_name),
        }
    )


def scope_corpus_sha256(chunks: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json(
        [
            {
                "source_id": str(chunk.get("source_id") or ""),
                "content": str(chunk.get("content") or ""),
                "index_content": str(
                    chunk.get("index_content") or chunk.get("content") or ""
                ),
                "metadata": dict(chunk.get("metadata") or {}),
            }
            for chunk in chunks
        ]
    )
