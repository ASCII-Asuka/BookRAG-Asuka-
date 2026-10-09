"""Append-only run identities without credentials, source text, or host paths.

Each invocation owns a new directory. Its manifest is created exclusively and
never revised; events and a separately created summary describe what happened.
These records identify newly generated artifacts, not historical cache origins.
"""

import dataclasses
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


_REDACTED = "***REDACTED***"
_SECRET_KEY = re.compile(r"apikey|apitoken|secret|password|passwd|credential|authorization|cookie|accesstoken|authtoken|bearertoken|refreshtoken|sessiontoken|clienttoken|^token$|^key$", re.I)
_ERROR_KEY = re.compile(r"exception|errormessage|traceback|stacktrace|^errors?$", re.I)
_LOCATION_KEY = re.compile(r"(?:^|_)(?:path|dir|directory|url|uri|endpoint|host|address|api_base|server)(?:$|_)|(?:path|dir|directory|url|uri|endpoint|host|address|apibase)$", re.I)
_PRIVATE_STRING = re.compile(r"[a-z][a-z0-9+.-]*://|[A-Za-z]:[\\/]|\\\\|(?:^|[\s\"'(])(?:/[^\s]+|~[\\/]|\.{1,2}[\\/])|\b(?:\d{1,3}\.){3}\d{1,3}\b|\b(?:sk-|Bearer\s)", re.I)
_METADATA_KEYS = {
    "dataset_name", "dataset_split", "strategy", "method_suffix", "models",
    "model_name", "model_version", "revision", "backend", "provider",
    "split", "nsplit", "num", "doc_uuid", "dataset_sha256", "manifest_sha256",
    "input_manifest_sha256", "source_dataset_sha256", "command",
}
_QID_KEYS = ("qasper_question_id", "hotpotqa_question_id", "question_id", "qid", "id", "_id")
_RESULT_FIELDS = {
    "output", "retrieved_node_ids", "retrieved_block_ids", "answer_short",
    "answer_rationale", "supporting_block_ids", "run_provenance",
    "generation_provenance", "support_context_validation",
    "answer_context_block_ids", "planned_answer_context_block_ids",
    "posthoc_supporting_evidence", "verification_trace",
}
_INPUT_HASH_SCHEMA = "json_safe_v1"


class CacheInputMismatchError(RuntimeError):
    """A saved result explicitly belongs to a different query input."""


def _plain(value):
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # NumPy values can appear in DataFrame records without making this module
    # depend on NumPy or a model SDK. Do not stringify arbitrary objects.
    if hasattr(value, "item"):
        try:
            return _plain(value.item())
        except (ValueError, TypeError):
            pass
    return {"status": "unsupported", "type": type(value).__name__}


def _sha256(value):
    payload = json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _redacted_location(value):
    return {"redacted": True, "sha256": _sha256(value)}


def safe_runtime_config(value):
    """Copy nested config, masking credentials and hashing private locations.

    Endpoint/path fingerprints distinguish configurations without publishing
    their values. Credentials are omitted from the fingerprint entirely.
    """
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, Path):
        return _redacted_location(str(value))
    if isinstance(value, dict):
        safe = {}
        for key, item in value.items():
            key = str(key)
            if _PRIVATE_STRING.search(key):
                key = "redacted_key_" + _sha256(key)
            normalized_key = re.sub(r"[^a-z0-9]", "", key.lower())
            if _SECRET_KEY.search(normalized_key) or _ERROR_KEY.search(normalized_key):
                safe[key] = _REDACTED
            elif _LOCATION_KEY.search(key) and isinstance(item, (str, Path, dict, list, tuple)):
                safe[key] = _redacted_location(item)
            else:
                safe[key] = safe_runtime_config(item)
        return safe
    if isinstance(value, (list, tuple)):
        return [safe_runtime_config(item) for item in value]
    value = _plain(value)
    if isinstance(value, str) and _PRIVATE_STRING.search(value):
        return _redacted_location(value)
    return value


def safe_run_metadata(value):
    """Keep only declared runtime identifiers, never arbitrary notes/errors."""
    raw = value if isinstance(value, dict) else _plain(value)
    if not isinstance(raw, dict):
        return {}
    return safe_runtime_config({key: item for key, item in raw.items() if key in _METADATA_KEYS})


def runtime_model_metadata(config):
    """Report configured model identifiers; absent version locks stay unknown."""
    raw = safe_runtime_config(config)
    models = {}

    def visit(value, prefix=""):
        if not isinstance(value, dict):
            return
        if "model_name" in value:
            revision = value.get("revision") or value.get("model_revision")
            version = value.get("model_version")
            models[prefix] = {
                "model_name": value.get("model_name"),
                "backend": value.get("backend"),
                "revision": revision,
                "revision_status": "known" if revision is not None else "unknown",
                "model_version": version,
                "version_status": "known" if version is not None else "unknown",
            }
        for key, item in value.items():
            if isinstance(item, dict):
                visit(item, f"{prefix}.{key}" if prefix else key)

    visit(raw)
    return safe_runtime_config(models)


def file_fingerprint(path):
    """Hash file contents and represent unavailable files without an error log."""
    try:
        digest = hashlib.sha256()
        size = 0
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        return {"status": "known", "sha256": digest.hexdigest(), "size_bytes": size}
    except OSError:
        return {"status": "unknown", "sha256": None, "size_bytes": None}


def _code_identity(repo_root):
    revision = None
    dirty = None
    try:
        response = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=10, check=False)
        if response.returncode == 0 and re.fullmatch(r"[a-fA-F0-9]{40,64}", response.stdout.strip()):
            revision = response.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        response = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=repo_root, capture_output=True, text=True, timeout=10, check=False)
        if response.returncode == 0:
            dirty = bool(response.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    paths = list((repo_root / "Core").rglob("*.py")) if (repo_root / "Core").is_dir() else []
    if (repo_root / "main.py").is_file():
        paths.append(repo_root / "main.py")
    files = {path.relative_to(repo_root).as_posix(): file_fingerprint(path) for path in sorted(paths)}
    return {
        "git_commit": revision,
        "git_dirty": dirty,
        "git_status": "known" if revision is not None and dirty is not None else "unknown",
        "source_files": files,
        "source_status": "known" if files and all(item["status"] == "known" for item in files.values()) else "unknown",
        "source_sha256": _sha256(files) if files else None,
        "python_version": platform.python_version(),
    }


def _configured_index_locations(config):
    # Read original config before redaction: safe snapshots intentionally omit
    # locations. Only index storage declarations are used, never model paths.
    raw = _plain(config) if config is not None else {}
    declarations = []

    def visit(value, prefix=""):
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            field = f"{prefix}.{key}" if prefix else key
            if key.endswith("vdb_dir_name"):
                declarations.append((field, item))
            elif isinstance(item, dict):
                visit(item, field)

    if isinstance(raw, dict):
        rag = raw.get("rag")
        if isinstance(rag, dict):
            visit(rag.get("strategy_config", rag), "rag.strategy_config")
            strategy = rag.get("strategy_config", rag).get("strategy")
            if strategy in {"gbc", "graph"}:
                visit(raw.get("vdb"), "vdb")
                visit(raw.get("graph"), "graph")
        elif "strategy" in raw:
            visit(raw, "strategy_config")
    return declarations


def _index_identity(index_path, runtime_config=None):
    raw_base = os.fspath(index_path) if index_path is not None else None
    root = Path(raw_base) if raw_base is not None else None
    if root is not None and root.is_file():
        raw_base = os.path.dirname(raw_base)
    files = {}
    fingerprint_cache = {}

    def fingerprint(path):
        canonical_path = path.absolute()
        if canonical_path not in fingerprint_cache:
            fingerprint_cache[canonical_path] = file_fingerprint(path)
        return fingerprint_cache[canonical_path]

    if root is not None:
        try:
            if root.is_file():
                files[root.name] = fingerprint(root)
            elif root.is_dir():
                paths = set()
                for pattern in ("tree.pkl", "tree.json", "graph_data*.json", "evibridge*.json", "evibridge*.pkl", "hri*.json", "hri*.pkl"):
                    paths.update(path for path in root.glob(pattern) if path.is_file())
                for folder in ("Tree_vdb", "kg_vdb", "kg_vdb_basic", "evibridge_vdb", "hri_vdb"):
                    directory = root / folder
                    if directory.is_dir():
                        paths.update(path for path in directory.rglob("*") if path.is_file())
                files.update({path.relative_to(root).as_posix(): fingerprint(path) for path in sorted(paths)})
        except OSError:
            pass

    configured_stores = {}
    for field, value in _configured_index_locations(runtime_config):
        label = "configured_" + _sha256(field)[:16]
        store = {"status": "unknown", "configuration_field_sha256": _sha256(field), "location_sha256": None, "sha256": None, "file_count": 0}
        configured_stores[label] = store
        if not isinstance(value, str) or not value:
            continue
        raw_store_path = value
        if not os.path.isabs(raw_store_path):
            if raw_base is None:
                continue
            # The loader compares original strings before joining. Preserve ./
            # and mixed slash spellings here; Path normalization changes that
            # prefix decision and can fingerprint a different existing store.
            if raw_base not in raw_store_path:
                raw_store_path = os.path.join(raw_base, raw_store_path)
        path = Path(raw_store_path)
        store["location_sha256"] = _sha256(str(path.absolute()))
        try:
            if path.is_file():
                store_files = {path.name: fingerprint(path)}
            elif path.is_dir():
                store_files = {file.relative_to(path).as_posix(): fingerprint(file) for file in sorted(path.rglob("*")) if file.is_file()}
            else:
                store_files = {}
        except OSError:
            store_files = {}
        if store_files:
            store["status"] = "known" if all(record["status"] == "known" for record in store_files.values()) else "partial"
            store["sha256"] = _sha256(store_files)
            store["file_count"] = len(store_files)
            files.update({f"{label}/{name}": record for name, record in store_files.items()})
    # This hashes common files and declared storage locations. It is not a
    # complete lock of arbitrary strategy dependencies or external services.
    return {
        "status": "partial" if files else "unknown",
        "sha256": _sha256({"files": files, "configured_stores": configured_stores}) if files else None,
        "files": files,
        "configured_stores": configured_stores,
        "coverage_status": "partial" if files else "unknown",
        "coverage_note": "Common index formats and configured VDB storage are fingerprinted; unenumerated dependencies and remote state are not fully locked. Configured missing or unreadable storage is unknown.",
    }


def _qid(item):
    for key in _QID_KEYS:
        value = item.get(key)
        if value is not None:
            return _plain(value)
    return None


def cache_input_validation(item, cached_result):
    """Compare known cached inputs; missing historical identity stays unknown.

    Values are only compared and hashed, never included in this report. A qid
    and question are required for a complete association, even when known
    overlapping input fields agree. Origin run IDs alone do not prove it.
    """
    current = _plain(item)
    cached = _plain(cached_result)
    current_fields = set(current) - _RESULT_FIELDS
    cached_fields = set(cached) - _RESULT_FIELDS
    common = current_fields & cached_fields
    missing_cached = current_fields - cached_fields
    missing_current = cached_fields - current_fields
    compared = []
    mismatched = []
    core_fields = {"question", "doc_uuid", "doc_path", *_QID_KEYS}
    for key in sorted(common):
        if key in core_fields and (current[key] is None or cached[key] is None or current[key] == "" or cached[key] == ""):
            if current[key] is None or current[key] == "":
                missing_current.add(key)
            if cached[key] is None or cached[key] == "":
                missing_cached.add(key)
            continue
        compared.append(key)
        if _sha256(current[key]) != _sha256(cached[key]):
            mismatched.append(key)
    current_qid, cached_qid = _qid(current), _qid(cached)
    if current_qid is not None and cached_qid is not None and _sha256(current_qid) != _sha256(cached_qid):
        mismatched.append("qid")
    advertised = cached.get("run_provenance")
    stored_hash = advertised.get("input_sha256") if isinstance(advertised, dict) else None
    stored_schema = advertised.get("input_hash_schema") if isinstance(advertised, dict) else None
    input_hash = _sha256(current)
    fingerprint_status = "unknown"
    if stored_schema == _INPUT_HASH_SCHEMA and isinstance(stored_hash, str) and re.fullmatch(r"[a-fA-F0-9]{64}", stored_hash):
        fingerprint_status = "matched" if stored_hash.lower() == input_hash else "mismatch"
        if fingerprint_status == "mismatch":
            mismatched.append("input_sha256")
    complete_identity = current_qid is not None and cached_qid is not None and "question" in compared
    if mismatched:
        status = "mismatch"
    elif complete_identity and (fingerprint_status == "matched" or not (missing_cached or missing_current)):
        status = "matched"
    elif compared or cached_qid is not None or fingerprint_status == "matched":
        status = "partial"
    else:
        status = "unknown"
    return {
        "status": status,
        "cached_qid": safe_runtime_config(cached_qid),
        "cached_qid_sha256": _sha256(cached_qid) if cached_qid is not None else None,
        "input_sha256": input_hash,
        "cached_saved_input_sha256": _sha256({key: cached[key] for key in cached_fields}) if cached_fields else None,
        "origin_input_fingerprint_status": fingerprint_status,
        "compared_fields": safe_runtime_config(sorted(compared)),
        "missing_cached_fields": safe_runtime_config(sorted(missing_cached)),
        "missing_current_fields": safe_runtime_config(sorted(missing_current)),
        "mismatched_fields": safe_runtime_config(sorted(set(mismatched))),
        "association_note": "This compares saved input fields and client fingerprints. Missing cached input/qid is not proof of a match; this invocation's config does not identify cache execution.",
    }


def _dataset_identity(dataset):
    if dataset is None:
        return {"status": "unknown", "count": None, "input_sha256": None, "ordered_qid_sha256": None, "items": []}
    identities = []
    qids = []
    for index, item in enumerate(dataset):
        qid = _qid(item)
        qids.append(qid)
        identities.append({"index": index, "qid": safe_runtime_config(qid), "qid_sha256": _sha256(qid) if qid is not None else None, "input_sha256": _sha256(item)})
    return {"status": "known", "count": len(dataset), "input_sha256": _sha256(dataset), "ordered_qid_sha256": _sha256(qids), "items": identities}


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _exclusive_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


class RunProvenance:
    def __init__(self, output_dir, dataset, *, runtime_config=None, run_metadata=None, index_path=None, force_reprocess=False, repo_root=None):
        self.output_dir = Path(output_dir)
        self.run_id = str(uuid.uuid4())
        self.run_dir = self.output_dir / ".runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.manifest_path = self.run_dir / "manifest.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.counts = {name: 0 for name in ("generated", "reused", "skipped", "failed")}
        repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
        config = safe_runtime_config(runtime_config) if runtime_config is not None else None
        self.manifest = {
            "schema_version": 1,
            "run_id": self.run_id,
            "created_at": _utc_now(),
            "scope": "run_rag_invocation",
            "method_suffix": safe_run_metadata(run_metadata).get("method_suffix"),
            "force_reprocess": bool(force_reprocess),
            "code": _code_identity(repo_root),
            "config": {"status": "known" if config is not None else "unknown", "sha256": _sha256(config) if config is not None else None, "safe_snapshot": config},
            "metadata": safe_run_metadata(run_metadata),
            "dataset": _dataset_identity(dataset),
            "indexes": _index_identity(index_path, runtime_config),
            "events_file": "events.jsonl",
            "summary_file": "summary.json",
            "identity_note": "Document-root config snapshots are compatibility files, not this run identity. Reused artifacts retain their original provenance; absent origins are unknown.",
        }
        _exclusive_json(self.manifest_path, self.manifest)
        with self.events_path.open("x", encoding="utf-8"):
            pass

    def result_provenance(self, *, item=None):
        record = {
            "run_id": self.run_id,
            "manifest_path": f"../.runs/{self.run_id}/manifest.json",
            "path_base": "result_directory",
            "code": {key: self.manifest["code"][key] for key in ("git_commit", "git_dirty", "git_status", "source_sha256")},
            "config_sha256": self.manifest["config"]["sha256"],
        }
        if item is not None:
            record["input_sha256"] = _sha256(item)
            record["input_hash_schema"] = _INPUT_HASH_SCHEMA
        return record

    def record_event(self, event, *, index=None, item=None, query_output_dir=None, origin=None, reason=None, error=None, generation_provenance=None, cache_validation=None):
        if event not in self.counts:
            raise ValueError("Unsupported run event")
        self.counts[event] += 1
        origin_id = origin.get("run_id") if isinstance(origin, dict) else None
        if not isinstance(origin_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", origin_id):
            origin_id = None
        files = {}
        if query_output_dir is not None:
            for name in ("result.json", "retrieval_res.json", "evidence_chain.json"):
                path = Path(query_output_dir) / name
                if path.is_file():
                    files[name] = file_fingerprint(path)
        qid = _qid(item) if item is not None else None
        record = {
            "event": event, "recorded_at": _utc_now(), "run_id": self.run_id,
            "index": index, "qid": safe_runtime_config(qid),
            "qid_sha256": _sha256(qid) if qid is not None else None,
            "query_directory": f"query_{index + 1:03d}" if index is not None else None,
            "origin_run_id": origin_id, "origin_status": "known" if origin_id is not None else "unknown",
            "referenced_run_id": origin_id, "files": files,
        }
        if reason is not None:
            record["reason"] = safe_runtime_config(reason)
        if error is not None:
            # Exception strings may embed credentials, URLs, or model paths.
            # Only the exception class is part of the auditable event.
            record["error_type"] = re.sub(r"[^A-Za-z0-9_]", "", type(error).__name__)
        if isinstance(generation_provenance, dict):
            record["generation_provenance"] = safe_runtime_config(generation_provenance)
        if isinstance(cache_validation, dict):
            record["cache_input_status"] = cache_validation["status"]
            record["cached_qid"] = cache_validation.get("cached_qid")
            record["cached_qid_sha256"] = cache_validation.get("cached_qid_sha256")
            record["cache_input_validation"] = safe_runtime_config(cache_validation)
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def finish(self, status, *, duration_seconds=None):
        summary = {"run_id": self.run_id, "status": status, "finished_at": _utc_now(), "events": dict(self.counts), "duration_seconds": duration_seconds, "artifacts": {}}
        for name in ("final_results.json", "token_cost.json"):
            path = self.output_dir / name
            if path.is_file():
                summary["artifacts"][name] = file_fingerprint(path)
        _exclusive_json(self.run_dir / "summary.json", summary)
