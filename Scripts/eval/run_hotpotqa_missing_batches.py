import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def load_rows(dataset_path: Path) -> List[dict]:
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def missing_ids(rows: Sequence[dict], working_dir: Path, method_dir: str) -> List[str]:
    missing = []
    for row in rows:
        qid = str(row["doc_uuid"])
        if not (working_dir / qid / method_dir / "final_results.json").exists():
            missing.append(qid)
    return missing


def write_batch_files(
    rows: Sequence[dict],
    processed_dir: Path,
    config_dir: Path,
    working_dir: Path,
    dataset_name: str,
    label: str,
    iteration: int,
) -> Tuple[Path, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    data_path = processed_dir / f"hotpotqa_validation_random600_{label}_orch_batch_{iteration}.json"
    cfg_path = config_dir / f"hotpotqa_validation_random600_{label}_orch_batch_{iteration}.yaml"
    data_path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    cfg_path.write_text(
        f"dataset_path: {data_path.as_posix()}\n"
        f"working_dir: {working_dir.as_posix()}\n"
        f"dataset_name: {dataset_name}\n",
        encoding="utf-8",
    )
    return data_path, cfg_path


def chunk(items: Sequence[str], size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def run_batches(args: argparse.Namespace) -> int:
    dataset_path = Path(args.full_dataset)
    working_dir = Path(args.working_dir)
    processed_dir = Path(args.processed_dir or dataset_path.parent)
    config_dir = Path(args.config_dir)
    log_dir = Path(args.log_dir or (working_dir / "0_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(dataset_path)
    row_by_id = {str(row["doc_uuid"]): row for row in rows}
    order = [str(row["doc_uuid"]) for row in rows]
    attempts = {}
    iteration = args.start_iteration
    started_at = time.time()

    while True:
        current_missing = missing_ids(rows, working_dir, args.method_dir)
        print(
            f"[{args.label}-orchestrator] missing={len(current_missing)} "
            f"elapsed={time.time() - started_at:.1f}s",
            flush=True,
        )
        if not current_missing:
            return 0

        candidates = sorted(current_missing, key=lambda qid: (attempts.get(qid, 0), order.index(qid)))
        candidates = [qid for qid in candidates if attempts.get(qid, 0) < args.max_attempts]
        if not candidates:
            print(
                f"[{args.label}-orchestrator] retry budget exceeded: {current_missing[:50]}",
                flush=True,
            )
            return 2

        batch_ids = candidates[: args.batch_size]
        iteration += 1
        batch_rows = [row_by_id[qid] for qid in batch_ids]
        _, batch_cfg = write_batch_files(
            rows=batch_rows,
            processed_dir=processed_dir,
            config_dir=config_dir,
            working_dir=working_dir,
            dataset_name=args.dataset_name,
            label=args.label,
            iteration=iteration,
        )
        before = set(current_missing)
        out_log = log_dir / f"{args.label}_orch_batch_{iteration}.out.log"
        err_log = log_dir / f"{args.label}_orch_batch_{iteration}.err.log"
        cmd = [
            args.python_executable,
            "main.py",
            "-c",
            args.system_config,
            "-d",
            str(batch_cfg),
            "--nsplit",
            "1",
            "--num",
            "1",
            "rag",
        ]
        print(
            f"[{args.label}-orchestrator] batch={iteration} "
            f"size={len(batch_ids)} first={batch_ids[0]}",
            flush=True,
        )
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        return_code = None
        timed_out = False
        with out_log.open("w", encoding="utf-8") as out, err_log.open("w", encoding="utf-8") as err:
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=Path.cwd(),
                    env=env,
                    stdout=out,
                    stderr=err,
                    timeout=args.batch_timeout,
                )
                return_code = completed.returncode
            except subprocess.TimeoutExpired:
                return_code = "timeout"
                timed_out = True

        after = set(missing_ids(rows, working_dir, args.method_dir))
        gained = len(before) - len(after)
        print(
            f"[{args.label}-orchestrator] batch={iteration} return={return_code} "
            f"gained={gained} timed_out={timed_out}",
            flush=True,
        )
        if gained == 0:
            attempts[batch_ids[0]] = attempts.get(batch_ids[0], 0) + 1
            print(
                f"[{args.label}-orchestrator] no progress; "
                f"qid={batch_ids[0]} attempts={attempts[batch_ids[0]]}",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run missing HotpotQA RAG samples in small batches.")
    parser.add_argument("--full-dataset", required=True)
    parser.add_argument("--working-dir", required=True)
    parser.add_argument("--system-config", required=True)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--method-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--dataset-name", default="hotpotqa")
    parser.add_argument("--processed-dir")
    parser.add_argument("--log-dir")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--batch-timeout", type=int, default=900)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--start-iteration", type=int, default=0)
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_batches(parse_args()))
