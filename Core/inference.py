import os
import pandas as pd
from Core.configs.system_config import load_system_config, SystemConfig
from Core.provider.TokenTracker import TokenTracker
from Core.rag import create_rag_agent
from Core.rag.base_rag import BaseRAG
from Core.utils.json_safety import make_json_safe
from Core.utils.resource_loader import prepare_rag_dependencies
from Core.utils.run_provenance import (
    CacheInputMismatchError,
    RunProvenance,
    cache_input_validation,
    runtime_model_metadata,
)

import json
from tqdm import tqdm
from pathlib import Path
import logging
import argparse
import time
from rich.logging import RichHandler

log = logging.getLogger(__name__)
# logging.basicConfig(
#     level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
# )


def run_rag(
    rag_agent: BaseRAG,
    output_dir: str,
    force_reprocess: bool = False,
    dataset_path: str = None,
    data_df: pd.DataFrame = None,
    run_metadata: dict = None,
    runtime_config=None,
    index_path=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Results will be saved to: {output_dir}")

    start_time = time.time()
    # The manifest identifies this invocation, including failed input loading.
    try:
        if dataset_path and os.path.exists(dataset_path):
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
        elif data_df is not None:
            dataset = data_df.to_dict(orient="records")
        else:
            log.error(f"Dataset file not found: {dataset_path}")
            log.error("Dataframe data not provided")
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    except Exception as error:
        provenance = RunProvenance(
            output_dir, None, runtime_config=runtime_config,
            run_metadata=run_metadata, index_path=index_path,
            force_reprocess=force_reprocess,
        )
        provenance.record_event("failed", reason="dataset_load", error=error)
        provenance.finish("failed", duration_seconds=time.time() - start_time)
        raise

    provenance = RunProvenance(
        output_dir, dataset, runtime_config=runtime_config,
        run_metadata=run_metadata, index_path=index_path,
        force_reprocess=force_reprocess,
    )

    results_list = []
    load_cnt = 0
    for i, item in enumerate(tqdm(dataset, desc=f"Processing Query")):
        query_index_str = f"query_{i+1:03d}"
        query_output_dir = output_dir / query_index_str
        query_result_file = query_output_dir / "result.json"

        if query_result_file.exists() and not force_reprocess:
            existing_result = None
            try:
                with open(query_result_file, "r", encoding="utf-8") as f:
                    existing_result = json.load(f)
                if not isinstance(existing_result, dict):
                    raise ValueError("Cached result is not a mapping")
            except (ValueError, KeyError):
                log.warning(
                    f"Found corrupted result file for {query_index_str}. Re-processing."
                )
            except Exception as error:
                provenance.record_event(
                    "failed", index=i, item=item,
                    query_output_dir=query_output_dir,
                    reason="cache_read", error=error,
                )
                provenance.finish("failed", duration_seconds=time.time() - start_time)
                raise
            if isinstance(existing_result, dict) and existing_result.get("output"):
                cache_validation = cache_input_validation(make_json_safe(item), existing_result)
                if cache_validation["status"] == "mismatch":
                    error = CacheInputMismatchError("Cached input mismatch; saved artifacts were preserved")
                    provenance.record_event(
                        "failed", index=i, item=item,
                        query_output_dir=query_output_dir,
                        origin=existing_result.get("run_provenance"),
                        reason="cache_input_mismatch", error=error,
                        cache_validation=cache_validation,
                    )
                    provenance.finish("failed", duration_seconds=time.time() - start_time)
                    raise error
                log.info(f"Skipping {query_index_str}, result already exists.")
                results_list.append(existing_result)
                load_cnt += 1
                provenance.record_event(
                    "reused", index=i, item=item,
                    query_output_dir=query_output_dir,
                    origin=existing_result.get("run_provenance"),
                    cache_validation=cache_validation,
                )
                continue

        query = item.get("question")
        if not query:
            log.warning(f"Skipping item {i} due to missing 'question' field.")
            provenance.record_event("skipped", index=i, item=item, reason="missing_question")
            continue

        # These optional audit fields must come from this call, never an earlier
        # question or a legacy cache. They do not affect retrieval/generation.
        try:
            for name in ("last_generation_provenance", "last_support_context_validation"):
                if hasattr(rag_agent, name):
                    setattr(rag_agent, name, None)
            query_output_dir.mkdir(exist_ok=True)
            answer, retrieved_node_ids = rag_agent.generation(query, query_output_dir)
            current_result = {
                **item,
                "output": answer,
                "retrieved_node_ids": retrieved_node_ids,
                "run_provenance": provenance.result_provenance(item=make_json_safe(item)),
            }
            retrieved_block_ids = getattr(rag_agent, "last_retrieved_block_ids", None)
            if retrieved_block_ids is not None:
                current_result["retrieved_block_ids"] = retrieved_block_ids
            answer_short = getattr(rag_agent, "last_answer_short", None)
            if answer_short:
                current_result["answer_short"] = answer_short
            answer_rationale = getattr(rag_agent, "last_answer_rationale", None)
            if answer_rationale:
                current_result["answer_rationale"] = answer_rationale
            supporting_block_ids = getattr(rag_agent, "last_supporting_block_ids", None)
            if supporting_block_ids is not None:
                current_result["supporting_block_ids"] = supporting_block_ids
            for field in ("generation_provenance", "support_context_validation"):
                value = getattr(rag_agent, "last_" + field, None)
                if isinstance(value, dict):
                    current_result[field] = value
            current_result = make_json_safe(current_result)
            with open(query_result_file, "w", encoding="utf-8") as f:
                json.dump(current_result, f, indent=2, ensure_ascii=False, allow_nan=False)
            results_list.append(current_result)
            provenance.record_event(
                "generated", index=i, item=item,
                query_output_dir=query_output_dir,
                origin=current_result["run_provenance"],
            )
        except Exception as error:
            provenance.record_event(
                "failed", index=i, item=item,
                query_output_dir=query_output_dir, error=error,
                generation_provenance=getattr(rag_agent, "last_generation_provenance", None),
            )
            provenance.finish("failed", duration_seconds=time.time() - start_time)
            raise

    end_time = time.time()
    total_time = end_time - start_time
    log.info(f"RAG processing complete in {total_time:.2f} seconds.")
    try:
        final_res_path = output_dir / "final_results.json"
        with open(final_res_path, "w", encoding="utf-8") as f:
            json.dump(
                make_json_safe(results_list),
                f,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        log.info(f"RAG complete. All results are saved to {final_res_path}")
        rag_agent.close()

        token_tracker = TokenTracker.get_instance()
        rag_cost = token_tracker.record_stage("rag_cost")
        log.info(f"The token cost of RAG in the current document: {rag_cost}")

        update_and_save_cost(
            output_dir=output_dir,
            new_cost=rag_cost,
            new_time=total_time,
            load_cnt=load_cnt,
            dataset_len=len(dataset),
            force_reprocess=force_reprocess,
        )
    except Exception as error:
        provenance.record_event("failed", reason="finalize", error=error)
        provenance.finish("failed", duration_seconds=time.time() - start_time)
        raise
    provenance.finish("complete", duration_seconds=time.time() - start_time)


def update_and_save_cost(
    output_dir: Path,
    new_cost: int,
    new_time: float,
    load_cnt: int,
    dataset_len: int,
    force_reprocess: bool,
):
    if load_cnt == dataset_len:
        log.info(f"All {load_cnt} samples were loaded from existing results.")
        log.info("Skipping saving token cost since no new inference was made.")
        return

    token_cost_path = output_dir / "token_cost.json"
    previous_cost = {}
    previous_time = 0

    if token_cost_path.exists() and load_cnt != 0 and not force_reprocess:
        log.info(
            f"Found existing cost file at {token_cost_path}. Reading previous values."
        )
        try:
            with open(token_cost_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            previous_cost = existing_data.get("rag_cost", {})
            previous_time = existing_data.get("time", 0)
            log.info(
                f"Previous cost: {previous_cost}, Previous time: {previous_time:.2f}s"
            )
        except (json.JSONDecodeError, KeyError):
            log.warning(
                f"Could not read or parse existing cost file. Starting from zero."
            )
            previous_cost = {}
            previous_time = 0

    total_rag_cost = previous_cost.copy()
    for key, value in new_cost.items():
        total_rag_cost[key] = total_rag_cost.get(key, 0) + value

    total_processing_time = previous_time + new_time

    final_token_cost = {
        "rag_cost": total_rag_cost,
        "time": total_processing_time,
    }

    log.info(
        f"Saving accumulated cost: {total_rag_cost}, Total time: {total_processing_time:.2f}s"
    )
    with open(token_cost_path, "w", encoding="utf-8") as f:
        json.dump(final_token_cost, f, indent=2, ensure_ascii=False)


def create_log_handler(cfg: SystemConfig, dataset_path: str):
    """
    Creates a logging handler that writes logs to a file in the specified output directory.
    The log file is named based on the dataset file name.
    Return: output_dir
    """
    rag_strategy = cfg.rag.strategy_config.strategy
    log.info(f"Using RAG strategy: {rag_strategy}")

    dataset_file = Path(dataset_path)
    method_suffix = rag_strategy
    if rag_strategy == "hri":
        ablation_variant = getattr(cfg.rag.strategy_config, "ablation_variant", "full")
        if ablation_variant and ablation_variant != "full":
            method_suffix = f"hri_{ablation_variant}"
    if rag_strategy == "evibridge":
        method_suffix = cfg.rag.strategy_config.method_suffix
    output_dir = Path(cfg.save_path) / f"eval_{dataset_file.stem}_{method_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = output_dir / "evaluation.log"

    # 给 root logger 添加 FileHandler
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            root_logger.removeHandler(h)
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(file_handler)
    root_logger.info(f"Logging to: {log_file_path}")

    return output_dir


def inference_base(cfg: SystemConfig, dataset_path: str):
    output_dir = create_log_handler(cfg, dataset_path)

    log.info(
        f"Successfully loaded config. Using RAG strategy: {cfg.rag.strategy_config.strategy}"
    )
    dependencies = prepare_rag_dependencies(cfg=cfg)

    rag_agent = create_rag_agent(
        strategy_config=cfg.rag.strategy_config,
        llm_config=cfg.llm,
        vlm_config=cfg.vlm,
        **dependencies,
    )
    log.info(f"RAG agent created with strategy: {rag_agent.name}")

    run_rag(
        rag_agent=rag_agent,
        dataset_path=dataset_path,
        output_dir=output_dir,
        force_reprocess=True,
        runtime_config=cfg,
        index_path=cfg.save_path,
        run_metadata={
            "dataset_name": Path(dataset_path).stem,
            "strategy": cfg.rag.strategy_config.strategy,
            "method_suffix": output_dir.name.removeprefix(f"eval_{Path(dataset_path).stem}_"),
            "models": runtime_model_metadata(cfg),
        },
    )


def inference(cfg: SystemConfig, data_df: pd.DataFrame, dataset_name: str, run_metadata: dict = None):
    dependencies = prepare_rag_dependencies(cfg=cfg)
    rag_agent = create_rag_agent(
        strategy_config=cfg.rag.strategy_config,
        llm_config=cfg.llm,
        vlm_config=cfg.vlm,
        **dependencies,
    )
    log.info(f"RAG agent created with strategy: {rag_agent.name}")

    rag_strategy = cfg.rag.strategy_config.strategy
    log.info(f"Using RAG strategy: {rag_strategy}")
    if rag_strategy == "vanilla":
        retrieval_method = cfg.rag.strategy_config.retrieval_method
        method_suffix = retrieval_method
        output_dir = output_dir = (
            Path(cfg.save_path) / f"eval_{dataset_name}_{retrieval_method}"
        )
    elif rag_strategy == "gbc":
        variant = cfg.rag.strategy_config.variant
        method_suffix = f"{rag_strategy}_{variant}"
        output_dir = output_dir = (
            Path(cfg.save_path) / f"eval_{dataset_name}_{rag_strategy}_{variant}"
        )
    elif rag_strategy == "hri":
        ablation_variant = getattr(cfg.rag.strategy_config, "ablation_variant", "full")
        method_suffix = rag_strategy
        if ablation_variant and ablation_variant != "full":
            method_suffix = f"hri_{ablation_variant}"
        output_dir = output_dir = (
            Path(cfg.save_path) / f"eval_{dataset_name}_{method_suffix}"
        )
    elif rag_strategy == "evibridge":
        method_suffix = cfg.rag.strategy_config.method_suffix
        output_dir = output_dir = (
            Path(cfg.save_path) / f"eval_{dataset_name}_{method_suffix}"
        )
    else:
        method_suffix = rag_strategy
        output_dir = output_dir = (
            Path(cfg.save_path) / f"eval_{dataset_name}_{rag_strategy}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rag(
        rag_agent=rag_agent,
        output_dir=output_dir,
        force_reprocess=cfg.rag_force_reprocess,
        data_df=data_df,
        runtime_config=cfg,
        index_path=cfg.save_path,
        run_metadata={
            **(run_metadata or {}),
            "dataset_name": dataset_name,
            "strategy": rag_strategy,
            "method_suffix": method_suffix,
            "models": runtime_model_metadata(cfg),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation on a dataset.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/wangshu/multimodal/GBC-RAG/config/gbc.yaml",
        # default="/home/wangshu/multimodal/GBC-RAG/config/mm.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the JSON dataset file with questions.",
        default="/home/wangshu/multimodal/GBC-RAG/test/test_qa/test_samples.json",
        # default="/home/wangshu/multimodal/GBC-RAG/test/sf/case-qa/sel_data_qa.json",
    )
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )

    args = parser.parse_args()
    cfg = load_system_config(args.config_path)
    inference_base(cfg, args.dataset_path)
