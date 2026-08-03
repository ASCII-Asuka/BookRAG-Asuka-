import yaml
from pathlib import Path
from Core.configs.mineru_config import MinerU
from Core.configs.llm_config import LLMConfig
from Core.configs.tree_config import TreeConfig
from Core.configs.graph_config import GraphConfig
from Core.configs.vlm_config import VLMConfig
from Core.configs.rag_config import RAGConfig
from Core.configs.vdb_config import VDBConfig
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, Set


class SystemConfig(BaseModel):
    """
    Top-level application configuration model.
    Pydantic will automatically handle nested validation and instantiation.
    """

    # LLM Configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    mineru: MinerU = Field(default_factory=MinerU)

    # Index Configurations
    tree: TreeConfig = Field(default_factory=TreeConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    vdb: VDBConfig = Field(default_factory=VDBConfig)

    # Other Index selection
    index_type: Optional[str] = "gbc"  # Options: "gbc", "tree", "vanilla", "bm25", "raptor", "pdf_vanilla"

    rag_force_reprocess: Optional[bool] = False

    # RAG Configurations
    rag: RAGConfig = Field(default_factory=RAGConfig)

    # Paths
    pdf_path: Optional[str] = "/home/wangshu/multimodal/GBC-RAG/test/double_paper.pdf"
    save_path: Optional[str] = "/home/wangshu/multimodal/GBC-RAG/test/tree_index"

    # # 新增: 专门用于存放评估结果的根目录
    # evaluation_output_path: Optional[str] = Field(
    #     default="/home/wangshu/multimodal/GBC-RAG/test/tree_index/evaluation_results",
    #     description="Root directory to save evaluation results."
    # )


def load_system_config(path: str = "../configs/default.yaml") -> SystemConfig:
    raw_config = _load_raw_config(Path(path).resolve(), seen=set())
    _inherit_matching_embedding_credentials(raw_config)

    if "rag" in raw_config:
        rag_data = raw_config["rag"]
        raw_config["rag"] = {"strategy_config": rag_data}

    cfg = SystemConfig(**raw_config)
    return cfg


def _inherit_matching_embedding_credentials(raw_config: Dict[str, Any]) -> None:
    llm_config = raw_config.get("llm")
    if not isinstance(llm_config, dict):
        return
    api_key = llm_config.get("api_key")
    api_base = llm_config.get("api_base")
    if not api_key or not api_base or str(api_key).strip().upper() == "TODO":
        return

    embedding_configs = []
    vdb_config = raw_config.get("vdb")
    if isinstance(vdb_config, dict):
        embedding_configs.append(vdb_config.get("embedding_config"))
    rag_config = raw_config.get("rag")
    if isinstance(rag_config, dict):
        rag_vdb_config = rag_config.get("vdb_config")
        if isinstance(rag_vdb_config, dict):
            embedding_configs.append(rag_vdb_config.get("embedding_config"))

    for embedding_config in embedding_configs:
        if not isinstance(embedding_config, dict):
            continue
        if str(embedding_config.get("backend", "")).lower() != "openai":
            continue
        if embedding_config.get("api_base") != api_base:
            continue
        if not embedding_config.get("api_key"):
            embedding_config["api_key"] = api_key


def _load_raw_config(path: Path, seen: Set[Path]) -> Dict[str, Any]:
    if path in seen:
        chain = " -> ".join(str(item) for item in [*seen, path])
        raise ValueError(f"Circular config inheritance detected: {chain}")
    seen = {*seen, path}
    with path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}
    if not isinstance(raw_config, dict):
        raise ValueError(f"System config must contain a YAML mapping: {path}")
    parent_value = raw_config.pop("extends", None)
    if not parent_value:
        return raw_config
    parent_path = Path(str(parent_value))
    if not parent_path.is_absolute():
        parent_path = (path.parent / parent_path).resolve()
    parent_config = _load_raw_config(parent_path, seen=seen)
    return _deep_merge(parent_config, raw_config)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
