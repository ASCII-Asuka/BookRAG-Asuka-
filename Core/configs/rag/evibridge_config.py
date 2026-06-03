from typing import Dict, Literal, Optional

from pydantic import Field

from Core.configs.vdb_config import VDBConfig

from .base_config import BaseRAGStrategyConfig

# 这是一个 全局常量字典 ，预定义了不同“问题意图/任务类型”（如事实检索 fact 、多跳推理 multi-hop 、图表对比 table-figure 等）在进行图谱或节点游走（如 Personalized PageRank，PPR）时，所依赖的三种类型边（ context 上下文边、 semantic 语义边、 hierarchy 层级边）的 权重分配比例 。
DEFAULT_TYPED_EDGE_WEIGHTS = {
    "fact": {"context": 0.45, "semantic": 0.35, "hierarchy": 0.20},
    "multi-hop": {"context": 0.25, "semantic": 0.55, "hierarchy": 0.20},
    "comparison": {"context": 0.25, "semantic": 0.55, "hierarchy": 0.20},
    "table-figure": {"context": 0.65, "semantic": 0.20, "hierarchy": 0.15},
    "aggregation": {"context": 0.35, "semantic": 0.20, "hierarchy": 0.45},
    "global-summary": {"context": 0.35, "semantic": 0.20, "hierarchy": 0.45},
}


class EviBridgeRAGConfig(BaseRAGStrategyConfig):
    strategy: Literal["evibridge"] = "evibridge"
    ablation_variant: Literal[
        "full",
        "wo_multi_granularity_seeds",
        "wo_context_edges",
        "wo_semantic_edges",
        "wo_hierarchy_edges",
        "wo_typed_weights",
        "wo_budgeted_selector",
        "wo_sufficiency_verifier",
        "static_topk",
    ] = "full"
    demand_parser: Literal["rule", "llm", "hybrid"] = "hybrid"
    demand_confidence_threshold: float = 0.7
    bm25_topk: int = 20
    embedding_topk: int = 20
    patch_topk: int = 5
    entity_topk: int = 10
    enable_vector_recall: bool = False
    hybrid_bm25_weight: float = 0.55
    hybrid_vector_weight: float = 0.45
    ppr_topk: int = 30
    ppr_restart_alpha: float = 0.15
    ppr_max_iter: int = 50
    enable_shortest_path_connector: bool = True
    connector_max_paths: int = 3
    max_context_blocks: int = 10
    max_context_tokens: int = 4000
    max_iterations: int = 2
    enable_llm_verifier: bool = True
    output_evidence_chain: bool = True
    evibridge_vdb_config: VDBConfig = Field(default_factory=VDBConfig)
    typed_edge_weights: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            key: value.copy() for key, value in DEFAULT_TYPED_EDGE_WEIGHTS.items()
        }
    )
    topk: Optional[int] = Field(
        default=None,
        description="Backward-compatible top-k override for BM25 and PPR.",
    )
