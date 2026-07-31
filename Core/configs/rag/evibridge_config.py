from typing import Dict, List, Literal, Optional

from pydantic import Field

from Core.configs.rerank_config import RerankerConfig
from Core.configs.vdb_config import VDBConfig

from .base_config import BaseRAGStrategyConfig

# 这是一个 全局常量字典 ，预定义了不同“问题意图/任务类型”（如事实检索 fact 、多跳推理 multi-hop 、图表对比 table-figure 等）在进行图谱或节点游走（如 Personalized PageRank，PPR）时，所依赖的三种类型边（ context 上下文边、 semantic 语义边、 hierarchy 层级边）的 权重分配比例 。
DEFAULT_TYPED_EDGE_WEIGHTS = {
    "fact": {"context": 0.45, "semantic": 0.35, "hierarchy": 0.20},
    "boolean": {"context": 0.5, "semantic": 0.25, "hierarchy": 0.25},
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
        "wo_demand_aware_seed_recall",
        "wo_context_edges",
        "wo_semantic_edges",
        "wo_hierarchy_edges",
        "wo_typed_weights",
        "wo_budgeted_selector",
        "wo_sufficiency_verifier",
        "wo_citation_reorder",
        "wo_support_reranker",
        "wo_verifier_repair",
        "wo_implicit_multihop",
        "static_topk",
    ] = "full"
    demand_parser: Literal["rule", "llm", "hybrid"] = "hybrid"
    demand_confidence_threshold: float = 0.7
    dataset_profile: Literal["auto", "qasper", "hotpotqa"] = "auto"
    qasper_demand_mode: Literal["default", "conservative"] = "conservative"
    enable_boolean_answer_hint: bool = True
    multi_hop_requires_explicit_bridge: bool = True
    bm25_topk: int = 20
    embedding_topk: int = 20
    patch_topk: int = 5
    entity_topk: int = 10
    enable_vector_recall: bool = False
    hybrid_bm25_weight: float = 0.55
    hybrid_vector_weight: float = 0.45
    enable_candidate_rerank: bool = False
    candidate_rerank_topk: int = 50
    candidate_rerank_weight: float = 0.65
    enable_supporting_rerank: bool = False
    rerank_batch_size: int = 50
    trust_answer_supporting_ids: bool = True
    ppr_topk: int = 30
    ppr_restart_alpha: float = 0.15
    ppr_max_iter: int = 50
    enable_shortest_path_connector: bool = True
    connector_max_paths: int = 3
    preserve_seed_topk: int = 40
    final_evidence_types: List[str] = Field(default_factory=lambda: ["paragraph", "table", "caption", "figure"])
    bridge_auxiliary_types: List[str] = Field(default_factory=lambda: ["entity", "summary", "patch", "title"])
    supporting_evidence_topk: int = 4
    dynamic_supporting_evidence_budget: bool = True
    supporting_evidence_types: List[str] = Field(default_factory=lambda: ["paragraph", "table", "caption", "figure"])
    paragraph_quota: int = 4
    auxiliary_quota: int = 2
    max_context_blocks: int = 10
    max_context_tokens: int = 4000
    max_iterations: int = 2
    enable_llm_verifier: bool = True
    enable_short_answer_extraction: bool = False
    enable_long_context_fallback: bool = False
    fallback_max_context_blocks: int = 30
    fallback_max_context_tokens: int = 12000
    output_evidence_chain: bool = True
    evibridge_vdb_config: VDBConfig = Field(default_factory=VDBConfig)
    reranker_config: RerankerConfig = Field(default_factory=RerankerConfig)
    typed_edge_weights: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            key: value.copy() for key, value in DEFAULT_TYPED_EDGE_WEIGHTS.items()
        }
    )
    topk: Optional[int] = Field(
        default=None,
        description="Backward-compatible top-k override for BM25 and PPR.",
    )

    @property
    def method_suffix(self) -> str:
        suffix = "evibridge"
        if self.ablation_variant and self.ablation_variant != "full":
            suffix = f"{suffix}_{self.ablation_variant}"
        if self.enable_long_context_fallback:
            suffix = f"{suffix}_fallback"
        return suffix
