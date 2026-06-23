from typing import Literal

from pydantic import Field

from Core.configs.rag.base_config import BaseRAGStrategyConfig


class HippoRAGConfig(BaseRAGStrategyConfig):
    strategy: Literal["hipporag"] = "hipporag"
    topk: int = Field(default=10, description="Number of passages used as generation context.")
    bm25_topk: int = Field(default=20, description="Number of lexical passage seeds before graph PPR.")
    ppr_topk: int = Field(default=30, description="Number of passage candidates retained after PPR.")
    supporting_evidence_topk: int = Field(
        default=3,
        description="Number of paragraph evidence items exported for official evidence metrics.",
    )
    answer_style: Literal["default", "short"] = Field(
        default="short",
        description="Use concise dataset-friendly answers when set to short.",
    )
    ppr_alpha: float = Field(default=0.85, description="Personalized PageRank damping factor.")
    ppr_max_iter: int = Field(default=100, description="Maximum PPR iterations.")
    passage_seed_weight: float = Field(default=1.0, description="Weight for BM25 passage seeds.")
    entity_seed_weight: float = Field(default=2.0, description="Weight for query entity seeds.")
    max_entities_per_passage: int = Field(default=24, description="Maximum lightweight entities per passage.")
    max_query_entities: int = Field(default=16, description="Maximum lightweight entities extracted from a query.")
    enable_context_edges: bool = Field(
        default=True,
        description="Connect neighboring passages with low-weight context edges.",
    )
    context_edge_weight: float = Field(default=0.15, description="Weight for neighboring passage context edges.")
    ppr_score_weight: float = Field(default=0.70, description="Final ranking weight for PPR score.")
    bm25_score_weight: float = Field(default=0.20, description="Final ranking weight for BM25 score.")
    entity_overlap_weight: float = Field(default=0.10, description="Final ranking weight for query entity overlap.")
