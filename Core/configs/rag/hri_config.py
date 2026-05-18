from typing import Literal

from pydantic import Field

from .base_config import BaseRAGStrategyConfig


class HRIRAGConfig(BaseRAGStrategyConfig):
    strategy: Literal["hri"] = "hri"
    topk: int = Field(default=8, description="Number of coarse BM25 candidates.")
    expand_depth: int = Field(default=1, description="Relation expansion depth.")
    context_window: int = Field(default=1, description="Sibling context window.")
    enable_relation_expansion: bool = Field(
        default=True, description="Whether to expand evidence along HRI relations."
    )
    output_evidence_chain: bool = Field(
        default=True, description="Whether to write evidence_chain.json per query."
    )
    max_context_nodes: int = Field(
        default=12, description="Maximum number of evidence nodes in generation context."
    )
