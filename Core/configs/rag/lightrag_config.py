from typing import Literal

from pydantic import Field

from Core.configs.embedding_config import EmbeddingConfig
from Core.configs.rag.base_config import BaseRAGStrategyConfig


class LightRAGConfig(BaseRAGStrategyConfig):
    strategy: Literal["lightrag"] = "lightrag"
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = Field(
        default="hybrid",
        description="LightRAG query mode.",
    )
    fallback_mode: Literal["local", "global", "hybrid", "naive", "mix", "none"] = Field(
        default="mix",
        description="Fallback query mode when the primary mode returns no chunks.",
    )
    topk: int = Field(default=10, description="Number of LightRAG retrieval candidates.")
    chunk_topk: int = Field(default=10, description="Number of text chunks used by LightRAG query.")
    answer_style: Literal["default", "short"] = Field(
        default="short",
        description="Use concise dataset-friendly answers when set to short.",
    )
    include_references: bool = Field(
        default=False,
        description="Ask LightRAG to include reference blocks in generated answers.",
    )
    supporting_evidence_topk: int = Field(
        default=3,
        description="Number of traceable paragraph evidence items exported for official evidence metrics.",
    )
    enable_graph_merge: bool = Field(
        default=True,
        description="Merge custom chunk entity extraction results into LightRAG graph storages.",
    )
    repair_empty_graph: bool = Field(
        default=True,
        description="Rebuild an existing LightRAG workdir when chunks exist but entity/relation graph storages are empty.",
    )
    working_dir_name: str = Field(
        default="lightrag_workdir",
        description="Per-document LightRAG storage directory under save_path.",
    )
    force_rebuild: bool = Field(default=False, description="Rebuild LightRAG storage even if a manifest exists.")
    chunk_token_size: int = Field(default=1200, description="LightRAG chunk token size.")
    chunk_overlap_token_size: int = Field(default=120, description="LightRAG chunk overlap token size.")
    embedding_dim: int = Field(
        default=1024,
        description="Embedding dimension for the configured embedding model.",
    )
    max_total_tokens: int = Field(default=12000, description="LightRAG maximum query context tokens.")
    embedding_config: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
