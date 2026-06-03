from .base_config import BaseRAGStrategyConfig
from typing import Literal
from pydantic import Field
from Core.configs.vdb_config import VDBConfig


class VanillaConfig(BaseRAGStrategyConfig):
    vdb_config: VDBConfig = Field(default_factory=VDBConfig)
    strategy: Literal["vanilla"] = "vanilla"
    topk: int = Field(
        default=5, description="The number of topk retrieval results for Vanilla RAG."
    )
    retrieval_method: Literal["vanilla", "bm25", "raptor", "pdf_vanilla"] = Field(
        default="vanilla",
        description="The retrieval method to use: vanilla (text-only), bm25, raptor (text-only), pdf_vanilla (supports PDF documents).",
    )
    bm25_corpus: Literal["chunk", "paragraph"] = Field(
        default="chunk",
        description="BM25 corpus unit. Use paragraph for Qasper official evidence alignment.",
    )
    answer_style: Literal["default", "short"] = Field(
        default="default",
        description="Answer generation style for vanilla baselines.",
    )
