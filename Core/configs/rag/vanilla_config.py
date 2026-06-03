from .base_config import BaseRAGStrategyConfig
from typing import Literal
from pydantic import Field
from Core.configs.vdb_config import VDBConfig
from Core.configs.rerank_config import RerankerConfig


class VanillaConfig(BaseRAGStrategyConfig):
    vdb_config: VDBConfig = Field(default_factory=VDBConfig)
    strategy: Literal["vanilla"] = "vanilla"
    topk: int = Field(
        default=5, description="The number of topk retrieval results for Vanilla RAG."
    )
    retrieval_method: Literal[
        "vanilla",
        "bm25",
        "raptor",
        "pdf_vanilla",
        "hybrid",
        "bm25_rerank",
        "abstract_only",
    ] = Field(
        default="vanilla",
        description="The retrieval method to use.",
    )
    corpus_unit: Literal["chunk", "paragraph"] = Field(
        default="chunk",
        description="Corpus unit for vector baselines. Use paragraph for Qasper evidence alignment.",
    )
    bm25_corpus: Literal["chunk", "paragraph"] = Field(
        default="chunk",
        description="BM25 corpus unit. Use paragraph for Qasper official evidence alignment.",
    )
    answer_style: Literal["default", "short"] = Field(
        default="default",
        description="Answer generation style for vanilla baselines.",
    )
    hybrid_bm25_topk: int = Field(default=50, description="BM25 candidate count for hybrid RRF retrieval.")
    hybrid_dense_topk: int = Field(default=50, description="Dense candidate count for hybrid RRF retrieval.")
    rrf_k: int = Field(default=60, description="RRF denominator constant for hybrid retrieval.")
    bm25_vdb_dir_name: str = Field(default="bm25_vdb", description="BM25 index dir used by hybrid/rerank baselines.")
    rerank_topk: int = Field(default=50, description="BM25 candidate count before reranking.")
    rerank_batch_size: int = Field(default=50, description="Batch size for remote/local reranker calls.")
    reranker_config: RerankerConfig = Field(default_factory=RerankerConfig)
