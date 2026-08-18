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
        "lead_only",
        "full_document",
        "longrag",
        "ircot",
        "react",
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
    supporting_evidence_topk: int = Field(
        default=4,
        ge=1,
        description="Maximum validated supporting evidence items submitted for short-answer evaluation.",
    )
    full_document_max_context_tokens: int = Field(
        default=30000,
        description="Maximum prompt context tokens for the full-document long-context baseline.",
    )
    longrag_unit_tokens: int = Field(
        default=4096,
        description="Approximate maximum tokens for one LongRAG retrieval unit.",
    )
    longrag_max_context_tokens: int = Field(
        default=30000,
        description="Maximum prompt context tokens for the LongRAG long-reader baseline.",
    )
    ircot_max_steps: int = Field(default=3, description="Maximum interleaved reasoning-retrieval steps.")
    ircot_step_topk: int = Field(default=3, description="BM25 evidence count retrieved at each IRCoT step.")
    ircot_final_topk: int = Field(default=10, description="Maximum deduplicated evidence count used for final IRCoT answer generation.")
    react_max_steps: int = Field(default=7, ge=1)
    react_search_topk: int = Field(default=1, ge=1)
    react_page_observation_units: int = Field(default=5, ge=1)
    react_prompt_file: str = Field(default="")
    react_dataset_name: Literal["qasper", "hotpotqa"] = "qasper"
