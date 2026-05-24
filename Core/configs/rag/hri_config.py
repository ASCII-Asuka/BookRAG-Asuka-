from typing import Dict, Literal, Optional

from pydantic import Field

from Core.configs.vdb_config import VDBConfig

from .base_config import BaseRAGStrategyConfig


DEFAULT_EVIDENCE_BUDGETS = {
    "locating": {
        "definition": 2,
        "condition": 1,
        "requirement": 4,
        "table": 1,
        "exception": 1,
        "supplement": 1,
        "article": 4,
    },
    "comprehensive": {
        "definition": 2,
        "condition": 2,
        "requirement": 6,
        "table": 3,
        "exception": 2,
        "supplement": 2,
        "article": 4,
    },
    "statistical": {
        "definition": 1,
        "condition": 1,
        "requirement": 3,
        "table": 5,
        "exception": 1,
        "supplement": 1,
        "article": 3,
    },
}


class HRIRAGConfig(BaseRAGStrategyConfig):
    strategy: Literal["hri"] = "hri"
    ablation_variant: Literal[
        "full",
        "wo_tree",
        "wo_relation",
        "wo_planner",
        "wo_evidence_chain",
    ] = Field(
        default="full",
        description="HRI ablation variant. 'full' keeps all HRI modules enabled.",
    )
    topk: int = Field(default=8, description="Backward-compatible coarse retrieval top-k.")
    question_classifier: Literal["rule", "llm", "hybrid"] = Field(
        default="hybrid", description="Question planning mode for HRI retrieval."
    )
    classification_model: Literal["llm"] = Field(
        default="llm", description="Planner model family. Currently only LLM is supported."
    )
    classification_confidence_threshold: float = Field(
        default=0.7, description="Rule confidence threshold before falling back to LLM."
    )
    enable_query_decomposition: bool = Field(
        default=True, description="Whether comprehensive queries can create retrieval sub-questions."
    )
    max_sub_questions: int = Field(default=4, description="Maximum HRI retrieval sub-questions.")
    bm25_topk: Optional[int] = Field(default=None, description="Number of BM25 coarse candidates.")
    embedding_topk: Optional[int] = Field(default=None, description="Number of vector coarse candidates.")
    enable_vector_recall: bool = Field(
        default=False, description="Whether to merge HRI vector candidates with BM25."
    )
    hybrid_bm25_weight: float = Field(default=0.55, description="Weight for normalized BM25 score.")
    hybrid_vector_weight: float = Field(default=0.45, description="Weight for normalized vector score.")
    hri_vdb_config: VDBConfig = Field(default_factory=VDBConfig)
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
    evidence_budgets: Dict[str, Dict[str, int]] = Field(
        default_factory=lambda: {
            question_type: budget.copy()
            for question_type, budget in DEFAULT_EVIDENCE_BUDGETS.items()
        },
        description="Per-question-type evidence budgets grouped by semantic role.",
    )
