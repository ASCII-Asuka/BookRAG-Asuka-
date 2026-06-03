# evibridge RAG 策略中的 证据选择器（Evidence Selector） 。在经历了 BM25 粗排、图游走（PPR）打分、最短路径连通之后，系统手头会有一大堆带有分数的候选节点。但是，大语言模型（LLM）的上下文窗口（Token 限制）和处理能力是有限的。
# 这个选择器的核心作用就是： 在有限的预算（最大节点数、最大 Token 数）内，像装背包一样，挑选出性价比最高、覆盖面最广、冗余度最低的证据组合，喂给 LLM。
from typing import Dict, List, Set

from pydantic import BaseModel, Field

from Core.Index.EvidenceBridgeIndex import EvidenceBlock, EvidenceBridgeIndex, evidence_tokenize
from Core.rag.evibridge_demand import EvidenceDemand

try:
    from Core.utils.utils import num_tokens
except Exception:
    num_tokens = None


class SelectedEvidence(BaseModel):
    block: EvidenceBlock
    score: float
    score_parts: Dict[str, float] = Field(default_factory=dict)
    bridge_types: List[str] = Field(default_factory=list)
    evidence_role: str = "answer_evidence"
    selection_rank: int = 0


def select_budgeted_evidence(
    index: EvidenceBridgeIndex,
    query: str,
    candidate_scores: Dict[int, float],
    demand: EvidenceDemand,
    max_blocks: int,
    max_tokens: int,
    final_evidence_types: List[str] | None = None,
    bridge_auxiliary_types: List[str] | None = None,
    paragraph_quota: int = 0,
    auxiliary_quota: int = 2,
) -> List[SelectedEvidence]:
    query_terms = set(evidence_tokenize(query))
    final_types = set(final_evidence_types or ["paragraph", "table", "caption", "figure"])
    auxiliary_types = set(bridge_auxiliary_types or ["entity", "summary", "patch", "title"])
    selected: List[SelectedEvidence] = []
    selected_ids: Set[int] = set()
    used_tokens = 0
    remaining = {
        block_id: score
        for block_id, score in candidate_scores.items()
        if block_id in index.blocks
    }

    while remaining and len(selected) < max_blocks:
        best_id = None
        best_score = float("-inf")
        best_parts: Dict[str, float] = {}
        best_role = "answer_evidence"
        paragraph_candidates = [
            block_id
            for block_id in remaining
            if index.blocks.get(block_id) and index.blocks[block_id].block_type == "paragraph"
        ]
        selected_paragraphs = sum(1 for item in selected if item.block.block_type == "paragraph")
        selected_auxiliary = sum(1 for item in selected if item.evidence_role == "bridge_auxiliary")
        require_paragraph = (
            paragraph_quota > 0
            and selected_paragraphs < paragraph_quota
            and bool(paragraph_candidates)
            and demand.intent == "fact"
            and "table" not in demand.modality
            and "figure" not in demand.modality
        )
        for block_id, base_score in remaining.items():
            block = index.blocks[block_id]
            role = _evidence_role(block, final_types, auxiliary_types)
            if require_paragraph and block.block_type != "paragraph":
                continue
            if role == "bridge_auxiliary" and selected_auxiliary >= auxiliary_quota:
                answer_remaining = any(
                    _evidence_role(index.blocks[item_id], final_types, auxiliary_types) == "answer_evidence"
                    for item_id in remaining
                    if item_id in index.blocks and item_id != block_id
                )
                if answer_remaining:
                    continue
            block_terms = set(evidence_tokenize(block.text))
            token_cost = _token_cost(block.text)
            if used_tokens + token_cost > max_tokens and selected:
                continue
            relevance = float(base_score) + _lexical_overlap(query_terms, block_terms)
            coverage = _coverage_bonus(block, demand)
            connectivity = _connectivity_bonus(index, block_id, selected_ids)
            diversity = _diversity_bonus(block_terms, selected)
            bridge_diversity = _bridge_diversity_bonus(index, block_id, demand, selected_ids)
            redundancy = _redundancy_penalty(block_terms, selected)
            role_bonus = 0.35 if role == "answer_evidence" else -0.2
            if block.block_type == "paragraph" and demand.intent == "fact":
                role_bonus += 0.25
            cost = token_cost / max(max_tokens, 1)
            total = relevance + coverage + connectivity + diversity + bridge_diversity + role_bonus - redundancy - 0.15 * cost
            if total > best_score:
                best_id = block_id
                best_score = total
                best_role = role
                best_parts = {
                    "relevance": round(relevance, 6),
                    "coverage": round(coverage, 6),
                    "connectivity": round(connectivity, 6),
                    "diversity": round(diversity, 6),
                    "bridge_diversity": round(bridge_diversity, 6),
                    "role_bonus": round(role_bonus, 6),
                    "redundancy": round(redundancy, 6),
                    "cost": round(cost, 6),
                }
        if best_id is None:
            break
        block = index.blocks[best_id]
        selected.append(
            SelectedEvidence(
                block=block,
                score=round(best_score, 6),
                score_parts=best_parts,
                bridge_types=_bridge_types(index, best_id),
                evidence_role=best_role,
                selection_rank=len(selected) + 1,
            )
        )
        selected_ids.add(best_id)
        used_tokens += _token_cost(block.text)
        remaining.pop(best_id, None)

    return selected


def _evidence_role(block: EvidenceBlock, final_types: Set[str], auxiliary_types: Set[str]) -> str:
    if block.block_type in final_types:
        return "answer_evidence"
    if block.block_type in auxiliary_types:
        return "bridge_auxiliary"
    return "answer_evidence"


def _lexical_overlap(query_terms: Set[str], block_terms: Set[str]) -> float:
    if not query_terms or not block_terms:
        return 0.0
    return len(query_terms & block_terms) / len(query_terms)


def _coverage_bonus(block: EvidenceBlock, demand: EvidenceDemand) -> float:
    bonus = 0.0
    if block.block_type in demand.modality:
        bonus += 0.35
    if "table" in demand.modality and block.block_type in {"table", "caption"}:
        bonus += 0.45
    if demand.granularity == "summary" and block.block_type == "summary":
        bonus += 0.35
    if demand.granularity == "entity" and block.block_type in {"entity", "paragraph"}:
        bonus += 0.15
    return bonus


def _connectivity_bonus(index: EvidenceBridgeIndex, block_id: int, selected_ids: Set[int]) -> float:
    if not selected_ids:
        return 0.0
    for bridge in index.get_related_bridges([block_id], expand_depth=1):
        if bridge.source_id in selected_ids or bridge.target_id in selected_ids:
            return 0.3
    return 0.0


def _bridge_diversity_bonus(
    index: EvidenceBridgeIndex,
    block_id: int,
    demand: EvidenceDemand,
    selected_ids: Set[int],
) -> float:
    available_types = set(_bridge_types(index, block_id))
    if not available_types:
        return 0.0
    demanded = set(demand.bridge_need or [])
    demanded_hit = len(available_types & demanded) / max(len(demanded), 1) if demanded else 0.0
    if not selected_ids:
        return 0.12 * demanded_hit
    selected_types = set()
    for selected_id in selected_ids:
        selected_types.update(_bridge_types(index, selected_id))
    new_types = available_types - selected_types
    return 0.12 * demanded_hit + 0.08 * (len(new_types) / max(len(available_types), 1))


def _diversity_bonus(block_terms: Set[str], selected: List[SelectedEvidence]) -> float:
    if not selected:
        return 0.1
    selected_terms = set()
    for item in selected:
        selected_terms.update(evidence_tokenize(item.block.text))
    if not block_terms:
        return 0.0
    novelty = len(block_terms - selected_terms) / len(block_terms)
    return 0.2 * novelty


def _redundancy_penalty(block_terms: Set[str], selected: List[SelectedEvidence]) -> float:
    if not selected or not block_terms:
        return 0.0
    max_overlap = 0.0
    for item in selected:
        other_terms = set(evidence_tokenize(item.block.text))
        overlap = len(block_terms & other_terms) / max(len(block_terms | other_terms), 1)
        max_overlap = max(max_overlap, overlap)
    return 0.45 * max_overlap


def _bridge_types(index: EvidenceBridgeIndex, block_id: int) -> List[str]:
    return sorted(
        {
            bridge.bridge_type
            for bridge in index.get_related_bridges([block_id], expand_depth=1)
        }
    )


def _token_cost(text: str) -> int:
    if num_tokens is not None:
        try:
            return max(int(num_tokens(text or "")), 1)
        except Exception:
            pass
    return max(len(evidence_tokenize(text or "")), 1)
