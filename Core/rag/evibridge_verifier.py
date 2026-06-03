# 是 evibridge RAG 策略中的 证据充足性校验器（Evidence Sufficiency Verifier） 。
# 在检索系统选出一批证据之后、将它们发给 LLM 生成最终答案之前，系统需要先做一次“自我反省”：
# 这批证据真的够回答用户的问题了吗？如果不够，缺了什么？下一步该怎么做（是该顺着表格去扩充，还是顺着语义去寻找实体）？
# 这个文件就是用来执行这种自我反省逻辑的。它主要分为规则引擎和 LLM 混合验证两个部分
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from Core.Index.EvidenceBridgeIndex import EvidenceBlock, EvidenceBridge, evidence_tokenize
from Core.rag.evibridge_demand import EvidenceDemand

ALLOWED_MISSING_TYPES = {"evidence", "table", "summary", "context", "entity", "noise"}
ALLOWED_BRIDGE_TYPES = {"context", "semantic", "hierarchy"}
ALLOWED_NEXT_ACTIONS = {
    "accept",
    "expand_context",
    "expand_semantic_bridge",
    "expand_hierarchy_context",
    "expand_table_caption",
    "reduce_noise",
    "expand_relevant_evidence",
}
HARD_RULE_MISSING = {"evidence", "table_or_caption_context", "too_noisy"}


class SufficiencyVerdict(BaseModel):
    sufficient: bool
    missing: List[str] = Field(default_factory=list)
    missing_types: List[str] = Field(default_factory=list)
    missing_bridge_types: List[str] = Field(default_factory=list)
    relevance: float = 0.0
    connectivity: float = 0.0
    coverage: float = 0.0
    specificity: float = 0.0
    noise: float = 0.0
    noise_warning: bool = False
    next_bridge: List[str] = Field(default_factory=list)
    next_action: str = "accept"
    reason: str = ""


class RuleBasedSufficiencyVerifier:
    def __init__(
        self,
        relevance_threshold: float = 0.35,
        coverage_threshold: float = 0.55,
        noise_threshold: float = 0.45,
    ):
        self.relevance_threshold = relevance_threshold
        self.coverage_threshold = coverage_threshold
        self.noise_threshold = noise_threshold

    def verify(
        self,
        query: str,
        demand: EvidenceDemand,
        evidence: List[EvidenceBlock],
        bridges: List[EvidenceBridge],
    ) -> SufficiencyVerdict:
        missing: List[str] = []
        query_terms = set(evidence_tokenize(query))
        evidence_terms = set()
        for block in evidence:
            evidence_terms.update(evidence_tokenize(block.text))
        relevance = len(query_terms & evidence_terms) / max(len(query_terms), 1)
        coverage = self._coverage(demand, evidence)
        connectivity = self._connectivity(evidence, bridges)
        specificity = self._specificity(evidence)
        noise = self._noise(query_terms, evidence)
        missing_types: List[str] = []
        missing_bridge_types: List[str] = []

        if not evidence:
            missing.append("evidence")
            missing_types.append("evidence")
        if relevance < self.relevance_threshold:
            missing.append("relevant_evidence")
        if coverage < self.coverage_threshold:
            missing.append("demand_coverage")
        if "table" in demand.modality and not any(
            block.block_type in {"table", "caption"} for block in evidence
        ):
            missing.append("table_or_caption_context")
            missing_types.append("table")
        if demand.intent in {"multi-hop", "comparison"} and len(evidence) >= 2 and connectivity < 0.5:
            missing.append("semantic_link")
            missing_bridge_types.append("semantic")
        if demand.granularity == "summary" and not any(block.block_type == "summary" for block in evidence):
            missing.append("hierarchy_context")
            missing_types.append("summary")
            missing_bridge_types.append("hierarchy")
        if noise > self.noise_threshold:
            missing.append("too_noisy")

        next_bridge = self._next_bridge(missing, demand)
        for bridge_type in next_bridge:
            if bridge_type not in missing_bridge_types:
                missing_bridge_types.append(bridge_type)
        next_action = self._next_action(missing, missing_types, missing_bridge_types)
        sufficient = not missing
        return SufficiencyVerdict(
            sufficient=sufficient,
            missing=list(dict.fromkeys(missing)),
            missing_types=list(dict.fromkeys(missing_types)),
            missing_bridge_types=list(dict.fromkeys(missing_bridge_types)),
            relevance=round(relevance, 6),
            connectivity=round(connectivity, 6),
            coverage=round(coverage, 6),
            specificity=round(specificity, 6),
            noise=round(noise, 6),
            noise_warning=noise > self.noise_threshold,
            next_bridge=next_bridge,
            next_action=next_action,
            reason="sufficient" if sufficient else ", ".join(list(dict.fromkeys(missing))[:3]),
        )

    @staticmethod
    def _coverage(demand: EvidenceDemand, evidence: List[EvidenceBlock]) -> float:
        if not evidence:
            return 0.0
        hits = 0
        total = 1
        if "text" in demand.modality:
            total += 1
            hits += int(any(block.block_type in {"paragraph", "title", "summary"} for block in evidence))
        if "table" in demand.modality:
            total += 1
            hits += int(any(block.block_type in {"table", "caption"} for block in evidence))
        if "figure" in demand.modality:
            total += 1
            hits += int(any(block.block_type in {"figure", "caption"} for block in evidence))
        if demand.scope in {"document", "multi-document"}:
            total += 1
            hits += int(any(block.block_type in {"summary", "title"} for block in evidence))
        hits += 1
        return hits / total

    @staticmethod
    def _connectivity(evidence: List[EvidenceBlock], bridges: List[EvidenceBridge]) -> float:
        if len(evidence) <= 1:
            return 1.0 if evidence else 0.0
        evidence_ids = {block.block_id for block in evidence}
        connected_ids = set()
        for bridge in bridges:
            if bridge.source_id in evidence_ids and bridge.target_id in evidence_ids:
                connected_ids.add(bridge.source_id)
                connected_ids.add(bridge.target_id)
        return len(connected_ids) / len(evidence_ids)

    @staticmethod
    def _specificity(evidence: List[EvidenceBlock]) -> float:
        if not evidence:
            return 0.0
        specific = sum(1 for block in evidence if block.block_type in {"paragraph", "table", "caption", "figure"})
        return specific / len(evidence)

    @staticmethod
    def _noise(query_terms: set, evidence: List[EvidenceBlock]) -> float:
        if not evidence:
            return 0.0
        irrelevant = 0
        for block in evidence:
            block_terms = set(evidence_tokenize(block.text))
            if query_terms and not (query_terms & block_terms):
                irrelevant += 1
        return irrelevant / len(evidence)

    @staticmethod
    def _next_bridge(missing: List[str], demand: EvidenceDemand) -> List[str]:
        if "table_or_caption_context" in missing:
            return ["context"]
        next_bridge: List[str] = []
        if any(item in missing for item in ["table_or_caption_context", "demand_coverage"]):
            next_bridge.append("context")
        if any(item in missing for item in ["semantic_link", "relevant_evidence"]):
            next_bridge.append("semantic")
        if any(item in missing for item in ["hierarchy_context", "demand_coverage"]):
            next_bridge.append("hierarchy")
        if not next_bridge:
            next_bridge.extend(demand.bridge_need)
        return list(dict.fromkeys(next_bridge))

    @staticmethod
    def _next_action(
        missing: List[str],
        missing_types: List[str],
        missing_bridge_types: List[str],
    ) -> str:
        if not missing:
            return "accept"
        if "table" in missing_types or "table_or_caption_context" in missing:
            return "expand_table_caption"
        if "semantic" in missing_bridge_types or "semantic_link" in missing:
            return "expand_semantic_bridge"
        if "hierarchy" in missing_bridge_types or "hierarchy_context" in missing:
            return "expand_hierarchy_context"
        if "too_noisy" in missing:
            return "reduce_noise"
        if "relevant_evidence" in missing:
            return "expand_relevant_evidence"
        return "expand_context"


class EvidenceSufficiencyVerifier:
    def __init__(self, llm: Optional[Any] = None, enable_llm: bool = False):
        self.llm = llm
        self.enable_llm = enable_llm
        self.rule_verifier = RuleBasedSufficiencyVerifier()

    def verify(
        self,
        query: str,
        demand: EvidenceDemand,
        evidence: List[EvidenceBlock],
        bridges: List[EvidenceBridge],
    ) -> SufficiencyVerdict:
        rule_verdict = self.rule_verifier.verify(query, demand, evidence, bridges)
        if not self.enable_llm or self.llm is None:
            return rule_verdict
        try:
            prompt = self._prompt(query, demand, evidence, rule_verdict)
            llm_verdict = self.llm.get_json_completion(prompt, SufficiencyVerdict)
            return self._sanitize_llm_verdict(llm_verdict, rule_verdict)
        except Exception:
            return rule_verdict

    @staticmethod
    def _sanitize_llm_verdict(
        llm_verdict: SufficiencyVerdict,
        rule_verdict: SufficiencyVerdict,
    ) -> SufficiencyVerdict:
        if set(rule_verdict.missing) & HARD_RULE_MISSING:
            return rule_verdict

        missing_types = [
            item
            for item in (llm_verdict.missing_types or [])
            if str(item).strip().lower() in ALLOWED_MISSING_TYPES
        ]
        missing_bridge_types = [
            item
            for item in (llm_verdict.missing_bridge_types or [])
            if str(item).strip().lower() in ALLOWED_BRIDGE_TYPES
        ]
        next_bridge = [
            item
            for item in (llm_verdict.next_bridge or [])
            if str(item).strip().lower() in ALLOWED_BRIDGE_TYPES
        ]
        next_action = str(llm_verdict.next_action or "accept").strip()
        if next_action not in ALLOWED_NEXT_ACTIONS:
            next_action = rule_verdict.next_action if rule_verdict.missing else "accept"
        missing = list(dict.fromkeys(str(item) for item in (llm_verdict.missing or [])))
        sufficient = bool(llm_verdict.sufficient) and not missing
        if sufficient:
            next_action = "accept"
        return SufficiencyVerdict(
            sufficient=sufficient,
            missing=[] if sufficient else missing,
            missing_types=list(dict.fromkeys(str(item).strip().lower() for item in missing_types)),
            missing_bridge_types=list(dict.fromkeys(str(item).strip().lower() for item in missing_bridge_types)),
            relevance=max(0.0, min(float(llm_verdict.relevance), 1.0)),
            connectivity=max(0.0, min(float(llm_verdict.connectivity), 1.0)),
            coverage=max(0.0, min(float(llm_verdict.coverage), 1.0)),
            specificity=max(0.0, min(float(llm_verdict.specificity), 1.0)),
            noise=max(0.0, min(float(llm_verdict.noise), 1.0)),
            noise_warning=bool(llm_verdict.noise_warning),
            next_bridge=list(dict.fromkeys(str(item).strip().lower() for item in next_bridge)),
            next_action=next_action,
            reason=str(llm_verdict.reason or ("sufficient" if sufficient else rule_verdict.reason)),
        )

    @staticmethod
    def _prompt(
        query: str,
        demand: EvidenceDemand,
        evidence: List[EvidenceBlock],
        rule_verdict: SufficiencyVerdict,
    ) -> str:
        evidence_text = "\n".join(
            f"- {block.block_id} [{block.block_type}] {block.text[:600]}"
            for block in evidence
        )
        return (
            "You are an evidence sufficiency verifier for complex document QA. "
            "Return only JSON matching the schema.\n"
            f"Question: {query}\n"
            f"Demand: {demand.model_dump_json()}\n"
            f"Rule verdict: {rule_verdict.model_dump_json()}\n"
            f"Evidence:\n{evidence_text}"
        )
