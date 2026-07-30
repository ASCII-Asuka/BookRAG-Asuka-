# 意图感知模块（Demand Parser）。
# 它的核心作用是：在收到用户的查询（Query）时，将其解析为一个 证据需求（Evidence Demand）对象 ，从而指导下游检索系统该采用何种游走策略、侧重哪种类型的边、以及需要什么粒度的证据。
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


EvidenceIntent = Literal[
    "fact",
    "boolean",
    "multi-hop",
    "comparison",
    "table-figure",
    "aggregation",
    "global-summary",
]
EvidenceScope = Literal["local", "section", "document", "multi-document"]
EvidenceGranularity = Literal["block", "patch", "entity", "summary"]


class EvidenceDemand(BaseModel):
    intent: EvidenceIntent = "fact"
    scope: EvidenceScope = "local"
    modality: List[str] = Field(default_factory=lambda: ["text"])
    granularity: EvidenceGranularity = "block"
    bridge_need: List[str] = Field(default_factory=lambda: ["context", "semantic"])
    confidence: float = 1.0
    source: str = "rule"
    rationale: str = ""
    subqueries: List[str] = Field(default_factory=list)
    provenance: Dict[str, Any] = Field(default_factory=dict)


ALLOWED_BRIDGE_NEEDS = {"context", "semantic", "hierarchy"}
ALLOWED_MODALITIES = {"text", "table", "figure", "caption"}


_TABLE_RE = re.compile(r"\b(table|tab\.|figure|fig\.|chart|caption|row|column)\b|表|图|图片|图表|图注", re.I)
_COMPARE_RE = re.compile(r"\b(compare(?:d|s|ing)?|difference|versus|vs\.?|better|worse)\b|比较|区别|差异|相比|对比", re.I)
_MULTIHOP_RE = re.compile(r"\b(why|how|relationship|connect|supporting|evidence)\b|关系|联系|依据|为什么|如何", re.I)
_EXPLICIT_MULTIHOP_RE = re.compile(
    r"\b(relationship|connect(?:ion|ed)?|supporting evidence|evidence for|link(?:ed)?|bridge)\b|关系|联系|依据",
    re.I,
)
_GLOBAL_RE = re.compile(r"\b(summarize|summary|overall|entire|whole|global)\b|总结|概括|全文|整体|全局", re.I)
_AGG_RE = re.compile(r"\b(how many|count|average|total|all|list)\b|多少|几个|统计|列出|全部", re.I)
_BOOLEAN_RE = re.compile(r"^\s*(is|are|do|does|did|can|was|were|has|have|should|would|could)\b", re.I)
_NUMERIC_FACT_RE = re.compile(r"^\s*(how many|how much|what (?:is|was) the size|what size)\b", re.I)
_LOCAL_HOW_FACT_RE = re.compile(r"^\s*how\s+(?:was|were|is|are|did|do|does)\b", re.I)
_IMPLICIT_ENTITY_CHAIN_RE = re.compile(
    r"^\s*(?:what|which|who|where|when)\b.+\b(?:who|whose|that|which)\b",
    re.I,
)
_NESTED_RELATION_RE = re.compile(
    r"^\s*(?:what|which|who|where|when)\b.+\bof\b.+\bof\b",
    re.I,
)
_PARALLEL_ENTITY_RE = re.compile(r"\b(?:both|same|respectively|versus|vs\.?)\b", re.I)


class DemandParser:
    def __init__(
        self,
        llm: Optional[Any] = None,
        mode: Literal["rule", "llm", "hybrid"] = "hybrid",
        confidence_threshold: float = 0.7,
        dataset_profile: Literal["auto", "qasper", "hotpotqa"] = "auto",
        qasper_demand_mode: Literal["default", "conservative"] = "default",
        enable_boolean_answer_hint: bool = True,
        multi_hop_requires_explicit_bridge: bool = False,
        enable_implicit_multihop: bool = True,
    ):
        self.llm = llm
        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self.dataset_profile = dataset_profile
        self.qasper_demand_mode = qasper_demand_mode
        self.enable_boolean_answer_hint = enable_boolean_answer_hint
        self.multi_hop_requires_explicit_bridge = multi_hop_requires_explicit_bridge
        self.enable_implicit_multihop = enable_implicit_multihop

    def parse(self, query: str) -> EvidenceDemand:
        rule_demand = self._parse_rule(query)
        if self.mode == "rule" or rule_demand.confidence >= self.confidence_threshold:
            return sanitize_demand(rule_demand, fallback=rule_demand)
        if self.llm is None or self.mode not in {"llm", "hybrid"}:
            return sanitize_demand(rule_demand, fallback=rule_demand)
        try:
            llm_demand = self.llm.get_json_completion(
                prompt_or_memory=self._prompt(query),
                schema=EvidenceDemand,
            )
            parsed = sanitize_demand(llm_demand, fallback=rule_demand)
            parsed.source = "hybrid" if self.mode == "hybrid" else "llm"
            parsed.provenance = {
                **parsed.provenance,
                "parser": parsed.source,
                "rule_fallback_intent": rule_demand.intent,
            }
            return parsed
        except TypeError:
            llm_demand = self.llm.get_json_completion(self._prompt(query), EvidenceDemand)
            parsed = sanitize_demand(llm_demand, fallback=rule_demand)
            parsed.source = "hybrid" if self.mode == "hybrid" else "llm"
            parsed.provenance = {
                **parsed.provenance,
                "parser": parsed.source,
                "rule_fallback_intent": rule_demand.intent,
            }
            return parsed
        except Exception:
            return sanitize_demand(rule_demand, fallback=rule_demand)

    def _parse_rule(self, query: str) -> EvidenceDemand:
        text = query or ""
        modality = ["text"]
        bridge_need = ["context", "semantic"]
        intent: EvidenceIntent = "fact"
        scope: EvidenceScope = "local"
        granularity: EvidenceGranularity = "block"
        confidence = 0.68
        rationale = "default fact demand"
        signal = "default_fact"
        conservative = (
            self.qasper_demand_mode == "conservative"
            or self.dataset_profile == "qasper"
        )
        hotpot_profile = self.dataset_profile == "hotpotqa"
        explicit_multihop = _EXPLICIT_MULTIHOP_RE.search(text)
        implicit_multihop = (
            (
                _IMPLICIT_ENTITY_CHAIN_RE.search(text)
                or _NESTED_RELATION_RE.search(text)
            )
            if self.enable_implicit_multihop
            else None
        )
        multihop_match = (
            explicit_multihop or implicit_multihop
            if self.multi_hop_requires_explicit_bridge or conservative
            else _MULTIHOP_RE.search(text) or implicit_multihop
        )
        subqueries: List[str] = []

        if _TABLE_RE.search(text):
            intent = "table-figure"
            modality = ["text", "table"]
            bridge_need = ["context"]
            confidence = 0.9
            rationale = "explicit table or figure signal"
            signal = "table_or_figure"
        elif _COMPARE_RE.search(text) or _PARALLEL_ENTITY_RE.search(text):
            intent = "comparison"
            bridge_need = ["semantic", "context"]
            granularity = "entity"
            confidence = 0.86
            rationale = "comparison signal"
            signal = "comparison"
            subqueries = self._decompose_comparison(text)
        elif conservative and self.enable_boolean_answer_hint and _BOOLEAN_RE.search(text):
            intent = "boolean"
            bridge_need = ["context"]
            confidence = 0.9
            rationale = "boolean question signal"
            signal = "boolean"
        elif conservative and (_NUMERIC_FACT_RE.search(text) or _LOCAL_HOW_FACT_RE.search(text)):
            intent = "fact"
            bridge_need = ["context", "semantic"]
            confidence = 0.9
            rationale = "qasper local fact question signal"
            signal = "local_fact"
        elif _GLOBAL_RE.search(text):
            intent = "global-summary"
            scope = "document"
            granularity = "summary"
            bridge_need = ["hierarchy", "context"]
            confidence = 0.88
            rationale = "global summary signal"
            signal = "global_summary"
        elif _AGG_RE.search(text) and not (conservative and _NUMERIC_FACT_RE.search(text)):
            intent = "aggregation"
            scope = "section"
            bridge_need = ["hierarchy", "context"]
            confidence = 0.82
            rationale = "aggregation signal"
            signal = "aggregation"
        elif multihop_match or hotpot_profile:
            intent = "multi-hop"
            scope = "multi-document"
            granularity = "entity"
            bridge_need = ["semantic", "context"]
            confidence = 0.78
            if implicit_multihop and not explicit_multihop:
                rationale = "implicit entity or attribute chain signal"
                signal = "implicit_entity_chain"
                subqueries = self._decompose_implicit_chain(text)
            elif hotpot_profile and not explicit_multihop:
                rationale = "HotpotQA implicit multi-hop prior"
                signal = "hotpotqa_implicit_multihop"
                subqueries = (
                    self._decompose_implicit_chain(text)
                    or self._decompose_hotpot_chain(text)
                )
            else:
                rationale = "explicit multi-hop signal"
                signal = "explicit_multi_hop"
                subqueries = self._decompose_implicit_chain(text)

        return EvidenceDemand(
            intent=intent,
            scope=scope,
            modality=modality,
            granularity=granularity,
            bridge_need=list(dict.fromkeys(bridge_need)),
            confidence=confidence,
            source="rule",
            rationale=rationale,
            subqueries=subqueries,
            provenance={"parser": "rule", "signals": [signal]},
        )

    @staticmethod
    def _decompose_implicit_chain(query: str) -> List[str]:
        text = str(query or "").strip().rstrip("?")
        relative = re.search(r"\b(who|whose|that|which)\b", text, re.I)
        if relative:
            marker = relative.group(1)
            inner = f"{marker.capitalize()} {text[relative.end():].strip()}?"
            outer = f"{text[:relative.start()].strip()}?"
            return list(dict.fromkeys(item for item in [inner, outer] if len(item) > 2))[:2]
        of_matches = list(re.finditer(r"\bof\b", text, re.I))
        if len(of_matches) >= 2:
            split_at = of_matches[0].end()
            inner = f"What is known about {text[split_at:].strip()}?"
            outer = f"{text[:of_matches[0].start()].strip()}?"
            return list(dict.fromkeys([inner, outer]))[:2]
        return []

    @staticmethod
    def _decompose_comparison(query: str) -> List[str]:
        text = str(query or "").strip().rstrip("?")
        parts = re.split(r"\b(?:and|versus|vs\.?)\b", text, maxsplit=1, flags=re.I)
        if len(parts) != 2:
            return []
        return [
            f"What evidence is relevant to {part.strip()}?"
            for part in parts
            if part.strip()
        ][:2]

    @staticmethod
    def _decompose_hotpot_chain(query: str) -> List[str]:
        text = str(query or "").strip().rstrip("?")
        possessive = re.search(
            r"((?:the\s+)?(?:[A-Za-z][\w'-]*\s+){0,3}"
            r"[A-Za-z][\w'-]*'s\s+[A-Za-z][\w'-]*)",
            text,
            re.I,
        )
        if possessive:
            relation = possessive.group(1).strip()
            outer = text[: possessive.start()] + "the bridge entity" + text[possessive.end() :]
            return [
                f"Who or what is {relation}?",
                f"{outer.strip()}?",
            ]
        return [
            f"What bridge entity is described in: {text}?",
            f"What attribute of that bridge entity answers: {text}?",
        ]

    @staticmethod
    def _prompt(query: str) -> str:
        return (
            "Parse the question into an evidence demand JSON with fields intent, "
            "scope, modality, granularity, bridge_need, confidence, source, rationale. "
            "bridge_need must only contain context, semantic, or hierarchy.\n"
            f"Question: {query}"
        )


def sanitize_demand(demand: EvidenceDemand, fallback: Optional[EvidenceDemand] = None) -> EvidenceDemand:
    fallback = fallback or EvidenceDemand()
    bridge_need = [
        str(item).strip().lower()
        for item in (demand.bridge_need or [])
        if str(item).strip().lower() in ALLOWED_BRIDGE_NEEDS
    ]
    if not bridge_need:
        bridge_need = [
            str(item).strip().lower()
            for item in (fallback.bridge_need or [])
            if str(item).strip().lower() in ALLOWED_BRIDGE_NEEDS
        ]
    if not bridge_need:
        bridge_need = ["context", "semantic"]

    modality = [
        str(item).strip().lower()
        for item in (demand.modality or [])
        if str(item).strip().lower() in ALLOWED_MODALITIES
    ]
    if not modality:
        modality = [
            str(item).strip().lower()
            for item in (fallback.modality or ["text"])
            if str(item).strip().lower() in ALLOWED_MODALITIES
        ] or ["text"]

    granularity = demand.granularity
    if demand.intent in {"fact", "boolean"}:
        granularity = "block"
    subqueries = []
    for item in demand.subqueries or fallback.subqueries or []:
        value = " ".join(str(item).split()).strip()
        if value and value not in subqueries:
            subqueries.append(value)
        if len(subqueries) >= 2:
            break
    source = str(demand.source or "").strip().lower()
    if source not in {"rule", "llm", "hybrid", "fallback"}:
        source = (
            str(fallback.source or "").strip().lower()
            if str(fallback.source or "").strip().lower() in {"rule", "llm", "hybrid", "fallback"}
            else "fallback"
        )

    return EvidenceDemand(
        intent=demand.intent,
        scope=demand.scope,
        modality=list(dict.fromkeys(modality)),
        granularity=granularity,
        bridge_need=list(dict.fromkeys(bridge_need)),
        confidence=max(0.0, min(float(demand.confidence), 1.0)),
        source=source,
        rationale=demand.rationale,
        subqueries=subqueries,
        provenance=dict(demand.provenance or fallback.provenance or {}),
    )


def weights_for_demand(
    demand: EvidenceDemand,
    typed_weights: Optional[Dict[str, Dict[str, float]]] = None,
    use_typed_weights: bool = True,
) -> Dict[str, float]:
    if not use_typed_weights:
        return {"context": 1 / 3, "semantic": 1 / 3, "hierarchy": 1 / 3}
    defaults = {
        "fact": {"context": 0.45, "semantic": 0.35, "hierarchy": 0.2},
        "boolean": {"context": 0.5, "semantic": 0.25, "hierarchy": 0.25},
        "multi-hop": {"context": 0.25, "semantic": 0.55, "hierarchy": 0.2},
        "comparison": {"context": 0.25, "semantic": 0.55, "hierarchy": 0.2},
        "table-figure": {"context": 0.65, "semantic": 0.2, "hierarchy": 0.15},
        "aggregation": {"context": 0.35, "semantic": 0.2, "hierarchy": 0.45},
        "global-summary": {"context": 0.35, "semantic": 0.2, "hierarchy": 0.45},
    }
    if typed_weights and demand.intent in typed_weights:
        raw_weights = typed_weights[demand.intent]
    else:
        raw_weights = defaults[demand.intent]
    total = sum(raw_weights.values()) or 1.0
    return {key: round(value / total, 6) for key, value in raw_weights.items()}
