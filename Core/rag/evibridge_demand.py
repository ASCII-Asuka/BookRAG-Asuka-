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


class DemandParser:
    def __init__(
        self,
        llm: Optional[Any] = None,
        mode: Literal["rule", "llm", "hybrid"] = "hybrid",
        confidence_threshold: float = 0.7,
        qasper_demand_mode: Literal["default", "conservative"] = "default",
        enable_boolean_answer_hint: bool = True,
        multi_hop_requires_explicit_bridge: bool = False,
    ):
        self.llm = llm
        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self.qasper_demand_mode = qasper_demand_mode
        self.enable_boolean_answer_hint = enable_boolean_answer_hint
        self.multi_hop_requires_explicit_bridge = multi_hop_requires_explicit_bridge

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
            return sanitize_demand(llm_demand, fallback=rule_demand)
        except TypeError:
            llm_demand = self.llm.get_json_completion(self._prompt(query), EvidenceDemand)
            return sanitize_demand(llm_demand, fallback=rule_demand)
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
        conservative = self.qasper_demand_mode == "conservative"
        multihop_match = (
            _EXPLICIT_MULTIHOP_RE.search(text)
            if self.multi_hop_requires_explicit_bridge or conservative
            else _MULTIHOP_RE.search(text)
        )

        if _TABLE_RE.search(text):
            intent = "table-figure"
            modality = ["text", "table"]
            bridge_need = ["context"]
            confidence = 0.9
            rationale = "explicit table or figure signal"
        elif conservative and self.enable_boolean_answer_hint and _BOOLEAN_RE.search(text):
            intent = "boolean"
            bridge_need = ["context"]
            confidence = 0.9
            rationale = "boolean question signal"
        elif conservative and (_NUMERIC_FACT_RE.search(text) or _LOCAL_HOW_FACT_RE.search(text)):
            intent = "fact"
            bridge_need = ["context", "semantic"]
            confidence = 0.9
            rationale = "qasper local fact question signal"
        elif _COMPARE_RE.search(text):
            intent = "comparison"
            bridge_need = ["semantic", "context"]
            granularity = "entity"
            confidence = 0.86
            rationale = "comparison signal"
        elif _GLOBAL_RE.search(text):
            intent = "global-summary"
            scope = "document"
            granularity = "summary"
            bridge_need = ["hierarchy", "context"]
            confidence = 0.88
            rationale = "global summary signal"
        elif _AGG_RE.search(text) and not (conservative and _NUMERIC_FACT_RE.search(text)):
            intent = "aggregation"
            scope = "section"
            bridge_need = ["hierarchy", "context"]
            confidence = 0.82
            rationale = "aggregation signal"
        elif multihop_match:
            intent = "multi-hop"
            scope = "multi-document"
            granularity = "entity"
            bridge_need = ["semantic", "context"]
            confidence = 0.78
            rationale = "multi-hop signal"

        return EvidenceDemand(
            intent=intent,
            scope=scope,
            modality=modality,
            granularity=granularity,
            bridge_need=list(dict.fromkeys(bridge_need)),
            confidence=confidence,
            source="rule",
            rationale=rationale,
        )

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

    return EvidenceDemand(
        intent=demand.intent,
        scope=demand.scope,
        modality=list(dict.fromkeys(modality)),
        granularity=granularity,
        bridge_need=list(dict.fromkeys(bridge_need)),
        confidence=max(0.0, min(float(demand.confidence), 1.0)),
        source=demand.source,
        rationale=demand.rationale,
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
