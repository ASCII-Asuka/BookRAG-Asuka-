import logging
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from Core.prompts.hri_plan_prompt import HRI_QUERY_PLAN_PROMPT
from Core.provider.llm import LLM

log = logging.getLogger(__name__)


HRIQueryType = Literal["locating", "comprehensive", "statistical"]
HRIIntent = Literal[
    "definition_lookup",
    "article_lookup",
    "requirement_lookup",
    "condition_requirement",
    "parameter_table",
    "exception_or_supplement",
    "multi_evidence_synthesis",
    "aggregation",
]
HRISubQuestionType = Literal["retrieval", "synthesis"]
HRIAggregationOperation = Literal["COUNT", "LIST", "SUMMARIZE", "ANALYZE"]
HRIPlanSource = Literal["rule", "llm", "fallback"]

EVIDENCE_ROLES = {
    "definition",
    "condition",
    "requirement",
    "table",
    "exception",
    "supplement",
    "article",
}
RELATION_TYPES = {
    "defines",
    "condition_of",
    "requires",
    "refers_to",
    "parameter_of",
    "supplements",
    "exception_to",
}


class HRISubQuestion(BaseModel):
    question: str
    type: HRISubQuestionType = "retrieval"
    evidence_roles: List[str] = Field(default_factory=list)
    relation_types: List[str] = Field(default_factory=list)


class HRIAggregationSpec(BaseModel):
    operation: HRIAggregationOperation = "COUNT"
    filters: List[str] = Field(default_factory=list)
    target: Optional[str] = None
    scope: Optional[str] = None


class HRIPlanResult(BaseModel):
    query_type: HRIQueryType
    intent: HRIIntent
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_roles: List[str] = Field(default_factory=list)
    relation_types: List[str] = Field(default_factory=list)
    retrieval_focus: List[str] = Field(default_factory=list)
    sub_questions: List[HRISubQuestion] = Field(default_factory=list)
    aggregation: Optional[HRIAggregationSpec] = None
    rationale: str = ""
    source: HRIPlanSource = "rule"


class HRIQueryPlanner:
    """Question planner for hydro regulation RAG."""

    def __init__(
        self,
        llm: Optional[LLM],
        question_classifier: str = "hybrid",
        classification_model: str = "llm",
        confidence_threshold: float = 0.7,
        enable_query_decomposition: bool = True,
        max_sub_questions: int = 4,
    ):
        self.llm = llm
        self.question_classifier = question_classifier
        self.classification_model = classification_model
        self.confidence_threshold = confidence_threshold
        self.enable_query_decomposition = enable_query_decomposition
        self.max_sub_questions = max(max_sub_questions, 0)

    def analyze(self, query: str) -> HRIPlanResult:
        rule_plan = self._rule_plan(query)
        mode = (self.question_classifier or "hybrid").lower()

        if mode == "rule":
            return rule_plan
        if mode == "hybrid" and rule_plan.confidence >= self.confidence_threshold:
            return rule_plan
        if self.llm is None or not hasattr(self.llm, "get_json_completion"):
            return self._with_source(rule_plan, "fallback")
        if (self.classification_model or "llm").lower() != "llm":
            return self._with_source(rule_plan, "fallback")

        try:
            prompt = HRI_QUERY_PLAN_PROMPT.format(
                query=query,
                max_sub_questions=self.max_sub_questions,
            )
            raw_plan = self.llm.get_json_completion(prompt, schema=HRIPlanResult)
            llm_plan = self._coerce_plan(raw_plan)
            llm_plan.source = "llm"
            return self._normalize_plan(llm_plan)
        except Exception as exc:
            log.warning("HRI query planner failed, falling back to rule plan: %s", exc)
            return self._with_source(rule_plan, "fallback")

    def _rule_plan(self, query: str) -> HRIPlanResult:
        text = query or ""
        focus = self._retrieval_focus(text)

        if self._has_any(text, ["多少", "几个", "几项", "数量", "统计", "总数", "一共有", "共几", "列出", "清单"]):
            operation: HRIAggregationOperation = "LIST" if self._has_any(text, ["列出", "清单", "有哪些"]) else "COUNT"
            return self._normalize_plan(
                HRIPlanResult(
                    query_type="statistical",
                    intent="aggregation",
                    confidence=0.95,
                    evidence_roles=["requirement", "table", "article"],
                    relation_types=["requires", "refers_to", "parameter_of"],
                    retrieval_focus=focus,
                    aggregation=HRIAggregationSpec(
                        operation=operation,
                        filters=self._aggregation_filters(text),
                        target=self._aggregation_target(text),
                        scope=self._section_scope(text),
                    ),
                    rationale="问题包含统计、计数或列举意图，需要按结构范围聚合。",
                )
            )

        if self._has_any(text, ["什么是", "定义", "含义", "是指", "指什么"]):
            return self._normalize_plan(
                HRIPlanResult(
                    query_type="locating",
                    intent="definition_lookup",
                    confidence=0.95,
                    evidence_roles=["definition", "article"],
                    relation_types=["defines"],
                    retrieval_focus=focus,
                    rationale="问题询问术语定义，通常由单个定义证据回答。",
                )
            )

        if self._has_any(text, ["哪一条", "第几页", "在哪里", "位置"]) or re.search(r"第\s*[\d一二三四五六七八九十.、-]+\s*[章节条页]", text):
            return self._normalize_plan(
                HRIPlanResult(
                    query_type="locating",
                    intent="article_lookup",
                    confidence=0.9,
                    evidence_roles=["article"],
                    relation_types=[],
                    retrieval_focus=focus,
                    rationale="问题询问条文、章节或页码位置。",
                )
            )

        if self._has_any(text, ["当", "若", "如果", "条件", "超过", "达到"]) and self._has_any(text, ["应", "措施", "要求", "采取", "处置"]):
            return self._normalize_plan(
                HRIPlanResult(
                    query_type="comprehensive",
                    intent="condition_requirement",
                    confidence=0.85,
                    evidence_roles=["condition", "requirement", "table", "article"],
                    relation_types=["condition_of", "requires", "refers_to", "parameter_of"],
                    retrieval_focus=focus,
                    sub_questions=self._rule_sub_questions(text, "condition_requirement"),
                    rationale="问题同时包含适用条件和规范措施，需要多类证据组合。",
                )
            )

        if self._has_any(text, ["参数", "阈值", "指标", "表", "附表", "依据"]):
            return self._normalize_plan(
                HRIPlanResult(
                    query_type="comprehensive",
                    intent="parameter_table",
                    confidence=0.82,
                    evidence_roles=["requirement", "table", "article"],
                    relation_types=["requires", "refers_to", "parameter_of"],
                    retrieval_focus=focus,
                    sub_questions=self._rule_sub_questions(text, "parameter_table"),
                    rationale="问题需要正文要求与参数表格共同支撑。",
                )
            )

        if self._has_any(text, ["除", "例外", "特殊情况", "但", "但是", "补充", "另", "同时", "还应"]):
            return self._normalize_plan(
                HRIPlanResult(
                    query_type="comprehensive",
                    intent="exception_or_supplement",
                    confidence=0.82,
                    evidence_roles=["requirement", "exception", "supplement", "article"],
                    relation_types=["requires", "exception_to", "supplements"],
                    retrieval_focus=focus,
                    sub_questions=self._rule_sub_questions(text, "exception_or_supplement"),
                    rationale="问题涉及例外或补充说明，需要沿规范关系扩展。",
                )
            )

        if self._has_any(text, ["要求", "应", "必须", "不得", "措施", "如何", "流程", "关系", "比较", "说明"]):
            return self._normalize_plan(
                HRIPlanResult(
                    query_type="comprehensive",
                    intent="requirement_lookup",
                    confidence=0.65,
                    evidence_roles=["requirement", "article"],
                    relation_types=["requires"],
                    retrieval_focus=focus,
                    sub_questions=self._rule_sub_questions(text, "requirement_lookup"),
                    rationale="问题可能需要规范要求证据；规则置信度较低时交给 LLM 进一步规划。",
                )
            )

        return self._normalize_plan(
            HRIPlanResult(
                query_type="locating",
                intent="article_lookup",
                confidence=0.4,
                evidence_roles=["article"],
                relation_types=[],
                retrieval_focus=focus,
                rationale="未命中强规则，默认按定位型问题处理。",
            )
        )

    def _normalize_plan(self, plan: HRIPlanResult) -> HRIPlanResult:
        defaults = self._defaults_for(plan.query_type, plan.intent)
        plan.evidence_roles = self._allowed_unique(
            plan.evidence_roles, EVIDENCE_ROLES, defaults["evidence_roles"]
        )
        plan.relation_types = self._allowed_unique(
            plan.relation_types, RELATION_TYPES, defaults["relation_types"]
        )
        plan.retrieval_focus = self._unique_nonempty(plan.retrieval_focus)
        if plan.query_type != "statistical":
            plan.aggregation = None
        elif plan.aggregation is None:
            plan.aggregation = HRIAggregationSpec(operation="COUNT")
        if not self.enable_query_decomposition or plan.query_type != "comprehensive":
            plan.sub_questions = []
        elif self.max_sub_questions:
            plan.sub_questions = plan.sub_questions[: self.max_sub_questions]
        return plan

    @staticmethod
    def _coerce_plan(raw_plan: Any) -> HRIPlanResult:
        if isinstance(raw_plan, HRIPlanResult):
            return raw_plan
        if hasattr(raw_plan, "model_dump"):
            raw_plan = raw_plan.model_dump()
        if isinstance(raw_plan, dict):
            return HRIPlanResult.model_validate(raw_plan)
        raise ValueError(f"Unsupported HRI plan response: {type(raw_plan)}")

    @staticmethod
    def _with_source(plan: HRIPlanResult, source: HRIPlanSource) -> HRIPlanResult:
        plan.source = source
        return plan

    @staticmethod
    def _has_any(text: str, keywords: List[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    @staticmethod
    def _allowed_unique(values: List[str], allowed: set, default: List[str]) -> List[str]:
        normalized = []
        for value in values or []:
            item = str(value).strip().lower()
            if item in allowed and item not in normalized:
                normalized.append(item)
        return normalized or default.copy()

    @staticmethod
    def _unique_nonempty(values: List[str]) -> List[str]:
        result = []
        for value in values or []:
            item = str(value).strip()
            if item and item not in result:
                result.append(item)
        return result

    @staticmethod
    def _defaults_for(query_type: str, intent: str) -> Dict[str, List[str]]:
        if intent == "definition_lookup":
            return {"evidence_roles": ["definition", "article"], "relation_types": ["defines"]}
        if intent == "condition_requirement":
            return {
                "evidence_roles": ["condition", "requirement", "table", "article"],
                "relation_types": ["condition_of", "requires", "refers_to", "parameter_of"],
            }
        if intent == "parameter_table":
            return {
                "evidence_roles": ["requirement", "table", "article"],
                "relation_types": ["requires", "refers_to", "parameter_of"],
            }
        if intent == "exception_or_supplement":
            return {
                "evidence_roles": ["requirement", "exception", "supplement", "article"],
                "relation_types": ["requires", "exception_to", "supplements"],
            }
        if query_type == "statistical":
            return {
                "evidence_roles": ["requirement", "table", "article"],
                "relation_types": ["requires", "refers_to", "parameter_of"],
            }
        if query_type == "comprehensive":
            return {
                "evidence_roles": ["requirement", "table", "article"],
                "relation_types": ["requires", "refers_to", "parameter_of"],
            }
        return {"evidence_roles": ["article"], "relation_types": []}

    @staticmethod
    def _retrieval_focus(text: str) -> List[str]:
        markers = re.findall(r"第\s*[\d一二三四五六七八九十.、-]+\s*[章节条页]", text)
        quoted = re.findall(r"[“\"]([^”\"]+)[”\"]", text)
        keywords = [
            word
            for word in ["数字底板", "四预", "预报", "预警", "预演", "预案", "指标", "阈值", "措施", "要求", "参数"]
            if word in text
        ]
        return HRIQueryPlanner._unique_nonempty([*markers, *quoted, *keywords, text[:40]])

    def _rule_sub_questions(self, text: str, intent: str) -> List[HRISubQuestion]:
        if not self.enable_query_decomposition or self.max_sub_questions <= 0:
            return []
        if intent == "condition_requirement":
            questions = [
                ("适用条件是什么？", ["condition"], ["condition_of"]),
                ("对应有哪些规范要求或措施？", ["requirement"], ["requires"]),
                ("相关参数、指标或表格依据是什么？", ["table"], ["refers_to", "parameter_of"]),
            ]
        elif intent == "parameter_table":
            questions = [
                ("正文中对应的规范要求是什么？", ["requirement"], ["requires"]),
                ("相关参数、阈值、指标或表格依据是什么？", ["table"], ["refers_to", "parameter_of"]),
            ]
        elif intent == "exception_or_supplement":
            questions = [
                ("正文中的基础规范要求是什么？", ["requirement"], ["requires"]),
                ("是否存在例外或补充说明？", ["exception", "supplement"], ["exception_to", "supplements"]),
            ]
        else:
            questions = [("相关规范要求是什么？", ["requirement"], ["requires"])]
        return [
            HRISubQuestion(
                question=f"{text}：{question}",
                type="retrieval",
                evidence_roles=roles,
                relation_types=relations,
            )
            for question, roles, relations in questions[: self.max_sub_questions]
        ]

    @staticmethod
    def _aggregation_filters(text: str) -> List[str]:
        filters = []
        section = HRIQueryPlanner._section_scope(text)
        if section:
            filters.append(f"section:{section}")
        if any(word in text for word in ["要求", "应", "必须", "不得"]):
            filters.append("node_type:Requirement")
        if any(word in text for word in ["表", "参数", "指标", "阈值"]):
            filters.append("node_type:Table")
        return filters

    @staticmethod
    def _section_scope(text: str) -> Optional[str]:
        match = re.search(r"第\s*[\d一二三四五六七八九十.、-]+\s*[章节条页]", text)
        return match.group(0) if match else None

    @staticmethod
    def _aggregation_target(text: str) -> Optional[str]:
        if "要求" in text:
            return "规范要求"
        if any(word in text for word in ["表", "参数", "指标", "阈值"]):
            return "参数表格"
        if "条" in text:
            return "条文"
        return None
