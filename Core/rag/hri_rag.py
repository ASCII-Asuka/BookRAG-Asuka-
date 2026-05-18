import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from Core.Index.HRIIndex import EvidenceAnchor, HRIIndex, HydroRelation, hydro_tokenize
from Core.Index.Tree import DocumentTree
from Core.configs.rag.hri_config import HRIRAGConfig
from Core.prompts.hri_prompt import HRI_ANSWER_PROMPT
from Core.provider.llm import LLM
from Core.rag.base_rag import BaseRAG

log = logging.getLogger(__name__)


QuestionType = str


def classify_question_type(query: str) -> QuestionType:
    text = query or ""
    statistical_keywords = ["多少", "几个", "几项", "数量", "统计", "总数", "一共有", "共几"]
    comprehensive_keywords = [
        "依据",
        "措施",
        "条件",
        "要求",
        "参数",
        "表",
        "附录",
        "如何",
        "为什么",
        "说明",
        "综合",
        "比较",
        "关系",
        "流程",
    ]
    locating_keywords = ["什么是", "定义", "含义", "是指", "哪一条", "在哪里", "第几页", "位置"]

    if any(keyword in text for keyword in statistical_keywords):
        return "statistical"
    if any(keyword in text for keyword in comprehensive_keywords):
        return "comprehensive"
    if any(keyword in text for keyword in locating_keywords):
        return "locating"
    return "locating"


class HRIRAG(BaseRAG):
    def __init__(
        self,
        config: HRIRAGConfig,
        llm: LLM,
        tree_index: DocumentTree,
        hri_index: HRIIndex,
        bm25: Any,
    ):
        super().__init__(
            llm=llm,
            name="Hydro HRI RAG",
            description="Hydro regulation tree-graph evidence-chain RAG",
        )
        self.config = config
        self.tree_index = tree_index
        self.hri_index = hri_index
        self.bm25 = bm25

    def _retrieve(self, query: str, **kwargs) -> Dict[str, Any]:
        question_type = classify_question_type(query)
        coarse_results = self.hri_index.search_bm25(
            bm25=self.bm25, query=query, top_k=self.config.topk
        )
        ranked_results = self._structure_rerank(query, question_type, coarse_results)
        seed_ids = [item["node_id"] for item in ranked_results]
        selected_ids, relations = self._expand_context(seed_ids, question_type)
        evidence_chain = self._build_evidence_chain(
            selected_ids=selected_ids,
            seed_ids=seed_ids,
            relations=relations,
            question_type=question_type,
        )
        return {
            "question_type": question_type,
            "coarse_results": coarse_results,
            "ranked_results": ranked_results,
            "selected_node_ids": selected_ids,
            "relations": relations,
            "evidence_chain": evidence_chain,
        }

    def _create_augmented_prompt(self, query: str, evidence_chain: List[Dict[str, Any]] = None) -> str:
        evidence_chain = evidence_chain or []
        question_type = classify_question_type(query)
        chain_text = self._format_chain_for_prompt(evidence_chain, with_relations=True)
        context_text = self._format_chain_for_prompt(evidence_chain, with_relations=False)
        return HRI_ANSWER_PROMPT.format(
            question_type=question_type,
            query=query,
            evidence_chain=chain_text or "未检索到可用证据。",
            context=context_text or "未检索到可用证据。",
        )

    def generation(self, query: str, query_output_dir: str) -> Tuple[str, List[Any]]:
        retrieval_info = self._retrieve(query)
        prompt = self._create_augmented_prompt(
            query, evidence_chain=retrieval_info["evidence_chain"]
        )
        answer = self.llm.get_completion(prompt=prompt, json_response=False)
        retrieval_ids = self._save_retrieval_res(retrieval_info, Path(query_output_dir))
        return answer, retrieval_ids

    def _structure_rerank(
        self, query: str, question_type: QuestionType, coarse_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        query_tokens = set(hydro_tokenize(query))
        type_weight = self._type_weights(question_type)
        reranked = []
        for rank, item in enumerate(coarse_results):
            anchor = self.hri_index.get_anchor(item["node_id"])
            if anchor is None:
                continue
            text_tokens = set(hydro_tokenize(anchor.section_id + " " + anchor.text))
            overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
            title_bonus = 0.2 if any(part and part in query for part in anchor.title_path) else 0.0
            score = (
                float(item.get("score", 0.0))
                + type_weight.get(anchor.node_type, 0.0)
                + overlap
                + title_bonus
                - rank * 0.01
            )
            reranked.append({**item, "rerank_score": score})
        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked

    def _expand_context(
        self, seed_ids: List[int], question_type: QuestionType
    ) -> Tuple[List[int], List[HydroRelation]]:
        selected_ids: List[int] = []

        def add(node_id: int):
            if node_id in self.hri_index.anchors and node_id not in selected_ids:
                selected_ids.append(node_id)

        for node_id in seed_ids:
            add(node_id)
            self._add_tree_context(node_id, add)

        relations: List[HydroRelation] = []
        if self.config.enable_relation_expansion:
            relations = self.hri_index.get_related_relations(
                seed_ids,
                expand_depth=self.config.expand_depth,
                relation_types=self._relation_types(question_type),
            )
            for relation in relations:
                add(relation.source_id)
                add(relation.target_id)

        return selected_ids[: self.config.max_context_nodes], relations

    def _add_tree_context(self, node_id: int, add):
        if self.config.context_window <= 0:
            return
        node = self.tree_index.get_node_by_index_id(node_id)
        if node is None:
            return
        if node.parent is not None:
            add(node.parent.index_id)
            siblings = node.parent.children
            current_idx = next(
                (idx for idx, sibling in enumerate(siblings) if sibling.index_id == node_id),
                None,
            )
            if current_idx is not None:
                start = max(0, current_idx - self.config.context_window)
                end = min(len(siblings), current_idx + self.config.context_window + 1)
                for sibling in siblings[start:end]:
                    add(sibling.index_id)

    def _build_evidence_chain(
        self,
        selected_ids: List[int],
        seed_ids: List[int],
        relations: List[HydroRelation],
        question_type: QuestionType,
    ) -> List[Dict[str, Any]]:
        relations_by_node: Dict[int, List[Dict[str, Any]]] = {}
        for relation in relations:
            rel_data = relation.model_dump()
            relations_by_node.setdefault(relation.source_id, []).append(rel_data)
            if relation.target_id != relation.source_id:
                relations_by_node.setdefault(relation.target_id, []).append(rel_data)

        chain = []
        seed_set = set(seed_ids)
        for node_id in selected_ids:
            anchor = self.hri_index.get_anchor(node_id)
            if anchor is None:
                continue
            role = self._node_role(anchor, node_id in seed_set, relations_by_node.get(node_id, []), question_type)
            chain.append(
                {
                    "node_id": node_id,
                    "role": role,
                    "anchor": anchor.model_dump(),
                    "relations": relations_by_node.get(node_id, []),
                }
            )
        return chain

    def _save_retrieval_res(
        self, retrieval_info: Dict[str, Any], query_output_dir: Path
    ) -> List[Any]:
        query_output_dir.mkdir(parents=True, exist_ok=True)
        retrieval_ids = retrieval_info["selected_node_ids"]
        serializable = {
            "question_type": retrieval_info["question_type"],
            "coarse_results": retrieval_info["coarse_results"],
            "ranked_results": retrieval_info["ranked_results"],
            "selected_node_ids": retrieval_ids,
            "relations": [relation.model_dump() for relation in retrieval_info["relations"]],
        }
        with open(query_output_dir / "retrieval_res.json", "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

        if self.config.output_evidence_chain:
            with open(query_output_dir / "evidence_chain.json", "w", encoding="utf-8") as f:
                json.dump(retrieval_info["evidence_chain"], f, ensure_ascii=False, indent=2)

        for item in retrieval_info["evidence_chain"]:
            anchor = item["anchor"]
            with open(query_output_dir / f"{item['node_id']}.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "id": item["node_id"],
                        "role": item["role"],
                        "type": anchor["node_type"],
                        "section_id": anchor["section_id"],
                        "page": anchor["page"],
                        "content": anchor["text"],
                        "attributes": anchor.get("attributes", {}),
                        "relations": item["relations"],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        log.info("Saved HRI retrieval results to %s", query_output_dir)
        return retrieval_ids

    def _format_chain_for_prompt(
        self, evidence_chain: List[Dict[str, Any]], with_relations: bool
    ) -> str:
        lines = []
        for idx, item in enumerate(evidence_chain, 1):
            anchor = EvidenceAnchor(**item["anchor"])
            title_path = " > ".join(anchor.title_path) if anchor.title_path else "无"
            text = anchor.text[:1500]
            lines.append(
                f"[证据{idx}] 作用：{item['role']}；类型：{anchor.node_type}；"
                f"位置：{anchor.section_id or anchor.node_id}；页码：{anchor.page}；路径：{title_path}"
            )
            if with_relations and item.get("relations"):
                relation_texts = [
                    f"{rel['source_id']} -{rel['relation_type']}-> {rel['target_id']} ({rel.get('evidence', '')})"
                    for rel in item["relations"]
                ]
                lines.append("关系：" + "；".join(relation_texts))
            if anchor.attributes:
                attrs = "；".join(
                    f"{key}：{value}" for key, value in anchor.attributes.items() if value
                )
                if attrs:
                    lines.append("结构化属性：" + attrs)
            lines.append("原文：" + text)
        return "\n".join(lines)

    @staticmethod
    def _type_weights(question_type: QuestionType) -> Dict[str, float]:
        if question_type == "statistical":
            return {"Table": 0.5, "Article": 0.3, "Chapter": 0.2}
        if question_type == "comprehensive":
            return {
                "Article": 0.4,
                "Table": 0.35,
                "TermDefinition": 0.25,
                "Requirement": 0.25,
                "Condition": 0.2,
                "Concept": 0.2,
                "Exception": 0.2,
                "Supplement": 0.15,
                "Appendix": 0.2,
            }
        return {
            "TermDefinition": 0.5,
            "Concept": 0.45,
            "Requirement": 0.35,
            "Article": 0.35,
            "Condition": 0.25,
            "Exception": 0.2,
            "Supplement": 0.15,
            "Table": 0.2,
        }

    @staticmethod
    def _relation_types(question_type: QuestionType) -> Iterable[str]:
        if question_type == "statistical":
            return ["refers_to", "parameter_of"]
        if question_type == "comprehensive":
            return [
                "defines",
                "condition_of",
                "requires",
                "refers_to",
                "parameter_of",
                "supplements",
                "exception_to",
            ]
        return ["defines", "refers_to", "parameter_of"]

    @staticmethod
    def _node_role(
        anchor: EvidenceAnchor,
        is_seed: bool,
        relations: List[Dict[str, Any]],
        question_type: QuestionType,
    ) -> str:
        if anchor.node_type == "Concept":
            return "定义对象"
        if anchor.node_type == "Requirement":
            return "规范要求对象"
        if anchor.node_type == "Condition":
            return "适用条件对象"
        if anchor.node_type == "Exception":
            return "例外限制对象"
        if anchor.node_type == "Supplement":
            return "补充说明对象"
        if anchor.node_type == "TermDefinition":
            return "术语定义"
        if anchor.node_type == "Table":
            return "参数依据"
        relation_types = {rel.get("relation_type") for rel in relations}
        if "condition_of" in relation_types:
            return "适用条件"
        if "requires" in relation_types:
            return "规范要求"
        if anchor.node_type == "Appendix":
            return "补充说明"
        if is_seed:
            return f"{question_type}问题命中证据"
        return "上下文证据"

    def close(self):
        return None
