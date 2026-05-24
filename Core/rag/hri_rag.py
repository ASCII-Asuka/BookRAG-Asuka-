import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from Core.Index.HRIIndex import EvidenceAnchor, HRIIndex, HydroRelation, hydro_tokenize
from Core.Index.Tree import DocumentTree
from Core.configs.rag.hri_config import HRIRAGConfig
from Core.prompts.hri_prompt import HRI_ANSWER_PROMPT
from Core.provider.llm import LLM
from Core.rag.base_rag import BaseRAG
from Core.rag.hri_plan import HRIPlanResult, HRIQueryPlanner

log = logging.getLogger(__name__)


QuestionType = str


def classify_question_type(query: str) -> QuestionType:
    return HRIQueryPlanner(llm=None, question_classifier="rule").analyze(query).query_type


class HRIRAG(BaseRAG):
    def __init__(
        self,
        config: HRIRAGConfig,
        llm: LLM,
        tree_index: DocumentTree,
        hri_index: HRIIndex,
        bm25: Any,
        hri_vector_store: Any = None,
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
        self.hri_vector_store = hri_vector_store
        self.planner = HRIQueryPlanner(
            llm=llm,
            question_classifier=config.question_classifier,
            classification_model=config.classification_model,
            confidence_threshold=config.classification_confidence_threshold,
            enable_query_decomposition=config.enable_query_decomposition,
            max_sub_questions=config.max_sub_questions,
        )

    def _ablation_variant(self) -> str:
        return getattr(self.config, "ablation_variant", "full") or "full"

    def _uses_tree_structure(self) -> bool:
        return self._ablation_variant() != "wo_tree"

    def _uses_relation_graph(self) -> bool:
        return self._ablation_variant() not in {"wo_tree", "wo_relation"}

    def _uses_query_planner(self) -> bool:
        return self._ablation_variant() != "wo_planner"

    def _outputs_evidence_trace(self) -> bool:
        return self._ablation_variant() != "wo_evidence_chain"

    def _analyze_query(self, query: str) -> HRIPlanResult:
        if self._uses_query_planner():
            return self.planner.analyze(query)
        return HRIPlanResult(
            query_type="comprehensive",
            intent="multi_evidence_synthesis",
            confidence=1.0,
            evidence_roles=[
                "definition",
                "condition",
                "requirement",
                "table",
                "exception",
                "supplement",
                "article",
            ],
            relation_types=list(self._relation_types("comprehensive")),
            retrieval_focus=[query],
            sub_questions=[],
            aggregation=None,
            rationale="Ablation wo_planner: use one static comprehensive retrieval plan for all questions.",
            source="fallback",
        )

    def _retrieve(self, query: str, **kwargs) -> Dict[str, Any]:
        query_plan = self._analyze_query(query)
        question_type = query_plan.query_type
        bm25_results, vector_results, hybrid_results = self._hybrid_recall(query, query_plan=query_plan)
        ranked_results = self._structure_rerank(
            query, question_type, hybrid_results, query_plan=query_plan
        )
        seed_ids = [item["node_id"] for item in ranked_results]
        relations = self._collect_relations(seed_ids, question_type, query_plan=query_plan)
        selected_ids = self._select_budgeted_nodes(
            ranked_results=ranked_results,
            relations=relations,
            question_type=question_type,
            query_plan=query_plan,
        )
        evidence_chain = self._build_evidence_chain(
            selected_ids=selected_ids,
            seed_ids=seed_ids,
            relations=relations,
            question_type=question_type,
        )
        selected_set = set(selected_ids)
        return {
            "question_type": question_type,
            "query_plan": query_plan,
            "coarse_results": bm25_results,
            "vector_results": vector_results,
            "hybrid_results": hybrid_results,
            "ranked_results": ranked_results,
            "budgeted_results": self._results_for_node_ids(selected_ids, ranked_results),
            "dropped_results": [
                item for item in ranked_results if item["node_id"] not in selected_set
            ],
            "selected_node_ids": selected_ids,
            "relations": relations,
            "evidence_chain": evidence_chain,
        }

    def _create_augmented_prompt(
        self,
        query: str,
        evidence_chain: List[Dict[str, Any]] = None,
        query_plan: Optional[HRIPlanResult] = None,
    ) -> str:
        evidence_chain = evidence_chain or []
        question_type = query_plan.query_type if query_plan else classify_question_type(query)
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
            query,
            evidence_chain=retrieval_info["evidence_chain"],
            query_plan=retrieval_info.get("query_plan"),
        )
        answer = self.llm.get_completion(prompt=prompt, json_response=False)
        retrieval_ids = self._save_retrieval_res(retrieval_info, Path(query_output_dir))
        return answer, retrieval_ids

    def _hybrid_recall(
        self, query: str, query_plan: Optional[HRIPlanResult] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        recall_queries = self._recall_queries(query, query_plan)
        bm25_results: List[Dict[str, Any]] = []
        vector_results: List[Dict[str, Any]] = []
        for recall_query in recall_queries:
            query_bm25_results = self.hri_index.search_bm25(
                bm25=self.bm25, query=recall_query, top_k=self._bm25_topk()
            )
            for item in query_bm25_results:
                item["source"] = "bm25"
                item["recall_query"] = recall_query
                item["bm25_score"] = float(item.get("score", 0.0))
                item["vector_score"] = 0.0
            bm25_results.extend(query_bm25_results)

            if self.config.enable_vector_recall and self.hri_vector_store is not None:
                query_vector_results = self._search_vector(recall_query, top_k=self._embedding_topk())
                for item in query_vector_results:
                    item["recall_query"] = recall_query
                vector_results.extend(query_vector_results)

        bm25_norm = self._normalized_scores(bm25_results, "bm25_score")
        vector_norm = self._normalized_scores(vector_results, "vector_score")
        combined: Dict[int, Dict[str, Any]] = {}

        for item in bm25_results:
            node_id = item["node_id"]
            item_score = bm25_norm.get(node_id, 0.0)
            if node_id not in combined or item_score > combined[node_id].get("bm25_norm", 0.0):
                combined[node_id] = {
                    **item,
                    "source": "bm25",
                    "bm25_score": item["bm25_score"],
                    "vector_score": 0.0,
                    "bm25_norm": item_score,
                    "vector_norm": 0.0,
                }

        for item in vector_results:
            node_id = item["node_id"]
            item_score = vector_norm.get(node_id, 0.0)
            if node_id in combined:
                combined[node_id]["source"] = "hybrid"
                if item_score > combined[node_id].get("vector_norm", 0.0):
                    combined[node_id]["vector_score"] = item["vector_score"]
                    combined[node_id]["vector_norm"] = item_score
                    combined[node_id]["recall_query"] = item.get(
                        "recall_query", combined[node_id].get("recall_query", query)
                    )
            else:
                combined[node_id] = {
                    **item,
                    "source": "vector",
                    "bm25_score": 0.0,
                    "vector_score": item["vector_score"],
                    "bm25_norm": 0.0,
                    "vector_norm": item_score,
                }

        for item in combined.values():
            hybrid_score = (
                self.config.hybrid_bm25_weight * item.get("bm25_norm", 0.0)
                + self.config.hybrid_vector_weight * item.get("vector_norm", 0.0)
            )
            item["hybrid_score"] = hybrid_score
            item["score"] = hybrid_score

        hybrid_results = sorted(
            combined.values(),
            key=lambda item: item.get("hybrid_score", 0.0),
            reverse=True,
        )
        return bm25_results, vector_results, hybrid_results

    def _recall_queries(
        self, query: str, query_plan: Optional[HRIPlanResult]
    ) -> List[str]:
        queries = [query]
        if (
            query_plan
            and self._uses_query_planner()
            and query_plan.query_type == "comprehensive"
            and self.config.enable_query_decomposition
        ):
            for sub_question in query_plan.sub_questions:
                if sub_question.type == "retrieval":
                    queries.append(sub_question.question)
        return list(dict.fromkeys(item for item in queries if item))

    def _search_vector(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        raw_results = self.hri_vector_store.search(query_text=query, top_k=top_k)
        mapped_results: List[Dict[str, Any]] = []
        for item in raw_results:
            metadata = item.get("metadata") or {}
            node_id = self._coerce_node_id(metadata.get("node_id"))
            if node_id is None:
                node_id = self._coerce_node_id(metadata.get("source_node_id"))
            if node_id is None:
                continue
            anchor = self.hri_index.get_anchor(node_id)
            if anchor is None:
                continue
            distance = item.get("distance")
            vector_score = self._distance_to_score(distance)
            mapped_results.append(
                {
                    "node_id": node_id,
                    "score": vector_score,
                    "vector_score": vector_score,
                    "bm25_score": 0.0,
                    "node_type": anchor.node_type,
                    "section_id": anchor.section_id,
                    "page": anchor.page,
                    "text": anchor.text,
                    "anchor": anchor.model_dump(),
                    "distance": distance,
                    "source": "vector",
                }
            )
        return mapped_results

    def _bm25_topk(self) -> int:
        return self.config.bm25_topk or self.config.topk

    def _embedding_topk(self) -> int:
        return self.config.embedding_topk or self.config.topk

    @staticmethod
    def _coerce_node_id(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _distance_to_score(distance: Any) -> float:
        if distance is None:
            return 0.0
        try:
            distance_value = max(float(distance), 0.0)
        except (TypeError, ValueError):
            return 0.0
        return 1.0 / (1.0 + distance_value)

    @staticmethod
    def _normalized_scores(items: List[Dict[str, Any]], score_key: str) -> Dict[int, float]:
        if not items:
            return {}
        values = [float(item.get(score_key, 0.0)) for item in items]
        min_value = min(values)
        max_value = max(values)
        normalized: Dict[int, float] = {}
        for item, value in zip(items, values):
            if max_value == min_value:
                score = 1.0 if value > 0 else 0.0
            else:
                score = (value - min_value) / (max_value - min_value)
            node_id = item["node_id"]
            normalized[node_id] = max(score, normalized.get(node_id, 0.0))
        return normalized

    def _structure_rerank(
        self,
        query: str,
        question_type: QuestionType,
        coarse_results: List[Dict[str, Any]],
        query_plan: Optional[HRIPlanResult] = None,
    ) -> List[Dict[str, Any]]:
        if not self._uses_tree_structure():
            reranked = []
            for rank, item in enumerate(coarse_results):
                score = float(item.get("hybrid_score", item.get("score", 0.0))) - rank * 0.01
                reranked.append({**item, "rerank_score": score})
            reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
            return reranked

        query_tokens = set(hydro_tokenize(query))
        type_weight = self._type_weights(question_type)
        seed_source_ids = [
            self._tree_source_id(self.hri_index.get_anchor(item["node_id"]))
            for item in coarse_results[:3]
        ]
        seed_source_ids = [node_id for node_id in seed_source_ids if node_id is not None]
        reranked = []
        for rank, item in enumerate(coarse_results):
            anchor = self.hri_index.get_anchor(item["node_id"])
            if anchor is None:
                continue
            text_tokens = set(hydro_tokenize(anchor.section_id + " " + anchor.text))
            overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
            title_bonus = 0.2 if any(part and part in query for part in anchor.title_path) else 0.0
            tree_bonus = self._tree_feature_score(query, anchor, seed_source_ids)
            role_bonus = 0.15 if query_plan and self._budget_role(anchor) in query_plan.evidence_roles else 0.0
            score = (
                float(item.get("hybrid_score", item.get("score", 0.0)))
                + type_weight.get(anchor.node_type, 0.0)
                + overlap
                + title_bonus
                + tree_bonus
                + role_bonus
                - rank * 0.01
            )
            reranked.append({**item, "rerank_score": score})
        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked

    def _tree_feature_score(
        self, query: str, anchor: EvidenceAnchor, seed_source_ids: List[int]
    ) -> float:
        if not self._uses_tree_structure():
            return 0.0
        score = 0.0
        source_id = self._tree_source_id(anchor)
        node = self.tree_index.get_node_by_index_id(source_id) if source_id is not None else None

        title_path = anchor.title_path or []
        if any(part and (part in query or query in part) for part in title_path):
            score += 0.3
        if self._marker_matches_query(query, anchor):
            score += 0.35

        if node is None:
            return score

        parent = node.parent
        if parent is not None and parent.meta_info.content and parent.meta_info.content in query:
            score += 0.25

        for seed_id in seed_source_ids:
            seed_node = self.tree_index.get_node_by_index_id(seed_id)
            if seed_node is None or seed_node == node:
                continue
            if seed_node.parent is not None and seed_node.parent == parent:
                score += 0.2
                if abs(seed_node.index_id - node.index_id) <= 3:
                    score += 0.1
            if seed_node.parent == node or node.parent == seed_node:
                score += 0.15

        if anchor.node_type == "Table" and self._has_table_relation_to_seed(anchor.node_id, seed_source_ids):
            score += 0.25
        return score

    @staticmethod
    def _marker_matches_query(query: str, anchor: EvidenceAnchor) -> bool:
        marker_pattern = re.compile(
            r"(第\s*[\d一二三四五六七八九十百千万.．\-－—]+\s*条|"
            r"表\s*[\dA-Za-z一二三四五六七八九十百千万]+(?:[-－—.．]\d+)*|"
            r"\b\d+(?:[.．\-－]\d+)+\b)"
        )
        markers = marker_pattern.findall(query or "")
        if not markers:
            return False
        haystack = "\n".join([anchor.section_id, anchor.text, " ".join(anchor.title_path)])
        normalized_haystack = re.sub(r"\s+", "", haystack)
        return any(re.sub(r"\s+", "", marker) in normalized_haystack for marker in markers)

    def _has_table_relation_to_seed(self, node_id: int, seed_source_ids: List[int]) -> bool:
        seed_set = set(seed_source_ids)
        for relation in self.hri_index.relations:
            if relation.relation_type not in {"refers_to", "parameter_of"}:
                continue
            if relation.source_id == node_id and self._relation_tree_id(relation.target_id) in seed_set:
                return True
            if relation.target_id == node_id and self._relation_tree_id(relation.source_id) in seed_set:
                return True
        return False

    def _expand_context(
        self, seed_ids: List[int], question_type: QuestionType
    ) -> Tuple[List[int], List[HydroRelation]]:
        ranked_results = [{"node_id": node_id, "rerank_score": len(seed_ids) - idx} for idx, node_id in enumerate(seed_ids)]
        relations = self._collect_relations(seed_ids, question_type)
        return self._select_budgeted_nodes(ranked_results, relations, question_type), relations

    def _collect_relations(
        self,
        seed_ids: List[int],
        question_type: QuestionType,
        query_plan: Optional[HRIPlanResult] = None,
    ) -> List[HydroRelation]:
        if not self.config.enable_relation_expansion or not self._uses_relation_graph():
            return []
        relation_types = self._planned_relation_types(question_type, query_plan)
        relations = self.hri_index.get_related_relations(
            seed_ids,
            expand_depth=self.config.expand_depth,
            relation_types=relation_types,
        )
        if question_type == "comprehensive" and relations:
            semantic_ids = list(
                {
                    related_id
                    for relation in relations
                    for related_id in (relation.source_id, relation.target_id)
                }
            )
            relations.extend(
                self.hri_index.get_related_relations(
                    semantic_ids,
                    expand_depth=1,
                    relation_types=relation_types,
                )
            )
        return self._dedupe_relations(relations)

    @staticmethod
    def _dedupe_relations(relations: List[HydroRelation]) -> List[HydroRelation]:
        seen = set()
        deduped = []
        for relation in relations:
            key = (relation.source_id, relation.target_id, relation.relation_type)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(relation)
        return deduped

    def _select_budgeted_nodes(
        self,
        ranked_results: List[Dict[str, Any]],
        relations: List[HydroRelation],
        question_type: QuestionType,
        query_plan: Optional[HRIPlanResult] = None,
    ) -> List[int]:
        if not self._uses_tree_structure():
            selected: List[int] = []
            for item in ranked_results:
                node_id = item["node_id"]
                if node_id in self.hri_index.anchors and node_id not in selected:
                    selected.append(node_id)
                if len(selected) >= self.config.max_context_nodes:
                    break
            return selected

        candidates: Dict[int, Dict[str, Any]] = {}
        order = 0

        def add_candidate(node_id: int, score: float, source: str):
            nonlocal order
            if node_id not in self.hri_index.anchors:
                return
            current = candidates.get(node_id)
            if current is None:
                candidates[node_id] = {"node_id": node_id, "score": score, "source": source, "order": order}
                order += 1
            elif score > current["score"]:
                current["score"] = score
                current["source"] = source

        ranked_score = {}
        for idx, item in enumerate(ranked_results):
            node_id = item["node_id"]
            score = float(item.get("rerank_score", item.get("hybrid_score", item.get("score", 0.0))))
            ranked_score[node_id] = score
            add_candidate(node_id, score, "seed")
            self._add_tree_context_candidates(node_id, score - 0.05, add_candidate)

        for relation in relations:
            relation_base = max(
                ranked_score.get(relation.source_id, 0.0),
                ranked_score.get(relation.target_id, 0.0),
                ranked_score.get(self._relation_tree_id(relation.source_id), 0.0),
                ranked_score.get(self._relation_tree_id(relation.target_id), 0.0),
            )
            relation_score = relation_base + relation.weight * 0.05
            add_candidate(relation.source_id, relation_score, "relation")
            add_candidate(relation.target_id, relation_score, "relation")

        sorted_candidates = sorted(
            candidates.values(), key=lambda item: (-item["score"], item["order"])
        )
        budget = self._budget_for(question_type, query_plan=query_plan)
        counts = {key: 0 for key in budget}
        selected: List[int] = []

        for item in sorted_candidates:
            if len(selected) >= self.config.max_context_nodes:
                break
            anchor = self.hri_index.get_anchor(item["node_id"])
            if anchor is None:
                continue
            role = self._budget_role(anchor)
            limit = budget.get(role, 0)
            if limit <= 0 or counts.get(role, 0) >= limit:
                continue
            selected.append(item["node_id"])
            counts[role] = counts.get(role, 0) + 1

        for item in sorted_candidates:
            if len(selected) >= self.config.max_context_nodes:
                break
            if item["node_id"] not in selected:
                selected.append(item["node_id"])
        return selected

    def _add_tree_context_candidates(self, node_id: int, score: float, add_candidate):
        if self.config.context_window <= 0 or not self._uses_tree_structure():
            return
        tree_node_id = self._relation_tree_id(node_id)
        node = self.tree_index.get_node_by_index_id(tree_node_id)
        if node is None or node.parent is None:
            return
        add_candidate(node.parent.index_id, score, "tree_context")
        siblings = node.parent.children
        current_idx = next(
            (idx for idx, sibling in enumerate(siblings) if sibling.index_id == tree_node_id),
            None,
        )
        if current_idx is None:
            return
        start = max(0, current_idx - self.config.context_window)
        end = min(len(siblings), current_idx + self.config.context_window + 1)
        for sibling in siblings[start:end]:
            add_candidate(sibling.index_id, score, "tree_context")

    def _budget_for(
        self, question_type: QuestionType, query_plan: Optional[HRIPlanResult] = None
    ) -> Dict[str, int]:
        base_budget = self.config.evidence_budgets.get(
            question_type,
            self.config.evidence_budgets.get("locating", {}),
        )
        if not query_plan or not query_plan.evidence_roles:
            return base_budget
        focused_budget = {
            role: limit
            for role, limit in base_budget.items()
            if role in query_plan.evidence_roles
        }
        if "article" in base_budget:
            focused_budget["article"] = min(base_budget["article"], 2)
        return focused_budget or base_budget

    def _planned_relation_types(
        self, question_type: QuestionType, query_plan: Optional[HRIPlanResult] = None
    ) -> Iterable[str]:
        if query_plan and query_plan.relation_types:
            return query_plan.relation_types
        return self._relation_types(question_type)

    @staticmethod
    def _budget_role(anchor: EvidenceAnchor) -> str:
        if anchor.node_type in {"Concept", "TermDefinition"}:
            return "definition"
        if anchor.node_type == "Condition":
            return "condition"
        if anchor.node_type == "Requirement":
            return "requirement"
        if anchor.node_type in {"Table", "Appendix"}:
            return "table"
        if anchor.node_type == "Exception":
            return "exception"
        if anchor.node_type == "Supplement":
            return "supplement"
        return "article"

    def _relation_tree_id(self, node_id: Optional[int]) -> Optional[int]:
        if node_id is None:
            return None
        return self._tree_source_id(self.hri_index.get_anchor(node_id))

    @staticmethod
    def _tree_source_id(anchor: Optional[EvidenceAnchor]) -> Optional[int]:
        if anchor is None:
            return None
        return anchor.source_node_id if anchor.source_node_id is not None else anchor.node_id

    def _results_for_node_ids(
        self, node_ids: List[int], ranked_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        ranked_by_id = {item["node_id"]: item for item in ranked_results}
        results = []
        for node_id in node_ids:
            if node_id in ranked_by_id:
                results.append(ranked_by_id[node_id])
                continue
            anchor = self.hri_index.get_anchor(node_id)
            if anchor is None:
                continue
            results.append(
                {
                    "node_id": node_id,
                    "score": 0.0,
                    "node_type": anchor.node_type,
                    "section_id": anchor.section_id,
                    "page": anchor.page,
                    "text": anchor.text,
                    "anchor": anchor.model_dump(),
                    "source": "budget",
                }
            )
        return results

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
        return self._sort_evidence_chain(chain)

    def _save_retrieval_res(
        self, retrieval_info: Dict[str, Any], query_output_dir: Path
    ) -> List[Any]:
        query_output_dir.mkdir(parents=True, exist_ok=True)
        retrieval_ids = retrieval_info["selected_node_ids"]
        if not self._outputs_evidence_trace():
            log.info("HRI evidence trace output disabled by ablation variant.")
            return []

        query_plan = retrieval_info.get("query_plan")
        serializable = {
            "question_type": retrieval_info["question_type"],
            "query_plan": query_plan.model_dump() if hasattr(query_plan, "model_dump") else query_plan,
            "coarse_results": retrieval_info["coarse_results"],
            "vector_results": retrieval_info.get("vector_results", []),
            "hybrid_results": retrieval_info.get("hybrid_results", []),
            "ranked_results": retrieval_info["ranked_results"],
            "budgeted_results": retrieval_info.get("budgeted_results", []),
            "dropped_results": retrieval_info.get("dropped_results", []),
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

    def _sort_evidence_chain(self, chain: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        original_order = {item["node_id"]: idx for idx, item in enumerate(chain)}

        def sort_key(item: Dict[str, Any]):
            anchor = EvidenceAnchor(**item["anchor"])
            role = self._budget_role(anchor)
            order = {
                "definition": 0,
                "condition": 1,
                "requirement": 2,
                "table": 3,
                "exception": 4,
                "supplement": 5,
                "article": 6,
            }.get(role, 9)
            page = anchor.page if anchor.page is not None else 10**9
            source_id = anchor.source_node_id if anchor.source_node_id is not None else anchor.node_id
            return (order, page, source_id, original_order[item["node_id"]])

        return sorted(chain, key=sort_key)

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
