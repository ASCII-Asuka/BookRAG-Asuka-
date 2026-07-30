import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Core.Index.EvidenceBridgeIndex import EvidenceBlock, EvidenceBridgeIndex, evidence_tokenize
from Core.configs.rag.evibridge_config import EviBridgeRAGConfig
from Core.rag.base_rag import BaseRAG
from Core.rag.evibridge_demand import DemandParser, EvidenceDemand
from Core.rag.evibridge_ppr import TypedPPRResult, run_typed_ppr_with_details, shortest_path_connector_with_paths
from Core.rag.evibridge_selector import SelectedEvidence, select_budgeted_evidence
from Core.rag.evibridge_verifier import EvidenceSufficiencyVerifier, SufficiencyVerdict

log = logging.getLogger(__name__)


class EviBridgeRAG(BaseRAG):
    def __init__(
        self,
        config: EviBridgeRAGConfig,
        llm: Any,
        evibridge_index: EvidenceBridgeIndex,
        bm25: Any,
        evibridge_vector_store: Any = None,
        reranker: Any = None,
    ):
        super().__init__(
            llm=llm,
            name="EviBridge RAG",
            description="Sufficiency-guided evidence bridging for complex document QA",
        )
        self.config = config
        self.evibridge_index = evibridge_index
        self.bm25 = bm25
        self.evibridge_vector_store = evibridge_vector_store
        self.reranker = reranker
        implicit_multihop_ablated = (
            config.ablation_variant == "wo_implicit_multihop"
        )
        self.demand_parser = DemandParser(
            llm=llm,
            mode=config.demand_parser,
            confidence_threshold=config.demand_confidence_threshold,
            dataset_profile=(
                "auto" if implicit_multihop_ablated else config.dataset_profile
            ),
            qasper_demand_mode=config.qasper_demand_mode,
            enable_boolean_answer_hint=config.enable_boolean_answer_hint,
            multi_hop_requires_explicit_bridge=(
                True
                if implicit_multihop_ablated
                else config.multi_hop_requires_explicit_bridge
            ),
            enable_implicit_multihop=not implicit_multihop_ablated,
        )
        self.verifier = EvidenceSufficiencyVerifier(
            llm=llm,
            enable_llm=(
                config.enable_llm_verifier
                and config.ablation_variant != "wo_verifier_repair"
            ),
        )
        self.last_retrieved_block_ids: List[int] = []
        self.last_answer_short: str = ""
        self.last_answer_rationale: str = ""
        self.last_supporting_block_ids: List[int] = []

    def _retrieve(self, query: str, **kwargs) -> Dict[str, Any]:
        demand = self.demand_parser.parse(query)
        return self._retrieve_with_demand(query, demand)

    def _retrieve_with_demand(self, query: str, demand: EvidenceDemand) -> Dict[str, Any]:
        seed_results = self._hybrid_seed_retrieval(query, demand)
        seed_scores = {item["block_id"]: item["score"] for item in seed_results}
        if not seed_scores:
            return self._empty_retrieval(query, demand)
        seed_ranks = {item["block_id"]: rank for rank, item in enumerate(seed_results, 1)}

        selected: List[SelectedEvidence] = []
        verdict: Optional[SufficiencyVerdict] = None
        iteration_records: List[Dict[str, Any]] = []
        candidate_scores: Dict[int, float] = seed_scores.copy()
        candidate_score_parts: Dict[int, Dict[str, float]] = {}
        connector_paths: List[Dict[str, Any]] = []
        connector_edges: List[Dict[str, Any]] = []
        selected_bridges = []
        seen_candidate_ids: set[int] = set()
        seen_selected_answer_ids: set[int] = set()
        previous_best_answer_score: Optional[float] = None
        stopping_reason = "max_iterations"

        max_iterations = 1 if self._ablation_variant() == "wo_sufficiency_verifier" else self.config.max_iterations
        for iteration in range(max(max_iterations, 1)):
            if self._ablation_variant() == "static_topk":
                candidate_scores = dict(
                    sorted(seed_scores.items(), key=lambda item: item[1], reverse=True)[
                        : self._max_context_blocks()
                    ]
                )
                candidate_score_parts = {
                    block_id: {"context": 0.0, "semantic": 0.0, "hierarchy": 0.0}
                    for block_id in candidate_scores
                }
            else:
                ppr_result = self._bridge_scores(
                    seed_scores=seed_scores,
                    demand=demand,
                )
                ppr_ranks = {block_id: rank for rank, block_id in enumerate(ppr_result.scores, 1)}
                candidate_scores = self._seed_preserved_candidate_scores(
                    seed_results,
                    ppr_result.scores,
                    active_seed_scores=seed_scores,
                )
                candidate_score_parts = ppr_result.score_parts
                for block_id in candidate_scores:
                    parts = candidate_score_parts.setdefault(
                        block_id,
                        {"context": 0.0, "semantic": 0.0, "hierarchy": 0.0},
                    )
                    if block_id in seed_ranks:
                        parts["direct_seed_rank"] = seed_ranks[block_id]
                        parts["direct_seed"] = round(float(seed_scores.get(block_id, 0.0)), 8)
                    if block_id in ppr_ranks:
                        parts["ppr_rank"] = ppr_ranks[block_id]
                if self.config.enable_shortest_path_connector:
                    connector_result = shortest_path_connector_with_paths(
                        index=self.evibridge_index,
                        seed_ids=seed_scores.keys(),
                        candidate_ids=candidate_scores.keys(),
                        max_paths=self.config.connector_max_paths,
                        bridge_types=self._enabled_bridge_types(),
                    )
                    connector_paths = connector_result.paths
                    connector_edges = connector_result.edges
                    for block_id, score in connector_result.scores.items():
                        candidate_scores[block_id] = max(candidate_scores.get(block_id, 0.0), score * 0.5)
                        candidate_score_parts.setdefault(
                            block_id,
                            {"context": 0.0, "semantic": 0.0, "hierarchy": 0.0},
                        )["connector"] = round(float(score), 8)
                self._expand_auxiliary_candidates(candidate_scores, candidate_score_parts)
                candidate_scores = self._rerank_candidate_scores(
                    query=query,
                    candidate_scores=candidate_scores,
                    candidate_score_parts=candidate_score_parts,
                )
            round_candidate_ids = set(candidate_scores)
            new_candidate_ids = sorted(round_candidate_ids - seen_candidate_ids)
            final_evidence_types = set(self.config.final_evidence_types)
            new_answer_candidate_ids = [
                block_id
                for block_id in new_candidate_ids
                if self.evibridge_index.blocks.get(block_id)
                and self.evibridge_index.blocks[block_id].block_type in final_evidence_types
            ]

            if self._ablation_variant() == "wo_budgeted_selector":
                selected = self._top_selected(candidate_scores)
            else:
                selected = select_budgeted_evidence(
                    index=self.evibridge_index,
                    query=query,
                    candidate_scores=candidate_scores,
                    demand=demand,
                    max_blocks=self._max_context_blocks(),
                    max_tokens=self.config.max_context_tokens,
                    final_evidence_types=self.config.final_evidence_types,
                    bridge_auxiliary_types=self.config.bridge_auxiliary_types,
                    paragraph_quota=self.config.paragraph_quota,
                    auxiliary_quota=self.config.auxiliary_quota,
                )

            selected_blocks = [item.block for item in selected]
            selected_ids = [block.block_id for block in selected_blocks]
            selected_bridges = self.evibridge_index.get_related_bridges(selected_ids, expand_depth=1)
            if self._ablation_variant() == "wo_sufficiency_verifier":
                verdict = SufficiencyVerdict(
                    sufficient=True,
                    missing=[],
                    relevance=1.0,
                    connectivity=1.0,
                    coverage=1.0,
                    specificity=1.0,
                    noise=0.0,
                    next_bridge=[],
                    next_action="accept",
                    reason="verifier disabled",
                )
            else:
                verdict = self.verifier.verify(query, demand, selected_blocks, selected_bridges)

            selected_answer_ids = {
                block.block_id
                for block in selected_blocks
                if block.block_type in final_evidence_types
            }
            best_answer_score = max(
                (
                    float(candidate_scores.get(block_id, 0.0))
                    for block_id in selected_answer_ids
                ),
                default=0.0,
            )
            evidence_score_improved = (
                previous_best_answer_score is None
                or best_answer_score > previous_best_answer_score + 1e-8
            )
            evidence_set_expanded = bool(selected_answer_ids - seen_selected_answer_ids)
            iteration_record = {
                "iteration": iteration + 1,
                "candidate_scores": candidate_scores,
                "candidate_score_parts": candidate_score_parts,
                "new_candidate_ids": new_candidate_ids,
                "new_answer_candidate_ids": new_answer_candidate_ids,
                "selected_block_ids": selected_ids,
                "selected": self._selected_payload(selected, candidate_score_parts),
                "verification": verdict.model_dump(),
                "connector_paths": connector_paths,
                "connector_edges": connector_edges,
                "next_action": verdict.next_action,
                "evidence_score_improved": evidence_score_improved,
                "evidence_set_expanded": evidence_set_expanded,
            }
            iteration_records.append(iteration_record)
            seen_candidate_ids.update(round_candidate_ids)
            if verdict.sufficient:
                stopping_reason = "sufficient"
                break
            if iteration + 1 >= max(max_iterations, 1):
                stopping_reason = "max_iterations"
                break
            if previous_best_answer_score is not None and not (
                evidence_score_improved or evidence_set_expanded
            ):
                stopping_reason = "no_evidence_gain"
                break
            refined_scores, refinement = self._refine_seed_scores_with_diagnostics(
                query=query,
                demand=demand,
                seed_scores=seed_scores,
                selected_ids=selected_ids,
                verdict=verdict,
                excluded_candidate_ids=seen_candidate_ids,
            )
            iteration_record["refinement"] = refinement
            if not refinement["new_answer_candidate_ids"]:
                stopping_reason = "no_new_candidates"
                break
            seed_scores = refined_scores
            seen_selected_answer_ids.update(selected_answer_ids)
            previous_best_answer_score = best_answer_score

        selected_ids = [item.block.block_id for item in selected]
        evidence_chain = self._build_evidence_chain(selected, demand, verdict)
        selected_payload = self._selected_payload(selected, candidate_score_parts)
        supporting_evidence = self._supporting_evidence_payload(selected_payload)
        supporting_block_ids = [item["block_id"] for item in supporting_evidence]
        return {
            "query": query,
            "demand": demand,
            "seed_results": seed_results,
            "typed_ppr_scores": candidate_scores,
            "typed_ppr_score_parts": candidate_score_parts,
            "connector_paths": connector_paths,
            "connector_edges": connector_edges,
            "selected": selected,
            "selected_payload": selected_payload,
            "selected_bridges": selected_bridges,
            "selected_block_ids": selected_ids,
            "retrieved_block_ids": selected_ids,
            "supporting_evidence": supporting_evidence,
            "supporting_block_ids": supporting_block_ids,
            "evidence_chain": evidence_chain,
            "verification": verdict,
            "iterations": iteration_records,
            "stopping_reason": stopping_reason,
        }

    def _create_augmented_prompt(
        self,
        query: str,
        evidence_chain: Optional[List[Dict[str, Any]]] = None,
        demand: Optional[EvidenceDemand] = None,
        verification: Optional[SufficiencyVerdict] = None,
    ) -> str:
        evidence_chain = evidence_chain or []
        evidence_text = "\n\n".join(
            [
                f"[block_id={item['block_id']}] type={item['block_type']} "
                f"page={item.get('page')} section={item.get('section_path')}\n"
                f"{item.get('text', '')}"
                for item in evidence_chain
            ]
        )
        demand_text = demand.model_dump_json() if demand else "{}"
        verdict_text = verification.model_dump_json() if verification else "{}"
        return (
            "You answer complex document questions using only the provided bridged evidence.\n"
            "Return only a JSON object with keys answer_short, answer_rationale, and supporting_block_ids.\n"
            "answer_short must be a concise Qasper-style answer: use exact spans when possible, "
            "answer exactly Yes or No for boolean questions, and use Unanswerable only when evidence is insufficient. "
            "Do not include evidence bullets or explanations in answer_short.\n"
            "For global-summary or abstractive questions, answer_short may be one concise synthesis sentence; "
            "synthesize only the strongest supporting_block_ids and do not add background knowledge.\n"
            "supporting_block_ids must contain 1 to 4 block_id values from the evidence chain that best support answer_short.\n"
            "Copy the exact integer shown in each [block_id=...] label; never use evidence order numbers.\n"
            f"Question: {query}\n"
            f"Evidence demand: {demand_text}\n"
            f"Sufficiency verdict: {verdict_text}\n"
            f"Evidence chain:\n{evidence_text or 'No evidence retrieved.'}\n"
            "Answer:"
        )

    def generation(self, query: str, query_output_dir: str) -> Tuple[str, List[Any]]:
        retrieval_info = self._retrieve(query)
        prompt = self._create_augmented_prompt(
            query=query,
            evidence_chain=retrieval_info["evidence_chain"],
            demand=retrieval_info["demand"],
            verification=retrieval_info["verification"],
        )
        try:
            answer = self.llm.get_completion(prompt=prompt, json_response=False)
        except TypeError:
            answer = self.llm.get_completion(prompt)
        answer_evidence = list(retrieval_info.get("evidence_chain") or [])
        fallback_info = {
            "enabled": bool(self.config.enable_long_context_fallback),
            "used": False,
            "trigger": "disabled",
            "context_block_ids": [],
            "estimated_tokens": 0,
        }
        verdict = retrieval_info.get("verification")
        if self.config.enable_long_context_fallback:
            if verdict is not None and not verdict.sufficient:
                fallback_context = self._build_long_context_fallback(
                    query=query,
                    retrieval_info=retrieval_info,
                )
                if fallback_context:
                    fallback_prompt = self._create_fallback_prompt(
                        query=query,
                        evidence_items=fallback_context,
                    )
                    try:
                        answer = self.llm.get_completion(
                            prompt=fallback_prompt,
                            json_response=False,
                        )
                    except TypeError:
                        answer = self.llm.get_completion(fallback_prompt)
                    answer_evidence = fallback_context
                    self._merge_fallback_context(retrieval_info, fallback_context)
                    fallback_info = {
                        "enabled": True,
                        "used": True,
                        "trigger": "insufficient_evidence",
                        "context_block_ids": [
                            item["block_id"] for item in fallback_context
                        ],
                        "estimated_tokens": sum(
                            len(evidence_tokenize(item.get("text", "")))
                            for item in fallback_context
                        ),
                    }
                else:
                    fallback_info["trigger"] = "no_fallback_context"
            else:
                fallback_info["trigger"] = "sufficient_evidence"
        retrieval_info["fallback"] = fallback_info
        answer_short, answer_rationale, answer_supporting_ids = self._parse_answer_payload(answer)
        answer_short = self._normalize_answer_short(answer_short, retrieval_info["demand"])
        if self.config.enable_short_answer_extraction:
            answer_short, extraction_info = self._extract_short_answer_from_evidence(
                query=query,
                answer_short=answer_short,
                demand=retrieval_info["demand"],
                evidence_items=answer_evidence,
            )
        else:
            extraction_info = {
                "enabled": False,
                "applied": False,
                "source_block_id": None,
                "original_answer": answer_short,
                "extracted_answer": answer_short,
            }
        retrieval_info["answer_extraction"] = extraction_info
        self._apply_answer_supporting_ids(
            retrieval_info,
            answer_supporting_ids,
            answer_short=answer_short,
        )

        output_dir = Path(query_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save_retrieval_outputs(retrieval_info, output_dir)
        retrieved_ids = retrieval_info["retrieved_block_ids"]
        supporting_ids = retrieval_info["supporting_block_ids"]
        self.last_retrieved_block_ids = retrieved_ids
        self.last_answer_short = answer_short
        self.last_answer_rationale = answer_rationale
        self.last_supporting_block_ids = supporting_ids
        return answer, retrieved_ids

    def _build_long_context_fallback(
        self,
        query: str,
        retrieval_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        query_terms = set(evidence_tokenize(query))
        selected_ids = set(retrieval_info.get("selected_block_ids") or [])
        selected_sections = {
            str(item.get("section_id") or "").strip()
            for item in retrieval_info.get("selected_payload", [])
            if str(item.get("section_id") or "").strip()
        }
        candidates: List[Tuple[float, EvidenceBlock]] = []
        final_types = set(self.config.final_evidence_types)
        for block in self.evibridge_index.blocks.values():
            if block.block_type not in final_types:
                continue
            block_terms = set(
                evidence_tokenize(
                    " ".join(
                        [
                            block.section_id,
                            " ".join(block.title_path),
                            block.text,
                        ]
                    )
                )
            )
            lexical = len(query_terms & block_terms) / max(len(query_terms), 1)
            same_section = bool(
                block.section_id and block.section_id in selected_sections
            )
            score = lexical + (0.75 if same_section else 0.0)
            if block.block_id in selected_ids:
                score += 1.0
            if score > 0:
                candidates.append((score, block))
        candidates.sort(
            key=lambda item: (
                -item[0],
                item[1].page if item[1].page is not None else 10**9,
                item[1].block_id,
            )
        )
        max_blocks = max(int(self.config.fallback_max_context_blocks or 0), 0)
        max_tokens = max(int(self.config.fallback_max_context_tokens or 0), 0)
        payload: List[Dict[str, Any]] = []
        used_tokens = 0
        for score, block in candidates:
            token_cost = max(len(evidence_tokenize(block.text)), 1)
            if max_tokens and used_tokens + token_cost > max_tokens and payload:
                continue
            metadata = dict(block.metadata or {})
            payload.append(
                {
                    "block_id": block.block_id,
                    "block_type": block.block_type,
                    "page": block.page,
                    "section_id": block.section_id,
                    "section_path": " > ".join(block.title_path) or block.section_id,
                    "metadata": metadata,
                    "hotpot_title": metadata.get("hotpot_title"),
                    "hotpot_sent_id": metadata.get("hotpot_sent_id"),
                    "text": block.text,
                    "score": round(float(score), 8),
                    "score_parts": {"fallback_relevance": round(float(score), 8)},
                    "bridge_types": [],
                    "evidence_role": "answer_evidence",
                    "selection_rank": len(payload) + 1,
                    "fallback_context": True,
                }
            )
            used_tokens += token_cost
            if max_blocks and len(payload) >= max_blocks:
                break
        return payload

    @staticmethod
    def _create_fallback_prompt(
        query: str,
        evidence_items: List[Dict[str, Any]],
    ) -> str:
        evidence_text = "\n\n".join(
            f"[block_id={item['block_id']}] section={item.get('section_path')} "
            f"page={item.get('page')}\n{item.get('text', '')}"
            for item in evidence_items
        )
        return (
            "Answer the question using only the chapter-level fallback evidence. "
            "Return only JSON with answer_short, answer_rationale, and supporting_block_ids. "
            "Use the shortest exact evidence span for extractive, entity, numeric, and yes/no "
            "questions; use Unanswerable if the fallback evidence is still insufficient.\n"
            f"Question: {query}\n"
            f"Fallback evidence:\n{evidence_text}\n"
            "Answer:"
        )

    @staticmethod
    def _merge_fallback_context(
        retrieval_info: Dict[str, Any],
        fallback_context: List[Dict[str, Any]],
    ) -> None:
        selected_payload = list(retrieval_info.get("selected_payload") or [])
        existing_ids = {
            int(item["block_id"])
            for item in selected_payload
            if item.get("block_id") is not None
        }
        for item in fallback_context:
            block_id = int(item["block_id"])
            if block_id not in existing_ids:
                payload = dict(item)
                payload["selection_rank"] = len(selected_payload) + 1
                selected_payload.append(payload)
                existing_ids.add(block_id)
        retrieval_info["selected_payload"] = selected_payload
        retrieved_ids = list(retrieval_info.get("retrieved_block_ids") or [])
        for item in fallback_context:
            block_id = int(item["block_id"])
            if block_id not in retrieved_ids:
                retrieved_ids.append(block_id)
        retrieval_info["retrieved_block_ids"] = retrieved_ids

    def close(self):
        if hasattr(self.evibridge_vector_store, "close"):
            self.evibridge_vector_store.close()
        if hasattr(self.reranker, "close"):
            self.reranker.close()

    def _hybrid_seed_retrieval(self, query: str, demand: EvidenceDemand) -> List[Dict[str, Any]]:
        retrieval_queries = list(
            dict.fromkeys(
                item.strip()
                for item in [query, *(demand.subqueries or [])]
                if str(item).strip()
            )
        )
        bm25_by_id: Dict[int, Dict[str, Any]] = {}
        for query_rank, retrieval_query in enumerate(retrieval_queries):
            for raw_item in self.evibridge_index.search_bm25(
                self.bm25,
                query=retrieval_query,
                top_k=self._bm25_topk(),
            ):
                item = dict(raw_item)
                block_id = int(item["block_id"])
                existing = bm25_by_id.get(block_id)
                if existing is None:
                    item["matched_queries"] = [retrieval_query]
                    item["matched_query_ranks"] = [query_rank]
                    bm25_by_id[block_id] = item
                else:
                    existing["score"] = max(
                        float(existing.get("score", 0.0)),
                        float(item.get("score", 0.0)),
                    )
                    if retrieval_query not in existing["matched_queries"]:
                        existing["matched_queries"].append(retrieval_query)
                        existing["matched_query_ranks"].append(query_rank)
        bm25_results = list(bm25_by_id.values())
        for item in bm25_results:
            item["source"] = "bm25"
            item["seed_family"] = "block"
            item["seed_families"] = ["block"]
            item["bm25_score"] = float(item.get("score", 0.0))
            item["vector_score"] = 0.0

        vector_results: List[Dict[str, Any]] = []
        if self.config.enable_vector_recall and self.evibridge_vector_store is not None:
            vector_by_id: Dict[int, Dict[str, Any]] = {}
            for query_rank, retrieval_query in enumerate(retrieval_queries):
                for raw_item in self._search_vector(
                    retrieval_query,
                    top_k=self._embedding_topk(),
                ):
                    item = dict(raw_item)
                    block_id = int(item["block_id"])
                    existing = vector_by_id.get(block_id)
                    if existing is None:
                        item["matched_queries"] = [retrieval_query]
                        item["matched_query_ranks"] = [query_rank]
                        vector_by_id[block_id] = item
                    else:
                        existing["vector_score"] = max(
                            float(existing.get("vector_score", 0.0)),
                            float(item.get("vector_score", 0.0)),
                        )
                        existing["score"] = max(
                            float(existing.get("score", 0.0)),
                            float(item.get("score", 0.0)),
                        )
                        if retrieval_query not in existing["matched_queries"]:
                            existing["matched_queries"].append(retrieval_query)
                            existing["matched_query_ranks"].append(query_rank)
            vector_results = list(vector_by_id.values())

        bm25_norm = self._normalized_scores(bm25_results, "bm25_score")
        vector_norm = self._normalized_scores(vector_results, "vector_score")
        combined: Dict[int, Dict[str, Any]] = {}
        for item in bm25_results:
            block_id = item["block_id"]
            combined[block_id] = {
                **item,
                "bm25_norm": bm25_norm.get(block_id, 0.0),
                "vector_norm": 0.0,
            }
        for item in vector_results:
            block_id = item["block_id"]
            if block_id in combined:
                combined[block_id]["source"] = "hybrid"
                combined[block_id]["vector_score"] = item["vector_score"]
                combined[block_id]["vector_norm"] = vector_norm.get(block_id, 0.0)
                combined[block_id]["matched_queries"] = list(
                    dict.fromkeys(
                        [
                            *(combined[block_id].get("matched_queries") or []),
                            *(item.get("matched_queries") or []),
                        ]
                    )
                )
            else:
                combined[block_id] = {
                    **item,
                    "source": "vector",
                    "bm25_score": 0.0,
                    "bm25_norm": 0.0,
                    "vector_norm": vector_norm.get(block_id, 0.0),
                }

        for item in combined.values():
            item["score"] = (
                self.config.hybrid_bm25_weight * item.get("bm25_norm", 0.0)
                + self.config.hybrid_vector_weight * item.get("vector_norm", 0.0)
            )
        ablation_variant = self._ablation_variant()
        if ablation_variant == "wo_demand_aware_seed_recall":
            for item in self._demand_agnostic_seed_results(query):
                self._merge_seed_item(combined, item)
        elif ablation_variant != "wo_multi_granularity_seeds":
            for item in self._multigranularity_seed_results(query, demand):
                self._merge_seed_item(combined, item)
        return sorted(combined.values(), key=lambda item: item.get("score", 0.0), reverse=True)

    def _search_vector(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        raw_results = self.evibridge_vector_store.search(query_text=query, top_k=top_k)
        mapped: List[Dict[str, Any]] = []
        for item in raw_results:
            metadata = item.get("metadata") or {}
            block_id = self._coerce_int(metadata.get("block_id") or metadata.get("source_node_id"))
            if block_id is None or block_id not in self.evibridge_index.blocks:
                continue
            block = self.evibridge_index.blocks[block_id]
            mapped.append(
                {
                    "block_id": block_id,
                    "score": float(item.get("score", item.get("distance", 0.0))),
                    "vector_score": float(item.get("score", item.get("distance", 0.0))),
                    "block_type": block.block_type,
                    "section_id": block.section_id,
                    "page": block.page,
                    "text": block.text,
                    "block": block.model_dump(),
                }
            )
        return mapped

    def _multigranularity_seed_results(
        self,
        query: str,
        demand: EvidenceDemand,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if demand.granularity in {"summary", "patch"} or demand.scope in {"document", "multi-document"}:
            results.extend(
                self._type_seed_results(
                    query=query,
                    block_types=["summary"],
                    top_k=self.config.patch_topk,
                    seed_family="summary",
                    score_boost=0.72 if demand.granularity == "summary" else 0.55,
                )
            )
            results.extend(
                self._type_seed_results(
                    query=query,
                    block_types=["patch"],
                    top_k=self.config.patch_topk,
                    seed_family="patch",
                    score_boost=0.6,
                )
            )
        if demand.granularity == "entity" or demand.intent in {"multi-hop", "comparison", "global-summary"}:
            results.extend(
                self._type_seed_results(
                    query=query,
                    block_types=["entity"],
                    top_k=self.config.entity_topk,
                    seed_family="entity",
                    score_boost=0.68,
                )
            )
        if "table" in demand.modality or demand.intent == "table-figure":
            results.extend(
                self._type_seed_results(
                    query=query,
                    block_types=["table", "figure", "caption"],
                    top_k=self.config.patch_topk,
                    seed_family="modality",
                    score_boost=0.7,
                )
            )
        for item in results:
            item["demand_aware_seed"] = True
        return results

    def _demand_agnostic_seed_results(self, query: str) -> List[Dict[str, Any]]:
        fixed_seed_specs = [
            (["summary"], self.config.patch_topk, "summary"),
            (["patch"], self.config.patch_topk, "patch"),
            (["entity"], self.config.entity_topk, "entity"),
            (["table", "figure", "caption"], self.config.patch_topk, "modality"),
        ]
        results: List[Dict[str, Any]] = []
        for block_types, top_k, seed_family in fixed_seed_specs:
            for item in self._type_seed_results(
                query=query,
                block_types=block_types,
                top_k=top_k,
                seed_family=seed_family,
                score_boost=0.55,
            ):
                item["demand_aware_seed"] = False
                results.append(item)
        return results

    def _type_seed_results(
        self,
        query: str,
        block_types: List[str],
        top_k: int,
        seed_family: str,
        score_boost: float,
    ) -> List[Dict[str, Any]]:
        query_terms = set(evidence_tokenize(query))
        candidates: List[Dict[str, Any]] = []
        for block in self.evibridge_index.blocks.values():
            if block.block_type not in block_types:
                continue
            text = " ".join([block.section_id, " ".join(block.title_path), block.text])
            block_terms = set(evidence_tokenize(text))
            overlap = len(query_terms & block_terms) / max(len(query_terms), 1) if query_terms else 0.0
            exact_entity = block.block_type == "entity" and block.text.lower() in (query or "").lower()
            if overlap <= 0 and not exact_entity and seed_family != "summary":
                continue
            score = score_boost + overlap + (0.25 if exact_entity else 0.0)
            candidates.append(
                {
                    "block_id": block.block_id,
                    "score": round(float(score), 8),
                    "block_type": block.block_type,
                    "section_id": block.section_id,
                    "page": block.page,
                    "text": block.text,
                    "bridge_types": sorted(
                        {
                            bridge.bridge_type
                            for bridge in self.evibridge_index.get_related_bridges([block.block_id], expand_depth=1)
                        }
                    ),
                    "block": block.model_dump(),
                    "source": f"{seed_family}_seed",
                    "seed_family": seed_family,
                    "seed_families": [seed_family],
                    "bm25_score": 0.0,
                    "vector_score": 0.0,
                    "bm25_norm": 0.0,
                    "vector_norm": 0.0,
                }
            )
        return sorted(candidates, key=lambda item: item["score"], reverse=True)[:top_k]

    @staticmethod
    def _merge_seed_item(combined: Dict[int, Dict[str, Any]], item: Dict[str, Any]) -> None:
        block_id = item["block_id"]
        if block_id not in combined:
            combined[block_id] = item
            return
        existing = combined[block_id]
        existing["score"] = max(float(existing.get("score", 0.0)), float(item.get("score", 0.0)))
        families = list(existing.get("seed_families") or [existing.get("seed_family", "block")])
        for family in item.get("seed_families") or [item.get("seed_family")]:
            if family and family not in families:
                families.append(family)
        existing["seed_families"] = families
        if item.get("seed_family") != "block":
            existing["seed_family"] = item.get("seed_family")
            existing["source"] = item.get("source", existing.get("source"))
            if "demand_aware_seed" in item:
                existing["demand_aware_seed"] = item.get("demand_aware_seed")

    def _bridge_scores(self, seed_scores: Dict[int, float], demand: EvidenceDemand) -> TypedPPRResult:
        bridge_types = self._enabled_bridge_types()
        return run_typed_ppr_with_details(
            index=self.evibridge_index,
            seed_scores=seed_scores,
            demand=demand,
            top_k=self.config.ppr_topk,
            restart_alpha=self.config.ppr_restart_alpha,
            max_iter=self.config.ppr_max_iter,
            bridge_types=bridge_types,
            use_typed_weights=self._ablation_variant() != "wo_typed_weights",
            typed_weights=self.config.typed_edge_weights,
        )

    def _seed_preserved_candidate_scores(
        self,
        seed_results: List[Dict[str, Any]],
        ppr_scores: Dict[int, float],
        active_seed_scores: Optional[Dict[int, float]] = None,
    ) -> Dict[int, float]:
        candidate_scores = dict(ppr_scores)
        preserve_limit = max(int(getattr(self.config, "preserve_seed_topk", 0) or 0), 0)
        final_types = set(getattr(self.config, "final_evidence_types", []) or [])
        if preserve_limit <= 0:
            return candidate_scores
        preserved_scores: Dict[int, float] = {}
        ordered_ids: List[int] = []
        for item in seed_results:
            block_id = int(item["block_id"])
            if item.get("block_type") not in final_types and item.get("block_type") != "paragraph":
                continue
            if block_id not in ordered_ids:
                ordered_ids.append(block_id)
            preserved_scores[block_id] = max(
                preserved_scores.get(block_id, 0.0),
                float(item.get("score", 0.0)),
            )
        for block_id, score in sorted(
            (active_seed_scores or {}).items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            block = self.evibridge_index.blocks.get(block_id)
            if not block or block.block_type not in final_types:
                continue
            if block_id not in ordered_ids:
                ordered_ids.append(block_id)
            preserved_scores[block_id] = max(
                preserved_scores.get(block_id, 0.0),
                float(score),
            )
        for block_id in ordered_ids[:preserve_limit]:
            seed_score = preserved_scores[block_id]
            candidate_scores[block_id] = max(float(candidate_scores.get(block_id, 0.0)), seed_score)
        return dict(sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True))

    def _expand_auxiliary_candidates(
        self,
        candidate_scores: Dict[int, float],
        candidate_score_parts: Dict[int, Dict[str, float]],
    ) -> None:
        for block_id, score in list(candidate_scores.items()):
            block = self.evibridge_index.blocks.get(block_id)
            if not block or block.block_type not in set(self.config.bridge_auxiliary_types):
                continue
            for target_id in self._answer_targets_for_auxiliary(block):
                target = self.evibridge_index.blocks.get(target_id)
                if not target or target.block_type not in set(self.config.final_evidence_types):
                    continue
                expanded_score = float(score) * 0.85
                if expanded_score > float(candidate_scores.get(target_id, 0.0)):
                    candidate_scores[target_id] = expanded_score
                    parts = candidate_score_parts.setdefault(
                        target_id,
                        {"context": 0.0, "semantic": 0.0, "hierarchy": 0.0},
                    )
                    parts["expanded_from_auxiliary"] = block.block_id
                    parts["auxiliary_seed"] = round(float(score), 8)

    def _rerank_candidate_scores(
        self,
        query: str,
        candidate_scores: Dict[int, float],
        candidate_score_parts: Dict[int, Dict[str, float]],
    ) -> Dict[int, float]:
        if not self.config.enable_candidate_rerank or self.reranker is None:
            return candidate_scores
        topk = max(int(getattr(self.config, "candidate_rerank_topk", 0) or 0), 0)
        if topk <= 0:
            return candidate_scores

        final_types = set(getattr(self.config, "final_evidence_types", []) or [])
        candidates: List[Tuple[int, EvidenceBlock, float]] = []
        for block_id, score in sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True):
            block = self.evibridge_index.blocks.get(block_id)
            if not block or block.block_type not in final_types:
                continue
            candidates.append((block_id, block, float(score)))
            if len(candidates) >= topk:
                break
        if not candidates:
            return candidate_scores

        documents = [self._rerank_document_text(block) for _, block, _ in candidates]
        try:
            scores = self.reranker.rerank(
                query=query,
                documents=documents,
                batch_size=max(int(getattr(self.config, "rerank_batch_size", 1) or 1), 1),
            )
        except Exception as exc:
            self._mark_rerank_failure(candidate_score_parts, candidates, str(exc))
            log.warning("EviBridge candidate rerank failed; falling back to original scores: %s", exc)
            return candidate_scores
        if len(scores) != len(candidates):
            self._mark_rerank_failure(
                candidate_score_parts,
                candidates,
                f"score count mismatch: got {len(scores)}, expected {len(candidates)}",
            )
            log.warning(
                "EviBridge candidate rerank returned %s scores for %s candidates; falling back.",
                len(scores),
                len(candidates),
            )
            return candidate_scores

        rerank_by_id = {
            block_id: float(score)
            for (block_id, _, _), score in zip(candidates, scores)
        }
        base_norm = self._normalized_score_map({block_id: score for block_id, _, score in candidates})
        rerank_norm = self._normalized_score_map(rerank_by_id)
        rerank_ranks = {
            block_id: rank
            for rank, block_id in enumerate(
                sorted(rerank_by_id, key=lambda item_id: rerank_by_id[item_id], reverse=True),
                1,
            )
        }
        weight = min(max(float(getattr(self.config, "candidate_rerank_weight", 0.0) or 0.0), 0.0), 1.0)
        reranked = dict(candidate_scores)
        for block_id, score in rerank_by_id.items():
            blended = (1.0 - weight) * base_norm.get(block_id, 0.0) + weight * rerank_norm.get(block_id, 0.0)
            reranked[block_id] = blended
            parts = candidate_score_parts.setdefault(
                block_id,
                {"context": 0.0, "semantic": 0.0, "hierarchy": 0.0},
            )
            parts["pre_rerank_score"] = round(float(candidate_scores.get(block_id, 0.0)), 8)
            parts["rerank_score"] = round(float(score), 8)
            parts["rerank_norm"] = round(float(rerank_norm.get(block_id, 0.0)), 8)
            parts["rerank_rank"] = rerank_ranks[block_id]
        return dict(sorted(reranked.items(), key=lambda item: item[1], reverse=True))

    @staticmethod
    def _mark_rerank_failure(
        candidate_score_parts: Dict[int, Dict[str, float]],
        candidates: List[Tuple[int, EvidenceBlock, float]],
        error: str,
    ) -> None:
        for block_id, _, _ in candidates:
            parts = candidate_score_parts.setdefault(
                block_id,
                {"context": 0.0, "semantic": 0.0, "hierarchy": 0.0},
            )
            parts["rerank_failed"] = True
            parts["rerank_error"] = str(error)[:300]

    @staticmethod
    def _rerank_document_text(block: EvidenceBlock) -> str:
        section = " > ".join(block.title_path) or block.section_id
        return f"section={section}\ntype={block.block_type}\n{block.text}"

    def _answer_targets_for_auxiliary(self, block: EvidenceBlock) -> List[int]:
        targets: List[int] = []
        if block.block_type == "patch":
            targets.extend(
                int(item)
                for item in block.metadata.get("contains_block_ids", [])
                if isinstance(item, int)
            )
        if block.block_type == "entity":
            targets.extend(
                int(item)
                for item in block.metadata.get("mentioned_by", [])
                if isinstance(item, int)
            )
        if block.block_type == "summary":
            for value in (block.metadata.get("summary_of"), block.parent_id):
                if isinstance(value, int):
                    targets.append(value)
        for bridge in self.evibridge_index.get_related_bridges([block.block_id], expand_depth=1):
            if bridge.relation_type in {
                "patch_contains",
                "contained_in_patch",
                "mentioned_by",
                "summary_of",
                "has_summary",
            }:
                targets.extend([bridge.source_id, bridge.target_id])
        unique: List[int] = []
        for target_id in targets:
            if target_id != block.block_id and target_id not in unique:
                unique.append(target_id)
        return unique

    def _top_selected(self, candidate_scores: Dict[int, float]) -> List[SelectedEvidence]:
        selected: List[SelectedEvidence] = []
        for block_id, score in sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)[
            : self._max_context_blocks()
        ]:
            block = self.evibridge_index.get_block(block_id)
            if block is None:
                continue
            selected.append(
                SelectedEvidence(
                    block=block,
                    score=round(float(score), 6),
                    score_parts={"relevance": round(float(score), 6), "cost": 0.0},
                    bridge_types=[],
                    evidence_role=self._evidence_role(block),
                    selection_rank=len(selected) + 1,
                )
            )
        return selected

    def _build_evidence_chain(
        self,
        selected: List[SelectedEvidence],
        demand: EvidenceDemand,
        verdict: Optional[SufficiencyVerdict],
    ) -> List[Dict[str, Any]]:
        items = sorted(selected, key=lambda item: ((item.block.page or 10**9), item.block.block_id))
        chain: List[Dict[str, Any]] = []
        for rank, item in enumerate(items, 1):
            block = item.block
            chain.append(
                {
                    "rank": rank,
                    "display_rank": rank,
                    "selection_rank": item.selection_rank,
                    "block_id": block.block_id,
                    "node_id": block.source_node_id,
                    "doc_id": block.doc_name,
                    "page": block.page,
                    "section_path": " > ".join(block.title_path) or block.section_id,
                    "block_type": block.block_type,
                    "role": self._role(block, demand),
                    "evidence_role": item.evidence_role,
                    "text": block.text,
                    "score": item.score,
                    "score_parts": item.score_parts,
                    "bridge_types": item.bridge_types,
                    "sufficiency_status": verdict.sufficient if verdict else None,
                }
            )
        return chain

    def _selected_payload(
        self,
        selected: List[SelectedEvidence],
        candidate_score_parts: Dict[int, Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        payload = []
        for item in sorted(selected, key=lambda value: value.selection_rank or 10**9):
            block = item.block
            diagnostics = candidate_score_parts.get(block.block_id, {})
            score_parts = {**diagnostics, **item.score_parts}
            payload.append(
                {
                    "block_id": block.block_id,
                    "block_type": block.block_type,
                    "page": block.page,
                    "section_id": block.section_id,
                    "metadata": dict(block.metadata or {}),
                    "hotpot_title": (block.metadata or {}).get("hotpot_title"),
                    "hotpot_sent_id": (block.metadata or {}).get("hotpot_sent_id"),
                    "text": block.text,
                    "score": item.score,
                    "score_parts": score_parts,
                    "bridge_types": item.bridge_types,
                    "evidence_role": item.evidence_role,
                    "selection_rank": item.selection_rank,
                    "direct_seed_rank": diagnostics.get("direct_seed_rank"),
                    "ppr_rank": diagnostics.get("ppr_rank"),
                }
            )
        return payload

    def _supporting_evidence_payload(
        self,
        selected_payload: List[Dict[str, Any]],
        preferred_ids: Optional[List[int]] = None,
        topk_override: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        allowed_types = set(self.config.supporting_evidence_types)
        topk = max(int(self.config.supporting_evidence_topk or 0), 0)
        if topk_override is not None:
            topk = max(int(topk_override), 0)
            if topk == 0:
                return []
        by_id = {int(item["block_id"]): item for item in selected_payload if item.get("block_id") is not None}
        ordered: List[Dict[str, Any]] = []
        for block_id in preferred_ids or []:
            item = by_id.get(block_id)
            if item is not None:
                ordered.append(item)
        remaining = [item for item in selected_payload if item not in ordered]
        if (
            self.config.enable_supporting_rerank
            and self._ablation_variant() != "wo_support_reranker"
        ):
            remaining = sorted(remaining, key=self._supporting_order_key)
        else:
            remaining = sorted(remaining, key=lambda value: value.get("selection_rank", 10**9))
        ordered.extend(remaining)

        supporting: List[Dict[str, Any]] = []
        seen = set()
        for item in ordered:
            block_id = int(item["block_id"])
            if block_id in seen:
                continue
            if item.get("block_type") not in allowed_types:
                continue
            if item.get("evidence_role") == "bridge_auxiliary":
                continue
            seen.add(block_id)
            payload = dict(item)
            payload["supporting_rank"] = len(supporting) + 1
            supporting.append(payload)
            if topk > 0 and len(supporting) >= topk:
                break
        return supporting

    def _apply_answer_supporting_ids(
        self,
        retrieval_info: Dict[str, Any],
        supporting_ids: List[int],
        answer_short: str = "",
    ) -> None:
        selected_payload = retrieval_info.get("selected_payload", [])
        requested_ids = list(dict.fromkeys(int(block_id) for block_id in supporting_ids))
        allowed_types = set(self.config.supporting_evidence_types)
        eligible_ids = {
            int(item["block_id"])
            for item in selected_payload
            if item.get("block_id") is not None
            and item.get("block_type") in allowed_types
            and item.get("evidence_role") != "bridge_auxiliary"
        }
        valid_ids = [block_id for block_id in requested_ids if block_id in eligible_ids]
        invalid_ids = [block_id for block_id in requested_ids if block_id not in eligible_ids]
        trust_answer_ids = (
            self.config.trust_answer_supporting_ids
            and self._ablation_variant() != "wo_citation_reorder"
        )
        preferred_ids = valid_ids if trust_answer_ids else None
        ignored_ids = invalid_ids if trust_answer_ids else requested_ids
        budget = self._supporting_evidence_budget(
            answer_short=answer_short,
            demand=retrieval_info.get("demand"),
        )
        supporting_evidence = self._supporting_evidence_payload(
            selected_payload,
            preferred_ids=preferred_ids,
            topk_override=(
                budget
                if answer_short and self.config.dynamic_supporting_evidence_budget
                else None
            ),
        )
        retrieval_info["supporting_evidence"] = supporting_evidence
        retrieval_info["supporting_block_ids"] = [item["block_id"] for item in supporting_evidence]
        retrieval_info["supporting_evidence_budget"] = budget
        retrieval_info["citation_validation"] = {
            "enabled": bool(trust_answer_ids),
            "requested_ids": requested_ids,
            "valid_ids": valid_ids,
            "invalid_ids": invalid_ids,
            "ignored_ids": ignored_ids,
            "used_ids": valid_ids if trust_answer_ids else [],
        }

    def _supporting_evidence_budget(
        self,
        answer_short: str,
        demand: Optional[EvidenceDemand],
    ) -> int:
        configured_maximum = max(int(self.config.supporting_evidence_topk or 0), 0)
        if not self.config.dynamic_supporting_evidence_budget or not answer_short:
            return configured_maximum
        maximum = configured_maximum or 4
        normalized = re.sub(r"[^a-z]+", " ", answer_short.lower()).strip()
        if normalized in {
            "unanswerable",
            "not answerable",
            "not enough information",
            "cannot be answered",
            "no answer",
            "unknown",
        }:
            return 0
        intent = demand.intent if demand is not None else "fact"
        if intent == "boolean" or normalized in {"yes", "no"}:
            return min(maximum, 3)
        if intent in {"global-summary", "aggregation"} or len(answer_short.split()) > 12:
            return min(maximum, 2)
        return min(maximum, 3)

    def _save_retrieval_outputs(self, retrieval_info: Dict[str, Any], output_dir: Path) -> None:
        retrieval_payload = {
            "query": retrieval_info["query"],
            "demand": retrieval_info["demand"].model_dump(),
            "demand_provenance": self._normalized_demand_provenance(
                retrieval_info["demand"]
            ),
            "seed_results": retrieval_info["seed_results"],
            "typed_ppr_scores": retrieval_info["typed_ppr_scores"],
            "typed_ppr_score_parts": retrieval_info["typed_ppr_score_parts"],
            "connector_paths": retrieval_info["connector_paths"],
            "connector_edges": retrieval_info["connector_edges"],
            "retrieved_block_ids": retrieval_info["retrieved_block_ids"],
            "supporting_block_ids": retrieval_info.get("supporting_block_ids", []),
            "supporting_evidence": retrieval_info.get("supporting_evidence", []),
            "supporting_evidence_budget": retrieval_info.get("supporting_evidence_budget"),
            "citation_validation": retrieval_info.get("citation_validation", {}),
            "answer_extraction": retrieval_info.get("answer_extraction", {}),
            "fallback": retrieval_info.get("fallback", {}),
            "selected": retrieval_info.get("selected_payload", []),
            "selected_bridges": self._bridge_payloads(retrieval_info.get("selected_bridges", [])),
            "verification": retrieval_info["verification"].model_dump()
            if retrieval_info["verification"]
            else None,
            "iterations": retrieval_info["iterations"],
            "stopping_reason": retrieval_info.get("stopping_reason"),
        }
        with open(output_dir / "retrieval_res.json", "w", encoding="utf-8") as f:
            json.dump(retrieval_payload, f, ensure_ascii=False, indent=2)
        if self.config.output_evidence_chain:
            with open(output_dir / "evidence_chain.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "evidence_chain": retrieval_info["evidence_chain"],
                        "connections": retrieval_payload["selected_bridges"],
                        "connector_paths": retrieval_info["connector_paths"],
                        "connector_edges": retrieval_info["connector_edges"],
                        "verification": retrieval_payload["verification"],
                        "iterations": retrieval_info["iterations"],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    @staticmethod
    def _normalized_demand_provenance(demand: EvidenceDemand) -> Dict[str, Any]:
        allowed_sources = {"rule", "llm", "hybrid", "fallback"}
        source = str(demand.source or "").strip().lower()
        if source not in allowed_sources:
            source = "fallback"
        raw_provenance = (
            demand.provenance if isinstance(demand.provenance, dict) else {}
        )
        parser = str(raw_provenance.get("parser") or source).strip().lower()
        if parser not in allowed_sources:
            parser = source
        signals = [
            str(item).strip()
            for item in raw_provenance.get("signals", [])
            if str(item).strip()
        ]
        return {
            "source": source,
            "parser": parser,
            "signals": list(dict.fromkeys(signals)),
            "intent": demand.intent,
            "confidence": float(demand.confidence),
            "rationale": str(demand.rationale or ""),
            "subqueries": list(demand.subqueries or []),
        }

    def _empty_retrieval(self, query: str, demand: EvidenceDemand) -> Dict[str, Any]:
        verdict = SufficiencyVerdict(
            sufficient=False,
            missing=["evidence"],
            next_bridge=demand.bridge_need,
            reason="no seed evidence",
        )
        return {
            "query": query,
            "demand": demand,
            "seed_results": [],
            "typed_ppr_scores": {},
            "typed_ppr_score_parts": {},
            "connector_paths": [],
            "connector_edges": [],
            "selected": [],
            "selected_bridges": [],
            "selected_block_ids": [],
            "retrieved_block_ids": [],
            "supporting_block_ids": [],
            "supporting_evidence": [],
            "evidence_chain": [],
            "verification": verdict,
            "iterations": [],
            "selected_payload": [],
            "stopping_reason": "no_seed_results",
        }

    def _enabled_bridge_types(self) -> List[str]:
        disabled = {
            "wo_context_edges": "context",
            "wo_semantic_edges": "semantic",
            "wo_hierarchy_edges": "hierarchy",
        }.get(self._ablation_variant())
        return [item for item in ["context", "semantic", "hierarchy"] if item != disabled]

    def _refine_seed_scores(
        self,
        query: str,
        demand: EvidenceDemand,
        seed_scores: Dict[int, float],
        selected_ids: List[int],
        verdict: SufficiencyVerdict,
    ) -> Dict[int, float]:
        refined, _ = self._refine_seed_scores_with_diagnostics(
            query=query,
            demand=demand,
            seed_scores=seed_scores,
            selected_ids=selected_ids,
            verdict=verdict,
            excluded_candidate_ids=set(selected_ids),
        )
        return refined

    def _refine_seed_scores_with_diagnostics(
        self,
        query: str,
        demand: EvidenceDemand,
        seed_scores: Dict[int, float],
        selected_ids: List[int],
        verdict: SufficiencyVerdict,
        excluded_candidate_ids: set[int],
    ) -> Tuple[Dict[int, float], Dict[str, Any]]:
        refined = seed_scores.copy()
        new_sources: Dict[int, List[str]] = {}

        def add_candidate(block_id: int, score: float, source: str) -> None:
            if (
                block_id in excluded_candidate_ids
                or block_id in refined
                or block_id not in self.evibridge_index.blocks
            ):
                return
            refined[block_id] = float(score)
            new_sources.setdefault(block_id, []).append(source)

        bridge_types = list(verdict.next_bridge) or self._bridge_types_for_action(verdict.next_action)
        for bridge in self.evibridge_index.get_related_bridges(
            selected_ids,
            expand_depth=1,
            bridge_types=bridge_types,
        ):
            for block_id in (bridge.source_id, bridge.target_id):
                add_candidate(
                    block_id,
                    max(0.35 * float(bridge.weight), 0.05),
                    f"{bridge.bridge_type}:{bridge.relation_type}",
                )
                block = self.evibridge_index.blocks.get(block_id)
                if block and block.block_type in set(self.config.bridge_auxiliary_types):
                    for target_id in self._answer_targets_for_auxiliary(block):
                        add_candidate(
                            target_id,
                            max(0.3 * float(bridge.weight), 0.05),
                            f"auxiliary:{block.block_id}",
                        )
        target_types = self._target_block_types_for_action(verdict)
        if target_types:
            for item in self._type_seed_results(
                query=query,
                block_types=target_types,
                top_k=max(self.config.patch_topk, self.config.entity_topk),
                seed_family=verdict.next_action,
                score_boost=0.45,
            ):
                add_candidate(
                    item["block_id"],
                    float(item["score"]),
                    f"targeted:{verdict.next_action}",
                )
        final_types = set(self.config.final_evidence_types)
        new_candidate_ids = sorted(new_sources)
        new_answer_candidate_ids = [
            block_id
            for block_id in new_candidate_ids
            if self.evibridge_index.blocks[block_id].block_type in final_types
        ]
        return refined, {
            "action": verdict.next_action,
            "bridge_types": bridge_types,
            "query": query,
            "new_candidate_ids": new_candidate_ids,
            "new_answer_candidate_ids": new_answer_candidate_ids,
            "candidate_sources": {
                str(block_id): sources
                for block_id, sources in sorted(new_sources.items())
            },
        }

    @staticmethod
    def _bridge_types_for_action(next_action: str) -> List[str]:
        if next_action in {"expand_context", "expand_table_caption"}:
            return ["context"]
        if next_action == "expand_semantic_bridge":
            return ["semantic"]
        if next_action == "expand_hierarchy_context":
            return ["hierarchy"]
        return ["context", "semantic", "hierarchy"]

    @staticmethod
    def _target_block_types_for_action(verdict: SufficiencyVerdict) -> List[str]:
        if verdict.next_action == "expand_table_caption":
            return ["table", "caption", "figure"]
        if verdict.next_action == "expand_semantic_bridge":
            return ["entity", "paragraph"]
        if verdict.next_action == "expand_hierarchy_context":
            return ["summary", "patch", "title"]
        if verdict.next_action == "expand_relevant_evidence":
            return ["paragraph", "table", "figure", "caption"]
        if verdict.next_action == "expand_context":
            return ["paragraph", "table", "figure", "caption"]
        return []

    @staticmethod
    def _bridge_payloads(bridges: List[Any]) -> List[Dict[str, Any]]:
        return [
            bridge.model_dump() if hasattr(bridge, "model_dump") else dict(bridge)
            for bridge in bridges
        ]

    def _ablation_variant(self) -> str:
        return getattr(self.config, "ablation_variant", "full") or "full"

    def _bm25_topk(self) -> int:
        return self.config.topk or self.config.bm25_topk

    def _embedding_topk(self) -> int:
        return self.config.topk or self.config.embedding_topk

    def _max_context_blocks(self) -> int:
        return self.config.topk or self.config.max_context_blocks

    @staticmethod
    def _normalized_scores(items: List[Dict[str, Any]], key: str) -> Dict[int, float]:
        if not items:
            return {}
        values = [float(item.get(key, 0.0)) for item in items]
        min_value = min(values)
        max_value = max(values)
        if max_value <= min_value:
            return {item["block_id"]: 1.0 for item in items}
        return {
            item["block_id"]: (float(item.get(key, 0.0)) - min_value) / (max_value - min_value)
            for item in items
        }

    @staticmethod
    def _normalized_score_map(scores: Dict[int, float]) -> Dict[int, float]:
        if not scores:
            return {}
        values = [float(value) for value in scores.values()]
        min_value = min(values)
        max_value = max(values)
        if max_value <= min_value:
            return {block_id: 1.0 for block_id in scores}
        return {
            block_id: (float(score) - min_value) / (max_value - min_value)
            for block_id, score in scores.items()
        }

    @staticmethod
    def _supporting_order_key(item: Dict[str, Any]) -> Tuple[int, float, int]:
        score_parts = item.get("score_parts") or {}
        rank = score_parts.get("rerank_rank")
        score = score_parts.get("rerank_score")
        if rank is not None:
            try:
                return (0, float(rank), int(item.get("selection_rank", 10**9)))
            except (TypeError, ValueError):
                pass
        if score is not None:
            try:
                return (1, -float(score), int(item.get("selection_rank", 10**9)))
            except (TypeError, ValueError):
                pass
        return (2, 0.0, int(item.get("selection_rank", 10**9)))

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_answer_payload(answer: Any) -> Tuple[str, str, List[int]]:
        text = str(answer or "").strip()
        cleaned = text
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            payload = json.loads(cleaned)
            if isinstance(payload, dict):
                answer_short = str(payload.get("answer_short") or payload.get("answer") or "").strip()
                answer_rationale = str(payload.get("answer_rationale") or payload.get("rationale") or "").strip()
                supporting_ids = EviBridgeRAG._parse_supporting_ids(payload.get("supporting_block_ids"))
                if answer_short:
                    return answer_short, answer_rationale, supporting_ids
        except Exception:
            pass
        for marker in ["Final Answer:", "Answer:", "answer_short:"]:
            if marker.lower() in text.lower():
                parts = text.split(marker, 1)
                if len(parts) == 2 and parts[1].strip():
                    return parts[1].strip(), "", []
        return text, "", []

    @staticmethod
    def _parse_supporting_ids(value: Any) -> List[int]:
        if value is None:
            return []
        raw_items = value if isinstance(value, list) else [value]
        ids: List[int] = []
        for item in raw_items:
            try:
                block_id = int(item)
            except (TypeError, ValueError):
                continue
            if block_id not in ids:
                ids.append(block_id)
        return ids

    @staticmethod
    def _normalize_answer_short(answer_short: str, demand: EvidenceDemand) -> str:
        text = str(answer_short or "").strip()
        if demand.intent != "boolean":
            return text
        normalized = text.lower()
        if normalized.startswith("yes") or normalized in {"true", "correct"}:
            return "Yes"
        if normalized.startswith("no") or normalized in {"false", "incorrect"}:
            return "No"
        return text

    @staticmethod
    def _extract_short_answer_from_evidence(
        query: str,
        answer_short: str,
        demand: EvidenceDemand,
        evidence_items: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        original = str(answer_short or "").strip()
        diagnostics = {
            "enabled": True,
            "applied": False,
            "source_block_id": None,
            "original_answer": original,
            "extracted_answer": original,
            "question": query,
        }
        normalized_unanswerable = re.sub(
            r"[^a-z]+",
            " ",
            original.lower(),
        ).strip()
        if normalized_unanswerable in {
            "unanswerable",
            "not answerable",
            "not enough information",
            "cannot be answered",
            "no answer",
            "unknown",
        }:
            diagnostics["extracted_answer"] = "Unanswerable"
            diagnostics["applied"] = original != "Unanswerable"
            return "Unanswerable", diagnostics
        if demand.intent == "boolean":
            normalized = EviBridgeRAG._normalize_answer_short(original, demand)
            diagnostics["extracted_answer"] = normalized
            diagnostics["applied"] = normalized != original
            return normalized, diagnostics
        if demand.intent in {"global-summary", "aggregation"} or not original:
            return original, diagnostics

        candidates: List[str] = []

        def add_candidate(value: str) -> None:
            cleaned = str(value or "").strip(" \t\r\n\"'`.,;:")
            if cleaned and cleaned not in candidates and len(cleaned.split()) <= 12:
                candidates.append(cleaned)

        stripped = re.sub(
            r"^\s*(?:the\s+answer\s+is|answer\s*:|it\s+is)\s+",
            "",
            original,
            flags=re.I,
        )
        add_candidate(
            re.split(
                r"\s*(?:,\s*)?\b(?:because|since|as the evidence|according to)\b",
                stripped,
                maxsplit=1,
                flags=re.I,
            )[0]
        )
        for quoted in re.findall(r"[\"']([^\"']{1,120})[\"']", original):
            add_candidate(quoted)
        numeric = re.search(
            r"\b\d+(?:\.\d+)?(?:\s*%)?(?:\s+(?:participants?|people|"
            r"samples?|documents?|papers?|years?|months?|days?|points?))?\b",
            stripped,
            re.I,
        )
        if numeric:
            add_candidate(numeric.group(0))
        add_candidate(stripped)

        for candidate in candidates:
            if candidate == original:
                continue
            for item in evidence_items:
                evidence_text = str(item.get("text") or "")
                if candidate.lower() in evidence_text.lower():
                    diagnostics.update(
                        {
                            "applied": True,
                            "source_block_id": item.get("block_id"),
                            "extracted_answer": candidate,
                        }
                    )
                    return candidate, diagnostics
        return original, diagnostics

    @staticmethod
    def _role(block: EvidenceBlock, demand: EvidenceDemand) -> str:
        if block.block_type in {"table", "figure", "caption"}:
            return "direct_modality_evidence"
        if block.block_type == "summary" or demand.granularity == "summary":
            return "global_context"
        if demand.intent in {"multi-hop", "comparison"}:
            return "semantic_bridge_evidence"
        return "local_evidence"

    def _evidence_role(self, block: EvidenceBlock) -> str:
        if block.block_type in set(self.config.final_evidence_types):
            return "answer_evidence"
        if block.block_type in set(self.config.bridge_auxiliary_types):
            return "bridge_auxiliary"
        return "answer_evidence"
