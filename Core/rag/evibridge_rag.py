import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Core.Index.EvidenceBridgeIndex import EvidenceBlock, EvidenceBridgeIndex, evidence_tokenize
from Core.configs.rag.evibridge_config import EviBridgeRAGConfig
from Core.rag.base_rag import BaseRAG
from Core.rag.evibridge_demand import DemandParser, EvidenceDemand
from Core.rag.evibridge_ppr import TypedPPRResult, run_typed_ppr_with_details, shortest_path_connector_with_paths
from Core.rag.evibridge_selector import SelectedEvidence, select_budgeted_evidence
from Core.rag.evibridge_verifier import EvidenceSufficiencyVerifier, SufficiencyVerdict


class EviBridgeRAG(BaseRAG):
    def __init__(
        self,
        config: EviBridgeRAGConfig,
        llm: Any,
        evibridge_index: EvidenceBridgeIndex,
        bm25: Any,
        evibridge_vector_store: Any = None,
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
        self.demand_parser = DemandParser(
            llm=llm,
            mode=config.demand_parser,
            confidence_threshold=config.demand_confidence_threshold,
        )
        self.verifier = EvidenceSufficiencyVerifier(
            llm=llm,
            enable_llm=config.enable_llm_verifier,
        )
        self.last_retrieved_block_ids: List[int] = []

    def _retrieve(self, query: str, **kwargs) -> Dict[str, Any]:
        demand = self.demand_parser.parse(query)
        seed_results = self._hybrid_seed_retrieval(query, demand)
        seed_scores = {item["block_id"]: item["score"] for item in seed_results}
        if not seed_scores:
            return self._empty_retrieval(query, demand)

        selected: List[SelectedEvidence] = []
        verdict: Optional[SufficiencyVerdict] = None
        iteration_records: List[Dict[str, Any]] = []
        candidate_scores: Dict[int, float] = seed_scores.copy()
        candidate_score_parts: Dict[int, Dict[str, float]] = {}
        connector_paths: List[Dict[str, Any]] = []
        connector_edges: List[Dict[str, Any]] = []
        selected_bridges = []

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
                candidate_scores = ppr_result.scores
                candidate_score_parts = ppr_result.score_parts
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

            iteration_records.append(
                {
                    "iteration": iteration + 1,
                    "candidate_scores": candidate_scores,
                    "candidate_score_parts": candidate_score_parts,
                    "selected_block_ids": selected_ids,
                    "verification": verdict.model_dump(),
                    "connector_paths": connector_paths,
                    "connector_edges": connector_edges,
                    "next_action": verdict.next_action,
                }
            )
            if verdict.sufficient:
                break
            seed_scores = self._refine_seed_scores(query, demand, seed_scores, selected_ids, verdict)

        selected_ids = [item.block.block_id for item in selected]
        evidence_chain = self._build_evidence_chain(selected, demand, verdict)
        return {
            "query": query,
            "demand": demand,
            "seed_results": seed_results,
            "typed_ppr_scores": candidate_scores,
            "typed_ppr_score_parts": candidate_score_parts,
            "connector_paths": connector_paths,
            "connector_edges": connector_edges,
            "selected": selected,
            "selected_bridges": selected_bridges,
            "selected_block_ids": selected_ids,
            "retrieved_block_ids": selected_ids,
            "evidence_chain": evidence_chain,
            "verification": verdict,
            "iterations": iteration_records,
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
                f"[{idx}] block_id={item['block_id']} type={item['block_type']} "
                f"page={item.get('page')} section={item.get('section_path')}\n"
                f"{item.get('text', '')}"
                for idx, item in enumerate(evidence_chain, 1)
            ]
        )
        demand_text = demand.model_dump_json() if demand else "{}"
        verdict_text = verification.model_dump_json() if verification else "{}"
        return (
            "You answer complex document questions using only the provided bridged evidence.\n"
            "If the evidence is insufficient, state that the document evidence is insufficient.\n"
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

        output_dir = Path(query_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save_retrieval_outputs(retrieval_info, output_dir)
        retrieved_ids = retrieval_info["retrieved_block_ids"]
        self.last_retrieved_block_ids = retrieved_ids
        return answer, retrieved_ids

    def close(self):
        if hasattr(self.evibridge_vector_store, "close"):
            self.evibridge_vector_store.close()

    def _hybrid_seed_retrieval(self, query: str, demand: EvidenceDemand) -> List[Dict[str, Any]]:
        bm25_results = self.evibridge_index.search_bm25(
            self.bm25,
            query=query,
            top_k=self._bm25_topk(),
        )
        for item in bm25_results:
            item["source"] = "bm25"
            item["seed_family"] = "block"
            item["seed_families"] = ["block"]
            item["bm25_score"] = float(item.get("score", 0.0))
            item["vector_score"] = 0.0

        vector_results: List[Dict[str, Any]] = []
        if self.config.enable_vector_recall and self.evibridge_vector_store is not None:
            vector_results = self._search_vector(query, top_k=self._embedding_topk())

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
        if self._ablation_variant() != "wo_multi_granularity_seeds":
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
                    "block_id": block.block_id,
                    "node_id": block.source_node_id,
                    "doc_id": block.doc_name,
                    "page": block.page,
                    "section_path": " > ".join(block.title_path) or block.section_id,
                    "block_type": block.block_type,
                    "role": self._role(block, demand),
                    "text": block.text,
                    "score": item.score,
                    "score_parts": item.score_parts,
                    "bridge_types": item.bridge_types,
                    "sufficiency_status": verdict.sufficient if verdict else None,
                }
            )
        return chain

    def _save_retrieval_outputs(self, retrieval_info: Dict[str, Any], output_dir: Path) -> None:
        retrieval_payload = {
            "query": retrieval_info["query"],
            "demand": retrieval_info["demand"].model_dump(),
            "seed_results": retrieval_info["seed_results"],
            "typed_ppr_scores": retrieval_info["typed_ppr_scores"],
            "typed_ppr_score_parts": retrieval_info["typed_ppr_score_parts"],
            "connector_paths": retrieval_info["connector_paths"],
            "connector_edges": retrieval_info["connector_edges"],
            "retrieved_block_ids": retrieval_info["retrieved_block_ids"],
            "selected": [
                {
                    "block_id": item.block.block_id,
                    "block_type": item.block.block_type,
                    "page": item.block.page,
                    "section_id": item.block.section_id,
                    "text": item.block.text,
                    "score": item.score,
                    "score_parts": item.score_parts,
                    "bridge_types": item.bridge_types,
                }
                for item in retrieval_info["selected"]
            ],
            "selected_bridges": self._bridge_payloads(retrieval_info.get("selected_bridges", [])),
            "verification": retrieval_info["verification"].model_dump()
            if retrieval_info["verification"]
            else None,
            "iterations": retrieval_info["iterations"],
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
            "evidence_chain": [],
            "verification": verdict,
            "iterations": [],
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
        refined = seed_scores.copy()
        for block_id in selected_ids:
            refined[block_id] = max(refined.get(block_id, 0.0), 0.5)
        for bridge in self.evibridge_index.get_related_bridges(
            selected_ids,
            expand_depth=1,
            bridge_types=verdict.next_bridge,
        ):
            refined[bridge.target_id] = max(refined.get(bridge.target_id, 0.0), 0.35 * bridge.weight)
            refined[bridge.source_id] = max(refined.get(bridge.source_id, 0.0), 0.35 * bridge.weight)
        target_types = self._target_block_types_for_action(verdict)
        if target_types:
            for item in self._type_seed_results(
                query=query,
                block_types=target_types,
                top_k=max(self.config.patch_topk, self.config.entity_topk),
                seed_family=verdict.next_action,
                score_boost=0.45,
            ):
                refined[item["block_id"]] = max(refined.get(item["block_id"], 0.0), item["score"])
        return refined

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
    def _coerce_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _role(block: EvidenceBlock, demand: EvidenceDemand) -> str:
        if block.block_type in {"table", "figure", "caption"}:
            return "direct_modality_evidence"
        if block.block_type == "summary" or demand.granularity == "summary":
            return "global_context"
        if demand.intent in {"multi-hop", "comparison"}:
            return "semantic_bridge_evidence"
        return "local_evidence"
