from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Tuple

import networkx as nx

from Core.configs.rag.hipporag_config import HippoRAGConfig
from Core.rag.base_rag import BaseRAG
from Core.utils.bm25 import BM25
from Core.utils.json_safety import make_json_safe
from Core.utils.utils import TextProcessor

if TYPE_CHECKING:
    from Core.provider.llm import LLM

log = logging.getLogger(__name__)


class HippoRAGRAG(BaseRAG):
    """
    Local HippoRAG-style baseline.

    This baseline intentionally keeps only the graph-retrieval idea: passage/entity
    graph construction, entity-aware seeds, and Personalized PageRank. It does not
    use EviBridge demand parsing, verifier, selector, or typed evidence graph.
    """

    _STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "more",
        "most",
        "of",
        "on",
        "or",
        "our",
        "paper",
        "show",
        "shows",
        "study",
        "than",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "would",
    }

    def __init__(
        self,
        config: HippoRAGConfig,
        llm: "LLM",
        tree_index: Any,
        save_path: str,
    ):
        super().__init__(
            llm=llm,
            name="HippoRAG",
            description="HippoRAG-style passage/entity graph retrieval baseline",
        )
        self.cfg = config
        self.tree_index = tree_index
        self.save_path = str(save_path)
        self.max_tokens = getattr(getattr(self.llm, "config", None), "max_tokens", 4096) - 200
        self._docs: List[Dict[str, Any]] = []
        self._bm25: BM25 | None = None
        self._graph: nx.Graph | None = None
        self._doc_entities: Dict[int, set[str]] = {}
        self._entity_to_passages: Dict[str, List[int]] = {}
        self._ready = False
        self.last_answer_short = ""
        self.last_answer_rationale = ""
        self.last_supporting_block_ids: List[int] = []
        self.last_retrieved_block_ids: List[int] = []
        self.last_graph_diagnostics: Dict[str, Any] = {}
        self.last_seed_diagnostics: Dict[str, Any] = {}

    @staticmethod
    def documents_from_tree(tree_index: Any) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        for node in tree_index.get_nodes(hasRoot=False):
            node_type = getattr(getattr(node, "type", None), "value", str(getattr(node, "type", ""))).lower()
            if node_type not in {"text", "paragraph"}:
                continue
            text = str(getattr(getattr(node, "meta_info", None), "content", "") or "").strip()
            if not text:
                continue
            source_id = HippoRAGRAG._normalize_source_id(getattr(node, "index_id"))
            docs.append(
                {
                    "id": f"node_{source_id}",
                    "content": text,
                    "metadata": {
                        "source": "hipporag",
                        "node_id": source_id,
                        "source_node_id": source_id,
                        "paragraph_id": source_id,
                        "evidence_id": source_id,
                        "qasper_evidence_text": text,
                        "node_type": "paragraph",
                        "section": HippoRAGRAG._section_name(node),
                        "title_path": HippoRAGRAG._title_path(node),
                    },
                }
            )
        return docs

    def _retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        self._ensure_ready()
        if not self._docs:
            return []

        bm25_results = self._bm25.search(  # type: ignore[union-attr]
            query_text=query,
            top_k=max(self.cfg.bm25_topk, self.cfg.topk),
        )
        query_entities = self._extract_entities(query, limit=self.cfg.max_query_entities)
        ppr_scores = self._run_ppr(query_entities=query_entities, bm25_results=bm25_results)
        bm25_by_id = self._bm25_scores_by_source_id(bm25_results)
        ranked: List[Dict[str, Any]] = []
        max_ppr = max((ppr_scores.get(self._passage_node(doc), 0.0) for doc in self._docs), default=0.0)
        max_bm25 = max(bm25_by_id.values(), default=0.0)
        query_entity_set = set(query_entities)

        for doc in self._docs:
            source_id = doc["metadata"]["source_node_id"]
            ppr_score = ppr_scores.get(self._passage_node(doc), 0.0)
            bm25_score = bm25_by_id.get(source_id, 0.0)
            entity_overlap = self._entity_overlap(query_entity_set, self._doc_entities.get(source_id, set()))
            ppr_norm = ppr_score / max_ppr if max_ppr > 0 else 0.0
            bm25_norm = bm25_score / max_bm25 if max_bm25 > 0 else 0.0
            final_score = (
                self.cfg.ppr_score_weight * ppr_norm
                + self.cfg.bm25_score_weight * bm25_norm
                + self.cfg.entity_overlap_weight * entity_overlap
            )
            ranked.append(
                {
                    **doc,
                    "score": final_score,
                    "source": "hipporag",
                    "ppr_score": ppr_score,
                    "bm25_score": bm25_score,
                    "entity_overlap": entity_overlap,
                    "metadata": {
                        **doc["metadata"],
                        "source": "hipporag",
                        "hipporag_entities": sorted(self._doc_entities.get(source_id, set()))[:10],
                    },
                }
            )

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[: self.cfg.topk]

    def _create_augmented_prompt(self, query: str, retrieved_docs: List[Dict[str, Any]] | None = None) -> str:
        if self.cfg.answer_style == "short":
            prompt = (
                "You answer Qasper-style document questions using only the retrieved passages.\n"
                "Return only a JSON object with keys answer_short and answer_rationale.\n"
                "answer_short must be concise: use exact spans when possible, answer Yes or No for boolean questions, "
                "and use Unanswerable only when the passages are insufficient.\n\n"
                f"Question: {query}\n\n"
                "Retrieved passages:\n"
            )
        else:
            prompt = (
                "Answer the question using only the retrieved passages. "
                "If the passages are insufficient, say so.\n\n"
                f"Question: {query}\n\n"
                "Retrieved passages:\n"
            )
        for index, doc in enumerate(retrieved_docs or [], start=1):
            meta = doc.get("metadata", {})
            source_id = meta.get("source_node_id")
            section = meta.get("section") or meta.get("title_path") or ""
            source = f"node={source_id}" if source_id is not None else f"rank={index}"
            if section:
                source += f", section={section}"
            prompt += f"Text {index} ({source}): {doc.get('content', '')}\n"
        chunks = TextProcessor.split_text_into_chunks(text=prompt, max_length=max(self.max_tokens - 400, 1000))
        return chunks[0] if chunks else prompt

    def generation(self, query: str, query_output_dir: str) -> Tuple[str, List[Any]]:
        query_output_path = Path(query_output_dir)
        query_output_path.mkdir(parents=True, exist_ok=True)
        ranked_results = self._retrieve(query)
        if not ranked_results:
            answer = self.llm.get_completion(query, json_response=False)
            self.last_answer_short, self.last_answer_rationale = self._parse_answer_payload(answer)
            self.last_retrieved_block_ids = []
            self.last_supporting_block_ids = []
            return str(answer), []

        prompt = self._create_augmented_prompt(query, ranked_results)
        answer = str(self.llm.get_completion(prompt, json_response=False) or "").strip()
        self.last_answer_short, self.last_answer_rationale = self._parse_answer_payload(answer)
        retrieved_ids = [item["metadata"]["source_node_id"] for item in ranked_results]
        supporting_evidence = self._supporting_evidence_from_ranked(
            ranked_results,
            answer_text=self.last_answer_short,
            query=query,
        )
        self.last_retrieved_block_ids = retrieved_ids
        self.last_supporting_block_ids = [item["source_node_id"] for item in supporting_evidence]
        self._save_retrieval_res(
            query=query,
            query_output_dir=query_output_path,
            ranked_results=ranked_results,
            supporting_evidence=supporting_evidence,
        )
        return answer, retrieved_ids

    def close(self):
        return None

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        self._docs = self.documents_from_tree(self.tree_index)
        self._bm25 = BM25(
            [doc["content"] for doc in self._docs],
            metadatas=[doc["metadata"] for doc in self._docs],
        )
        self._bm25.initialize()
        self._graph = self._build_graph()
        self.last_graph_diagnostics = self._graph_diagnostics()
        self._save_manifest()
        self._ready = True

    def _build_graph(self) -> nx.Graph:
        graph = nx.Graph()
        entity_to_passages: Dict[str, List[int]] = defaultdict(list)
        for doc in self._docs:
            source_id = doc["metadata"]["source_node_id"]
            passage_node = self._passage_node(doc)
            graph.add_node(passage_node, node_type="passage", source_node_id=source_id)
            entities = set(
                self._extract_entities(
                    doc["content"],
                    limit=self.cfg.max_entities_per_passage,
                )
            )
            self._doc_entities[source_id] = entities
            for entity in entities:
                entity_to_passages[entity].append(source_id)
                entity_node = self._entity_node(entity)
                graph.add_node(entity_node, node_type="entity", entity=entity)
                graph.add_edge(passage_node, entity_node, weight=1.0, relation="mentions")

        if self.cfg.enable_context_edges:
            ordered = sorted(self._docs, key=lambda item: str(item["metadata"]["source_node_id"]))
            for left, right in zip(ordered, ordered[1:]):
                graph.add_edge(
                    self._passage_node(left),
                    self._passage_node(right),
                    weight=self.cfg.context_edge_weight,
                    relation="neighbor",
                )
        self._entity_to_passages = dict(entity_to_passages)
        return graph

    def _run_ppr(self, query_entities: List[str], bm25_results: List[Dict[str, Any]]) -> Dict[str, float]:
        graph = self._graph
        if graph is None or graph.number_of_nodes() == 0:
            return {}
        personalization = {node: 0.0 for node in graph.nodes}
        max_bm25 = max((float(item.get("score", 0.0)) for item in bm25_results), default=0.0)
        passage_seeds = []
        for rank, item in enumerate(bm25_results[: self.cfg.bm25_topk], start=1):
            meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            source_id = meta.get("source_node_id") or meta.get("node_id") or item.get("id")
            source_id = self._normalize_source_id(source_id)
            node = f"p:{source_id}"
            if node not in personalization:
                continue
            raw_score = float(item.get("score", 0.0))
            normalized = raw_score / max_bm25 if max_bm25 > 0 else 1.0 / rank
            weight = self.cfg.passage_seed_weight * max(normalized, 1.0 / (rank + 1))
            personalization[node] += weight
            passage_seeds.append({"source_node_id": source_id, "rank": rank, "weight": weight})

        entity_seeds = []
        for entity in query_entities:
            node = self._entity_node(entity)
            if node in personalization:
                personalization[node] += self.cfg.entity_seed_weight
                entity_seeds.append(
                    {
                        "entity": entity,
                        "weight": self.cfg.entity_seed_weight,
                        "num_passages": len(self._entity_to_passages.get(entity, [])),
                    }
                )

        if sum(personalization.values()) <= 0:
            for doc in self._docs[: max(self.cfg.topk, 1)]:
                node = self._passage_node(doc)
                if node in personalization:
                    personalization[node] = 1.0

        self.last_seed_diagnostics = {
            "query_entities": query_entities,
            "passage_seeds": passage_seeds,
            "entity_seeds": entity_seeds,
        }
        try:
            return nx.pagerank(
                graph,
                alpha=self.cfg.ppr_alpha,
                personalization=personalization,
                weight="weight",
                max_iter=self.cfg.ppr_max_iter,
                tol=1.0e-6,
            )
        except Exception as exc:
            log.warning("HippoRAG PPR failed; falling back to seed scores: %s", exc)
            return personalization

    def _supporting_evidence_from_ranked(
        self,
        ranked_results: List[Dict[str, Any]],
        answer_text: str = "",
        query: str = "",
    ) -> List[Dict[str, Any]]:
        limit = max(0, int(self.cfg.supporting_evidence_topk or 0))
        answer_terms = self._tokens(answer_text)
        query_terms = self._tokens(query)
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        seen = set()
        for rank, item in enumerate(ranked_results, start=1):
            meta = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
            source_id = meta.get("source_node_id") or meta.get("node_id")
            source_id = self._normalize_source_id(source_id)
            if source_id in seen:
                continue
            seen.add(source_id)
            text = str(meta.get("qasper_evidence_text") or item.get("content") or "")
            terms = self._tokens(text)
            answer_overlap = len(answer_terms & terms) / max(len(answer_terms), 1) if answer_terms else 0.0
            query_overlap = len(query_terms & terms) / max(len(query_terms), 1) if query_terms else 0.0
            score = (2.0 * answer_overlap) + (0.4 * query_overlap) + float(item.get("score", 0.0)) - (0.001 * rank)
            candidates.append(
                (
                    score,
                    {
                        "supporting_rank": 0,
                        "source_node_id": source_id,
                        "node_id": source_id,
                        "paragraph_id": source_id,
                        "block_type": "paragraph",
                        "node_type": "paragraph",
                        "qasper_evidence_text": text,
                        "content": text,
                        "source": "hipporag",
                        "supporting_score": score,
                        "answer_overlap": answer_overlap,
                        "query_overlap": query_overlap,
                        "retrieval_rank": rank,
                    },
                )
            )
        if answer_terms or query_terms:
            candidates.sort(key=lambda pair: pair[0], reverse=True)
        supporting = [item for _, item in candidates]
        if limit:
            supporting = supporting[:limit]
        for index, item in enumerate(supporting, start=1):
            item["supporting_rank"] = index
        return supporting

    def _save_retrieval_res(
        self,
        query: str,
        query_output_dir: Path,
        ranked_results: List[Dict[str, Any]],
        supporting_evidence: List[Dict[str, Any]],
    ) -> None:
        payload = {
            "query": query,
            "strategy": "hipporag",
            "topk": self.cfg.topk,
            "bm25_topk": self.cfg.bm25_topk,
            "ppr_alpha": self.cfg.ppr_alpha,
            "graph_diagnostics": self.last_graph_diagnostics,
            "seed_diagnostics": self.last_seed_diagnostics,
            "supporting_evidence": supporting_evidence,
            "ranked_results": self._public_ranked_results(ranked_results),
        }
        with (query_output_dir / "retrieval_res.json").open("w", encoding="utf-8") as f:
            json.dump(make_json_safe(payload), f, indent=2, ensure_ascii=False, allow_nan=False)

    @staticmethod
    def _public_ranked_results(ranked_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        public = []
        for rank, item in enumerate(ranked_results, start=1):
            meta = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
            public.append(
                {
                    "rank": rank,
                    "id": meta.get("source_node_id") or item.get("id"),
                    "node_id": meta.get("source_node_id") or meta.get("node_id"),
                    "paragraph_id": meta.get("paragraph_id") or meta.get("source_node_id"),
                    "block_type": "paragraph",
                    "content": item.get("content", ""),
                    "qasper_evidence_text": meta.get("qasper_evidence_text") or item.get("content", ""),
                    "score": item.get("score", 0.0),
                    "ppr_score": item.get("ppr_score", 0.0),
                    "bm25_score": item.get("bm25_score", 0.0),
                    "entity_overlap": item.get("entity_overlap", 0.0),
                    "source": "hipporag",
                    "hipporag_entities": meta.get("hipporag_entities", []),
                }
            )
        return public

    def _save_manifest(self) -> None:
        try:
            os.makedirs(self.save_path, exist_ok=True)
            payload = {
                "strategy": "hipporag",
                "source": "DocumentTree",
                "graph_diagnostics": self.last_graph_diagnostics,
                "config": {
                    "topk": self.cfg.topk,
                    "bm25_topk": self.cfg.bm25_topk,
                    "ppr_alpha": self.cfg.ppr_alpha,
                    "max_entities_per_passage": self.cfg.max_entities_per_passage,
                },
            }
            with open(os.path.join(self.save_path, "hipporag_manifest.json"), "w", encoding="utf-8") as f:
                json.dump(make_json_safe(payload), f, indent=2, ensure_ascii=False, allow_nan=False)
        except Exception as exc:
            log.debug("Could not save HippoRAG manifest: %s", exc)

    def _graph_diagnostics(self) -> Dict[str, Any]:
        graph = self._graph
        if graph is None:
            return {"passages": 0, "entities": 0, "edges": 0, "graph_ready": False}
        entities = [node for node, data in graph.nodes(data=True) if data.get("node_type") == "entity"]
        passages = [node for node, data in graph.nodes(data=True) if data.get("node_type") == "passage"]
        return {
            "passages": len(passages),
            "entities": len(entities),
            "edges": graph.number_of_edges(),
            "graph_ready": bool(passages and entities and graph.number_of_edges() > 0),
        }

    def _bm25_scores_by_source_id(self, bm25_results: List[Dict[str, Any]]) -> Dict[int, float]:
        scores = {}
        for item in bm25_results:
            meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            source_id = self._normalize_source_id(meta.get("source_node_id") or meta.get("node_id") or item.get("id"))
            scores[source_id] = max(scores.get(source_id, 0.0), float(item.get("score", 0.0)))
        return scores

    @classmethod
    def _extract_entities(cls, text: str, limit: int = 16) -> List[str]:
        normalized_text = str(text or "")
        entities: List[str] = []
        for match in re.findall(r"\b[A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*){0,4}\b", normalized_text):
            cleaned = cls._normalize_entity(match)
            if cleaned and cleaned not in cls._STOPWORDS:
                entities.append(cleaned)
        tokens = list(cls._tokens(normalized_text))
        entities.extend(tokens)
        token_sequence = [token for token in re.findall(r"[A-Za-z][A-Za-z0-9-]*", normalized_text.lower()) if token not in cls._STOPWORDS]
        for size in (2, 3):
            for index in range(0, max(len(token_sequence) - size + 1, 0)):
                phrase = " ".join(token_sequence[index : index + size])
                if len(phrase) >= 7:
                    entities.append(phrase)
        ranked = Counter(entities)
        ordered = sorted(ranked, key=lambda item: (ranked[item], len(item), item), reverse=True)
        return ordered[: max(limit, 0)]

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9-]*", str(text).lower())
            if len(token) > 2 and token not in cls._STOPWORDS
        }

    @staticmethod
    def _normalize_entity(entity: str) -> str:
        return re.sub(r"\s+", " ", str(entity or "").strip()).lower()

    @staticmethod
    def _entity_overlap(query_entities: set[str], doc_entities: set[str]) -> float:
        if not query_entities:
            return 0.0
        return len(query_entities & doc_entities) / max(len(query_entities), 1)

    @staticmethod
    def _passage_node(doc: Dict[str, Any]) -> str:
        return f"p:{doc['metadata']['source_node_id']}"

    @staticmethod
    def _entity_node(entity: str) -> str:
        return f"e:{entity}"

    @staticmethod
    def _normalize_source_id(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return abs(hash(str(value))) % (10**12)

    @staticmethod
    def _section_name(node: Any) -> str:
        parent = getattr(node, "parent", None)
        content = str(getattr(getattr(parent, "meta_info", None), "content", "") or "").strip()
        return content

    @staticmethod
    def _title_path(node: Any) -> str:
        titles = []
        parent = getattr(node, "parent", None)
        while parent is not None:
            text = str(getattr(getattr(parent, "meta_info", None), "content", "") or "").strip()
            if text:
                titles.append(text)
            parent = getattr(parent, "parent", None)
        return " > ".join(reversed(titles))

    @staticmethod
    def _parse_answer_payload(answer: Any) -> Tuple[str, str]:
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
                if answer_short:
                    return answer_short, answer_rationale
        except Exception:
            pass
        for marker in ["Final Answer:", "Answer:", "answer_short:"]:
            if marker.lower() in text.lower():
                parts = re.split(re.escape(marker), text, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 2 and parts[1].strip():
                    return parts[1].strip(), ""
        return text, ""
