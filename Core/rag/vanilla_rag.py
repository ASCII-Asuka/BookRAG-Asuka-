from typing import TYPE_CHECKING

from Core.rag.base_rag import BaseRAG
from Core.configs.rag.vanilla_config import VanillaConfig
from Core.utils.bm25 import BM25
from Core.utils.utils import TextProcessor
from Core.utils.json_safety import make_json_safe

from typing import Dict, Any, List, Tuple
from dataclasses import asdict
import json
import logging
from pathlib import Path

if TYPE_CHECKING:
    from Core.provider.llm import LLM
    from Core.provider.vdb import VectorStore

log = logging.getLogger(__name__)


class VanillaRAG(BaseRAG):
    """
    Text-only Vanilla Retrieval Augmented Generation,
    supports vanilla, BM25, RAPTOR, PDF+Vanilla
    """

    def __init__(
        self,
        config: VanillaConfig,
        llm: "LLM",
        vector_store: "VectorStore" = None,
        bm25: BM25 = None,
        reranker: Any = None,
        tree_index: Any = None,
    ):
        super().__init__(
            llm=llm,
            name="MM RAG",
            description="Text-only Vanilla Retrieval Augmented Generation",
        )
        self.cfg = config
        self.max_tokens = self.llm.config.max_tokens - 200
        log.info("Vanilla RAG initialized.")
        self.topk = self.cfg.topk
        self.last_answer_short = ""
        self.last_answer_rationale = ""
        self.last_supporting_block_ids = []
        self.last_retrieved_block_ids = []
        self.bm25 = bm25
        self.vdb = vector_store
        self.reranker = reranker
        self.tree_index = tree_index
        self._longrag_units = None

    def _retrieve(self, query: str, top_k: int = 3):
        if self.cfg.retrieval_method == "bm25":
            return self.bm25.search(query_text=query, top_k=top_k)
        if self.cfg.retrieval_method == "hybrid":
            return self._hybrid_retrieve(query, top_k=top_k)
        if self.cfg.retrieval_method == "bm25_rerank":
            return self._bm25_rerank(query, top_k=top_k)
        if self.cfg.retrieval_method == "abstract_only":
            return self._abstract_only_context()
        if self.cfg.retrieval_method == "full_document":
            return self._full_document_context()
        if self.cfg.retrieval_method == "longrag":
            return self._longrag_retrieve(query, top_k=top_k)
        return self.vdb.search(query_text=query, top_k=top_k)

    def _hybrid_retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        bm25_results = self.bm25.search(
            query_text=query,
            top_k=int(getattr(self.cfg, "hybrid_bm25_topk", top_k) or top_k),
        )
        dense_results = self.vdb.search(
            query_text=query,
            top_k=int(getattr(self.cfg, "hybrid_dense_topk", top_k) or top_k),
        )
        fused: Dict[str, Dict[str, Any]] = {}
        self._add_rrf_results(fused, bm25_results, source="bm25")
        self._add_rrf_results(fused, dense_results, source="dense")
        ranked = sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)
        for item in ranked:
            item["score"] = item["rrf_score"]
            item["source"] = "hybrid" if len(item.get("sources", [])) > 1 else item["sources"][0]
            item.setdefault("metadata", {})["source"] = item["source"]
        return ranked[:top_k]

    def _add_rrf_results(
        self,
        fused: Dict[str, Dict[str, Any]],
        results: List[Dict[str, Any]],
        source: str,
    ) -> None:
        rrf_k = float(getattr(self.cfg, "rrf_k", 60) or 60)
        for rank, doc in enumerate(results, start=1):
            key = self._document_key(doc)
            if key not in fused:
                fused[key] = {
                    **doc,
                    "rrf_score": 0.0,
                    "sources": [],
                    "source_ranks": {},
                }
            fused[key]["rrf_score"] += 1.0 / (rrf_k + rank)
            fused[key]["sources"].append(source)
            fused[key]["source_ranks"][source] = rank

    @staticmethod
    def _document_key(doc: Dict[str, Any]) -> str:
        meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        for key in ["source_node_id", "node_id", "paragraph_id", "evidence_id"]:
            if meta.get(key) is not None:
                return f"{key}:{meta[key]}"
        return str(doc.get("id", doc.get("content", "")))

    def _bm25_rerank(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        candidates = self.bm25.search(
            query_text=query,
            top_k=int(getattr(self.cfg, "rerank_topk", max(top_k, 50)) or max(top_k, 50)),
        )
        if not candidates or self.reranker is None:
            return candidates[:top_k]
        documents = [item.get("content", "") for item in candidates]
        try:
            scores = self.reranker.rerank(
                query=query,
                documents=documents,
                batch_size=int(getattr(self.cfg, "rerank_batch_size", 50) or 50),
            )
        except Exception as exc:
            log.warning("BM25 reranker failed; falling back to BM25 order: %s", exc)
            fallback = []
            for item in candidates[:top_k]:
                fallback.append(
                    {
                        **item,
                        "bm25_score": float(item.get("score", 0.0)),
                        "rerank_failed": True,
                        "rerank_error": str(exc)[:300],
                        "source": "bm25_rerank_fallback",
                    }
                )
            return fallback
        if len(scores) != len(candidates):
            log.warning(
                "BM25 reranker returned %s scores for %s candidates; falling back to BM25 order.",
                len(scores),
                len(candidates),
            )
            fallback = []
            for item in candidates[:top_k]:
                fallback.append(
                    {
                        **item,
                        "bm25_score": float(item.get("score", 0.0)),
                        "rerank_failed": True,
                        "rerank_error": f"score count mismatch: got {len(scores)}, expected {len(candidates)}",
                        "source": "bm25_rerank_fallback",
                    }
                )
            return fallback
        reranked = []
        for item, score in zip(candidates, scores):
            reranked.append(
                {
                    **item,
                    "bm25_score": float(item.get("score", 0.0)),
                    "rerank_score": float(score),
                    "score": float(score),
                    "source": "bm25_rerank",
                }
            )
        return sorted(reranked, key=lambda item: item["rerank_score"], reverse=True)[:top_k]

    def _abstract_only_context(self) -> List[Dict[str, Any]]:
        if self.tree_index is None:
            return []
        docs = []
        for node in self.tree_index.get_nodes(hasRoot=False):
            text = str(getattr(node.meta_info, "content", "") or "").strip()
            if not text:
                continue
            if self._is_abstract_node(node):
                docs.append(
                    {
                        "id": node.index_id,
                        "score": 1.0,
                        "content": text,
                        "metadata": {
                            "source": "abstract_only",
                            "node_id": node.index_id,
                            "source_node_id": node.index_id,
                            "paragraph_id": node.index_id,
                            "evidence_id": node.index_id,
                            "qasper_evidence_text": text,
                            "node_type": getattr(getattr(node, "type", None), "value", str(getattr(node, "type", ""))),
                            "section_id": "Abstract",
                            "section": "Abstract",
                            "title_path": "Abstract",
                        },
                    }
                )
        return docs[: self.topk]

    def _full_document_context(self) -> List[Dict[str, Any]]:
        if self.tree_index is None:
            return []
        docs: List[Dict[str, Any]] = []
        for rank, node in enumerate(self.tree_index.get_nodes(hasRoot=False), start=1):
            text = self._node_text(node)
            if not text:
                continue
            node_type = self._node_type_value(node)
            block_type = "paragraph" if node_type == "text" else node_type
            metadata = {
                "source": "full_document",
                "node_id": node.index_id,
                "source_node_id": node.index_id,
                "paragraph_id": node.index_id,
                "evidence_id": node.index_id,
                "qasper_evidence_text": text,
                "node_type": node_type,
                "block_type": block_type,
                "page": getattr(node.meta_info, "page_idx", None),
                "rank": rank,
            }
            metadata.update(self._section_metadata(node))
            metadata.update(self._hotpot_sentence_metadata(node))
            docs.append(
                {
                    "id": node.index_id,
                    "score": 1.0,
                    "content": text,
                    "metadata": metadata,
                }
            )
        return docs

    def _longrag_retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        units = self._build_longrag_units()
        if not units:
            return []
        bm25 = BM25([item["content"] for item in units], metadatas=[item["metadata"] for item in units])
        bm25.initialize()
        ranked = bm25.search(query_text=query, top_k=min(top_k, len(units)))
        results: List[Dict[str, Any]] = []
        for item in ranked:
            unit = units[int(item["id"])]
            results.append(
                {
                    **unit,
                    "score": float(item.get("score", 0.0)),
                }
            )
        return results

    def _build_longrag_units(self) -> List[Dict[str, Any]]:
        if self._longrag_units is not None:
            return self._longrag_units
        if self.tree_index is None:
            self._longrag_units = []
            return self._longrag_units

        unit_token_limit = int(getattr(self.cfg, "longrag_unit_tokens", 4096) or 4096)
        unit_token_limit = max(128, unit_token_limit)
        sections = self._longrag_sections()
        units: List[Dict[str, Any]] = []
        for section in sections:
            lines: List[str] = []
            child_items: List[Dict[str, Any]] = []
            token_count = 0

            def flush() -> None:
                nonlocal lines, child_items, token_count
                if not lines:
                    return
                units.append(self._make_longrag_unit(units, section["section_id"], lines, child_items))
                lines = []
                child_items = []
                token_count = 0

            for item in section["items"]:
                line = item["line"]
                line_tokens = self._approx_token_count(line)
                if lines and token_count + line_tokens > unit_token_limit:
                    flush()
                lines.append(line)
                token_count += line_tokens
                if item.get("is_evidence"):
                    child_items.append(item)
            flush()

        self._longrag_units = units
        return units

    def _longrag_sections(self) -> List[Dict[str, Any]]:
        sections: List[Dict[str, Any]] = []
        current = {"section_id": "", "items": []}

        def flush_current() -> None:
            nonlocal current
            if current["items"]:
                sections.append(current)
            current = {"section_id": "", "items": []}

        for node in self.tree_index.get_nodes(hasRoot=False):
            text = self._node_text(node)
            if not text:
                continue
            node_type = self._node_type_value(node)
            if node_type == "title":
                flush_current()
                current = {"section_id": text, "items": []}
                current["items"].append(
                    {
                        "line": f"[Section] {text}",
                        "node_id": node.index_id,
                        "text": text,
                        "block_type": "title",
                        "is_evidence": False,
                    }
                )
                continue

            metadata = {
                "node_id": node.index_id,
                "text": text,
                "block_type": "paragraph" if node_type == "text" else node_type,
                "page": getattr(node.meta_info, "page_idx", None),
            }
            metadata.update(self._section_metadata(node))
            metadata.update(self._hotpot_sentence_metadata(node))
            if not current["section_id"]:
                current["section_id"] = metadata.get("section_id") or metadata.get("section") or ""
            current["items"].append(
                {
                    **metadata,
                    "line": f"[source_id={node.index_id}] {text}",
                    "is_evidence": True,
                }
            )
        flush_current()
        return sections

    def _make_longrag_unit(
        self,
        units: List[Dict[str, Any]],
        section_id: str,
        lines: List[str],
        child_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        unit_id = f"longrag_{len(units)}"
        child_ids = [item["node_id"] for item in child_items]
        child_texts = [item["text"] for item in child_items]
        child_pages = [item.get("page") for item in child_items]
        child_sections = [item.get("section_id") or item.get("section") or section_id for item in child_items]
        child_block_types = [item.get("block_type", "") for item in child_items]
        child_hotpot_facts = []
        for item in child_items:
            title = item.get("hotpot_title") or item.get("title")
            sent_id = item.get("hotpot_sent_id")
            if sent_id is None:
                sent_id = item.get("sent_id")
            child_hotpot_facts.append([title, sent_id] if title is not None and sent_id is not None else None)
        metadata = {
            "source": "longrag",
            "node_id": unit_id,
            "longrag_unit_id": unit_id,
            "block_type": "long_unit",
            "section_id": section_id,
            "section": section_id,
            "title_path": section_id,
            "child_source_node_ids": child_ids,
            "child_qasper_evidence_texts": child_texts,
            "child_pages": child_pages,
            "child_sections": child_sections,
            "child_block_types": child_block_types,
            "child_hotpot_facts": child_hotpot_facts,
        }
        return {
            "id": unit_id,
            "score": 0.0,
            "content": "\n".join(lines),
            "metadata": metadata,
        }

    @staticmethod
    def _is_abstract_node(node: Any) -> bool:
        parent = getattr(node, "parent", None)
        parent_text = str(getattr(getattr(parent, "meta_info", None), "content", "") or "").strip().lower()
        own_text = str(getattr(getattr(node, "meta_info", None), "content", "") or "").strip().lower()
        return parent_text == "abstract" and own_text != "abstract"

    def _create_augmented_prompt(self, query: str, retrieved_docs=None) -> str:
        short_answer = getattr(self.cfg, "answer_style", "default") == "short"
        retrieval_method = getattr(self.cfg, "retrieval_method", "")
        full_document = retrieval_method == "full_document"
        longrag = retrieval_method == "longrag"
        long_context_reader = full_document or longrag
        if short_answer and full_document:
            context_text = (
                "You answer long-context document questions using only the provided full document.\n"
                "Return only a JSON object with keys answer_short, answer_rationale, and supporting_block_ids.\n"
                "answer_short must be concise: use exact spans when possible, answer Yes or No for boolean questions, "
                "and use Not answerable only when the full document is insufficient. Do not include evidence bullets or explanations in answer_short.\n"
                "supporting_block_ids must be a list of 1 to 4 integer source ids from the provided texts that best support answer_short.\n\n"
                "--- Background Information ---\n"
            )
        elif short_answer and longrag:
            context_text = (
                "You answer long-context document questions using only the retrieved long document units.\n"
                "Return only a JSON object with keys answer_short, answer_rationale, and supporting_block_ids.\n"
                "answer_short must be concise: use exact spans when possible, answer Yes or No for boolean questions, "
                "and use Not answerable only when the retrieved long units are insufficient. Do not include evidence bullets or explanations in answer_short.\n"
                "supporting_block_ids must be a list of 1 to 4 integer source ids from the inline [source_id=...] markers that best support answer_short.\n\n"
                "--- Background Information ---\n"
            )
        elif short_answer:
            context_text = (
                "You answer Qasper-style document questions using only the provided retrieved documents.\n"
                "Return only a JSON object with keys answer_short and answer_rationale.\n"
                "answer_short must be concise: use exact spans when possible, answer Yes or No for boolean questions, "
                "and use Not answerable only when evidence is insufficient. Do not include evidence bullets or explanations in answer_short.\n\n"
                "--- Background Information ---\n"
            )
        else:
            # context_text = "Please refer to the following background information to answer the question.\n\n--- Background Information ---\n"
            context_text = "Please refer to the following background information to answer the question. You should base your answer strictly on the provided information and not supplement it with outside knowledge. If the background information is insufficient to answer the question, please state that the provided information is not enough.\n\n--- Background Information ---\n"
        question_text = f"--- User Question ---\n{query}\n\n"
        context_text += question_text
        if retrieved_docs is None:
            context_text += "No relevant documents found.\n"
            return context_text

        context_text += "\n--- Retrieved Documents ---\n"
        for i, doc in enumerate(retrieved_docs):
            source = self._format_source(doc)
            source_id = self._doc_node_id(doc)
            label = f"Text {i+1}"
            if long_context_reader and source_id is not None:
                label += f" [source_id={source_id}]"
            if source:
                context_text += f"{label} ({source}): {doc['content']}\n"
            else:
                context_text += f"{label}: {doc['content']}\n"

        max_context_tokens = self.max_tokens - 400
        if full_document:
            configured_limit = int(getattr(self.cfg, "full_document_max_context_tokens", 30000) or 30000)
            max_context_tokens = max(1000, min(self.max_tokens - 800, configured_limit))
        elif longrag:
            configured_limit = int(getattr(self.cfg, "longrag_max_context_tokens", 30000) or 30000)
            max_context_tokens = max(1000, min(self.max_tokens - 800, configured_limit))
        context_text = TextProcessor.split_text_into_chunks(
            text=context_text, max_length=max_context_tokens
        )
        context_text = context_text[0]  # take the first chunk only
        return context_text

    @staticmethod
    def _format_source(doc: Dict[str, Any]) -> str:
        meta = doc.get("metadata") or {}
        source_parts = []
        page = meta.get("page") or doc.get("page")
        section = meta.get("section_id") or meta.get("section") or doc.get("section_id")
        title_path = meta.get("title_path") or doc.get("title_path")
        if page:
            source_parts.append(f"page={page}")
        if section:
            source_parts.append(f"section={section}")
        elif title_path:
            source_parts.append(f"title_path={title_path}")
        return ", ".join(source_parts)

    def _save_retrieval_res(self, context_nodes, query_output_dir, supporting_ids=None) -> List[Dict]:
        retrieval_ids = []
        ranked_results = []
        for rank, doc in enumerate(context_nodes, start=1):
            meta = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
            if "metadata" in doc:
                if "node_id" in meta:
                    node_id = meta["node_id"]
                elif "chunk_id" in meta:
                    node_id = meta["chunk_id"]
                else:
                    node_id = meta.get("node_id", meta.get("id", None))
            elif "node_id" in doc:
                node_id = doc["node_id"]
            else:
                node_id = doc["id"]
            meta_info_dict = {
                "id": node_id,
                "content": doc["content"],
            }
            if "score" in doc:
                meta_info_dict["score"] = doc["score"]
            meta_info_dict["rank"] = rank
            for key in ["distance", "bm25_score", "rerank_score", "rrf_score", "source", "sources", "source_ranks"]:
                if key in doc:
                    meta_info_dict[key] = doc[key]
            for key in [
                "source",
                "chunk_id",
                "source_chunk_index",
                "source_node_id",
                "node_id",
                "paragraph_id",
                "evidence_id",
                "qasper_evidence_text",
                "pdf_id",
                "page",
                "node_type",
                "raptor_depth",
                "child_source_node_ids",
                "child_qasper_evidence_texts",
                "child_pages",
                "child_sections",
                "child_hotpot_facts",
                "child_block_types",
                "longrag_unit_id",
                "section_id",
                "section",
                "title_path",
                "title",
                "hotpot_title",
                "sent_id",
                "hotpot_sent_id",
                "block_type",
            ]:
                if key in meta:
                    meta_info_dict[key] = meta[key]
            block_type = self._block_type_from_metadata(meta, doc)
            if block_type:
                meta_info_dict["block_type"] = block_type
            child_ids = self._parse_int_values(meta.get("child_source_node_ids"))
            if str(meta.get("source") or doc.get("source") or "").lower() == "longrag" and child_ids:
                for child_id in child_ids:
                    if child_id not in retrieval_ids:
                        retrieval_ids.append(child_id)
            else:
                retrieval_ids.append(node_id)
            ranked_results.append(meta_info_dict)
            node_file_path = query_output_dir / f"{node_id}.json"
            with open(node_file_path, "w", encoding="utf-8") as f:
                json.dump(
                    make_json_safe(meta_info_dict),
                    f,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )

        retrieval_payload = {"ranked_results": ranked_results}
        supporting_evidence = self._supporting_evidence_from_ranked(ranked_results, supporting_ids)
        if supporting_evidence:
            retrieval_payload["supporting_evidence"] = supporting_evidence
            retrieval_payload["supporting_block_ids"] = [item["id"] for item in supporting_evidence]
        with open(query_output_dir / "retrieval_res.json", "w", encoding="utf-8") as f:
            json.dump(
                make_json_safe(retrieval_payload),
                f,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        log.info("Saved retrieval results to output directory.")

        return retrieval_ids

    @staticmethod
    def _block_type_from_metadata(meta: Dict[str, Any], doc: Dict[str, Any]) -> str:
        source = str(meta.get("source") or doc.get("source") or "").lower()
        node_type = str(meta.get("node_type") or "").lower()
        if source == "longrag":
            return "long_unit"
        if source == "raptor_summary" or node_type == "raptor_summary":
            return "summary"
        if source == "qasper_paragraph" or node_type in {"text", "paragraph"}:
            return "paragraph"
        if node_type in {"table", "figure", "caption", "title", "summary", "entity", "patch"}:
            return node_type
        return ""

    def generation(self, query: str, query_output_dir: str) -> tuple:
        """
        Generates an answer for a given query and returns the answer along with the context used.
        Returns:
            Tuple[str, List[Any]]: A tuple containing the final answer string and a list of the context nodes.
        """
        if self.cfg.retrieval_method == "ircot":
            return self._ircot_generation(query, query_output_dir)
        if self.cfg.retrieval_method == "react":
            return self._react_generation(query, query_output_dir)

        retrieved_docs = self._retrieve(query, top_k=self.topk)
        if not retrieved_docs:
            # not found any relevant documents, fallback to LLM generation
            final_answer = self.llm.get_completion(query, json_response=False)
            answer_short, answer_rationale = self._parse_answer_payload(final_answer)
            self.last_answer_short = answer_short
            self.last_answer_rationale = answer_rationale
            self.last_retrieved_block_ids = []
            self.last_supporting_block_ids = []
            return final_answer, []

        context_text = self._create_augmented_prompt(query, retrieved_docs)

        final_answer = self.llm.get_completion(context_text, json_response=False)
        answer_short, answer_rationale = self._parse_answer_payload(final_answer)
        answer_supporting_ids = self._parse_supporting_ids(final_answer)

        retrieval_ids = self._save_retrieval_res(
            retrieved_docs,
            query_output_dir=query_output_dir,
            supporting_ids=answer_supporting_ids,
        )
        self.last_answer_short = answer_short
        self.last_answer_rationale = answer_rationale
        self.last_retrieved_block_ids = retrieval_ids
        self.last_supporting_block_ids = answer_supporting_ids or retrieval_ids
        return final_answer, retrieval_ids

    def _react_generation(self, query: str, query_output_dir: str) -> tuple:
        if self.bm25 is None:
            raise ValueError("ReAct requires a BM25 retriever.")

        from Core.rag.react_env import ReactLocalEnvironment
        from Core.rag.react_runner import ReactRunner

        docs = []
        for index, content in enumerate(self.bm25.original_docs):
            metadata = (
                dict(self.bm25.metadatas[index])
                if index < len(self.bm25.metadatas)
                and isinstance(self.bm25.metadatas[index], dict)
                else {}
            )
            docs.append(
                {
                    "id": index,
                    "content": str(content or ""),
                    "score": 0.0,
                    "metadata": metadata,
                }
            )

        environment = ReactLocalEnvironment(
            docs=docs,
            bm25=self.bm25,
            page_observation_units=int(
                getattr(
                    self.cfg,
                    "react_page_observation_units",
                    5,
                )
                or 5
            ),
            search_topk=int(
                getattr(self.cfg, "react_search_topk", 1)
                or 1
            ),
        )
        run = ReactRunner(
            llm=self.llm,
            environment=environment,
            dataset_name=getattr(
                self.cfg,
                "react_dataset_name",
                "qasper",
            ),
            max_steps=int(
                getattr(self.cfg, "react_max_steps", 7)
                or 7
            ),
            prompt_file=str(
                getattr(self.cfg, "react_prompt_file", "")
                or ""
            ),
        ).run(query)

        observed_items = []
        block_ids = []
        for evidence in run.observed_evidence:
            metadata = dict(evidence.metadata)
            metadata["node_id"] = evidence.block_id
            metadata.setdefault("source_node_id", evidence.block_id)
            observed_items.append(
                {
                    "id": evidence.block_id,
                    "content": str(
                        self.bm25.original_docs[evidence.corpus_index]
                    ),
                    "score": 0.0,
                    "metadata": metadata,
                }
            )
            block_ids.append(evidence.block_id)

        query_output_dir = Path(query_output_dir)
        self._save_retrieval_res(
            observed_items,
            query_output_dir=query_output_dir,
            supporting_ids=block_ids,
        )
        retrieval_path = query_output_dir / "retrieval_res.json"
        with open(retrieval_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        step_payloads = [asdict(step) for step in run.steps]
        payload.update(
            {
                "strategy": "react",
                "selected": payload.get("ranked_results", []),
                "supporting_evidence": payload.get(
                    "supporting_evidence",
                    [],
                ),
                "retrieved_block_ids": block_ids,
                "supporting_block_ids": block_ids,
                "react_steps": step_payloads,
                "react_num_calls": run.num_calls,
                "react_num_bad_calls": run.num_bad_calls,
                "react_search_count": run.search_count,
                "react_lookup_count": run.lookup_count,
                "react_termination_reason": run.termination_reason,
            }
        )
        with open(retrieval_path, "w", encoding="utf-8") as file:
            json.dump(
                make_json_safe(payload),
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
        with open(
            query_output_dir / "evidence_chain.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                make_json_safe(
                    {
                        "strategy": "react",
                        "steps": step_payloads,
                        "retrieved_block_ids": block_ids,
                        "supporting_block_ids": block_ids,
                        "termination_reason": run.termination_reason,
                    }
                ),
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        self.last_answer_short = run.answer
        self.last_answer_rationale = ""
        self.last_retrieved_block_ids = block_ids
        self.last_supporting_block_ids = block_ids
        answer = json.dumps(
            {
                "answer_short": run.answer,
                "answer_rationale": "",
                "supporting_block_ids": block_ids,
            },
            ensure_ascii=False,
        )
        return answer, block_ids

    def _ircot_generation(self, query: str, query_output_dir: str) -> tuple:
        if self.bm25 is None:
            raise ValueError("IRCoT requires a BM25 retriever.")
        max_steps = max(1, int(getattr(self.cfg, "ircot_max_steps", 3) or 3))
        step_topk = max(1, int(getattr(self.cfg, "ircot_step_topk", 3) or 3))
        final_topk = max(1, int(getattr(self.cfg, "ircot_final_topk", self.topk) or self.topk))

        thoughts: List[str] = []
        steps: List[Dict[str, Any]] = []
        evidence_by_key: Dict[str, Dict[str, Any]] = {}
        ranked_docs: List[Dict[str, Any]] = []

        for step_idx in range(1, max_steps + 1):
            retrieval_query = self._ircot_retrieval_query(query, thoughts)
            docs = self.bm25.search(query_text=retrieval_query, top_k=step_topk)
            step_docs = []
            for doc in docs:
                key = self._document_key(doc)
                if key not in evidence_by_key:
                    evidence_by_key[key] = doc
                    ranked_docs.append(doc)
                step_docs.append(self._ircot_doc_trace(doc))

            thought_payload = self._generate_ircot_thought(
                query=query,
                thoughts=thoughts,
                docs=docs,
                step_idx=step_idx,
            )
            thought = str(thought_payload.get("thought") or "").strip()
            stop = bool(thought_payload.get("stop"))
            if thought:
                thoughts.append(thought)
            steps.append(
                {
                    "step": step_idx,
                    "retrieval_query": retrieval_query,
                    "thought": thought,
                    "stop": stop,
                    "retrieved": step_docs,
                }
            )
            if stop:
                break

        final_docs = ranked_docs[:final_topk]
        if not final_docs:
            final_answer = self.llm.get_completion(query, json_response=False)
            answer_short, answer_rationale = self._parse_answer_payload(final_answer)
            self.last_answer_short = answer_short
            self.last_answer_rationale = answer_rationale
            self.last_retrieved_block_ids = []
            self.last_supporting_block_ids = []
            return final_answer, []

        context_text = self._create_ircot_answer_prompt(query, thoughts, final_docs)
        final_answer = self.llm.get_completion(context_text, json_response=False)
        answer_short, answer_rationale = self._parse_answer_payload(final_answer)
        answer_supporting_ids = self._parse_supporting_ids(final_answer)
        retrieval_ids = self._save_ircot_retrieval_res(
            final_docs,
            query_output_dir=query_output_dir,
            steps=steps,
            thoughts=thoughts,
            supporting_ids=answer_supporting_ids,
        )
        self.last_answer_short = answer_short
        self.last_answer_rationale = answer_rationale
        self.last_retrieved_block_ids = retrieval_ids
        self.last_supporting_block_ids = answer_supporting_ids or retrieval_ids
        return final_answer, retrieval_ids

    @staticmethod
    def _ircot_retrieval_query(query: str, thoughts: List[str]) -> str:
        if not thoughts:
            return query
        return query + "\n" + "\n".join(f"Thought {idx + 1}: {thought}" for idx, thought in enumerate(thoughts))

    def _generate_ircot_thought(
        self,
        query: str,
        thoughts: List[str],
        docs: List[Dict[str, Any]],
        step_idx: int,
    ) -> Dict[str, Any]:
        evidence_text = "\n".join(f"[{idx + 1}] {doc.get('content', '')}" for idx, doc in enumerate(docs))
        previous = "\n".join(f"Thought {idx + 1}: {thought}" for idx, thought in enumerate(thoughts)) or "None"
        prompt = (
            "Generate the next reasoning step for iterative retrieval.\n"
            "Return only JSON with keys thought and stop.\n"
            "thought should be a concise intermediate reasoning sentence or retrieval clue. "
            "stop should be true only when the evidence seems sufficient to answer.\n\n"
            f"Question: {query}\n"
            f"Previous thoughts:\n{previous}\n\n"
            f"Newly retrieved evidence at step {step_idx}:\n{evidence_text}\n"
        )
        raw = self.llm.get_completion(prompt, json_response=False)
        try:
            payload = json.loads(str(raw).strip().strip("`"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        text = str(raw or "").strip()
        return {"thought": text, "stop": False}

    def _create_ircot_answer_prompt(self, query: str, thoughts: List[str], docs: List[Dict[str, Any]]) -> str:
        prompt = (
            "You answer questions using evidence gathered by interleaved retrieval and reasoning.\n"
            "Return only a JSON object with keys answer_short, answer_rationale, and supporting_block_ids.\n"
            "answer_short must be concise: use exact spans when possible, answer Yes or No for boolean questions, "
            "and use Not answerable only when evidence is insufficient.\n"
            "supporting_block_ids must be a list of 1 to 4 integer source ids from the retrieved documents.\n\n"
            f"Question: {query}\n\n"
            "--- Reasoning Trace ---\n"
        )
        if thoughts:
            prompt += "\n".join(f"Thought {idx + 1}: {thought}" for idx, thought in enumerate(thoughts))
        else:
            prompt += "None"
        prompt += "\n\n--- Retrieved Documents ---\n"
        for idx, doc in enumerate(docs, start=1):
            source_id = self._doc_node_id(doc)
            source = self._format_source(doc)
            label = f"Text {idx}"
            if source_id is not None:
                label += f" [source_id={source_id}]"
            if source:
                prompt += f"{label} ({source}): {doc.get('content', '')}\n"
            else:
                prompt += f"{label}: {doc.get('content', '')}\n"
        chunks = TextProcessor.split_text_into_chunks(text=prompt, max_length=self.max_tokens - 400)
        return chunks[0]

    def _save_ircot_retrieval_res(
        self,
        docs: List[Dict[str, Any]],
        query_output_dir: str,
        steps: List[Dict[str, Any]],
        thoughts: List[str],
        supporting_ids=None,
    ) -> List[Any]:
        retrieval_ids = self._save_retrieval_res(
            docs,
            query_output_dir=query_output_dir,
            supporting_ids=supporting_ids,
        )
        payload_path = query_output_dir / "retrieval_res.json"
        with open(payload_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["strategy"] = "ircot"
        payload["ircot_steps"] = steps
        payload["ircot_thoughts"] = thoughts
        with open(payload_path, "w", encoding="utf-8") as f:
            json.dump(make_json_safe(payload), f, indent=2, ensure_ascii=False, allow_nan=False)
        with open(query_output_dir / "evidence_chain.json", "w", encoding="utf-8") as f:
            json.dump(
                make_json_safe(
                    {
                        "strategy": "ircot",
                        "thoughts": thoughts,
                        "steps": steps,
                        "retrieved_block_ids": retrieval_ids,
                        "supporting_block_ids": payload.get("supporting_block_ids", []),
                    }
                ),
                f,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
        return retrieval_ids

    def _ircot_doc_trace(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        meta = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
        return {
            "id": self._doc_node_id(doc),
            "score": float(doc.get("score", 0.0)),
            "content": str(doc.get("content", "")),
            "source": meta.get("source") or doc.get("source") or "bm25",
            "qasper_evidence_text": meta.get("qasper_evidence_text"),
            "hotpot_title": meta.get("hotpot_title") or meta.get("title"),
            "sent_id": meta.get("hotpot_sent_id") if meta.get("hotpot_sent_id") is not None else meta.get("sent_id"),
        }

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
                parts = text.split(marker, 1)
                if len(parts) == 2 and parts[1].strip():
                    return parts[1].strip(), ""
        return text, ""

    @staticmethod
    def _parse_supporting_ids(answer: Any) -> List[int]:
        text = str(answer or "").strip()
        cleaned = text
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            payload = json.loads(cleaned)
        except Exception:
            return []
        if not isinstance(payload, dict):
            return []
        raw_ids = payload.get("supporting_block_ids") or payload.get("supporting_node_ids") or []
        if not isinstance(raw_ids, list):
            raw_ids = [raw_ids]
        ids: List[int] = []
        for value in raw_ids:
            try:
                node_id = int(value)
            except (TypeError, ValueError):
                continue
            if node_id not in ids:
                ids.append(node_id)
        return ids[:4]

    @staticmethod
    def _supporting_evidence_from_ranked(ranked_results: List[Dict[str, Any]], supporting_ids=None) -> List[Dict[str, Any]]:
        if not supporting_ids:
            return []
        by_id = {str(item.get("id")): item for item in ranked_results if isinstance(item, dict)}
        supporting = []
        for rank, node_id in enumerate(supporting_ids, start=1):
            item = by_id.get(str(node_id))
            payload = dict(item) if item else {}
            if not payload:
                payload = VanillaRAG._child_supporting_evidence_from_ranked(
                    ranked_results,
                    node_id,
                )
            if not payload:
                continue
            payload["supporting_rank"] = rank
            supporting.append(payload)
        return supporting

    @staticmethod
    def _child_supporting_evidence_from_ranked(
        ranked_results: List[Dict[str, Any]],
        node_id: Any,
    ) -> Dict[str, Any]:
        try:
            target = int(node_id)
        except (TypeError, ValueError):
            return {}
        for item in ranked_results:
            if not isinstance(item, dict):
                continue
            child_ids = VanillaRAG._parse_int_values(item.get("child_source_node_ids"))
            if target not in child_ids:
                continue
            child_idx = child_ids.index(target)
            child_texts = VanillaRAG._list_value(item.get("child_qasper_evidence_texts"))
            child_pages = VanillaRAG._list_value(item.get("child_pages"))
            child_sections = VanillaRAG._list_value(item.get("child_sections"))
            child_block_types = VanillaRAG._list_value(item.get("child_block_types"))
            child_hotpot_facts = VanillaRAG._list_value(item.get("child_hotpot_facts"))
            text = VanillaRAG._value_at(child_texts, child_idx, "")
            payload: Dict[str, Any] = {
                "id": target,
                "content": text,
                "qasper_evidence_text": text,
                "source_node_id": target,
                "node_id": target,
                "paragraph_id": target,
                "evidence_id": target,
                "source": "longrag_support",
                "parent_longrag_unit_id": item.get("longrag_unit_id") or item.get("id"),
                "rank": item.get("rank"),
                "score": item.get("score"),
                "block_type": VanillaRAG._value_at(child_block_types, child_idx, "paragraph") or "paragraph",
                "page": VanillaRAG._value_at(child_pages, child_idx, None),
                "section_id": VanillaRAG._value_at(child_sections, child_idx, item.get("section_id")),
                "section": VanillaRAG._value_at(child_sections, child_idx, item.get("section")),
            }
            fact = VanillaRAG._value_at(child_hotpot_facts, child_idx, None)
            if isinstance(fact, list) and len(fact) >= 2 and fact[0] is not None and fact[1] is not None:
                payload["title"] = str(fact[0])
                payload["hotpot_title"] = str(fact[0])
                payload["sent_id"] = int(fact[1])
                payload["hotpot_sent_id"] = int(fact[1])
            return payload
        return {}

    @staticmethod
    def _doc_node_id(doc: Dict[str, Any]) -> Any:
        meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        for key in ("node_id", "source_node_id", "paragraph_id", "evidence_id"):
            if meta.get(key) is not None:
                return meta.get(key)
        return doc.get("id")

    @staticmethod
    def _node_type_value(node: Any) -> str:
        node_type = getattr(node, "type", "")
        return str(getattr(node_type, "value", node_type) or "").lower()

    @staticmethod
    def _node_text(node: Any) -> str:
        text = str(getattr(getattr(node, "meta_info", None), "content", "") or "").strip()
        if text:
            return text
        table_body = str(getattr(getattr(node, "meta_info", None), "table_body", "") or "").strip()
        caption = str(getattr(getattr(node, "meta_info", None), "caption", "") or "").strip()
        return "\n".join(part for part in [caption, table_body] if part).strip()

    @staticmethod
    def _approx_token_count(text: str) -> int:
        return max(1, len(BM25([])._tokenize(str(text or ""))))

    @staticmethod
    def _parse_int_values(value: Any) -> List[int]:
        values = VanillaRAG._list_value(value)
        parsed: List[int] = []
        for item in values:
            try:
                parsed.append(int(item))
            except (TypeError, ValueError):
                continue
        return parsed

    @staticmethod
    def _list_value(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        try:
            parsed = json.loads(str(value))
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [item.strip() for item in str(value).split(",") if item.strip()]

    @staticmethod
    def _value_at(values: List[Any], index: int, default: Any = None) -> Any:
        return values[index] if 0 <= index < len(values) else default

    @classmethod
    def _section_metadata(cls, node: Any) -> Dict[str, Any]:
        titles = []
        parent = getattr(node, "parent", None)
        while parent is not None:
            if cls._node_type_value(parent) == "title":
                title = cls._node_text(parent)
                if title:
                    titles.append(title)
            parent = getattr(parent, "parent", None)
        titles.reverse()
        section = titles[-1] if titles else ""
        return {
            "section_id": section,
            "section": section,
            "title_path": " > ".join(titles),
        }

    @classmethod
    def _hotpot_sentence_metadata(cls, node: Any) -> Dict[str, Any]:
        if cls._node_type_value(node) != "text":
            return {}
        parent = getattr(node, "parent", None)
        if parent is None or cls._node_type_value(parent) != "title":
            return {}
        title = cls._node_text(parent)
        if not title:
            return {}
        sent_id = 0
        for child in getattr(parent, "children", []) or []:
            if cls._node_type_value(child) != "text":
                continue
            if child is node:
                break
            if cls._node_text(child):
                sent_id += 1
        return {
            "title": title,
            "hotpot_title": title,
            "sent_id": sent_id,
            "hotpot_sent_id": sent_id,
        }

    def close(self):
        if self.cfg.retrieval_method in {"bm25", "bm25_rerank"} and self.bm25 is not None:
            log.info("Closing BM25 resources...")
            if hasattr(self.bm25, "close"):
                self.bm25.close()
        if self.vdb is not None:
            if hasattr(self.vdb, "close"):
                self.vdb.close()
            elif hasattr(self.vdb, "embedding_model"):
                self.vdb.embedding_model.close()
        if self.reranker is not None and hasattr(self.reranker, "close"):
            self.reranker.close()
