from typing import TYPE_CHECKING

from Core.rag.base_rag import BaseRAG
from Core.configs.rag.vanilla_config import VanillaConfig
from Core.utils.bm25 import BM25
from Core.utils.utils import TextProcessor
from Core.utils.json_safety import make_json_safe

from typing import Dict, Any, List, Tuple
import json
import logging

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
        self.bm25 = bm25
        self.vdb = vector_store
        self.reranker = reranker
        self.tree_index = tree_index

    def _retrieve(self, query: str, top_k: int = 3):
        if self.cfg.retrieval_method == "bm25":
            return self.bm25.search(query_text=query, top_k=top_k)
        if self.cfg.retrieval_method == "hybrid":
            return self._hybrid_retrieve(query, top_k=top_k)
        if self.cfg.retrieval_method == "bm25_rerank":
            return self._bm25_rerank(query, top_k=top_k)
        if self.cfg.retrieval_method == "abstract_only":
            return self._abstract_only_context()
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

    @staticmethod
    def _is_abstract_node(node: Any) -> bool:
        parent = getattr(node, "parent", None)
        parent_text = str(getattr(getattr(parent, "meta_info", None), "content", "") or "").strip().lower()
        own_text = str(getattr(getattr(node, "meta_info", None), "content", "") or "").strip().lower()
        return parent_text == "abstract" and own_text != "abstract"

    def _create_augmented_prompt(self, query: str, retrieved_docs=None) -> str:
        short_answer = getattr(self.cfg, "answer_style", "default") == "short"
        if short_answer:
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
            if source:
                context_text += f"Text {i+1} ({source}): {doc['content']}\n"
            else:
                context_text += f"Text {i+1}: {doc['content']}\n"

        context_text = TextProcessor.split_text_into_chunks(
            text=context_text, max_length=self.max_tokens-400
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

    def _save_retrieval_res(self, context_nodes, query_output_dir) -> List[Dict]:
        retrieval_ids = []
        ranked_results = []
        for doc in context_nodes:
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
                "child_pages",
                "child_sections",
                "section_id",
                "section",
                "title_path",
            ]:
                if key in meta:
                    meta_info_dict[key] = meta[key]
            block_type = self._block_type_from_metadata(meta, doc)
            if block_type:
                meta_info_dict["block_type"] = block_type
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

        with open(query_output_dir / "retrieval_res.json", "w", encoding="utf-8") as f:
            json.dump(
                make_json_safe({"ranked_results": ranked_results}),
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
        retrieved_docs = self._retrieve(query, top_k=self.topk)
        if not retrieved_docs:
            # not found any relevant documents, fallback to LLM generation
            final_answer = self.llm.get_completion(query, json_response=False)
            answer_short, answer_rationale = self._parse_answer_payload(final_answer)
            self.last_answer_short = answer_short
            self.last_answer_rationale = answer_rationale
            self.last_supporting_block_ids = []
            return final_answer, []

        context_text = self._create_augmented_prompt(query, retrieved_docs)

        final_answer = self.llm.get_completion(context_text, json_response=False)
        answer_short, answer_rationale = self._parse_answer_payload(final_answer)

        retrieval_ids = self._save_retrieval_res(
            retrieved_docs, query_output_dir=query_output_dir
        )
        self.last_answer_short = answer_short
        self.last_answer_rationale = answer_rationale
        self.last_supporting_block_ids = retrieval_ids
        return final_answer, retrieval_ids

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

    def close(self):
        if self.cfg.retrieval_method in {"bm25", "bm25_rerank"} and self.bm25 is not None:
            log.info("Closing BM25 resources...")
            if hasattr(self.bm25, "close"):
                self.bm25.close()
        if self.vdb is not None and hasattr(self.vdb, "embedding_model"):
            self.vdb.embedding_model.close()
        if self.reranker is not None and hasattr(self.reranker, "close"):
            self.reranker.close()
