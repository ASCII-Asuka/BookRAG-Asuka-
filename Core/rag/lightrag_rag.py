from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from Core.configs.rag.lightrag_config import LightRAGConfig
from Core.rag.base_rag import BaseRAG
from Core.utils.json_safety import make_json_safe

if TYPE_CHECKING:
    from Core.provider.llm import LLM

log = logging.getLogger(__name__)
merge_nodes_and_edges = None


class LightRAGRAG(BaseRAG):
    def __init__(
        self,
        config: LightRAGConfig,
        llm: "LLM",
        tree_index: Any,
        save_path: str,
    ):
        super().__init__(
            llm=llm,
            name="LightRAG",
            description="Graph-enhanced LightRAG baseline",
        )
        self.cfg = config
        self.tree_index = tree_index
        self.save_path = str(save_path)
        self.working_dir = os.path.join(self.save_path, self.cfg.working_dir_name)
        self.manifest_path = os.path.join(self.working_dir, "bookrag_lightrag_manifest.json")
        self._rag = None
        self._docs: List[Dict[str, Any]] = []
        self.last_answer_short = ""
        self.last_answer_rationale = ""
        self.last_supporting_block_ids: List[int] = []
        self.last_retrieved_block_ids: List[int] = []

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
            node_id = int(getattr(node, "index_id"))
            docs.append(
                {
                    "id": f"node_{node_id}",
                    "content": text,
                    "metadata": {
                        "source": "lightrag",
                        "node_id": node_id,
                        "source_node_id": node_id,
                        "paragraph_id": node_id,
                        "evidence_id": node_id,
                        "qasper_evidence_text": text,
                        "node_type": "paragraph",
                    },
                }
            )
        return docs

    def _retrieve(self, query: str, **kwargs):
        self._ensure_ready()
        data, _ = self._query_data_with_fallback(query)
        return self._ranked_results_from_query_data(data)

    def _create_augmented_prompt(self, query: str) -> str:
        if self.cfg.answer_style == "short":
            return (
                "Answer the question using the LightRAG indexed document. "
                "Return a concise answer only. Use exact spans when possible, "
                "answer Yes or No for boolean questions, and use Unanswerable only if needed.\n\n"
                f"Question: {query}"
            )
        return query

    def generation(self, query: str, query_output_dir: str) -> Tuple[str, List[Any]]:
        self._ensure_ready()
        query_output_path = Path(query_output_dir)
        query_output_path.mkdir(parents=True, exist_ok=True)

        retrieval_data, effective_mode = self._query_data_with_fallback(query)

        ranked_results = self._ranked_results_from_query_data(retrieval_data)
        answer_prompt = self._create_augmented_prompt(query)
        try:
            answer = self._rag.query(answer_prompt, param=self._query_param(mode=effective_mode))
        except Exception as exc:
            log.warning("LightRAG query failed: %s", exc)
            answer = "Unanswerable"

        answer = str(answer or "").strip()
        self.last_answer_short = self._shorten_answer(answer)
        self.last_answer_rationale = "" if self.last_answer_short == answer else answer
        retrieved_ids = [item["metadata"]["source_node_id"] for item in ranked_results]
        supporting_evidence = self._supporting_evidence_from_ranked(
            ranked_results,
            answer_text=self.last_answer_short,
            query=query,
        )
        self.last_retrieved_block_ids = retrieved_ids
        self.last_supporting_block_ids = [item["source_node_id"] for item in supporting_evidence]

        retrieval_payload = {
            "query": query,
            "mode": self.cfg.mode,
            "effective_mode": effective_mode,
            "fallback_mode": self.cfg.fallback_mode,
            "fallback_used": effective_mode != self.cfg.mode,
            "topk": self.cfg.topk,
            "chunk_topk": self.cfg.chunk_topk,
            "supporting_evidence_topk": self.cfg.supporting_evidence_topk,
            "graph_diagnostics": self._storage_diagnostics(self.working_dir),
            "supporting_evidence": supporting_evidence,
            "ranked_results": ranked_results,
            "raw_query_data": retrieval_data,
        }
        with (query_output_path / "retrieval_res.json").open("w", encoding="utf-8") as f:
            json.dump(make_json_safe(retrieval_payload), f, indent=2, ensure_ascii=False, allow_nan=False)

        return answer, retrieved_ids

    def close(self):
        if self._rag is not None and hasattr(self._rag, "finalize_storages"):
            try:
                self._run_on_lightrag_loop(
                    lambda: self._rag.finalize_storages(),
                    sync_name="finalize_storages",
                    async_name="finalize_storages",
                )
            except Exception as exc:
                log.debug("LightRAG finalize_storages failed: %s", exc)

    def _ensure_ready(self) -> None:
        if self._rag is not None:
            return
        self._docs = self.documents_from_tree(self.tree_index)
        self._rag = self._build_lightrag()
        self._initialize_storages()
        self._ensure_indexed()

    def _build_lightrag(self):
        try:
            from lightrag import LightRAG
            from lightrag.llm.openai import openai_complete_if_cache, openai_embed
            from lightrag.utils import EmbeddingFunc
        except ImportError as exc:
            raise ImportError(
                "LightRAG baseline requires the optional dependency 'lightrag-hku'. "
                "Install it with: pip install lightrag-hku"
            ) from exc

        embed_cfg = self.cfg.embedding_config

        async def llm_model_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: List[Dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> str:
            return await openai_complete_if_cache(
                model=self.llm.config.model_name,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                base_url=self.llm.config.api_base,
                api_key=self.llm.config.api_key,
                **kwargs,
            )

        async def embedding_func(texts: List[str]):
            return await openai_embed.func(
                texts,
                model=embed_cfg.model_name,
                base_url=embed_cfg.api_base,
                api_key=embed_cfg.api_key,
                embedding_dim=self.cfg.embedding_dim,
                max_token_size=embed_cfg.max_length,
            )

        embedding = EmbeddingFunc(
            embedding_dim=self.cfg.embedding_dim,
            max_token_size=embed_cfg.max_length,
            model_name=embed_cfg.model_name,
            func=embedding_func,
        )
        return LightRAG(
            working_dir=self.working_dir,
            llm_model_func=llm_model_func,
            llm_model_name=self.llm.config.model_name,
            embedding_func=embedding,
            chunk_token_size=self.cfg.chunk_token_size,
            chunk_overlap_token_size=self.cfg.chunk_overlap_token_size,
            top_k=self.cfg.topk,
            chunk_top_k=self.cfg.chunk_topk,
            max_total_tokens=self.cfg.max_total_tokens,
        )

    def _initialize_storages(self) -> None:
        if hasattr(self._rag, "initialize_storages"):
            self._run_on_lightrag_loop(
                lambda: self._rag.initialize_storages(),
                sync_name="initialize_storages",
                async_name="initialize_storages",
            )
        try:
            from lightrag.kg.shared_storage import initialize_pipeline_status

            self._run_on_lightrag_loop(
                lambda: initialize_pipeline_status(),
                sync_name="initialize_pipeline_status",
                async_name="initialize_pipeline_status",
            )
        except Exception as exc:
            log.debug("LightRAG initialize_pipeline_status skipped: %s", exc)

    def _run_on_lightrag_loop(self, coro_factory, *, sync_name: str, async_name: str) -> Any:
        try:
            from lightrag.lightrag import _run_sync as lightrag_run_sync

            return lightrag_run_sync(
                coro_factory,
                sync_name=sync_name,
                async_name=async_name,
                owning_loop=getattr(self._rag, "_owning_loop", None),
            )
        except ImportError:
            return self._run_sync(coro_factory())

    @staticmethod
    def _run_sync(value: Any) -> Any:
        if not inspect.isawaitable(value):
            return value
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(value)

        import queue
        import threading

        result_queue: "queue.Queue[Tuple[bool, Any]]" = queue.Queue(maxsize=1)

        def runner() -> None:
            try:
                result_queue.put((True, asyncio.run(value)))
            except Exception as exc:
                result_queue.put((False, exc))

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()
        ok, result = result_queue.get()
        if ok:
            return result
        raise result

    def _ensure_indexed(self) -> None:
        if self.cfg.force_rebuild and os.path.isdir(self.working_dir):
            self._reset_lightrag_storage()
        if os.path.exists(self.manifest_path):
            diagnostics = self._storage_diagnostics(self.working_dir)
            if not self._should_repair_empty_graph(diagnostics):
                return
            log.warning(
                "LightRAG workdir has chunks but empty graph storages; rebuilding graph-enabled index at %s",
                self.working_dir,
            )
            self._reset_lightrag_storage()
        os.makedirs(self.working_dir, exist_ok=True)
        chunks = [doc["content"] for doc in self._docs]
        full_text = "\n\n".join(chunks)
        if chunks:
            if self.cfg.enable_graph_merge and hasattr(self._rag, "_process_extract_entities"):
                self._run_on_lightrag_loop(
                    lambda: self._ainsert_custom_chunks_with_graph_merge(
                        full_text=full_text,
                        text_chunks=chunks,
                        doc_id="document_tree",
                        force_insert=True,
                    ),
                    sync_name="insert_custom_chunks_graph_merge",
                    async_name="ainsert_custom_chunks_graph_merge",
                )
            elif hasattr(self._rag, "insert_custom_chunks"):
                self._rag.insert_custom_chunks(full_text=full_text, text_chunks=chunks, doc_id="document_tree")
            else:
                self._rag.insert(chunks, ids=[doc["id"] for doc in self._docs])
        manifest = {
            "num_chunks": len(chunks),
            "source": "DocumentTree",
            "strategy": "lightrag",
            "mode": self.cfg.mode,
            "enable_graph_merge": self.cfg.enable_graph_merge,
            "graph_diagnostics": self._storage_diagnostics(self.working_dir),
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    def _reset_lightrag_storage(self) -> None:
        if os.path.isdir(self.working_dir):
            shutil.rmtree(self.working_dir)
        self._rag = self._build_lightrag()
        self._initialize_storages()

    def _should_repair_empty_graph(self, diagnostics: Dict[str, Any]) -> bool:
        if not (self.cfg.enable_graph_merge and self.cfg.repair_empty_graph):
            return False
        if self.cfg.mode not in {"local", "global", "hybrid", "mix"}:
            return False
        return not diagnostics.get("graph_ready", False)

    async def _ainsert_custom_chunks_with_graph_merge(
        self,
        full_text: str,
        text_chunks: List[str],
        doc_id: str | None = None,
        force_insert: bool = False,
    ) -> None:
        global merge_nodes_and_edges
        if merge_nodes_and_edges is None:
            from lightrag.operate import merge_nodes_and_edges as _merge_nodes_and_edges

            merge_nodes_and_edges = _merge_nodes_and_edges
        from lightrag.lightrag import normalize_document_file_path
        from lightrag.utils import compute_mdhash_id, sanitize_text_for_encoding

        full_text = sanitize_text_for_encoding(full_text)
        text_chunks = [sanitize_text_for_encoding(chunk) for chunk in text_chunks]
        file_path = normalize_document_file_path("")
        doc_key = doc_id or compute_mdhash_id(full_text, prefix="doc-")
        new_docs = {doc_key: {"content": full_text, "file_path": file_path}}

        if not force_insert and hasattr(self._rag.full_docs, "filter_keys"):
            add_doc_keys = await self._rag.full_docs.filter_keys({doc_key})
            new_docs = {key: value for key, value in new_docs.items() if key in add_doc_keys}
            if not new_docs:
                log.warning("LightRAG document already exists in storage; skip custom graph insert.")
                return

        inserting_chunks: Dict[str, Any] = {}
        for index, chunk_text in enumerate(text_chunks):
            chunk_key = compute_mdhash_id(chunk_text, prefix="chunk-")
            tokenizer = getattr(self._rag, "tokenizer", None)
            if tokenizer is not None and hasattr(tokenizer, "encode"):
                tokens = len(tokenizer.encode(chunk_text))
            else:
                tokens = len(chunk_text.split())
            inserting_chunks[chunk_key] = {
                "content": chunk_text,
                "full_doc_id": doc_key,
                "tokens": tokens,
                "chunk_order_index": index,
                "file_path": file_path,
            }

        if not force_insert and hasattr(self._rag.text_chunks, "filter_keys"):
            add_chunk_keys = await self._rag.text_chunks.filter_keys(set(inserting_chunks.keys()))
            inserting_chunks = {
                key: value for key, value in inserting_chunks.items() if key in add_chunk_keys
            }
            if not inserting_chunks:
                log.warning("LightRAG chunks already exist in storage; skip custom graph insert.")
                return

        await self._rag.chunks_vdb.upsert(inserting_chunks)
        await self._rag.full_docs.upsert(new_docs)
        await self._rag.text_chunks.upsert(inserting_chunks)
        chunk_results = await self._rag._process_extract_entities(inserting_chunks)
        pipeline_status = {
            "latest_message": "",
            "history_messages": [],
            "cancellation_requested": False,
        }
        pipeline_status_lock = asyncio.Lock()
        await merge_nodes_and_edges(
            chunk_results=chunk_results,
            knowledge_graph_inst=self._rag.chunk_entity_relation_graph,
            entity_vdb=self._rag.entities_vdb,
            relationships_vdb=self._rag.relationships_vdb,
            global_config=self._rag._build_global_config(),
            full_entities_storage=getattr(self._rag, "full_entities", None),
            full_relations_storage=getattr(self._rag, "full_relations", None),
            doc_id=doc_key,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            llm_response_cache=getattr(self._rag, "llm_response_cache", None),
            entity_chunks_storage=getattr(self._rag, "entity_chunks", None),
            relation_chunks_storage=getattr(self._rag, "relation_chunks", None),
            current_file_number=1,
            total_files=1,
            file_path=file_path,
        )
        if hasattr(self._rag, "_insert_done_with_cleanup"):
            await self._rag._insert_done_with_cleanup()

    def _query_data_with_fallback(self, query: str) -> Tuple[Dict[str, Any], str]:
        primary_mode = self.cfg.mode
        payload = self._safe_query_data(query, primary_mode)
        if self._extract_chunks(payload):
            return payload, primary_mode
        fallback_mode = getattr(self.cfg, "fallback_mode", "mix") or "none"
        if fallback_mode == "none" or fallback_mode == primary_mode:
            return payload, primary_mode
        fallback_payload = self._safe_query_data(query, fallback_mode)
        if self._extract_chunks(fallback_payload):
            return fallback_payload, fallback_mode
        return payload, primary_mode

    def _safe_query_data(self, query: str, mode: str) -> Dict[str, Any]:
        try:
            return self._rag.query_data(query, param=self._query_param(mode=mode))
        except Exception as exc:
            log.warning("LightRAG query_data failed for mode=%s: %s", mode, exc)
            return {"status": "failure", "message": str(exc), "data": {}}

    def _query_param(self, mode: str | None = None):
        from lightrag import QueryParam

        return QueryParam(
            mode=mode or self.cfg.mode,
            top_k=self.cfg.topk,
            chunk_top_k=self.cfg.chunk_topk,
            max_total_tokens=self.cfg.max_total_tokens,
            response_type="Single concise answer" if self.cfg.answer_style == "short" else "Multiple Paragraphs",
            include_references=self.cfg.include_references,
            enable_rerank=False,
        )

    def _supporting_evidence_from_ranked(
        self,
        ranked_results: List[Dict[str, Any]],
        answer_text: str = "",
        query: str = "",
    ) -> List[Dict[str, Any]]:
        limit = max(0, int(getattr(self.cfg, "supporting_evidence_topk", 3) or 0))
        answer_terms = self._overlap_terms(answer_text)
        query_terms = self._overlap_terms(query)
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        seen_ids = set()
        for rank, item in enumerate(ranked_results):
            metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
            source_id = metadata.get("source_node_id") or metadata.get("node_id") or item.get("source_node_id")
            if source_id is None:
                continue
            try:
                source_id = int(source_id)
            except (TypeError, ValueError):
                continue
            if source_id in seen_ids:
                continue
            seen_ids.add(source_id)
            text = (
                metadata.get("qasper_evidence_text")
                or item.get("qasper_evidence_text")
                or item.get("content")
                or ""
            )
            if not str(text).strip():
                continue
            candidate_terms = self._overlap_terms(str(text))
            answer_overlap = len(answer_terms & candidate_terms) / max(len(answer_terms), 1) if answer_terms else 0.0
            query_overlap = len(query_terms & candidate_terms) / max(len(query_terms), 1) if query_terms else 0.0
            # Preserve LightRAG ranking when overlap evidence is unavailable; otherwise let answer-aligned
            # paragraphs survive the official evidence truncation.
            score = (2.0 * answer_overlap) + (0.4 * query_overlap) - (0.001 * rank)
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
                        "qasper_evidence_text": str(text),
                        "content": str(text),
                        "source": "lightrag",
                        "supporting_score": score,
                        "answer_overlap": answer_overlap,
                        "query_overlap": query_overlap,
                        "retrieval_rank": rank + 1,
                    },
                )
            )
        if answer_terms or query_terms:
            candidates.sort(key=lambda pair: pair[0], reverse=True)
        supporting = [item for _, item in candidates]
        if limit:
            supporting = supporting[:limit]
        for idx, item in enumerate(supporting, start=1):
            item["supporting_rank"] = idx
        return supporting

    @staticmethod
    def _overlap_terms(text: str) -> set[str]:
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "be",
            "by",
            "do",
            "does",
            "for",
            "from",
            "how",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "the",
            "they",
            "to",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "with",
        }
        return {
            token
            for token in re.findall(r"[a-zA-Z0-9]+", str(text).lower())
            if len(token) > 2 and token not in stopwords
        }

    def _ranked_results_from_query_data(self, payload: Any) -> List[Dict[str, Any]]:
        chunks = self._extract_chunks(payload)
        ranked: List[Dict[str, Any]] = []
        used_ids = set()
        for chunk in chunks:
            content = self._chunk_content(chunk)
            match = self._match_doc(content)
            if match is None:
                continue
            source_id = match["metadata"]["source_node_id"]
            if source_id in used_ids:
                continue
            used_ids.add(source_id)
            ranked.append({**match, "score": float(len(ranked)), "source": "lightrag"})
            if len(ranked) >= self.cfg.topk:
                break
        return ranked

    @staticmethod
    def _extract_chunks(payload: Any) -> List[Any]:
        if not isinstance(payload, dict):
            return []
        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        for key in ("chunks", "sources", "text_units", "retrieved_chunks"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        return []

    @staticmethod
    def _chunk_content(chunk: Any) -> str:
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, dict):
            for key in ("content", "text", "chunk", "description"):
                if chunk.get(key):
                    return str(chunk[key])
        return str(chunk or "")

    def _match_doc(self, content: str) -> Dict[str, Any] | None:
        normalized = " ".join(content.split())
        if not normalized:
            return None
        for doc in self._docs:
            doc_text = " ".join(doc["content"].split())
            if doc_text and (doc_text in normalized or normalized in doc_text):
                return doc
        for doc in self._docs:
            prefix = " ".join(doc["content"].split())[:160]
            if prefix and prefix in normalized:
                return doc
        return None

    @staticmethod
    def _storage_diagnostics(working_dir: str) -> Dict[str, Any]:
        path = Path(working_dir)

        def json_data_count(name: str) -> int:
            file_path = path / name
            if not file_path.exists():
                return 0
            try:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                return 0
            data = payload.get("data") if isinstance(payload, dict) else payload
            return len(data) if isinstance(data, list) else 0

        chunks = json_data_count("vdb_chunks.json")
        entities = json_data_count("vdb_entities.json")
        relationships = json_data_count("vdb_relationships.json")
        graph_nodes = 0
        graph_path = path / "graph_chunk_entity_relation.graphml"
        if graph_path.exists():
            try:
                graph_text = graph_path.read_text(encoding="utf-8", errors="ignore")
                graph_nodes = len(re.findall(r"<node\b", graph_text))
            except Exception:
                graph_nodes = 0
        return {
            "chunks": chunks,
            "entities": entities,
            "relationships": relationships,
            "graph_nodes": graph_nodes,
            "graph_ready": entities > 0 and (relationships > 0 or graph_nodes > 0),
        }

    @staticmethod
    def _shorten_answer(answer: str) -> str:
        stripped = answer.strip()
        if not stripped:
            return stripped
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and parsed.get("answer_short"):
                return str(parsed["answer_short"]).strip()
        except Exception:
            pass
        content_lines = []
        for raw_line in stripped.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if re.match(r"^#{1,6}\s*references\b", line, flags=re.IGNORECASE):
                break
            if re.match(r"^references\s*:?\s*$", line, flags=re.IGNORECASE):
                break
            content_lines.append(line)
        lines = [line.strip(" -\t") for line in content_lines if line.strip()]
        if not lines:
            return stripped
        first = lines[0]
        for prefix in ("Answer:", "answer:", "Final answer:", "final answer:"):
            if first.startswith(prefix):
                first = first[len(prefix) :].strip()
        numbered_items: List[str] = []
        for line in lines[1:]:
            match = re.match(r"^(?:\d+[.)]|[-*])\s*(.+)$", line)
            if match:
                numbered_items.append(match.group(1).strip())
            elif numbered_items:
                break
        if first.endswith(":") and numbered_items:
            return "; ".join(numbered_items[:5])
        return first
