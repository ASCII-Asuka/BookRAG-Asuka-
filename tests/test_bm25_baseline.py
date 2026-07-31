import importlib
import json
import requests
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class BM25BaselineTests(unittest.TestCase):
    def test_bm25_index_smoke_log_does_not_emit_document_text(self):
        from Core.pipelines.vdb_index import build_other_vdb_index

        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(
                save_path=tmp,
                index_type="bm25",
                vdb=SimpleNamespace(
                    vdb_dir_name="bm25_vdb",
                    force_rebuild=True,
                ),
            )
            with patch(
                "Core.pipelines.vdb_index.get_all_chunks",
                return_value=(["Frøya supporting text"], [{"node_id": 7}]),
            ), self.assertLogs("Core.pipelines.vdb_index", level="INFO") as captured:
                build_other_vdb_index(cfg)

        messages = "\n".join(captured.output)
        self.assertNotIn("Frøya supporting text", messages)
        self.assertIn("BM25 smoke test completed with 1 results", messages)

    def test_vdb_index_import_does_not_require_modelscope(self):
        sentinel = object()
        original_modelscope = sys.modules.get("modelscope", sentinel)
        original_vdb_index = sys.modules.pop("Core.pipelines.vdb_index", None)
        sys.modules["modelscope"] = None
        try:
            module = importlib.import_module("Core.pipelines.vdb_index")
        finally:
            sys.modules.pop("Core.pipelines.vdb_index", None)
            if original_vdb_index is not None:
                sys.modules["Core.pipelines.vdb_index"] = original_vdb_index
            if original_modelscope is sentinel:
                sys.modules.pop("modelscope", None)
            else:
                sys.modules["modelscope"] = original_modelscope

        self.assertTrue(hasattr(module, "build_other_vdb_index"))

    def test_vanilla_rag_import_does_not_require_vector_optional_deps(self):
        sentinel = object()
        originals = {
            name: sys.modules.get(name, sentinel)
            for name in ("modelscope", "chromadb", "zmq")
        }
        original_vanilla = sys.modules.pop("Core.rag.vanilla_rag", None)
        for name in originals:
            sys.modules[name] = None
        try:
            module = importlib.import_module("Core.rag.vanilla_rag")
        finally:
            sys.modules.pop("Core.rag.vanilla_rag", None)
            if original_vanilla is not None:
                sys.modules["Core.rag.vanilla_rag"] = original_vanilla
            for name, original in originals.items():
                if original is sentinel:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original

        self.assertTrue(hasattr(module, "VanillaRAG"))

    def test_vllm_reranker_import_does_not_require_modelscope(self):
        sentinel = object()
        original_modelscope = sys.modules.get("modelscope", sentinel)
        original_rerank = sys.modules.pop("Core.provider.rerank", None)
        sys.modules["modelscope"] = None
        try:
            module = importlib.import_module("Core.provider.rerank")
            reranker = module.TextRerankerProvider(
                model_name="reranker",
                backend="vllm",
                api_base="http://localhost:8011/v1",
                api_key="test-key",
            )
        finally:
            sys.modules.pop("Core.provider.rerank", None)
            if original_rerank is not None:
                sys.modules["Core.provider.rerank"] = original_rerank
            if original_modelscope is sentinel:
                sys.modules.pop("modelscope", None)
            else:
                sys.modules["modelscope"] = original_modelscope

        self.assertEqual(reranker.rerank_url, "http://localhost:8011/v1/rerank")
        self.assertEqual(reranker.session.headers.get("Authorization"), "Bearer test-key")
        reranker.close()

    def test_vllm_reranker_retries_transient_request_failure(self):
        from Core.provider.rerank import TextRerankerProvider

        class FakeResponse:
            def __init__(self, ok):
                self.ok = ok

            def raise_for_status(self):
                if not self.ok:
                    raise requests.exceptions.HTTPError("temporary")

            def json(self):
                return {"results": [{"index": 0, "relevance_score": 0.77}]}

        class FakeSession:
            def __init__(self):
                self.calls = 0
                self.headers = {}

            def post(self, url, json=None, timeout=None):
                self.calls += 1
                return FakeResponse(ok=self.calls > 1)

            def close(self):
                pass

        reranker = TextRerankerProvider(
            model_name="reranker",
            backend="vllm",
            api_base="http://localhost:8011/v1",
            max_retries=2,
            retry_backoff=0.0,
            request_timeout=1.0,
        )
        fake_session = FakeSession()
        reranker.session = fake_session

        scores = reranker.rerank("query", ["document"], batch_size=1)

        self.assertEqual(scores, [0.77])
        self.assertEqual(fake_session.calls, 2)

    def test_openai_embedding_backend_does_not_require_modelscope(self):
        sentinel = object()
        original_modelscope = sys.modules.get("modelscope", sentinel)
        original_embedding = sys.modules.pop("Core.provider.embedding", None)
        sys.modules["modelscope"] = None
        try:
            module = importlib.import_module("Core.provider.embedding")
            embedder = module.TextEmbeddingProvider(
                model_name="Qwen/Qwen3-Embedding-0.6B",
                backend="openai",
                api_base="https://api.siliconflow.cn/v1",
                api_key="test-key",
            )
        finally:
            sys.modules.pop("Core.provider.embedding", None)
            if original_embedding is not None:
                sys.modules["Core.provider.embedding"] = original_embedding
            if original_modelscope is sentinel:
                sys.modules.pop("modelscope", None)
            else:
                sys.modules["modelscope"] = original_modelscope

        self.assertEqual(embedder.backend, "openai")
        self.assertEqual(embedder.model_name, "Qwen/Qwen3-Embedding-0.6B")

    def test_qasper_bm25_corpus_keeps_whole_paragraph_text(self):
        from Core.Index.Tree import DocumentTree, NodeType, TreeNode
        from Core.pipelines.vdb_index import get_tree_chunks

        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(
                save_path=tmp,
                pdf_path="qasper://paper-1",
                index_type="bm25",
                rag=SimpleNamespace(
                    strategy_config=SimpleNamespace(bm25_corpus="paragraph")
                ),
            )
            tree = DocumentTree(
                meta_dict={"file_name": "paper-1.pdf", "file_path": "qasper://paper-1"},
                cfg=cfg,
            )
            title = TreeNode({"content": "Methods", "page_idx": 0, "pdf_id": 1})
            title.type = NodeType.TITLE
            title.outline_node = True
            tree.add_node(title)
            tree.root_node.add_child(title)
            paragraph = "Qasper paragraph evidence " + ("with repeated terms " * 80)
            para_node = TreeNode({"content": paragraph, "page_idx": 0, "pdf_id": 2})
            para_node.type = NodeType.TEXT
            tree.add_node(para_node)
            title.add_child(para_node)
            tree.save_to_file()

            chunks, metadatas = get_tree_chunks(cfg)

        self.assertEqual(chunks, [paragraph])
        self.assertEqual(metadatas[0]["source"], "qasper_paragraph")
        self.assertEqual(metadatas[0]["paragraph_id"], 2)
        self.assertEqual(metadatas[0]["source_node_id"], 2)
        self.assertEqual(metadatas[0]["qasper_evidence_text"], paragraph)

    def test_hotpotqa_bm25_corpus_preserves_title_sentence_metadata(self):
        from Core.Index.Tree import DocumentTree, NodeType, TreeNode
        from Core.pipelines.vdb_index import get_tree_chunks

        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(
                save_path=tmp,
                pdf_path="hotpotqa://question-1",
                index_type="bm25",
                rag=SimpleNamespace(
                    strategy_config=SimpleNamespace(bm25_corpus="paragraph")
                ),
            )
            tree = DocumentTree(
                meta_dict={
                    "file_name": "question-1.json",
                    "file_path": "hotpotqa://question-1",
                },
                cfg=cfg,
            )
            title = TreeNode({"content": "Popular Science", "page_idx": 0, "pdf_id": 1})
            title.type = NodeType.TITLE
            title.outline_node = True
            tree.add_node(title)
            tree.root_node.add_child(title)
            sentence = TreeNode(
                {
                    "content": "The supporting sentence.",
                    "page_idx": 0,
                    "pdf_id": 2,
                    "pdf_para_block": {
                        "hotpot_title": "Popular Science",
                        "hotpot_sent_id": 3,
                        "source": "hotpotqa_sentence",
                    },
                }
            )
            sentence.type = NodeType.TEXT
            tree.add_node(sentence)
            title.add_child(sentence)
            tree.save_to_file()

            _, metadatas = get_tree_chunks(cfg)

        self.assertEqual(metadatas[0]["source"], "hotpotqa_sentence")
        self.assertEqual(metadatas[0]["hotpot_title"], "Popular Science")
        self.assertEqual(metadatas[0]["hotpot_sent_id"], 3)

    def test_qasper_dense_paragraph_corpus_keeps_whole_paragraph_text(self):
        from Core.Index.Tree import DocumentTree, NodeType, TreeNode
        from Core.pipelines.vdb_index import get_tree_chunks

        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(
                save_path=tmp,
                pdf_path="qasper://paper-1",
                index_type="vanilla",
                rag=SimpleNamespace(
                    strategy_config=SimpleNamespace(corpus_unit="paragraph")
                ),
            )
            tree = DocumentTree(
                meta_dict={"file_name": "paper-1.pdf", "file_path": "qasper://paper-1"},
                cfg=cfg,
            )
            title = TreeNode({"content": "Abstract", "page_idx": 0, "pdf_id": 1})
            title.type = NodeType.TITLE
            title.outline_node = True
            tree.add_node(title)
            tree.root_node.add_child(title)
            paragraph = "Dense paragraph evidence " + ("keeps full text " * 60)
            para_node = TreeNode({"content": paragraph, "page_idx": 0, "pdf_id": 2})
            para_node.type = NodeType.TEXT
            tree.add_node(para_node)
            title.add_child(para_node)
            tree.save_to_file()

            chunks, metadatas = get_tree_chunks(cfg)

        self.assertEqual(chunks, [paragraph])
        self.assertEqual(metadatas[0]["source"], "qasper_paragraph")
        self.assertEqual(metadatas[0]["qasper_evidence_text"], paragraph)

    def test_vanilla_retrieval_res_preserves_bm25_paragraph_metadata(self):
        from Core.rag.vanilla_rag import VanillaRAG

        with tempfile.TemporaryDirectory() as tmp:
            query_dir = Path(tmp)
            rag = VanillaRAG.__new__(VanillaRAG)
            retrieved_ids = rag._save_retrieval_res(
                [
                    {
                        "id": 0,
                        "score": 3.25,
                        "content": "truncated chunk text",
                        "metadata": {
                            "node_id": 12,
                            "source_node_id": 12,
                            "paragraph_id": 12,
                            "qasper_evidence_text": "Original Qasper paragraph.",
                            "section_id": "Methods",
                            "node_type": "text",
                        },
                    }
                ],
                query_dir,
            )
            payload = json.loads((query_dir / "retrieval_res.json").read_text(encoding="utf-8"))

        self.assertEqual(retrieved_ids, [12])
        self.assertEqual(payload["ranked_results"][0]["score"], 3.25)
        self.assertEqual(payload["ranked_results"][0]["paragraph_id"], 12)
        self.assertEqual(
            payload["ranked_results"][0]["qasper_evidence_text"],
            "Original Qasper paragraph.",
        )

    def test_vanilla_retrieval_res_preserves_raptor_summary_traceability(self):
        from Core.rag.vanilla_rag import VanillaRAG

        with tempfile.TemporaryDirectory() as tmp:
            query_dir = Path(tmp)
            rag = VanillaRAG.__new__(VanillaRAG)
            rag._save_retrieval_res(
                [
                    {
                        "id": 99,
                        "score": 0.42,
                        "content": "RAPTOR summary.",
                        "metadata": {
                            "chunk_id": 99,
                            "source": "raptor_summary",
                            "node_type": "raptor_summary",
                            "raptor_depth": 1,
                            "child_source_node_ids": "7,8",
                        },
                    }
                ],
                query_dir,
            )
            payload = json.loads((query_dir / "retrieval_res.json").read_text(encoding="utf-8"))

        result = payload["ranked_results"][0]
        self.assertEqual(result["block_type"], "summary")
        self.assertEqual(result["raptor_depth"], 1)
        self.assertEqual(result["child_source_node_ids"], "7,8")

    def test_official_export_prefers_qasper_evidence_text_over_chunk_content(self):
        from Scripts.eval.qasper_official import _prediction_evidence

        with tempfile.TemporaryDirectory() as tmp:
            query_dir = Path(tmp)
            (query_dir / "retrieval_res.json").write_text(
                json.dumps(
                    {
                        "ranked_results": [
                            {
                                "content": "chunk text should not be exported",
                                "qasper_evidence_text": "Original Qasper paragraph.",
                                "rank": 1,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            evidence = _prediction_evidence(
                query_dir=query_dir,
                paragraph_evidence_only=True,
                top_k_evidence=0,
            )

        self.assertEqual(evidence, ["Original Qasper paragraph."])

    def test_vanilla_bm25_short_answer_sets_answer_short(self):
        from Core.rag.vanilla_rag import VanillaRAG

        class FakeBM25:
            def search(self, query_text, top_k):
                return [
                    {
                        "id": 0,
                        "score": 1.0,
                        "content": "The answer is No.",
                        "metadata": {"node_id": 1, "qasper_evidence_text": "The answer is No."},
                    }
                ]

        class FakeLLM:
            config = SimpleNamespace(max_tokens=1000)

            def get_completion(self, prompt, json_response=False):
                self.prompt = prompt
                return '{"answer_short": "No", "answer_rationale": "The paragraph says no."}'

        cfg = SimpleNamespace(retrieval_method="bm25", topk=1, answer_style="short")
        llm = FakeLLM()
        rag = VanillaRAG(config=cfg, vector_store=None, llm=llm, bm25=FakeBM25())
        with tempfile.TemporaryDirectory() as tmp:
            answer, retrieved_ids = rag.generation("Is it yes?", Path(tmp))

        self.assertIn("Return only a JSON object", llm.prompt)
        self.assertEqual(answer, '{"answer_short": "No", "answer_rationale": "The paragraph says no."}')
        self.assertEqual(retrieved_ids, [1])
        self.assertEqual(rag.last_answer_short, "No")
        self.assertEqual(rag.last_answer_rationale, "The paragraph says no.")

    def test_vanilla_short_answer_prompt_uses_canonical_unanswerable_label(self):
        from Core.rag.vanilla_rag import VanillaRAG

        llm = SimpleNamespace(config=SimpleNamespace(max_tokens=4096))
        cfg = SimpleNamespace(retrieval_method="bm25", topk=2, answer_style="short")
        rag = VanillaRAG(config=cfg, llm=llm)

        prompt = rag._create_augmented_prompt(
            "What is not stated?",
            [{"id": 1, "content": "A paragraph.", "metadata": {"node_id": 1}}],
        )

        self.assertIn("use Unanswerable only", prompt)
        self.assertNotIn("use Not answerable only", prompt)

    def test_vanilla_short_answer_prompt_requests_traceable_supporting_ids(self):
        from Core.rag.vanilla_rag import VanillaRAG

        llm = SimpleNamespace(config=SimpleNamespace(max_tokens=4096))
        cfg = SimpleNamespace(
            retrieval_method="bm25_rerank",
            topk=2,
            answer_style="short",
        )
        rag = VanillaRAG(config=cfg, llm=llm)

        prompt = rag._create_augmented_prompt(
            "Which group is described?",
            [
                {
                    "id": 7,
                    "content": "The evidence.",
                    "metadata": {"node_id": 7},
                }
            ],
        )

        self.assertIn(
            "keys answer_short, answer_rationale, and supporting_block_ids",
            prompt,
        )
        self.assertIn("[source_id=7]", prompt)
        self.assertIn("1 to 4 integer source ids", prompt)

    def test_vanilla_generation_validates_and_records_supporting_ids(self):
        from Core.rag.vanilla_rag import VanillaRAG

        class FakeBM25:
            def search(self, query_text, top_k):
                return [
                    {
                        "id": 7,
                        "score": 2.0,
                        "content": "First evidence.",
                        "metadata": {"node_id": 7},
                    },
                    {
                        "id": 8,
                        "score": 1.0,
                        "content": "Second evidence.",
                        "metadata": {"node_id": 8},
                    },
                ]

        class FakeLLM:
            config = SimpleNamespace(max_tokens=4096)

            def get_completion(self, prompt, json_response=False):
                return json.dumps(
                    {
                        "answer_short": "answer",
                        "answer_rationale": "Second evidence supports it.",
                        "supporting_block_ids": [8, 999],
                    }
                )

        cfg = SimpleNamespace(
            retrieval_method="bm25",
            topk=2,
            answer_style="short",
            supporting_evidence_topk=4,
        )
        rag = VanillaRAG(config=cfg, llm=FakeLLM(), bm25=FakeBM25())
        with tempfile.TemporaryDirectory() as tmp:
            rag.generation("Question?", Path(tmp))
            payload = json.loads(
                (Path(tmp) / "retrieval_res.json").read_text(encoding="utf-8")
            )

        self.assertEqual(payload["supporting_block_ids"], [8])
        self.assertEqual(
            payload["citation_validation"],
            {
                "enabled": True,
                "requested_ids": [8, 999],
                "valid_ids": [8],
                "invalid_ids": [999],
                "ignored_ids": [999],
                "used_ids": [8],
                "fallback_used": False,
            },
        )
        self.assertEqual(rag.last_supporting_block_ids, [8])

    def test_vanilla_generation_caps_fallback_supporting_evidence(self):
        from Core.rag.vanilla_rag import VanillaRAG

        class FakeBM25:
            def search(self, query_text, top_k):
                return [
                    {
                        "id": node_id,
                        "score": float(10 - node_id),
                        "content": f"Evidence {node_id}.",
                        "metadata": {"node_id": node_id},
                    }
                    for node_id in range(1, 7)
                ]

        class FakeLLM:
            config = SimpleNamespace(max_tokens=4096)

            def get_completion(self, prompt, json_response=False):
                return json.dumps(
                    {
                        "answer_short": "answer",
                        "answer_rationale": "The evidence supports it.",
                    }
                )

        cfg = SimpleNamespace(
            retrieval_method="bm25",
            topk=6,
            answer_style="short",
            supporting_evidence_topk=4,
        )
        rag = VanillaRAG(config=cfg, llm=FakeLLM(), bm25=FakeBM25())
        with tempfile.TemporaryDirectory() as tmp:
            rag.generation("Question?", Path(tmp))
            payload = json.loads(
                (Path(tmp) / "retrieval_res.json").read_text(encoding="utf-8")
            )

        self.assertEqual(payload["supporting_block_ids"], [1, 2, 3, 4])
        self.assertTrue(payload["citation_validation"]["fallback_used"])
        self.assertEqual(rag.last_supporting_block_ids, [1, 2, 3, 4])

    def test_vanilla_hotpot_supporting_ids_reject_non_sentence_nodes(self):
        from Core.rag.vanilla_rag import VanillaRAG

        rag = VanillaRAG.__new__(VanillaRAG)
        rag.cfg = SimpleNamespace(
            retrieval_method="full_document",
            supporting_evidence_topk=4,
        )
        ranked = [
            {
                "id": 1,
                "content": "Article A",
                "metadata": {"node_id": 1, "node_type": "title"},
            },
            {
                "id": 2,
                "content": "The supporting sentence.",
                "metadata": {
                    "node_id": 2,
                    "node_type": "text",
                    "hotpot_title": "Article A",
                    "hotpot_sent_id": 0,
                },
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            rag._save_retrieval_res(ranked, Path(tmp), supporting_ids=[1, 2])
            payload = json.loads(
                (Path(tmp) / "retrieval_res.json").read_text(encoding="utf-8")
            )

        self.assertEqual(payload["supporting_block_ids"], [2])
        self.assertEqual(payload["citation_validation"]["valid_ids"], [2])
        self.assertEqual(payload["citation_validation"]["invalid_ids"], [1])

    def test_vanilla_hotpot_fallback_skips_non_sentence_nodes(self):
        from Core.rag.vanilla_rag import VanillaRAG

        rag = VanillaRAG.__new__(VanillaRAG)
        rag.cfg = SimpleNamespace(
            retrieval_method="full_document",
            supporting_evidence_topk=4,
        )
        ranked = [
            {
                "id": 1,
                "content": "Article A",
                "metadata": {"node_id": 1, "node_type": "title"},
            },
            {
                "id": 2,
                "content": "The supporting sentence.",
                "metadata": {
                    "node_id": 2,
                    "node_type": "text",
                    "hotpot_title": "Article A",
                    "hotpot_sent_id": 0,
                },
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            rag._save_retrieval_res(ranked, Path(tmp), supporting_ids=[])
            payload = json.loads(
                (Path(tmp) / "retrieval_res.json").read_text(encoding="utf-8")
            )

        self.assertEqual(payload["supporting_block_ids"], [2])
        self.assertEqual(payload["supporting_evidence"][0]["hotpot_title"], "Article A")
        self.assertEqual(payload["supporting_evidence"][0]["hotpot_sent_id"], 0)
        self.assertTrue(payload["citation_validation"]["fallback_used"])

    def test_all_short_answer_rag_prompts_use_canonical_unanswerable_label(self):
        repo_root = Path(__file__).resolve().parents[1]
        for relative_path in [
            "Core/rag/vanilla_rag.py",
            "Core/rag/hipporag_rag.py",
            "Core/rag/lightrag_rag.py",
        ]:
            source = (repo_root / relative_path).read_text(encoding="utf-8")
            self.assertNotIn("use Not answerable", source, relative_path)
        lightrag_source = (repo_root / "Core/rag/lightrag_rag.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('answer = "Not answerable"', lightrag_source)

    def test_vanilla_hybrid_rrf_merges_bm25_and_dense_rankings(self):
        from Core.rag.vanilla_rag import VanillaRAG

        class FakeBM25:
            def search(self, query_text, top_k):
                return [
                    {
                        "id": 1,
                        "score": 8.0,
                        "content": "BM25 first paragraph.",
                        "metadata": {"node_id": 1, "qasper_evidence_text": "BM25 first paragraph."},
                    },
                    {
                        "id": 2,
                        "score": 5.0,
                        "content": "Shared paragraph.",
                        "metadata": {"node_id": 2, "qasper_evidence_text": "Shared paragraph."},
                    },
                ][:top_k]

        class FakeVector:
            def search(self, query_text, top_k):
                return [
                    {
                        "id": "dense-2",
                        "distance": 0.1,
                        "content": "Shared paragraph.",
                        "metadata": {"node_id": 2, "qasper_evidence_text": "Shared paragraph."},
                    },
                    {
                        "id": "dense-3",
                        "distance": 0.2,
                        "content": "Dense second paragraph.",
                        "metadata": {"node_id": 3, "qasper_evidence_text": "Dense second paragraph."},
                    },
                ][:top_k]

        cfg = SimpleNamespace(
            retrieval_method="hybrid",
            topk=3,
            hybrid_bm25_topk=2,
            hybrid_dense_topk=2,
            rrf_k=60,
            answer_style="short",
        )
        rag = VanillaRAG(config=cfg, vector_store=FakeVector(), llm=SimpleNamespace(config=SimpleNamespace(max_tokens=1000)), bm25=FakeBM25())

        results = rag._retrieve("query", top_k=3)

        self.assertEqual([item["metadata"]["node_id"] for item in results], [2, 1, 3])
        self.assertEqual(results[0]["source"], "hybrid")
        self.assertIn("rrf_score", results[0])

    def test_vanilla_bm25_rerank_orders_by_reranker_score(self):
        from Core.rag.vanilla_rag import VanillaRAG

        class FakeBM25:
            def search(self, query_text, top_k):
                return [
                    {"id": 1, "score": 9.0, "content": "Less relevant.", "metadata": {"node_id": 1}},
                    {"id": 2, "score": 4.0, "content": "Most relevant.", "metadata": {"node_id": 2}},
                ]

        class FakeReranker:
            def rerank(self, query, documents, instruction=None, batch_size=4):
                return [0.1, 0.9]

        cfg = SimpleNamespace(retrieval_method="bm25_rerank", topk=1, rerank_topk=20, answer_style="short")
        rag = VanillaRAG(
            config=cfg,
            vector_store=None,
            llm=SimpleNamespace(config=SimpleNamespace(max_tokens=1000)),
            bm25=FakeBM25(),
            reranker=FakeReranker(),
        )

        results = rag._retrieve("query", top_k=1)

        self.assertEqual(results[0]["metadata"]["node_id"], 2)
        self.assertEqual(results[0]["rerank_score"], 0.9)
        self.assertEqual(results[0]["bm25_score"], 4.0)

    def test_vanilla_bm25_rerank_falls_back_to_bm25_order_on_failure(self):
        from Core.rag.vanilla_rag import VanillaRAG

        class FakeBM25:
            def search(self, query_text, top_k):
                return [
                    {"id": 1, "score": 9.0, "content": "Best BM25.", "metadata": {"node_id": 1}},
                    {"id": 2, "score": 4.0, "content": "Second BM25.", "metadata": {"node_id": 2}},
                ]

        class FailingReranker:
            def rerank(self, query, documents, instruction=None, batch_size=4):
                raise RuntimeError("reranker down")

        cfg = SimpleNamespace(retrieval_method="bm25_rerank", topk=1, rerank_topk=20, answer_style="short")
        rag = VanillaRAG(
            config=cfg,
            vector_store=None,
            llm=SimpleNamespace(config=SimpleNamespace(max_tokens=1000)),
            bm25=FakeBM25(),
            reranker=FailingReranker(),
        )

        results = rag._retrieve("query", top_k=1)

        self.assertEqual(results[0]["metadata"]["node_id"], 1)
        self.assertTrue(results[0]["rerank_failed"])
        self.assertIn("reranker down", results[0]["rerank_error"])

    def test_vanilla_close_prefers_vector_store_close(self):
        from Core.rag.vanilla_rag import VanillaRAG

        class FakeVector:
            def __init__(self):
                self.close_calls = 0
                self.embedding_model = SimpleNamespace(close=lambda: None)

            def close(self):
                self.close_calls += 1

        vector = FakeVector()
        cfg = SimpleNamespace(retrieval_method="vanilla", topk=1, answer_style="short")
        rag = VanillaRAG(
            config=cfg,
            vector_store=vector,
            llm=SimpleNamespace(config=SimpleNamespace(max_tokens=1000)),
        )

        rag.close()

        self.assertEqual(vector.close_calls, 1)

    def test_vanilla_abstract_only_uses_tree_abstract_without_retrieval(self):
        from Core.Index.Tree import DocumentTree, NodeType, TreeNode
        from Core.rag.vanilla_rag import VanillaRAG

        with tempfile.TemporaryDirectory() as tmp:
            tree = DocumentTree(
                meta_dict={"file_name": "paper-1.pdf", "file_path": "qasper://paper-1"},
                cfg=SimpleNamespace(save_path=tmp),
            )
            title = TreeNode({"content": "Abstract", "page_idx": 0, "pdf_id": 1})
            title.type = NodeType.TITLE
            tree.add_node(title)
            tree.root_node.add_child(title)
            para = TreeNode({"content": "This abstract describes the answer.", "page_idx": 0, "pdf_id": 2})
            para.type = NodeType.TEXT
            tree.add_node(para)
            title.add_child(para)

            cfg = SimpleNamespace(retrieval_method="abstract_only", topk=1, answer_style="short")
            rag = VanillaRAG(
                config=cfg,
                vector_store=None,
                llm=SimpleNamespace(config=SimpleNamespace(max_tokens=1000)),
                tree_index=tree,
            )
            results = rag._retrieve("query", top_k=1)

        self.assertEqual(results[0]["content"], "This abstract describes the answer.")
        self.assertEqual(results[0]["metadata"]["source"], "abstract_only")
        self.assertEqual(results[0]["metadata"]["node_id"], 2)

    def test_vanilla_full_document_uses_tree_nodes_without_retrieval(self):
        from Core.Index.Tree import DocumentTree, NodeType, TreeNode
        from Core.rag.vanilla_rag import VanillaRAG

        with tempfile.TemporaryDirectory() as tmp:
            tree = DocumentTree(
                meta_dict={"file_name": "paper-1.pdf", "file_path": "qasper://paper-1"},
                cfg=SimpleNamespace(save_path=tmp),
            )
            title = TreeNode({"content": "Methods", "page_idx": 0, "pdf_id": 1})
            title.type = NodeType.TITLE
            tree.add_node(title)
            tree.root_node.add_child(title)
            para = TreeNode({"content": "The full document contains the answer.", "page_idx": 0, "pdf_id": 2})
            para.type = NodeType.TEXT
            tree.add_node(para)
            title.add_child(para)

            cfg = SimpleNamespace(retrieval_method="full_document", topk=0, answer_style="short")
            rag = VanillaRAG(
                config=cfg,
                vector_store=None,
                llm=SimpleNamespace(config=SimpleNamespace(max_tokens=1000)),
                tree_index=tree,
            )
            results = rag._retrieve("query", top_k=10)

        self.assertEqual([item["metadata"]["node_id"] for item in results], [1, 2])
        self.assertEqual(results[0]["metadata"]["block_type"], "title")
        self.assertEqual(results[1]["metadata"]["source"], "full_document")
        self.assertEqual(results[1]["metadata"]["block_type"], "paragraph")
        self.assertEqual(
            results[1]["metadata"]["qasper_evidence_text"],
            "The full document contains the answer.",
        )

    def test_vanilla_full_document_preserves_hotpotqa_fact_metadata(self):
        from Core.Index.Tree import DocumentTree, NodeType, TreeNode
        from Core.rag.vanilla_rag import VanillaRAG

        with tempfile.TemporaryDirectory() as tmp:
            tree = DocumentTree(
                meta_dict={"file_name": "hotpot-1.json", "file_path": "hotpotqa://hotpot-1"},
                cfg=SimpleNamespace(save_path=tmp),
            )
            title = TreeNode({"content": "Article A", "page_idx": 0, "pdf_id": 1})
            title.type = NodeType.TITLE
            tree.add_node(title)
            tree.root_node.add_child(title)
            first = TreeNode({"content": "First sentence.", "page_idx": 0, "pdf_id": 2})
            first.type = NodeType.TEXT
            tree.add_node(first)
            title.add_child(first)
            second = TreeNode({"content": "Second sentence has the answer.", "page_idx": 0, "pdf_id": 3})
            second.type = NodeType.TEXT
            tree.add_node(second)
            title.add_child(second)

            cfg = SimpleNamespace(retrieval_method="full_document", topk=0, answer_style="short")
            rag = VanillaRAG(
                config=cfg,
                vector_store=None,
                llm=SimpleNamespace(config=SimpleNamespace(max_tokens=1000)),
                tree_index=tree,
            )
            results = rag._retrieve("query", top_k=10)

        sentence_items = [item for item in results if item["metadata"].get("block_type") == "paragraph"]
        self.assertEqual(sentence_items[0]["metadata"]["hotpot_title"], "Article A")
        self.assertEqual(sentence_items[0]["metadata"]["sent_id"], 0)
        self.assertEqual(sentence_items[1]["metadata"]["hotpot_title"], "Article A")
        self.assertEqual(sentence_items[1]["metadata"]["sent_id"], 1)

    def test_vanilla_longrag_retrieves_long_units_from_tree(self):
        from Core.Index.Tree import DocumentTree, NodeType, TreeNode
        from Core.rag.vanilla_rag import VanillaRAG

        with tempfile.TemporaryDirectory() as tmp:
            tree = DocumentTree(
                meta_dict={"file_name": "paper-1.pdf", "file_path": "qasper://paper-1"},
                cfg=SimpleNamespace(save_path=tmp),
            )
            methods = TreeNode({"content": "Methods", "page_idx": 0, "pdf_id": 1})
            methods.type = NodeType.TITLE
            tree.add_node(methods)
            tree.root_node.add_child(methods)
            method_para = TreeNode({"content": "We used alpha beta retrieval units.", "page_idx": 0, "pdf_id": 2})
            method_para.type = NodeType.TEXT
            tree.add_node(method_para)
            methods.add_child(method_para)
            results_title = TreeNode({"content": "Results", "page_idx": 1, "pdf_id": 3})
            results_title.type = NodeType.TITLE
            tree.add_node(results_title)
            tree.root_node.add_child(results_title)
            result_para = TreeNode({"content": "The final answer is zephyr bridge.", "page_idx": 1, "pdf_id": 4})
            result_para.type = NodeType.TEXT
            tree.add_node(result_para)
            results_title.add_child(result_para)

            cfg = SimpleNamespace(
                retrieval_method="longrag",
                topk=1,
                answer_style="short",
                longrag_unit_tokens=100,
                longrag_max_context_tokens=1000,
            )
            rag = VanillaRAG(
                config=cfg,
                vector_store=None,
                llm=SimpleNamespace(config=SimpleNamespace(max_tokens=1200)),
                tree_index=tree,
            )
            retrieved = rag._retrieve("zephyr bridge", top_k=1)

        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0]["metadata"]["source"], "longrag")
        self.assertEqual(retrieved[0]["metadata"]["section_id"], "Results")
        self.assertIn("The final answer is zephyr bridge.", retrieved[0]["content"])
        self.assertIn(4, retrieved[0]["metadata"]["child_source_node_ids"])

    def test_vanilla_longrag_generation_uses_source_ids_for_supporting_evidence(self):
        from Core.Index.Tree import DocumentTree, NodeType, TreeNode
        from Core.rag.vanilla_rag import VanillaRAG

        class FakeLLM:
            config = SimpleNamespace(max_tokens=1400)
            supporting_id = 0

            def get_completion(self, prompt, json_response=False):
                self.prompt = prompt
                return json.dumps(
                    {
                        "answer_short": "zephyr bridge",
                        "answer_rationale": "The text states it.",
                        "supporting_block_ids": [self.supporting_id],
                    }
                )

        with tempfile.TemporaryDirectory() as tmp:
            tree = DocumentTree(
                meta_dict={"file_name": "paper-1.pdf", "file_path": "qasper://paper-1"},
                cfg=SimpleNamespace(save_path=tmp),
            )
            title = TreeNode({"content": "Results", "page_idx": 1, "pdf_id": 3})
            title.type = NodeType.TITLE
            tree.add_node(title)
            tree.root_node.add_child(title)
            para = TreeNode({"content": "The final answer is zephyr bridge.", "page_idx": 1, "pdf_id": 4})
            para.type = NodeType.TEXT
            tree.add_node(para)
            title.add_child(para)
            para_id = para.index_id

            cfg = SimpleNamespace(
                retrieval_method="longrag",
                topk=1,
                answer_style="short",
                longrag_unit_tokens=100,
                longrag_max_context_tokens=1000,
            )
            llm = FakeLLM()
            llm.supporting_id = para_id
            rag = VanillaRAG(
                config=cfg,
                vector_store=None,
                llm=llm,
                tree_index=tree,
            )
            answer, retrieved_ids = rag.generation("What is the answer?", Path(tmp))
            payload = json.loads((Path(tmp) / "retrieval_res.json").read_text(encoding="utf-8"))

        self.assertIn(f"[source_id={para_id}]", llm.prompt)
        self.assertNotIn("[source_id=longrag_", llm.prompt)
        self.assertEqual(retrieved_ids, [para_id])
        self.assertEqual(rag.last_answer_short, "zephyr bridge")
        self.assertEqual(payload["supporting_block_ids"], [para_id])
        self.assertEqual(payload["supporting_evidence"][0]["qasper_evidence_text"], "The final answer is zephyr bridge.")
        self.assertIn('"answer_short": "zephyr bridge"', answer)

    def test_vanilla_longrag_fallback_expands_child_evidence(self):
        from Core.rag.vanilla_rag import VanillaRAG

        rag = VanillaRAG.__new__(VanillaRAG)
        rag.cfg = SimpleNamespace(
            retrieval_method="longrag",
            supporting_evidence_topk=4,
        )
        ranked = [
            {
                "id": "longrag_0",
                "content": "[source_id=4] The final answer is zephyr bridge.",
                "metadata": {
                    "source": "longrag",
                    "node_id": "longrag_0",
                    "longrag_unit_id": "longrag_0",
                    "child_source_node_ids": [4],
                    "child_qasper_evidence_texts": ["The final answer is zephyr bridge."],
                    "child_pages": [1],
                    "child_sections": ["Results"],
                    "child_block_types": ["paragraph"],
                    "child_hotpot_facts": [["Results", 0]],
                },
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            retrieved_ids = rag._save_retrieval_res(
                ranked,
                Path(tmp),
                supporting_ids=[],
            )
            payload = json.loads(
                (Path(tmp) / "retrieval_res.json").read_text(encoding="utf-8")
            )

        self.assertEqual(retrieved_ids, [4])
        self.assertEqual(payload["supporting_block_ids"], [4])
        self.assertEqual(payload["supporting_evidence"][0]["hotpot_title"], "Results")
        self.assertEqual(payload["supporting_evidence"][0]["hotpot_sent_id"], 0)
        self.assertTrue(payload["citation_validation"]["fallback_used"])

    def test_abstract_only_resource_loader_does_not_require_modelscope(self):
        from Core.Index.Tree import DocumentTree, NodeType, TreeNode

        sentinel = object()
        original_modelscope = sys.modules.get("modelscope", sentinel)
        original_resource_loader = sys.modules.pop("Core.utils.resource_loader", None)
        sys.modules["modelscope"] = None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tree = DocumentTree(
                    meta_dict={"file_name": "paper-1.pdf", "file_path": "qasper://paper-1"},
                    cfg=SimpleNamespace(save_path=tmp),
                )
                title = TreeNode({"content": "Abstract", "page_idx": 0, "pdf_id": 1})
                title.type = NodeType.TITLE
                tree.add_node(title)
                tree.root_node.add_child(title)
                para = TreeNode({"content": "Abstract text.", "page_idx": 0, "pdf_id": 2})
                para.type = NodeType.TEXT
                tree.add_node(para)
                title.add_child(para)
                tree.save_to_file()

                module = importlib.import_module("Core.utils.resource_loader")
                cfg = SimpleNamespace(
                    save_path=tmp,
                    rag=SimpleNamespace(
                        strategy_config=SimpleNamespace(
                            strategy="vanilla",
                            retrieval_method="abstract_only",
                        )
                    ),
                )
                dependencies = module.prepare_rag_dependencies(cfg)
        finally:
            sys.modules.pop("Core.utils.resource_loader", None)
            if original_resource_loader is not None:
                sys.modules["Core.utils.resource_loader"] = original_resource_loader
            if original_modelscope is sentinel:
                sys.modules.pop("modelscope", None)
            else:
                sys.modules["modelscope"] = original_modelscope

        self.assertIn("tree_index", dependencies)

    def test_main_dataset_reader_preserves_qasper_arxiv_id_string(self):
        from main import read_dataset_dataframe

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qasper.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "doc_uuid": "1809.00540",
                            "doc_path": "qasper://1809.00540",
                            "question": "q",
                            "answer": [],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            df = read_dataset_dataframe(str(path))

        self.assertEqual(df.loc[0, "doc_uuid"], "1809.00540")
        self.assertEqual(df.loc[0, "doc_path"], "qasper://1809.00540")


if __name__ == "__main__":
    unittest.main()
