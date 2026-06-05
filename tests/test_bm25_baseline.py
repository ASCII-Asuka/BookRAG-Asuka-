import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


class BM25BaselineTests(unittest.TestCase):
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
