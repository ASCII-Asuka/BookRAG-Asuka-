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


if __name__ == "__main__":
    unittest.main()
