import json
import tempfile
import unittest
from pathlib import Path


class GraphRAGWiringTests(unittest.TestCase):
    def test_save_retrieval_res_exports_ranked_results_for_official_eval(self):
        from Core.rag.graph_rag import GraphRAG

        class FakeTreeIndex:
            def get_nodes_data(self, node_ids):
                return [
                    {
                        "index_id": 11,
                        "type": "text",
                        "content": "GraphRAG evidence paragraph.",
                        "page": 3,
                    }
                ]

        rag = GraphRAG.__new__(GraphRAG)
        rag.gbc_index = type("FakeGBC", (), {"TreeIndex": FakeTreeIndex()})()

        with tempfile.TemporaryDirectory() as tmp:
            query_dir = Path(tmp)
            retrieval_ids = rag._save_retrieval_res(
                {"TreeNode_ids": [11], "EntNode_name": ["Entity::GraphRAG"]},
                query_output_dir=query_dir,
            )
            payload = json.loads((query_dir / "retrieval_res.json").read_text(encoding="utf-8"))

        self.assertEqual(retrieval_ids, [11])
        self.assertEqual(payload["TreeNode_ids"], [11])
        self.assertEqual(payload["EntNode_name"], ["Entity::GraphRAG"])
        self.assertEqual(payload["ranked_results"][0]["id"], 11)
        self.assertEqual(payload["ranked_results"][0]["source"], "graph")
        self.assertEqual(payload["ranked_results"][0]["block_type"], "paragraph")
        self.assertEqual(payload["ranked_results"][0]["qasper_evidence_text"], "GraphRAG evidence paragraph.")


if __name__ == "__main__":
    unittest.main()
