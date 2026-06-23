import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


class HippoRAGModuleTests(unittest.TestCase):
    @staticmethod
    def _tree_with_paragraphs(paragraphs):
        text_type = SimpleNamespace(value="text")
        return SimpleNamespace(
            get_nodes=lambda hasRoot=False: [
                SimpleNamespace(
                    index_id=index + 1,
                    type=text_type,
                    parent=None,
                    meta_info=SimpleNamespace(content=text),
                )
                for index, text in enumerate(paragraphs)
            ]
        )

    def test_documents_from_tree_preserve_qasper_metadata(self):
        from Core.rag.hipporag_rag import HippoRAGRAG

        tree = self._tree_with_paragraphs(["First paragraph.", "Second paragraph."])

        docs = HippoRAGRAG.documents_from_tree(tree)

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0]["id"], "node_1")
        self.assertEqual(docs[0]["metadata"]["source_node_id"], 1)
        self.assertEqual(docs[0]["metadata"]["qasper_evidence_text"], "First paragraph.")
        self.assertEqual(docs[0]["metadata"]["node_type"], "paragraph")

    def test_entity_bridge_ppr_promotes_connected_passage(self):
        from Core.configs.rag.hipporag_config import HippoRAGConfig
        from Core.rag.hipporag_rag import HippoRAGRAG

        tree = self._tree_with_paragraphs(
            [
                "The dataset was collected from crowd workers in a dialogue game.",
                "Dialogue game participants produced the final annotations.",
                "A neural parser was evaluated with unrelated lexical overlap.",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            rag = HippoRAGRAG(
                config=HippoRAGConfig(topk=2, bm25_topk=1, entity_seed_weight=3.0),
                llm=SimpleNamespace(config=SimpleNamespace(max_tokens=4096)),
                tree_index=tree,
                save_path=tmp,
            )

            ranked = rag._retrieve("Who produced the dialogue game annotations?")

        ids = [item["metadata"]["source_node_id"] for item in ranked]
        self.assertIn(2, ids)
        self.assertEqual(ranked[0]["source"], "hipporag")
        self.assertGreaterEqual(rag.last_graph_diagnostics["entities"], 1)

    def test_generation_saves_supporting_evidence_and_answer_short(self):
        from Core.configs.rag.hipporag_config import HippoRAGConfig
        from Core.rag.hipporag_rag import HippoRAGRAG

        tree = self._tree_with_paragraphs(["The answer is alpha beta."])
        llm = SimpleNamespace(
            config=SimpleNamespace(max_tokens=4096),
            get_completion=lambda prompt, json_response=False: '{"answer_short": "alpha beta", "answer_rationale": "from paragraph"}',
        )
        with tempfile.TemporaryDirectory() as tmp:
            rag = HippoRAGRAG(
                config=HippoRAGConfig(topk=1, supporting_evidence_topk=1),
                llm=llm,
                tree_index=tree,
                save_path=tmp,
            )
            answer, ids = rag.generation("What is the answer?", tmp)
            retrieval_payload = json.loads((Path(tmp) / "retrieval_res.json").read_text(encoding="utf-8"))

        self.assertIn("alpha beta", answer)
        self.assertEqual(ids, [1])
        self.assertEqual(rag.last_answer_short, "alpha beta")
        self.assertEqual(rag.last_supporting_block_ids, [1])
        self.assertEqual(retrieval_payload["supporting_evidence"][0]["source_node_id"], 1)
        self.assertEqual(retrieval_payload["graph_diagnostics"]["passages"], 1)


if __name__ == "__main__":
    unittest.main()
