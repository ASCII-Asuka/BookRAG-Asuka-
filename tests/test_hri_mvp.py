import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from Core.Index.Tree import DocumentTree, NodeType, TreeNode
from Core.Index.HRIIndex import HRIIndex
from Core.configs.rag.hri_config import HRIRAGConfig
from Core.rag.hri_rag import HRIRAG
from Core.rag.hri_rag import classify_question_type


class FakeLLM:
    def get_completion(self, prompt, json_response=False):
        return "应开启泄洪设施。依据：第3.2条和表3-1。"


class HRIMVPTests(unittest.TestCase):
    def _build_tree(self):
        tmp = tempfile.TemporaryDirectory()
        cfg = SimpleNamespace(save_path=tmp.name)
        tree = DocumentTree(
            meta_dict={"file_name": "hydro-standard.pdf", "file_path": "hydro-standard.pdf"},
            cfg=cfg,
        )

        def add_node(parent, node_type, content, page_idx=0, table_body=None):
            node = TreeNode(
                {
                    "content": content,
                    "page_idx": page_idx,
                    "pdf_id": len(tree.nodes),
                    "table_body": table_body,
                }
            )
            node.type = node_type
            node.outline_node = node_type == NodeType.TITLE
            tree.add_node(node)
            parent.add_child(node)
            return node

        chapter = add_node(tree.root_node, NodeType.TITLE, "第3章 调度管理", page_idx=1)
        add_node(chapter, NodeType.TEXT, "汛限水位是指水库在汛期允许兴利蓄水的上限水位。", page_idx=2)
        article = add_node(
            chapter,
            NodeType.TEXT,
            "第3.2条 当水位超过汛限水位时，应开启泄洪设施，具体下泄流量见表3-1。",
            page_idx=3,
        )
        table = add_node(
            chapter,
            NodeType.TABLE,
            "表3-1 下泄流量参数",
            page_idx=4,
            table_body="水位 | 下泄流量\n超过汛限水位 | 500 m3/s",
        )
        return tmp, tree, article.index_id, table.index_id

    def test_hri_index_extracts_anchors_and_hydro_relations(self):
        tmp, tree, article_id, table_id = self._build_tree()
        self.addCleanup(tmp.cleanup)

        hri = HRIIndex.from_tree(tree, save_dir=tmp.name)

        self.assertEqual(hri.anchors[article_id].node_type, "Article")
        self.assertEqual(hri.anchors[article_id].page, 4)
        self.assertEqual(hri.anchors[table_id].node_type, "Table")

        relation_types = {rel.relation_type for rel in hri.relations}
        self.assertIn("defines", relation_types)
        self.assertIn("condition_of", relation_types)
        self.assertIn("requires", relation_types)
        self.assertIn("refers_to", relation_types)
        self.assertIn("parameter_of", relation_types)

        self.assertTrue(
            any(
                rel.relation_type == "refers_to"
                and rel.source_id == article_id
                and rel.target_id == table_id
                for rel in hri.relations
            )
        )

    def test_hri_relations_target_semantic_object_anchors(self):
        tmp, tree, article_id, _ = self._build_tree()
        self.addCleanup(tmp.cleanup)

        hri = HRIIndex.from_tree(tree, save_dir=tmp.name)

        semantic_relations = [
            rel
            for rel in hri.relations
            if rel.relation_type in {"defines", "condition_of", "requires"}
        ]
        self.assertTrue(semantic_relations)
        self.assertTrue(all(rel.source_id != rel.target_id for rel in semantic_relations))

        concept_targets = [
            hri.anchors[rel.target_id]
            for rel in hri.relations
            if rel.relation_type == "defines"
        ]
        self.assertTrue(any(anchor.node_type == "Concept" and "汛限水位" in anchor.text for anchor in concept_targets))

        requirement_targets = [
            hri.anchors[rel.target_id]
            for rel in hri.relations
            if rel.relation_type == "requires" and rel.source_id == article_id
        ]
        self.assertTrue(
            any(anchor.node_type == "Requirement" and "开启泄洪设施" in anchor.text for anchor in requirement_targets)
        )

        condition_sources = [
            hri.anchors[rel.source_id]
            for rel in hri.relations
            if rel.relation_type == "condition_of"
        ]
        self.assertTrue(
            any(anchor.node_type == "Condition" and "水位超过汛限水位" in anchor.text for anchor in condition_sources)
        )

    def test_hri_bm25_retrieves_chinese_term_node(self):
        tmp, tree, _, _ = self._build_tree()
        self.addCleanup(tmp.cleanup)
        hri = HRIIndex.from_tree(tree, save_dir=tmp.name)
        bm25 = hri.build_bm25()

        results = hri.search_bm25(bm25, "汛限水位定义", top_k=2)

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["node_type"], "TermDefinition")
        self.assertIn("汛限水位", results[0]["text"])

    def test_question_type_rules_cover_locating_comprehensive_and_statistical(self):
        self.assertEqual(classify_question_type("什么是汛限水位？"), "locating")
        self.assertEqual(classify_question_type("超过汛限水位时应采取哪些措施，并依据哪个参数表？"), "comprehensive")
        self.assertEqual(classify_question_type("第3章中一共有多少条调度要求？"), "statistical")

    def test_hri_rag_writes_retrieval_and_evidence_chain_files(self):
        tmp, tree, _, _ = self._build_tree()
        self.addCleanup(tmp.cleanup)
        hri = HRIIndex.from_tree(tree, save_dir=tmp.name)
        bm25 = hri.build_bm25()
        rag = HRIRAG(
            config=HRIRAGConfig(topk=3, max_context_nodes=5),
            llm=FakeLLM(),
            tree_index=tree,
            hri_index=hri,
            bm25=bm25,
        )

        answer, node_ids = rag.generation("超过汛限水位时应采取哪些措施，并依据哪个参数表？", tmp.name)

        self.assertIn("泄洪设施", answer)
        self.assertGreaterEqual(len(node_ids), 1)
        self.assertTrue((Path(tmp.name) / "retrieval_res.json").exists())
        self.assertTrue((Path(tmp.name) / "evidence_chain.json").exists())


if __name__ == "__main__":
    unittest.main()
