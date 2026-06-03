import tempfile
import unittest
from types import SimpleNamespace

from Core.Index.Tree import DocumentTree, NodeType, TreeNode


class EviBridgeIndexTests(unittest.TestCase):
    def _build_tree(self):
        tmp = tempfile.TemporaryDirectory()
        cfg = SimpleNamespace(save_path=tmp.name)
        tree = DocumentTree(
            meta_dict={"file_name": "guide.pdf", "file_path": "guide.pdf"},
            cfg=cfg,
        )

        def add_node(parent, node_type, content, page_idx=0, table_body=None, caption=None):
            node = TreeNode(
                {
                    "content": content,
                    "caption": caption,
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

        section = add_node(tree.root_node, NodeType.TITLE, "1 Reservoir Operation", page_idx=0)
        paragraph = add_node(
            section,
            NodeType.TEXT,
            "Flood control operation uses the spillway discharge rule in Table 1.",
            page_idx=1,
        )
        table = add_node(
            section,
            NodeType.TABLE,
            "Table 1. Spillway discharge parameters",
            page_idx=2,
            table_body="Water level | Discharge\nAbove flood limit | 500 m3/s",
        )
        follow_up = add_node(
            section,
            NodeType.TEXT,
            "The spillway discharge rule must be checked before gate opening.",
            page_idx=3,
        )
        figure = add_node(
            section,
            NodeType.IMAGE,
            "Figure 1. Spillway layout",
            page_idx=4,
            caption="Figure 1 caption describes the spillway gate arrangement.",
        )
        return (
            tmp,
            tree,
            section.index_id,
            paragraph.index_id,
            table.index_id,
            follow_up.index_id,
            figure.index_id,
        )

    def test_builds_multiview_blocks_and_bridges_from_tree(self):
        from Core.Index.EvidenceBridgeIndex import EvidenceBridgeIndex

        tmp, tree, section_id, paragraph_id, table_id, follow_up_id, _ = self._build_tree()
        self.addCleanup(tmp.cleanup)

        index = EvidenceBridgeIndex.from_tree(tree=tree, save_dir=tmp.name)

        self.assertEqual(index.blocks[section_id].block_type, "title")
        self.assertEqual(index.blocks[paragraph_id].block_type, "paragraph")
        self.assertEqual(index.blocks[table_id].block_type, "table")
        self.assertEqual(index.blocks[paragraph_id].page, 2)
        self.assertIn("Reservoir Operation", index.blocks[follow_up_id].title_path)

        bridges = {
            (bridge.source_id, bridge.target_id, bridge.bridge_type, bridge.relation_type)
            for bridge in index.bridges
        }
        self.assertIn((paragraph_id, table_id, "context", "reference_to_table"), bridges)
        self.assertIn((paragraph_id, table_id, "context", "next"), bridges)
        self.assertIn((follow_up_id, paragraph_id, "semantic", "shared_terms"), bridges)
        self.assertIn((paragraph_id, section_id, "hierarchy", "belongs_to_section"), bridges)

    def test_round_trips_bm25_and_exports_typed_graph(self):
        from Core.Index.EvidenceBridgeIndex import EvidenceBridgeIndex

        tmp, tree, _, _, table_id, _, _ = self._build_tree()
        self.addCleanup(tmp.cleanup)

        index = EvidenceBridgeIndex.from_tree(tree=tree, save_dir=tmp.name)
        bm25 = index.build_bm25()
        index.save_bm25(bm25)
        index.save_to_dir()

        loaded = EvidenceBridgeIndex.load_from_dir(tmp.name)
        graph = loaded.to_typed_graph()
        results = loaded.search_bm25(EvidenceBridgeIndex.load_bm25(tmp.name), "spillway discharge", top_k=3)

        self.assertEqual(len(loaded.blocks), len(index.blocks))
        self.assertEqual(len(loaded.bridges), len(index.bridges))
        self.assertIn(table_id, graph.nodes)
        self.assertTrue(any(data["bridge_type"] == "context" for _, _, data in graph.edges(data=True)))
        self.assertTrue(any(item["block_id"] == table_id for item in results))
        self.assertTrue(all("bridge_types" in item for item in results))

    def test_generates_summary_caption_entity_and_patch_blocks(self):
        from Core.Index.EvidenceBridgeIndex import EvidenceBridgeIndex

        tmp, tree, section_id, paragraph_id, table_id, _, figure_id = self._build_tree()
        self.addCleanup(tmp.cleanup)

        index = EvidenceBridgeIndex.from_tree(tree=tree, save_dir=tmp.name)
        block_types = {block.block_type for block in index.blocks.values()}
        relation_types = {bridge.relation_type for bridge in index.bridges}

        self.assertIn("summary", block_types)
        self.assertIn("caption", block_types)
        self.assertIn("entity", block_types)
        self.assertIn("patch", block_types)
        self.assertIn("caption_of", relation_types)
        self.assertIn("mentions_entity", relation_types)
        self.assertIn("summary_of", relation_types)
        self.assertIn("patch_contains", relation_types)

        caption_blocks = [
            block for block in index.blocks.values()
            if block.block_type == "caption" and block.source_node_id == figure_id
        ]
        entity_blocks = [
            block for block in index.blocks.values()
            if block.block_type == "entity" and "spillway" in block.text.lower()
        ]
        patch_blocks = [
            block for block in index.blocks.values()
            if block.block_type == "patch" and block.source_node_id == section_id
        ]

        self.assertTrue(caption_blocks)
        self.assertTrue(entity_blocks)
        self.assertTrue(patch_blocks)
        self.assertTrue(
            any(
                bridge.source_id == paragraph_id
                and bridge.target_id in {block.block_id for block in entity_blocks}
                and bridge.bridge_type == "semantic"
                for bridge in index.bridges
            )
        )
        self.assertTrue(
            any(
                bridge.source_id == table_id
                and bridge.relation_type in {"patch_contains", "contained_in_patch"}
                for bridge in index.bridges
            )
        )


if __name__ == "__main__":
    unittest.main()
