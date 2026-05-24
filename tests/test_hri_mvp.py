import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace

from Core.Index.Tree import DocumentTree, NodeType, TreeNode
from Core.Index.HRIIndex import HRIIndex
from Core.configs.rag.hri_config import HRIRAGConfig
from Core.rag.hri_plan import HRIQueryPlanner
from Core.rag.hri_rag import HRIRAG
from Core.rag.hri_rag import classify_question_type


class FakeLLM:
    def get_completion(self, prompt, json_response=False):
        return "应开启泄洪设施。依据：第3.2条和表3-1。"


class FakePlannerLLM(FakeLLM):
    def __init__(self, json_result=None, raise_json=False):
        self.json_result = json_result
        self.raise_json = raise_json
        self.json_calls = 0
        self.prompts = []

    def get_json_completion(self, prompt, schema, images=None, think_mode=False):
        self.json_calls += 1
        self.prompts.append(prompt)
        if self.raise_json:
            raise RuntimeError("bad planner response")
        return schema(**self.json_result)


class FakeVectorStore:
    def __init__(self, results):
        self.results = results

    def search(self, query_text, top_k=3):
        return self.results[:top_k]


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
        add_node(
            chapter,
            NodeType.TEXT,
            "相关文件要求等标准制定工作由主管部门统筹推进。",
            page_idx=2,
        )
        article = add_node(
            chapter,
            NodeType.TEXT,
            "第3.2条 当水位超过汛限水位时，应开启泄洪设施，具体下泄流量见表3-1。",
            page_idx=3,
        )
        add_node(
            chapter,
            NodeType.TEXT,
            "第3.3条 除特殊情况外，应及时报送调度结果，同时还应补充记录会商意见。",
            page_idx=3,
        )
        table = add_node(
            chapter,
            NodeType.TABLE,
            "表3-1 下泄流量参数",
            page_idx=4,
            table_body="水位 | 下泄流量\n超过汛限水位 | 500 m3/s",
        )
        add_node(
            chapter,
            NodeType.TEXT,
            "第3.4条 调度参数详见附录A；预警分级见下表。",
            page_idx=4,
        )
        next_table = add_node(
            chapter,
            NodeType.TABLE,
            "表3-2 预警分级参数",
            page_idx=5,
            table_body="等级 | 水位\n红色 | 超保证水位",
        )
        appendix = add_node(
            tree.root_node,
            NodeType.TITLE,
            "附录A 调度参数说明",
            page_idx=6,
        )
        add_node(
            appendix,
            NodeType.TEXT,
            "附录A用于补充说明调度参数的计算口径。",
            page_idx=6,
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
        self.assertTrue(
            any(
                anchor.attributes.get("trigger") == "应"
                and anchor.attributes.get("action")
                and anchor.attributes.get("object")
                for anchor in requirement_targets
            )
        )

        condition_sources = [
            hri.anchors[rel.source_id]
            for rel in hri.relations
            if rel.relation_type == "condition_of"
        ]
        self.assertTrue(
            any(anchor.node_type == "Condition" and "水位超过汛限水位" in anchor.text for anchor in condition_sources)
        )

    def test_hri_filters_false_requirements_and_extracts_extra_relation_types(self):
        tmp, tree, _, _ = self._build_tree()
        self.addCleanup(tmp.cleanup)

        hri = HRIIndex.from_tree(tree, save_dir=tmp.name)
        requirement_texts = [
            anchor.text
            for anchor in hri.anchors.values()
            if anchor.node_type == "Requirement"
        ]
        self.assertFalse(any("要求等标准制定" in text for text in requirement_texts))

        relation_types = {rel.relation_type for rel in hri.relations}
        self.assertIn("exception_to", relation_types)
        self.assertIn("supplements", relation_types)

        exception_targets = [
            hri.anchors[rel.source_id]
            for rel in hri.relations
            if rel.relation_type == "exception_to"
        ]
        self.assertTrue(any(anchor.node_type == "Exception" and "特殊情况" in anchor.text for anchor in exception_targets))

        supplement_targets = [
            hri.anchors[rel.target_id]
            for rel in hri.relations
            if rel.relation_type == "supplements"
        ]
        self.assertTrue(any(anchor.node_type == "Supplement" and "会商意见" in anchor.text for anchor in supplement_targets))

    def test_hri_links_appendix_and_following_table_references(self):
        tmp, tree, _, _ = self._build_tree()
        self.addCleanup(tmp.cleanup)

        hri = HRIIndex.from_tree(tree, save_dir=tmp.name)
        referenced_targets = [
            hri.anchors[rel.target_id]
            for rel in hri.relations
            if rel.relation_type == "refers_to"
        ]

        self.assertTrue(any(anchor.node_type == "Appendix" and "附录A" in anchor.text for anchor in referenced_targets))
        self.assertTrue(any(anchor.node_type == "Table" and "表3-2" in anchor.text for anchor in referenced_targets))

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

    def test_hri_query_planner_rule_high_confidence_definition(self):
        llm = FakePlannerLLM()
        planner = HRIQueryPlanner(llm=llm, confidence_threshold=0.7)

        plan = planner.analyze("什么是数字底板？")

        self.assertEqual(plan.query_type, "locating")
        self.assertEqual(plan.intent, "definition_lookup")
        self.assertIn("definition", plan.evidence_roles)
        self.assertIn("defines", plan.relation_types)
        self.assertEqual(llm.json_calls, 0)

    def test_hri_query_planner_uses_llm_structured_plan_for_ambiguous_query(self):
        llm = FakePlannerLLM(
            json_result={
                "query_type": "comprehensive",
                "intent": "condition_requirement",
                "confidence": 0.82,
                "evidence_roles": ["condition", "requirement", "table"],
                "relation_types": ["condition_of", "requires", "parameter_of"],
                "retrieval_focus": ["预警风险", "指标"],
                "sub_questions": [
                    {"question": "预警风险出现时有哪些条件？", "type": "retrieval"},
                    {"question": "对应需要采取哪些措施？", "type": "retrieval"},
                ],
                "aggregation": None,
                "rationale": "需要条件、要求和参数证据共同回答。",
            }
        )
        planner = HRIQueryPlanner(llm=llm, question_classifier="llm")

        plan = planner.analyze("出现风险后处置方案怎么确定？")

        self.assertEqual(plan.query_type, "comprehensive")
        self.assertEqual(plan.intent, "condition_requirement")
        self.assertEqual(len(plan.sub_questions), 2)
        self.assertEqual(llm.json_calls, 1)
        self.assertIn("只输出一个合法 JSON 对象", llm.prompts[0])

    def test_hri_query_planner_falls_back_when_llm_plan_is_invalid(self):
        planner = HRIQueryPlanner(
            llm=FakePlannerLLM(raise_json=True),
            question_classifier="llm",
        )

        plan = planner.analyze("第1.3节有多少项预警要求？")

        self.assertEqual(plan.query_type, "statistical")
        self.assertEqual(plan.intent, "aggregation")
        self.assertIsNotNone(plan.aggregation)
        self.assertEqual(plan.aggregation.operation, "COUNT")
        self.assertEqual(plan.source, "fallback")

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
        retrieval = json.loads((Path(tmp.name) / "retrieval_res.json").read_text(encoding="utf-8"))
        self.assertEqual(retrieval["query_plan"]["query_type"], "comprehensive")
        self.assertIn("requirement", retrieval["query_plan"]["evidence_roles"])

    def test_hri_hybrid_recall_merges_vector_candidates_with_bm25(self):
        tmp, tree, _, _ = self._build_tree()
        self.addCleanup(tmp.cleanup)
        hri = HRIIndex.from_tree(tree, save_dir=tmp.name)
        bm25 = hri.build_bm25()
        vector_only_anchor = next(
            anchor
            for anchor in hri.anchors.values()
            if anchor.node_type == "Table" and "预警分级" in anchor.text
        )
        vector_store = FakeVectorStore(
            [
                {
                    "id": "vector-hit",
                    "distance": 0.05,
                    "content": vector_only_anchor.text,
                    "metadata": {"node_id": vector_only_anchor.node_id},
                }
            ]
        )
        rag = HRIRAG(
            config=HRIRAGConfig(
                enable_vector_recall=True,
                bm25_topk=1,
                embedding_topk=1,
                max_context_nodes=4,
            ),
            llm=FakeLLM(),
            tree_index=tree,
            hri_index=hri,
            bm25=bm25,
            hri_vector_store=vector_store,
        )

        retrieval_info = rag._retrieve("人员转移安置方案")

        hybrid_ids = [item["node_id"] for item in retrieval_info["hybrid_results"]]
        self.assertIn(vector_only_anchor.node_id, hybrid_ids)
        vector_item = next(
            item
            for item in retrieval_info["hybrid_results"]
            if item["node_id"] == vector_only_anchor.node_id
        )
        self.assertIn(vector_item["source"], {"vector", "hybrid"})

    def test_hri_evidence_budget_keeps_relation_roles_before_truncation(self):
        tmp, tree, article_id, table_id = self._build_tree()
        self.addCleanup(tmp.cleanup)
        hri = HRIIndex.from_tree(tree, save_dir=tmp.name)
        bm25 = hri.build_bm25()
        rag = HRIRAG(
            config=HRIRAGConfig(
                topk=6,
                max_context_nodes=4,
                enable_relation_expansion=True,
                evidence_budgets={
                    "comprehensive": {
                        "definition": 1,
                        "condition": 1,
                        "requirement": 1,
                        "table": 1,
                        "exception": 0,
                        "supplement": 0,
                        "article": 1,
                    }
                },
            ),
            llm=FakeLLM(),
            tree_index=tree,
            hri_index=hri,
            bm25=bm25,
        )
        ranked_results = [
            {"node_id": article_id, "rerank_score": 1.0},
            {"node_id": table_id, "rerank_score": 0.9},
        ]
        relations = rag._collect_relations([article_id, table_id], "comprehensive")

        selected_ids = rag._select_budgeted_nodes(
            ranked_results=ranked_results,
            relations=relations,
            question_type="comprehensive",
        )
        selected_types = {hri.anchors[node_id].node_type for node_id in selected_ids}

        self.assertIn("Condition", selected_types)
        self.assertIn("Requirement", selected_types)
        self.assertIn("Table", selected_types)
        self.assertLessEqual(len(selected_ids), 4)

    def test_hri_evidence_chain_uses_logical_order_for_prompt(self):
        tmp, tree, article_id, table_id = self._build_tree()
        self.addCleanup(tmp.cleanup)
        hri = HRIIndex.from_tree(tree, save_dir=tmp.name)
        bm25 = hri.build_bm25()
        rag = HRIRAG(
            config=HRIRAGConfig(topk=6, max_context_nodes=8, enable_relation_expansion=True),
            llm=FakeLLM(),
            tree_index=tree,
            hri_index=hri,
            bm25=bm25,
        )
        relations = rag._collect_relations([article_id, table_id], "comprehensive")
        unordered_ids = [table_id, article_id]
        unordered_ids.extend(rel.target_id for rel in relations)
        unordered_ids.extend(rel.source_id for rel in relations)
        deduped_ids = list(dict.fromkeys(unordered_ids))

        chain = rag._build_evidence_chain(
            selected_ids=deduped_ids,
            seed_ids=[article_id, table_id],
            relations=relations,
            question_type="comprehensive",
        )
        ordered_types = [item["anchor"]["node_type"] for item in chain]

        self.assertLess(ordered_types.index("Condition"), ordered_types.index("Requirement"))
        self.assertLess(ordered_types.index("Requirement"), ordered_types.index("Table"))

    def test_hri_wo_relation_disables_normative_relation_expansion(self):
        tmp, tree, article_id, table_id = self._build_tree()
        self.addCleanup(tmp.cleanup)
        hri = HRIIndex.from_tree(tree, save_dir=tmp.name)
        bm25 = hri.build_bm25()
        rag = HRIRAG(
            config=HRIRAGConfig(
                ablation_variant="wo_relation",
                enable_relation_expansion=True,
            ),
            llm=FakeLLM(),
            tree_index=tree,
            hri_index=hri,
            bm25=bm25,
        )

        relations = rag._collect_relations([article_id, table_id], "comprehensive")

        self.assertEqual(relations, [])

    def test_hri_wo_tree_uses_flat_retrieval_without_tree_context(self):
        tmp, tree, article_id, _ = self._build_tree()
        self.addCleanup(tmp.cleanup)
        hri = HRIIndex.from_tree(tree, save_dir=tmp.name)
        bm25 = hri.build_bm25()
        rag = HRIRAG(
            config=HRIRAGConfig(
                ablation_variant="wo_tree",
                max_context_nodes=4,
                context_window=2,
                enable_relation_expansion=True,
            ),
            llm=FakeLLM(),
            tree_index=tree,
            hri_index=hri,
            bm25=bm25,
        )
        ranked_results = [{"node_id": article_id, "rerank_score": 1.0}]

        selected_ids = rag._select_budgeted_nodes(
            ranked_results=ranked_results,
            relations=[],
            question_type="comprehensive",
        )

        self.assertEqual(selected_ids, [article_id])
        self.assertEqual(rag._collect_relations([article_id], "comprehensive"), [])

    def test_hri_wo_planner_uses_static_plan_without_llm_classification(self):
        tmp, tree, _, _ = self._build_tree()
        self.addCleanup(tmp.cleanup)
        hri = HRIIndex.from_tree(tree, save_dir=tmp.name)
        bm25 = hri.build_bm25()
        llm = FakePlannerLLM(
            json_result={
                "query_type": "locating",
                "intent": "definition_lookup",
                "confidence": 0.99,
                "evidence_roles": ["definition"],
                "relation_types": ["defines"],
                "retrieval_focus": [],
                "sub_questions": [],
                "aggregation": None,
                "rationale": "unused",
            }
        )
        rag = HRIRAG(
            config=HRIRAGConfig(
                ablation_variant="wo_planner",
                question_classifier="llm",
                topk=2,
                max_context_nodes=3,
            ),
            llm=llm,
            tree_index=tree,
            hri_index=hri,
            bm25=bm25,
        )

        retrieval_info = rag._retrieve("definition query")

        self.assertEqual(llm.json_calls, 0)
        self.assertEqual(retrieval_info["question_type"], "comprehensive")
        self.assertEqual(retrieval_info["query_plan"].intent, "multi_evidence_synthesis")
        self.assertEqual(retrieval_info["query_plan"].sub_questions, [])

    def test_hri_wo_evidence_chain_outputs_answer_without_trace_files(self):
        tmp, tree, _, _ = self._build_tree()
        self.addCleanup(tmp.cleanup)
        hri = HRIIndex.from_tree(tree, save_dir=tmp.name)
        bm25 = hri.build_bm25()
        rag = HRIRAG(
            config=HRIRAGConfig(
                ablation_variant="wo_evidence_chain",
                topk=3,
                max_context_nodes=5,
            ),
            llm=FakeLLM(),
            tree_index=tree,
            hri_index=hri,
            bm25=bm25,
        )

        answer, node_ids = rag.generation("condition requirement query", tmp.name)

        self.assertTrue(answer)
        self.assertEqual(node_ids, [])
        self.assertFalse((Path(tmp.name) / "retrieval_res.json").exists())
        self.assertFalse((Path(tmp.name) / "evidence_chain.json").exists())


if __name__ == "__main__":
    unittest.main()
