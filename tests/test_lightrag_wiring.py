import tempfile
import textwrap
import unittest
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


class LightRAGWiringTests(unittest.TestCase):
    def test_system_config_accepts_lightrag_strategy(self):
        from Core.configs.system_config import load_system_config

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "lightrag.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    pdf_path: TODO
                    save_path: TODO
                    index_type: lightrag
                    mineru:
                      backend: pipeline
                      method: auto
                      lang: en
                    rag:
                      strategy: lightrag
                      mode: hybrid
                      topk: 7
                      answer_style: short
                      working_dir_name: lightrag_workdir
                    """
                ).strip(),
                encoding="utf-8",
            )

            cfg = load_system_config(str(config_path))

        self.assertEqual(cfg.rag.strategy_config.strategy, "lightrag")
        self.assertEqual(cfg.rag.strategy_config.mode, "hybrid")
        self.assertEqual(cfg.rag.strategy_config.topk, 7)
        self.assertEqual(cfg.rag.strategy_config.answer_style, "short")

    def test_create_rag_agent_dispatches_lightrag(self):
        from Core.configs.llm_config import LLMConfig
        from Core.configs.rag.lightrag_config import LightRAGConfig
        from Core.configs.vlm_config import VLMConfig
        from Core.rag import create_rag_agent

        tree_index = SimpleNamespace()

        with patch("Core.rag.LightRAGRAG", create=True) as agent_cls, patch("Core.provider.llm.LLM"), patch(
            "Core.provider.vlm.VLM"
        ):
            create_rag_agent(
                strategy_config=LightRAGConfig(),
                llm_config=LLMConfig(),
                vlm_config=VLMConfig(),
                tree_index=tree_index,
                save_path="doc-work",
            )

        kwargs = agent_cls.call_args.kwargs
        self.assertEqual(kwargs["config"].strategy, "lightrag")
        self.assertIs(kwargs["tree_index"], tree_index)
        self.assertEqual(kwargs["save_path"], "doc-work")

    def test_resource_loader_provides_tree_and_save_path_for_lightrag(self):
        from Core.configs.system_config import SystemConfig
        from Core.configs.mineru_config import MinerU
        from Core.configs.rag_config import RAGConfig
        from Core.configs.rag.lightrag_config import LightRAGConfig
        from Core.utils.resource_loader import prepare_rag_dependencies

        cfg = SystemConfig(
            save_path="doc-work",
            mineru=MinerU(backend="pipeline", method="auto", lang="en"),
            rag=RAGConfig(strategy_config=LightRAGConfig()),
        )

        with patch("Core.Index.Tree.DocumentTree.get_save_path", return_value="doc-work/tree.pkl"), patch(
            "Core.Index.Tree.DocumentTree.load_from_file", return_value="tree"
        ):
            deps = prepare_rag_dependencies(cfg)

        self.assertEqual(deps["tree_index"], "tree")
        self.assertEqual(deps["save_path"], "doc-work")

    def test_tree_documents_preserve_qasper_evidence_text(self):
        from Core.rag.lightrag_rag import LightRAGRAG

        text_type = SimpleNamespace(value="text")
        title_type = SimpleNamespace(value="title")
        tree = SimpleNamespace(
            get_nodes=lambda hasRoot=False: [
                SimpleNamespace(
                    index_id=1,
                    type=title_type,
                    meta_info=SimpleNamespace(content="Introduction"),
                ),
                SimpleNamespace(
                    index_id=2,
                    type=text_type,
                    meta_info=SimpleNamespace(content="First paragraph."),
                ),
            ]
        )

        docs = LightRAGRAG.documents_from_tree(tree)

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["content"], "First paragraph.")
        self.assertEqual(docs[0]["metadata"]["source_node_id"], 2)
        self.assertEqual(docs[0]["metadata"]["qasper_evidence_text"], "First paragraph.")

    def test_run_sync_awaits_light_rag_coroutines(self):
        from Core.rag.lightrag_rag import LightRAGRAG

        async def returns_value():
            await asyncio.sleep(0)
            return "ok"

        self.assertEqual(LightRAGRAG._run_sync(returns_value()), "ok")

    def test_initialize_storages_uses_lightrag_owning_loop_runner(self):
        from Core.rag.lightrag_rag import LightRAGRAG

        class FakeLightRAG:
            async def initialize_storages(self):
                return "initialized"

        rag = LightRAGRAG.__new__(LightRAGRAG)
        rag._rag = FakeLightRAG()

        with patch.object(rag, "_run_on_lightrag_loop", return_value="initialized") as run_mock, patch(
            "lightrag.kg.shared_storage.initialize_pipeline_status",
            return_value=None,
        ):
            rag._initialize_storages()

        self.assertGreaterEqual(run_mock.call_count, 1)
        self.assertEqual(run_mock.call_args_list[0].kwargs["sync_name"], "initialize_storages")

    def test_query_data_falls_back_when_hybrid_returns_no_chunks(self):
        from Core.configs.rag.lightrag_config import LightRAGConfig
        from Core.rag.lightrag_rag import LightRAGRAG

        calls = []

        class FakeLightRAG:
            def query_data(self, query, param):
                calls.append(param.mode)
                if param.mode == "hybrid":
                    return {"status": "failure", "data": {}}
                return {"status": "success", "data": {"chunks": [{"content": "useful"}]}}

        rag = LightRAGRAG.__new__(LightRAGRAG)
        rag.cfg = LightRAGConfig(mode="hybrid", fallback_mode="mix")
        rag._rag = FakeLightRAG()

        payload, effective_mode = rag._query_data_with_fallback("question")

        self.assertEqual(calls, ["hybrid", "mix"])
        self.assertEqual(effective_mode, "mix")
        self.assertEqual(payload["data"]["chunks"][0]["content"], "useful")

    def test_query_param_disables_references_for_qasper_short_answers(self):
        from Core.configs.rag.lightrag_config import LightRAGConfig
        from Core.rag.lightrag_rag import LightRAGRAG

        rag = LightRAGRAG.__new__(LightRAGRAG)
        rag.cfg = LightRAGConfig(answer_style="short")

        param = rag._query_param(mode="mix")

        self.assertFalse(param.include_references)
        self.assertEqual(param.response_type, "Single concise answer")

    def test_shorten_answer_removes_references_and_keeps_numbered_items(self):
        from Core.rag.lightrag_rag import LightRAGRAG

        answer = textwrap.dedent(
            """
            The three regularization terms are:
            1. A regularization term associated with neutral features.
            2. The maximum entropy of class distribution regularization term.
            3. The KL divergence between reference and predicted class distribution.

            ### References
            - [1] Irrelevant citation text.
            """
        ).strip()

        shortened = LightRAGRAG._shorten_answer(answer)

        self.assertIn("neutral features", shortened)
        self.assertIn("maximum entropy", shortened)
        self.assertIn("KL divergence", shortened)
        self.assertNotIn("References", shortened)
        self.assertNotEqual(shortened, "The three regularization terms are:")

    def test_supporting_evidence_uses_configured_topk_and_keeps_traceable_items(self):
        from Core.configs.rag.lightrag_config import LightRAGConfig
        from Core.rag.lightrag_rag import LightRAGRAG

        rag = LightRAGRAG.__new__(LightRAGRAG)
        rag.cfg = LightRAGConfig(supporting_evidence_topk=2)
        ranked = [
            {
                "id": "node_1",
                "content": "First paragraph.",
                "metadata": {
                    "source_node_id": 1,
                    "qasper_evidence_text": "First paragraph.",
                    "node_type": "paragraph",
                },
            },
            {
                "id": "node_2",
                "content": "Second paragraph.",
                "metadata": {
                    "source_node_id": 2,
                    "qasper_evidence_text": "Second paragraph.",
                    "node_type": "paragraph",
                },
            },
            {
                "id": "node_3",
                "content": "Third paragraph.",
                "metadata": {
                    "source_node_id": 3,
                    "qasper_evidence_text": "Third paragraph.",
                    "node_type": "paragraph",
                },
            },
        ]

        supporting = rag._supporting_evidence_from_ranked(ranked)

        self.assertEqual([item["source_node_id"] for item in supporting], [1, 2])
        self.assertEqual([item["supporting_rank"] for item in supporting], [1, 2])
        self.assertEqual(supporting[0]["qasper_evidence_text"], "First paragraph.")

    def test_supporting_evidence_prefers_answer_overlap_before_truncation(self):
        from Core.configs.rag.lightrag_config import LightRAGConfig
        from Core.rag.lightrag_rag import LightRAGRAG

        rag = LightRAGRAG.__new__(LightRAGRAG)
        rag.cfg = LightRAGConfig(supporting_evidence_topk=2)
        ranked = [
            {
                "id": "node_82",
                "content": "The maximum entropy regularization term shows the strong ability of controlling unbalance.",
                "metadata": {
                    "source_node_id": 82,
                    "qasper_evidence_text": "The maximum entropy regularization term shows the strong ability of controlling unbalance.",
                    "node_type": "paragraph",
                },
            },
            {
                "id": "node_11",
                "content": "The rest of the paper describes the proposed regularization terms.",
                "metadata": {
                    "source_node_id": 11,
                    "qasper_evidence_text": "The rest of the paper describes the proposed regularization terms.",
                    "node_type": "paragraph",
                },
            },
            {
                "id": "node_9",
                "content": (
                    "More specifically, we explore three regularization terms: a regularization term "
                    "associated with neutral features, the maximum entropy of class distribution "
                    "regularization term, and the KL divergence between reference and predicted class distribution."
                ),
                "metadata": {
                    "source_node_id": 9,
                    "qasper_evidence_text": (
                        "More specifically, we explore three regularization terms: a regularization term "
                        "associated with neutral features, the maximum entropy of class distribution "
                        "regularization term, and the KL divergence between reference and predicted class distribution."
                    ),
                    "node_type": "paragraph",
                },
            },
        ]

        supporting = rag._supporting_evidence_from_ranked(
            ranked,
            answer_text=(
                "A regularization term associated with neutral features; "
                "the maximum entropy of class distribution regularization term; "
                "the KL divergence between reference and predicted class distribution."
            ),
        )

        self.assertEqual(supporting[0]["source_node_id"], 9)
        self.assertIn(9, [item["source_node_id"] for item in supporting])

    def test_storage_diagnostics_report_empty_lightrag_graph(self):
        from Core.rag.lightrag_rag import LightRAGRAG

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "vdb_chunks.json").write_text(
                json.dumps({"data": [{"id": "chunk-1"}]}),
                encoding="utf-8",
            )
            (workdir / "vdb_entities.json").write_text(
                json.dumps({"data": []}),
                encoding="utf-8",
            )
            (workdir / "vdb_relationships.json").write_text(
                json.dumps({"data": []}),
                encoding="utf-8",
            )
            (workdir / "graph_chunk_entity_relation.graphml").write_text(
                '<graphml><graph edgedefault="undirected"></graph></graphml>',
                encoding="utf-8",
            )

            stats = LightRAGRAG._storage_diagnostics(str(workdir))

        self.assertEqual(stats["chunks"], 1)
        self.assertEqual(stats["entities"], 0)
        self.assertEqual(stats["relationships"], 0)
        self.assertFalse(stats["graph_ready"])

    def test_custom_chunk_insert_merges_extracted_entities_into_lightrag_graph(self):
        from Core.configs.rag.lightrag_config import LightRAGConfig
        from Core.rag.lightrag_rag import LightRAGRAG

        class FakeStorage:
            def __init__(self):
                self.upserted = None

            async def filter_keys(self, keys):
                return set(keys)

            async def upsert(self, payload):
                self.upserted = payload

        class FakeLightRAG:
            def __init__(self):
                self.chunks_vdb = FakeStorage()
                self.full_docs = FakeStorage()
                self.text_chunks = FakeStorage()
                self.chunk_entity_relation_graph = object()
                self.entities_vdb = object()
                self.relationships_vdb = object()
                self.full_entities = object()
                self.full_relations = object()
                self.llm_response_cache = object()
                self.entity_chunks = object()
                self.relation_chunks = object()
                self.insert_done = False

            async def _process_extract_entities(self, inserting_chunks):
                self.extracted_chunks = inserting_chunks
                return [({"Entity": [{"entity_name": "Entity"}]}, {})]

            def _build_global_config(self):
                return {"tokenizer": "fake"}

            async def _insert_done_with_cleanup(self):
                self.insert_done = True

        fake_rag = FakeLightRAG()
        agent = LightRAGRAG.__new__(LightRAGRAG)
        agent.cfg = LightRAGConfig()
        agent._rag = fake_rag

        with patch("Core.rag.lightrag_rag.merge_nodes_and_edges", new_callable=AsyncMock) as merge_mock:
            agent._run_sync(
                agent._ainsert_custom_chunks_with_graph_merge(
                    full_text="Full document text.",
                    text_chunks=["First paragraph."],
                    doc_id="document_tree",
                )
            )

        self.assertTrue(fake_rag.insert_done)
        self.assertIsNotNone(fake_rag.chunks_vdb.upserted)
        merge_mock.assert_awaited_once()
        kwargs = merge_mock.await_args.kwargs
        self.assertEqual(kwargs["chunk_results"], [({"Entity": [{"entity_name": "Entity"}]}, {})])
        self.assertIs(kwargs["knowledge_graph_inst"], fake_rag.chunk_entity_relation_graph)
        self.assertIs(kwargs["entity_vdb"], fake_rag.entities_vdb)
        self.assertIs(kwargs["relationships_vdb"], fake_rag.relationships_vdb)

    def test_force_custom_chunk_insert_bypasses_stale_existing_doc_filter(self):
        from Core.configs.rag.lightrag_config import LightRAGConfig
        from Core.rag.lightrag_rag import LightRAGRAG

        class StaleFilterStorage:
            def __init__(self):
                self.upserted = None

            async def filter_keys(self, keys):
                return set()

            async def upsert(self, payload):
                self.upserted = payload

        class FakeLightRAG:
            def __init__(self):
                self.chunks_vdb = StaleFilterStorage()
                self.full_docs = StaleFilterStorage()
                self.text_chunks = StaleFilterStorage()
                self.chunk_entity_relation_graph = object()
                self.entities_vdb = object()
                self.relationships_vdb = object()

            async def _process_extract_entities(self, inserting_chunks):
                return [({}, {})]

            def _build_global_config(self):
                return {}

            async def _insert_done_with_cleanup(self):
                pass

        fake_rag = FakeLightRAG()
        agent = LightRAGRAG.__new__(LightRAGRAG)
        agent.cfg = LightRAGConfig()
        agent._rag = fake_rag

        with patch("Core.rag.lightrag_rag.merge_nodes_and_edges", new_callable=AsyncMock):
            agent._run_sync(
                agent._ainsert_custom_chunks_with_graph_merge(
                    full_text="Full document text.",
                    text_chunks=["First paragraph."],
                    doc_id="document_tree",
                    force_insert=True,
                )
            )

        self.assertIsNotNone(fake_rag.full_docs.upserted)
        self.assertIsNotNone(fake_rag.text_chunks.upserted)
        self.assertIsNotNone(fake_rag.chunks_vdb.upserted)

    def test_repair_empty_graph_also_rebuilds_when_manifest_has_no_chunks(self):
        from Core.configs.rag.lightrag_config import LightRAGConfig
        from Core.rag.lightrag_rag import LightRAGRAG

        agent = LightRAGRAG.__new__(LightRAGRAG)
        agent.cfg = LightRAGConfig(mode="hybrid", enable_graph_merge=True, repair_empty_graph=True)

        self.assertTrue(agent._should_repair_empty_graph({"chunks": 0, "graph_ready": False}))


if __name__ == "__main__":
    unittest.main()
