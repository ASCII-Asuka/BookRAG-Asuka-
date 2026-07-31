import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from Core.Index.EvidenceBridgeIndex import EvidenceBlock, EvidenceBridgeIndex
from Core.configs.rag.evibridge_config import EviBridgeRAGConfig


def _stub_runtime_imports():
    class FakeOpenAIClient:
        def __init__(self, *args, **kwargs):
            pass

    openai_module = types.ModuleType("openai")
    openai_module.OpenAI = FakeOpenAIClient
    tiktoken_module = types.ModuleType("tiktoken")
    tiktoken_module.Encoding = object
    tiktoken_module.get_encoding = lambda name: types.SimpleNamespace(
        encode=lambda text: str(text).split()
    )
    json_repair_module = types.ModuleType("json_repair")
    json_repair_module.repair_json = lambda json_str, return_objects=False: json_str
    sys.modules.setdefault("openai", openai_module)
    sys.modules.setdefault("ollama", types.ModuleType("ollama"))
    sys.modules.setdefault("tiktoken", tiktoken_module)
    sys.modules.setdefault("json_repair", json_repair_module)


class FakeRAG:
    name = "fake"
    last_retrieved_block_ids = [7, 8]

    def generation(self, query, query_output_dir):
        return "answer", [7, 8]

    def close(self):
        pass


class EviBridgeWiringTests(unittest.TestCase):
    def test_config_accepts_demand_aware_seed_ablation(self):
        cfg = EviBridgeRAGConfig(ablation_variant="wo_demand_aware_seed_recall")

        self.assertEqual(cfg.ablation_variant, "wo_demand_aware_seed_recall")

    def test_evibridge_ablation_config_inherits_runnable_base_config(self):
        from Core.configs.system_config import load_system_config

        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_system_config(str(repo_root / "config" / "evibridge_wo_context.yaml"))

        self.assertEqual(cfg.rag.strategy_config.ablation_variant, "wo_context_edges")
        self.assertEqual(cfg.mineru.backend, "pipeline")
        self.assertEqual(cfg.rag.strategy_config.method_suffix, "evibridge_wo_context_edges")

    def test_main_evibridge_config_enables_validated_evidence_reranking(self):
        from Core.configs.system_config import load_system_config

        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_system_config(str(repo_root / "config" / "evibridge.yaml"))
        rag = cfg.rag.strategy_config

        self.assertTrue(rag.enable_candidate_rerank)
        self.assertTrue(rag.enable_supporting_rerank)
        self.assertTrue(rag.trust_answer_supporting_ids)
        self.assertTrue(rag.dynamic_supporting_evidence_budget)
        self.assertFalse(rag.enable_short_answer_extraction)
        self.assertEqual(rag.candidate_rerank_topk, 50)

    def test_short_answer_extraction_is_opt_in(self):
        from Core.configs.rag.evibridge_config import EviBridgeRAGConfig

        self.assertFalse(EviBridgeRAGConfig().enable_short_answer_extraction)

    def test_construct_evibridge_index_builds_tree_then_evibridge_index(self):
        _stub_runtime_imports()
        from Core.construct_index import construct_evibridge_index

        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(save_path=tmp)
            tree = object()
            with patch("Core.construct_index.build_tree_from_pdf", return_value=tree) as build_tree, patch(
                "Core.pipelines.evibridge_builder.build_evibridge_index"
            ) as build_evibridge:
                construct_evibridge_index(cfg)

        build_tree.assert_called_once_with(cfg)
        build_evibridge.assert_called_once_with(tree_index=tree, cfg=cfg)

    def test_construct_evibridge_index_skips_existing_complete_index(self):
        _stub_runtime_imports()
        from Core.construct_index import construct_evibridge_index

        with tempfile.TemporaryDirectory() as tmp:
            index = EvidenceBridgeIndex(
                save_dir=tmp,
                blocks={
                    1: EvidenceBlock(
                        block_id=1,
                        block_type="paragraph",
                        text="retrieval evidence",
                    )
                },
            )
            bm25 = index.build_bm25()
            index.save_to_dir()
            index.save_bm25(bm25)
            cfg = SimpleNamespace(
                save_path=tmp,
                rag=SimpleNamespace(
                    strategy_config=EviBridgeRAGConfig(enable_vector_recall=False)
                ),
            )
            with patch("Core.construct_index.build_tree_from_pdf") as build_tree, patch(
                "Core.pipelines.evibridge_builder.build_evibridge_index"
            ) as build_evibridge:
                construct_evibridge_index(cfg)

        build_tree.assert_not_called()
        build_evibridge.assert_not_called()

    def test_resource_loader_loads_evibridge_index_and_bm25(self):
        _stub_runtime_imports()
        from Core.utils.resource_loader import prepare_rag_dependencies

        with tempfile.TemporaryDirectory() as tmp:
            index = EvidenceBridgeIndex(
                save_dir=tmp,
                blocks={
                    1: EvidenceBlock(
                        block_id=1,
                        block_type="paragraph",
                        text="retrieval evidence",
                    )
                },
            )
            bm25 = index.build_bm25()
            index.save_to_dir()
            index.save_bm25(bm25)
            cfg = SimpleNamespace(
                save_path=tmp,
                rag=SimpleNamespace(
                    strategy_config=EviBridgeRAGConfig(enable_vector_recall=False)
                ),
            )

            deps = prepare_rag_dependencies(cfg)

        self.assertIn("evibridge_index", deps)
        self.assertIn("bm25", deps)
        self.assertEqual(deps["evibridge_index"].blocks[1].text, "retrieval evidence")

    def test_resource_loader_loads_evibridge_reranker_when_candidate_rerank_enabled(self):
        _stub_runtime_imports()
        from Core.utils.resource_loader import prepare_rag_dependencies

        with tempfile.TemporaryDirectory() as tmp:
            index = EvidenceBridgeIndex(
                save_dir=tmp,
                blocks={
                    1: EvidenceBlock(
                        block_id=1,
                        block_type="paragraph",
                        text="retrieval evidence",
                    )
                },
            )
            bm25 = index.build_bm25()
            index.save_to_dir()
            index.save_bm25(bm25)
            cfg = SimpleNamespace(
                save_path=tmp,
                rag=SimpleNamespace(
                    strategy_config=EviBridgeRAGConfig(
                        enable_vector_recall=False,
                        enable_candidate_rerank=True,
                    )
                ),
            )
            with patch("Core.provider.rerank.TextRerankerProvider", return_value="reranker") as provider:
                deps = prepare_rag_dependencies(cfg)

        self.assertEqual(deps["reranker"], "reranker")
        provider.assert_called_once()

    def test_run_rag_preserves_node_ids_and_adds_block_ids(self):
        _stub_runtime_imports()
        from Core.inference import run_rag

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "data.json"
            dataset_path.write_text(
                json.dumps([{"question": "q", "answer": "a"}]),
                encoding="utf-8",
            )
            run_rag(
                rag_agent=FakeRAG(),
                output_dir=Path(tmp) / "out",
                dataset_path=str(dataset_path),
                force_reprocess=True,
            )
            result = json.loads(
                ((Path(tmp) / "out" / "query_001" / "result.json").read_text(encoding="utf-8"))
            )

        self.assertEqual(result["retrieved_node_ids"], [7, 8])
        self.assertEqual(result["retrieved_block_ids"], [7, 8])

    def test_main_accepts_evibridge_stage(self):
        _stub_runtime_imports()
        import main

        argv = [
            "main.py",
            "-c",
            "config/evibridge.yaml",
            "index",
            "--stage",
            "evibridge",
        ]
        with patch.object(sys, "argv", argv):
            args = main.create_args()

        self.assertEqual(args.stage, "evibridge")


if __name__ == "__main__":
    unittest.main()
