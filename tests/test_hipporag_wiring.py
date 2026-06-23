import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class HippoRAGWiringTests(unittest.TestCase):
    def test_system_config_accepts_hipporag_strategy(self):
        from Core.configs.system_config import load_system_config

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hipporag.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    pdf_path: TODO
                    save_path: TODO
                    index_type: hipporag
                    mineru:
                      backend: pipeline
                      method: auto
                      lang: en
                    rag:
                      strategy: hipporag
                      topk: 7
                      bm25_topk: 11
                      ppr_alpha: 0.73
                      answer_style: short
                    """
                ).strip(),
                encoding="utf-8",
            )

            cfg = load_system_config(str(config_path))

        self.assertEqual(cfg.rag.strategy_config.strategy, "hipporag")
        self.assertEqual(cfg.rag.strategy_config.topk, 7)
        self.assertEqual(cfg.rag.strategy_config.bm25_topk, 11)
        self.assertAlmostEqual(cfg.rag.strategy_config.ppr_alpha, 0.73)
        self.assertEqual(cfg.rag.strategy_config.answer_style, "short")

    def test_create_rag_agent_dispatches_hipporag(self):
        from Core.configs.llm_config import LLMConfig
        from Core.configs.rag.hipporag_config import HippoRAGConfig
        from Core.configs.vlm_config import VLMConfig
        from Core.rag import create_rag_agent

        tree_index = SimpleNamespace()

        with patch("Core.rag.HippoRAGRAG", create=True) as agent_cls, patch("Core.provider.llm.LLM"), patch(
            "Core.provider.vlm.VLM"
        ):
            create_rag_agent(
                strategy_config=HippoRAGConfig(),
                llm_config=LLMConfig(),
                vlm_config=VLMConfig(),
                tree_index=tree_index,
                save_path="doc-work",
            )

        kwargs = agent_cls.call_args.kwargs
        self.assertEqual(kwargs["config"].strategy, "hipporag")
        self.assertIs(kwargs["tree_index"], tree_index)
        self.assertEqual(kwargs["save_path"], "doc-work")

    def test_resource_loader_provides_tree_and_save_path_for_hipporag(self):
        from Core.configs.mineru_config import MinerU
        from Core.configs.rag.hipporag_config import HippoRAGConfig
        from Core.configs.rag_config import RAGConfig
        from Core.configs.system_config import SystemConfig
        from Core.utils.resource_loader import prepare_rag_dependencies

        cfg = SystemConfig(
            save_path="doc-work",
            mineru=MinerU(backend="pipeline", method="auto", lang="en"),
            rag=RAGConfig(strategy_config=HippoRAGConfig()),
        )

        with patch("Core.Index.Tree.DocumentTree.get_save_path", return_value="doc-work/tree.pkl"), patch(
            "Core.Index.Tree.DocumentTree.load_from_file", return_value="tree"
        ):
            deps = prepare_rag_dependencies(cfg)

        self.assertEqual(deps["tree_index"], "tree")
        self.assertEqual(deps["save_path"], "doc-work")


if __name__ == "__main__":
    unittest.main()
