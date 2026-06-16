import unittest
from types import SimpleNamespace
from unittest.mock import patch

from Core.configs.rag.gbc_config import GBCRAGConfig
from Core.configs.rerank_config import RerankerConfig


class GBCWiringTests(unittest.TestCase):
    def test_gbc_rag_passes_reranker_api_credentials_and_retry_options(self):
        from Core.rag.gbc_rag import GBCRAG

        reranker_config = RerankerConfig(
            model_name="reranker-model",
            max_length=2048,
            device="cpu",
            backend="vllm",
            api_base="https://example.test/v1",
            api_key="secret-key",
            max_retries=5,
            retry_backoff=0.25,
            request_timeout=12.0,
        )
        cfg = GBCRAGConfig(reranker_config=reranker_config)
        gbc_index = SimpleNamespace(embedder="embedder")

        with patch("Core.rag.gbc_rag.TextRerankerProvider", return_value="reranker") as provider, patch(
            "Core.rag.gbc_rag.TaskPlanner"
        ), patch("Core.rag.gbc_rag.AnswerAgent"), patch("Core.rag.gbc_rag.Retriever"):
            GBCRAG(
                llm=object(),
                vlm=object(),
                config=cfg,
                gbc_index=gbc_index,
            )

        kwargs = provider.call_args.kwargs
        self.assertEqual(kwargs["model_name"], "reranker-model")
        self.assertEqual(kwargs["api_base"], "https://example.test/v1")
        self.assertEqual(kwargs["api_key"], "secret-key")
        self.assertEqual(kwargs["max_retries"], 5)
        self.assertEqual(kwargs["retry_backoff"], 0.25)
        self.assertEqual(kwargs["request_timeout"], 12.0)


if __name__ == "__main__":
    unittest.main()
