import tempfile
import unittest
from pathlib import Path

from Core.configs.system_config import load_system_config


class SystemConfigCredentialTests(unittest.TestCase):
    def test_matching_openai_embeddings_inherit_llm_api_key(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(
                """
llm:
  backend: openai
  api_base: https://api.example.test/v1
  api_key: shared-secret
mineru:
  backend: pipeline
  method: auto
  lang: en
vdb:
  embedding_config:
    backend: openai
    api_base: https://api.example.test/v1
rag:
  strategy: vanilla
  retrieval_method: vanilla
  vdb_config:
    embedding_config:
      backend: openai
      api_base: https://api.example.test/v1
""".strip(),
                encoding="utf-8",
            )

            config = load_system_config(str(config_path))

            self.assertEqual(config.vdb.embedding_config.api_key, "shared-secret")
            self.assertEqual(
                config.rag.strategy_config.vdb_config.embedding_config.api_key,
                "shared-secret",
            )


if __name__ == "__main__":
    unittest.main()
