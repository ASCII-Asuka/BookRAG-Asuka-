import tempfile
import unittest

from chromadb.api.client import SharedSystemClient

from Core.provider.vdb import VectorStore


class _FakeEmbedder:
    MM_EMBEDDER = False

    def embed_texts(self, texts):
        return []

    def close(self):
        return None


class VectorStoreCloseTests(unittest.TestCase):
    def test_close_releases_persistent_chroma_system(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = VectorStore(
                embedding_model=_FakeEmbedder(),
                db_path=temporary_directory,
                collection_name="close_regression",
            )
            identifier = store.client._identifier

            self.assertIn(identifier, SharedSystemClient._identifier_to_system)
            try:
                store.close()
                self.assertNotIn(identifier, SharedSystemClient._identifier_to_system)
            finally:
                system = SharedSystemClient._identifier_to_system.pop(identifier, None)
                SharedSystemClient._identifier_to_refcount.pop(identifier, None)
                if system is not None:
                    system.stop()


if __name__ == "__main__":
    unittest.main()
