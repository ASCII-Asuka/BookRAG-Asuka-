import builtins
import unittest


class RaptorUtilsTests(unittest.TestCase):
    def test_gmm_cluster_falls_back_when_umap_is_unavailable(self):
        import numpy as np
        from Core.utils.raptor_utils import GMM_cluster

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "umap":
                raise ImportError("No module named 'umap'")
            return original_import(name, *args, **kwargs)

        embeddings = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 0.9],
                [4.0, 4.1, 0.0],
                [4.1, 4.0, 0.0],
                [8.0, 0.0, 0.0],
                [8.1, 0.1, 0.0],
            ]
        )
        builtins.__import__ = fake_import
        try:
            labels, n_clusters = GMM_cluster(embeddings, dim=2)
        finally:
            builtins.__import__ = original_import

        self.assertEqual(len(labels), len(embeddings))
        self.assertGreaterEqual(n_clusters, 1)
        self.assertTrue(all(len(label) >= 1 for label in labels))


if __name__ == "__main__":
    unittest.main()
