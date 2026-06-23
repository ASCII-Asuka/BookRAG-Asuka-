import json
import tempfile
import unittest
from pathlib import Path


class LightRAGDiagnosticsTests(unittest.TestCase):
    def test_collects_graph_and_fallback_counts_from_result_dir(self):
        from Scripts.eval.lightrag_diagnostics import collect_lightrag_diagnostics

        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "eval_qasper_lightrag"
            for index, graph_ready in enumerate([True, False], start=1):
                query_dir = result_dir / f"query_{index:03d}"
                query_dir.mkdir(parents=True)
                (query_dir / "retrieval_res.json").write_text(
                    json.dumps(
                        {
                            "mode": "hybrid",
                            "effective_mode": "hybrid" if graph_ready else "mix",
                            "fallback_used": not graph_ready,
                            "graph_diagnostics": {
                                "chunks": 10,
                                "entities": 5 if graph_ready else 0,
                                "relationships": 4 if graph_ready else 0,
                                "graph_nodes": 5 if graph_ready else 0,
                                "graph_ready": graph_ready,
                            },
                            "supporting_evidence": [{"source_node_id": 1}],
                        }
                    ),
                    encoding="utf-8",
                )
            (result_dir / "final_results.json").write_text(
                json.dumps([{"qasper_question_id": "q1"}, {"qasper_question_id": "q2"}]),
                encoding="utf-8",
            )

            summary = collect_lightrag_diagnostics([result_dir])

        self.assertEqual(summary["total_queries"], 2)
        self.assertEqual(summary["graph_ready_queries"], 1)
        self.assertEqual(summary["fallback_used_queries"], 1)
        self.assertEqual(summary["mode_counts"], {"hybrid": 1, "mix": 1})
        self.assertEqual(summary["avg_entities"], 2.5)
        self.assertEqual(summary["avg_supporting_evidence"], 1.0)

    def test_require_graph_ready_rejects_fallback_or_empty_graph(self):
        from Scripts.eval.lightrag_diagnostics import LightRAGDiagnosticError, validate_lightrag_graph_ready

        summary = {
            "total_queries": 2,
            "graph_ready_queries": 1,
            "fallback_used_queries": 1,
            "missing_retrieval_res_queries": 0,
        }

        with self.assertRaises(LightRAGDiagnosticError) as ctx:
            validate_lightrag_graph_ready(summary)

        self.assertIn("graph-ready", str(ctx.exception))
        self.assertIn("fallback", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
