import json
import tempfile
import unittest
from pathlib import Path


class QasperBaselineTableTests(unittest.TestCase):
    @staticmethod
    def _write_manifests(eval_dir: Path, *, dynamic=True):
        (eval_dir / "coverage_manifest.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "expected_questions": 1,
                    "predictions_questions": 1,
                }
            ),
            encoding="utf-8",
        )
        (eval_dir / "export_summary.json").write_text(
            json.dumps(
                {
                    "answer_source": "output",
                    "paragraph_evidence_only": True,
                    "top_k_evidence": 0,
                    "dynamic_evidence_topk": dynamic,
                }
            ),
            encoding="utf-8",
        )

    def test_collect_rows_reads_official_eval_metrics(self):
        from Scripts.eval.qasper_baseline_table import collect_rows

        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = Path(tmp) / "qasper_official_dense_output"
            eval_dir.mkdir()
            eval_path = eval_dir / "official_eval.json"
            eval_path.write_text(
                json.dumps(
                    {
                        "Answer F1": 0.42,
                        "Evidence F1": 0.18,
                        "Missing predictions": 0,
                    }
                ),
                encoding="utf-8",
            )
            self._write_manifests(eval_dir)

            rows = collect_rows([str(eval_path)])

        self.assertEqual(rows[0]["Method"], "dense")
        self.assertEqual(rows[0]["Answer F1"], 0.42)
        self.assertEqual(rows[0]["Evidence F1"], 0.18)
        self.assertEqual(rows[0]["Missing"], 0)

    def test_collect_rows_rejects_missing_predictions_by_default(self):
        from Scripts.eval.qasper_baseline_table import collect_rows

        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = Path(tmp) / "qasper_official_partial_output"
            eval_dir.mkdir()
            eval_path = eval_dir / "official_eval.json"
            eval_path.write_text(
                json.dumps(
                    {
                        "Answer F1": 0.1,
                        "Evidence F1": 0.2,
                        "Missing predictions": 1,
                    }
                ),
                encoding="utf-8",
            )
            self._write_manifests(eval_dir)

            with self.assertRaises(ValueError):
                collect_rows([str(eval_path)])

            rows = collect_rows([str(eval_path)], allow_missing=True)

        self.assertEqual(rows[0]["Missing"], 1)

    def test_collect_rows_rejects_nonuniform_evidence_policies(self):
        from Scripts.eval.qasper_baseline_table import collect_rows

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for method, dynamic in [("dense", True), ("evibridge", False)]:
                eval_dir = Path(tmp) / f"qasper_official_{method}"
                eval_dir.mkdir()
                eval_path = eval_dir / "official_eval.json"
                eval_path.write_text(
                    json.dumps(
                        {
                            "Answer F1": 0.4,
                            "Evidence F1": 0.2,
                            "Missing predictions": 0,
                        }
                    ),
                    encoding="utf-8",
                )
                self._write_manifests(eval_dir, dynamic=dynamic)
                paths.append(str(eval_path))

            with self.assertRaisesRegex(ValueError, "uniform export policy"):
                collect_rows(paths)

    def test_collect_rows_requires_dynamic_evidence_policy_for_paper_table(self):
        from Scripts.eval.qasper_baseline_table import collect_rows

        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = Path(tmp) / "qasper_official_legacy"
            eval_dir.mkdir()
            eval_path = eval_dir / "official_eval.json"
            eval_path.write_text(
                json.dumps(
                    {
                        "Answer F1": 0.4,
                        "Evidence F1": 0.2,
                        "Missing predictions": 0,
                    }
                ),
                encoding="utf-8",
            )
            self._write_manifests(eval_dir, dynamic=False)

            with self.assertRaisesRegex(ValueError, "dynamic_evidence_topk"):
                collect_rows([str(eval_path)])

    def test_format_markdown_table(self):
        from Scripts.eval.qasper_baseline_table import format_markdown_table

        table = format_markdown_table(
            [
                {
                    "Method": "EviBridge-RAG",
                    "Answer F1": 0.4298,
                    "Evidence F1": 0.182,
                    "Missing": 0,
                }
            ]
        )

        self.assertIn("| EviBridge-RAG | 0.4298 | 0.1820 | 0 |", table)


if __name__ == "__main__":
    unittest.main()
