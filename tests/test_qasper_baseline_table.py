import json
import tempfile
import unittest
from pathlib import Path


class QasperBaselineTableTests(unittest.TestCase):
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

            rows = collect_rows([str(eval_path)])

        self.assertEqual(rows[0]["Method"], "dense")
        self.assertEqual(rows[0]["Answer F1"], 0.42)
        self.assertEqual(rows[0]["Evidence F1"], 0.18)
        self.assertEqual(rows[0]["Missing"], 0)

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
