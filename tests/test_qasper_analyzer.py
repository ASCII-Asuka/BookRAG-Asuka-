import json
import tempfile
import unittest
from pathlib import Path


class QasperAnalyzerTests(unittest.TestCase):
    def test_analyzer_reports_topk_curve_and_method_comparison(self):
        from Scripts.eval.analyze_qasper_evibridge import analyze_qasper_runs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold_path = root / "gold.json"
            bm25_dir = root / "bm25"
            evibridge_dir = root / "evibridge"
            bm25_dir.mkdir()
            evibridge_dir.mkdir()
            gold_path.write_text(
                json.dumps(
                    {
                        "paper-1": {
                            "title": "Paper",
                            "abstract": "",
                            "full_text": [],
                            "figures_and_tables": [],
                            "qas": [
                                {
                                    "question_id": "q1",
                                    "question": "What is the answer?",
                                    "answers": [
                                        {
                                            "answer": {
                                                "unanswerable": False,
                                                "extractive_spans": ["The answer"],
                                                "free_form_answer": "",
                                                "yes_no": None,
                                                "evidence": ["Gold paragraph."],
                                            }
                                        }
                                    ],
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            (bm25_dir / "predictions.jsonl").write_text(
                json.dumps(
                    {
                        "question_id": "q1",
                        "predicted_answer": "wrong",
                        "predicted_evidence": ["Wrong paragraph."],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (evibridge_dir / "predictions.jsonl").write_text(
                json.dumps(
                    {
                        "question_id": "q1",
                        "predicted_answer": "The answer",
                        "predicted_evidence": ["Gold paragraph.", "Wrong paragraph."],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = analyze_qasper_runs(
                gold_path=str(gold_path),
                bm25_predictions_path=str(bm25_dir / "predictions.jsonl"),
                evibridge_predictions_path=str(evibridge_dir / "predictions.jsonl"),
                topk_values=[1, 2],
            )

        self.assertEqual(report["overall"]["num_questions"], 1)
        self.assertGreater(report["overall"]["evibridge"]["Answer F1"], report["overall"]["bm25"]["Answer F1"])
        self.assertEqual(report["topk_curve"]["evibridge"]["1"]["Evidence F1"], 1.0)
        self.assertEqual(report["comparison"]["answer_wins"]["evibridge"], 1)


if __name__ == "__main__":
    unittest.main()
