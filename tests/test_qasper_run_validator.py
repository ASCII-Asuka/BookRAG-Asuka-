import json
import tempfile
import unittest
from pathlib import Path


class QasperRunValidatorTests(unittest.TestCase):
    def _write_dataset_and_results(self, root: Path):
        dataset_path = root / "dataset.json"
        working_dir = root / "work"
        result_dir = working_dir / "paper-1" / "eval_qasper_evibridge"
        result_dir.mkdir(parents=True)
        rows = [
            {
                "question": "Question 1?",
                "doc_uuid": "paper-1",
                "doc_path": "qasper://paper-1",
                "qasper_question_id": "q1",
                "answer": [],
            },
            {
                "question": "Question 2?",
                "doc_uuid": "paper-1",
                "doc_path": "qasper://paper-1",
                "qasper_question_id": "q2",
                "answer": [],
            },
        ]
        dataset_path.write_text(json.dumps(rows), encoding="utf-8")
        for idx, qid in enumerate(["q1", "q2"], start=1):
            query_dir = result_dir / f"query_{idx:03d}"
            query_dir.mkdir()
            (query_dir / "result.json").write_text(
                json.dumps({"question": f"Question {idx}?", "qasper_question_id": qid, "output": "answer"}),
                encoding="utf-8",
            )
        (result_dir / "final_results.json").write_text(
            json.dumps(
                [
                    {"question": "Question 1?", "qasper_question_id": "q1", "output": "answer"},
                    {"question": "Question 2?", "qasper_question_id": "q2", "output": "answer"},
                ]
            ),
            encoding="utf-8",
        )
        return dataset_path, working_dir

    def test_validator_passes_complete_run_and_reports_counts(self):
        from Scripts.eval.qasper_run_validator import validate_qasper_run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path, working_dir = self._write_dataset_and_results(root)
            predictions_path = root / "predictions.jsonl"
            predictions_path.write_text(
                "\n".join(
                    [
                        json.dumps({"question_id": "q1", "predicted_answer": "answer", "predicted_evidence": []}),
                        json.dumps({"question_id": "q2", "predicted_answer": "answer", "predicted_evidence": []}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = validate_qasper_run(
                dataset_path=str(dataset_path),
                working_dir=str(working_dir),
                dataset_name="qasper",
                method="evibridge",
                predictions_path=str(predictions_path),
            )

        self.assertEqual(summary["expected_questions"], 2)
        self.assertEqual(summary["final_results_questions"], 2)
        self.assertEqual(summary["predictions_questions"], 2)
        self.assertEqual(summary["issues"], [])

    def test_validator_rejects_partial_final_results(self):
        from Scripts.eval.qasper_run_validator import QasperRunValidationError, validate_qasper_run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path, working_dir = self._write_dataset_and_results(root)
            result_dir = working_dir / "paper-1" / "eval_qasper_evibridge"
            (result_dir / "final_results.json").write_text(
                json.dumps([{"question": "Question 1?", "qasper_question_id": "q1", "output": "answer"}]),
                encoding="utf-8",
            )

            with self.assertRaises(QasperRunValidationError) as ctx:
                validate_qasper_run(
                    dataset_path=str(dataset_path),
                    working_dir=str(working_dir),
                    dataset_name="qasper",
                    method="evibridge",
                )

        self.assertIn("paper-1", str(ctx.exception))
        self.assertIn("final_results", str(ctx.exception))

    def test_validator_rejects_prediction_question_id_mismatch(self):
        from Scripts.eval.qasper_run_validator import QasperRunValidationError, validate_qasper_run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path, working_dir = self._write_dataset_and_results(root)
            predictions_path = root / "predictions.jsonl"
            predictions_path.write_text(
                json.dumps({"question_id": "q1", "predicted_answer": "answer", "predicted_evidence": []}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(QasperRunValidationError) as ctx:
                validate_qasper_run(
                    dataset_path=str(dataset_path),
                    working_dir=str(working_dir),
                    dataset_name="qasper",
                    method="evibridge",
                    predictions_path=str(predictions_path),
                )

        self.assertIn("missing prediction question ids", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
