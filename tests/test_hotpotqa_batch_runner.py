import json
import tempfile
import unittest
from pathlib import Path


class HotpotQABatchRunnerTests(unittest.TestCase):
    def test_missing_ids_detects_absent_final_results(self):
        from Scripts.eval.run_hotpotqa_missing_batches import missing_ids

        rows = [{"doc_uuid": "q1"}, {"doc_uuid": "q2"}, {"doc_uuid": "q3"}]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            done_dir = work / "q2" / "eval_hotpotqa_raptor"
            done_dir.mkdir(parents=True)
            (done_dir / "final_results.json").write_text("[]", encoding="utf-8")

            missing = missing_ids(rows, work, "eval_hotpotqa_raptor")

        self.assertEqual(missing, ["q1", "q3"])

    def test_write_batch_files_uses_shared_working_dir(self):
        from Scripts.eval.run_hotpotqa_missing_batches import write_batch_files

        rows = [
            {"doc_uuid": "q1", "question": "one"},
            {"doc_uuid": "q2", "question": "two"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path, cfg_path = write_batch_files(
                rows=rows,
                processed_dir=tmp_path / "processed",
                config_dir=tmp_path / "config",
                working_dir=tmp_path / "work",
                dataset_name="hotpotqa",
                label="raptor",
                iteration=3,
            )

            saved_rows = json.loads(data_path.read_text(encoding="utf-8"))
            config_text = cfg_path.read_text(encoding="utf-8")

        self.assertEqual(saved_rows, rows)
        self.assertIn("dataset_name: hotpotqa", config_text)
        self.assertIn("working_dir:", config_text)
        self.assertIn("hotpotqa_validation_random600_raptor_orch_batch_3.json", str(data_path))


if __name__ == "__main__":
    unittest.main()
