import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class FakeSplit:
    def __init__(self, rows):
        self._rows = rows

    def to_list(self):
        return list(self._rows)


class QasperDownloadTests(unittest.TestCase):
    def test_write_qasper_splits_saves_json_and_manifest(self):
        from Scripts.preprocess.download_qasper import write_qasper_splits

        dataset = {
            "train": FakeSplit(
                [
                    {
                        "id": "paper-train",
                        "qas": {
                            "question": ["Question 1?", "Question 2?"],
                            "answers": [
                                {"answer": [{"evidence": ["Paragraph one."]}]},
                                {"answer": [{"evidence": ["Paragraph two."]}]},
                            ],
                        },
                    }
                ]
            ),
            "validation": [
                {
                    "id": "paper-valid",
                    "qas": {
                        "question": ["Question 3?"],
                        "answers": [{"answer": [{"evidence": ["Valid paragraph."]}]}],
                    },
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_qasper_splits(dataset, output_root=tmp)
            output_root = Path(tmp)
            train_rows = json.loads(
                (output_root / "raw" / "qasper_train.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest_json = json.loads(
                (output_root / "raw" / "manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(train_rows[0]["id"], "paper-train")
        self.assertEqual(manifest["train"]["paper_count"], 1)
        self.assertEqual(manifest["train"]["question_count"], 2)
        self.assertTrue(manifest["train"]["has_answers"])
        self.assertTrue(manifest["train"]["has_evidence"])
        self.assertEqual(manifest_json["validation"]["question_count"], 1)

    def test_download_qasper_falls_back_to_converted_parquet_branch(self):
        from Scripts.preprocess import download_qasper

        fake_datasets = types.ModuleType("datasets")

        def broken_load_dataset(*args, **kwargs):
            raise RuntimeError("Dataset scripts are no longer supported")

        fake_datasets.load_dataset = broken_load_dataset
        fallback_dataset = {
            "validation": [
                {
                    "id": "paper-valid",
                    "qas": {
                        "question": ["Question?"],
                        "answers": [{"answer": [{"evidence": ["Evidence."]}]}],
                    },
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(sys.modules, {"datasets": fake_datasets}):
                with patch.object(
                    download_qasper,
                    "_download_parquet_splits",
                    return_value=fallback_dataset,
                ):
                    manifest = download_qasper.download_qasper(output_root=tmp)

        self.assertEqual(manifest["validation"]["paper_count"], 1)
        self.assertEqual(manifest["validation"]["question_count"], 1)


if __name__ == "__main__":
    unittest.main()
