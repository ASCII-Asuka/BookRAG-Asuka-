import json
import tempfile
import unittest
from pathlib import Path


class FakeSplit:
    def __init__(self, rows):
        self._rows = rows

    def to_list(self):
        return list(self._rows)


class HotpotQADownloadTests(unittest.TestCase):
    def test_write_hotpotqa_splits_saves_json_and_manifest(self):
        from Scripts.preprocess.download_hotpotqa import write_hotpotqa_splits

        dataset = {
            "train": FakeSplit(
                [
                    {
                        "id": "train-1",
                        "question": "Question?",
                        "answer": "Answer",
                        "supporting_facts": {"title": ["A"], "sent_id": [0]},
                    }
                ]
            ),
            "validation": [
                {
                    "id": "valid-1",
                    "question": "Valid question?",
                    "answer": "yes",
                    "supporting_facts": {"title": ["B"], "sent_id": [1]},
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_hotpotqa_splits(dataset, output_root=tmp)
            root = Path(tmp)
            validation_rows = json.loads(
                (root / "raw" / "hotpotqa_distractor_validation.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest_json = json.loads(
                (root / "raw" / "manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(validation_rows[0]["id"], "valid-1")
        self.assertEqual(manifest["train"]["question_count"], 1)
        self.assertTrue(manifest["validation"]["has_answers"])
        self.assertTrue(manifest["validation"]["has_supporting_facts"])
        self.assertEqual(
            Path(manifest_json["validation"]["path"]).name,
            "hotpotqa_distractor_validation.json",
        )


if __name__ == "__main__":
    unittest.main()
