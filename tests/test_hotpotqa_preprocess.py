import json
import tempfile
import unittest
from pathlib import Path


class HotpotQAPreprocessTests(unittest.TestCase):
    def _raw_rows(self):
        return [
            {
                "id": "hotpot-1",
                "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
                "answer": "yes",
                "type": "comparison",
                "level": "hard",
                "supporting_facts": {
                    "title": ["Scott Derrickson", "Ed Wood"],
                    "sent_id": [0, 0],
                },
                "context": {
                    "title": ["Scott Derrickson", "Ed Wood", "Distractor"],
                    "sentences": [
                        [
                            "Scott Derrickson is an American director.",
                            "He directed Doctor Strange.",
                        ],
                        [
                            "Ed Wood was an American filmmaker.",
                        ],
                        [
                            "This sentence is irrelevant.",
                        ],
                    ],
                },
            },
            {
                "id": "hotpot-2",
                "question": "What position did Shirley Temple hold?",
                "answer": "Chief of Protocol",
                "type": "bridge",
                "level": "medium",
                "supporting_facts": {
                    "title": ["Kiss and Tell (1945 film)", "Shirley Temple"],
                    "sent_id": [0, 1],
                },
                "context": {
                    "title": ["Kiss and Tell (1945 film)", "Shirley Temple"],
                    "sentences": [
                        ["Kiss and Tell starred Shirley Temple as Corliss Archer."],
                        [
                            "Shirley Temple was an American actress.",
                            "She served as Chief of Protocol of the United States.",
                        ],
                    ],
                },
            },
        ]

    def test_converts_hotpotqa_rows_to_unified_dataset_and_sentence_evidence(self):
        from Scripts.preprocess.hotpotqa_evibridge import convert_hotpotqa_to_unified

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "hotpotqa_validation.json"
            output_path = tmp_path / "processed" / "hotpotqa_validation_2.json"
            raw_path.write_text(json.dumps(self._raw_rows()), encoding="utf-8")

            rows = convert_hotpotqa_to_unified(
                raw_path=str(raw_path),
                output_path=str(output_path),
                sample_size=2,
                seed=13,
            )
            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(len(rows), 2)
        self.assertEqual(saved[0]["doc_uuid"], "hotpot-1")
        self.assertEqual(saved[0]["doc_path"], "hotpotqa://distractor/validation/hotpot-1")
        self.assertEqual(saved[0]["hotpotqa_question_id"], "hotpot-1")
        self.assertEqual(saved[0]["answer"], "yes")
        self.assertEqual(saved[0]["hotpot_answer_type"], "yes_no")
        self.assertEqual(saved[0]["hotpot_supporting_facts"], [["Scott Derrickson", 0], ["Ed Wood", 0]])
        self.assertEqual(saved[0]["evidence_block_ids"], [2, 5])

    def test_builds_hotpotqa_document_trees_and_dataset_config(self):
        from Core.Index.Tree import DocumentTree
        from Scripts.preprocess.hotpotqa_evibridge import prepare_hotpotqa_sample

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "hotpotqa_validation.json"
            output_path = tmp_path / "processed" / "hotpotqa_validation_2.json"
            work_dir = tmp_path / "work"
            cfg_path = tmp_path / "config" / "hotpotqa_validation_2.yaml"
            raw_path.write_text(json.dumps(self._raw_rows()), encoding="utf-8")

            summary = prepare_hotpotqa_sample(
                raw_path=str(raw_path),
                output_path=str(output_path),
                working_dir=str(work_dir),
                dataset_config_path=str(cfg_path),
                sample_size=2,
                seed=42,
            )
            tree = DocumentTree.load_from_file(str(work_dir / "hotpot-1" / "tree.pkl"))
            cfg_text = cfg_path.read_text(encoding="utf-8")
            tree_json_exists = (work_dir / "hotpot-1" / "tree.json").exists()

        self.assertEqual(summary["document_count"], 2)
        self.assertEqual(summary["question_count"], 2)
        self.assertTrue(tree_json_exists)
        self.assertIn("dataset_name: hotpotqa", cfg_text)
        self.assertIn("hotpotqa_validation_2.json", cfg_text)
        contents = [node.meta_info.content for node in tree.nodes if node.meta_info.content]
        self.assertIn("Scott Derrickson", contents)
        self.assertIn("Scott Derrickson is an American director.", contents)


if __name__ == "__main__":
    unittest.main()
