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
        from Core.Index.EvidenceBridgeIndex import EvidenceBridgeIndex
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
            index = EvidenceBridgeIndex.from_tree(
                tree,
                save_dir=str(work_dir / "hotpot-1"),
            )
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
        self.assertEqual(index.blocks[2].metadata["hotpot_title"], "Scott Derrickson")
        self.assertEqual(index.blocks[2].metadata["hotpot_sent_id"], 0)
        self.assertEqual(index.blocks[2].metadata["source"], "hotpotqa_sentence")

    def test_prepares_exclusive_random_subset_with_report(self):
        from Scripts.preprocess.hotpotqa_evibridge import prepare_hotpotqa_exclusive_sample

        raw_rows = self._raw_rows() + [
            {
                "id": "hotpot-3",
                "question": "Who wrote the book?",
                "answer": "Ada",
                "type": "bridge",
                "level": "hard",
                "supporting_facts": {
                    "title": ["Book", "Ada"],
                    "sent_id": [0, 0],
                },
                "context": {
                    "title": ["Book", "Ada", "Distractor"],
                    "sentences": [
                        ["The book was written by Ada."],
                        ["Ada was a writer."],
                        ["Noise."],
                    ],
                },
            },
            {
                "id": "hotpot-4",
                "question": "Where was the singer born?",
                "answer": "Paris",
                "type": "bridge",
                "level": "hard",
                "supporting_facts": {
                    "title": ["Singer", "Paris"],
                    "sent_id": [0, 0],
                },
                "context": {
                    "title": ["Singer", "Paris"],
                    "sentences": [
                        ["The singer was born in Paris."],
                        ["Paris is a city."],
                    ],
                },
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "hotpotqa_validation.json"
            exclude_path = tmp_path / "processed" / "done.json"
            output_path = tmp_path / "processed" / "random2.json"
            work_dir = tmp_path / "work"
            cfg_path = tmp_path / "config" / "random2.yaml"
            report_path = tmp_path / "processed" / "random2_report.md"
            raw_path.write_text(json.dumps(raw_rows), encoding="utf-8")
            exclude_path.parent.mkdir(parents=True)
            exclude_path.write_text(
                json.dumps([{"hotpotqa_question_id": "hotpot-1"}, {"doc_uuid": "hotpot-2"}]),
                encoding="utf-8",
            )

            summary = prepare_hotpotqa_exclusive_sample(
                raw_path=str(raw_path),
                output_path=str(output_path),
                working_dir=str(work_dir),
                dataset_config_path=str(cfg_path),
                exclude_dataset_paths=[str(exclude_path)],
                sample_size=2,
                seed=42,
                report_path=str(report_path),
            )
            rows = json.loads(output_path.read_text(encoding="utf-8"))
            manifest = json.loads(
                output_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
            )
            cfg_text = cfg_path.read_text(encoding="utf-8")
            report_text = report_path.read_text(encoding="utf-8")
            tree_exists = (work_dir / "hotpot-3" / "tree.pkl").exists()

        self.assertEqual(summary["question_count"], 2)
        self.assertEqual(summary["excluded_count"], 2)
        self.assertEqual({row["hotpotqa_question_id"] for row in rows}, {"hotpot-3", "hotpot-4"})
        self.assertEqual(manifest["seed"], 42)
        self.assertEqual(manifest["selected_question_ids"], ["hotpot-3", "hotpot-4"])
        self.assertEqual(len(manifest["dataset_sha256"]), 64)
        self.assertTrue(tree_exists)
        self.assertIn("dataset_name: hotpotqa", cfg_text)
        self.assertIn("Selected questions: 2", report_text)
        self.assertIn("Excluded completed questions: 2", report_text)


if __name__ == "__main__":
    unittest.main()
