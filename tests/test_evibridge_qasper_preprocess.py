import json
import tempfile
import unittest
from pathlib import Path


class EviBridgeQasperPreprocessTests(unittest.TestCase):
    def _raw_qasper(self):
        return {
            "paper-1": {
                "title": "Retrieval Reasoning Paper",
                "abstract": ["This paper studies retrieval reasoning."],
                "full_text": [
                    {
                        "section_name": "Introduction",
                        "paragraphs": [
                            "Retrieval reasoning connects evidence across paragraphs.",
                            "The method uses bridge evidence for question answering.",
                        ],
                    }
                ],
                "qas": [
                    {
                        "question_id": "q1",
                        "question": "What does retrieval reasoning connect?",
                        "answers": [
                            {
                                "answer": {
                                    "unanswerable": False,
                                    "extractive_spans": ["evidence across paragraphs"],
                                    "free_form_answer": "",
                                    "yes_no": None,
                                    "evidence": [
                                        "Retrieval reasoning connects evidence across paragraphs."
                                    ],
                                }
                            }
                        ],
                    }
                ],
            }
        }

    def _hf_qasper_rows(self):
        return [
            {
                "id": "paper-hf-1",
                "title": "HF Qasper Paper",
                "abstract": "This abstract introduces bridge retrieval.",
                "full_text": {
                    "section_name": ["Introduction", "Method"],
                    "paragraphs": [
                        [
                            "HF evidence paragraph supports the answer.",
                            "A nearby paragraph gives background.",
                        ],
                        ["The method section mentions entity bridges."],
                    ],
                },
                "qas": {
                    "question": ["What supports the answer?"],
                    "question_id": ["hf-q1"],
                    "answers": [
                        {
                            "answer": [
                                {
                                    "unanswerable": False,
                                    "extractive_spans": ["HF evidence paragraph"],
                                    "free_form_answer": "",
                                    "yes_no": None,
                                    "evidence": [
                                        "HF evidence paragraph supports the answer."
                                    ],
                                    "highlighted_evidence": [
                                        "HF evidence paragraph supports the answer."
                                    ],
                                }
                            ],
                            "annotation_id": ["ann-1"],
                            "worker_id": ["worker-1"],
                        }
                    ],
                },
            },
            {
                "id": "paper-hf-2",
                "title": "Second HF Paper",
                "abstract": "Second abstract.",
                "full_text": {
                    "section_name": ["Overview"],
                    "paragraphs": [["Second paper evidence."]],
                },
                "qas": {
                    "question": ["What is present?"],
                    "question_id": ["hf-q2"],
                    "answers": [
                        {
                            "answer": [
                                {
                                    "unanswerable": False,
                                    "extractive_spans": ["Second paper evidence"],
                                    "free_form_answer": "",
                                    "yes_no": None,
                                    "evidence": ["Second paper evidence."],
                                }
                            ]
                        }
                    ],
                },
            },
            {
                "id": "paper-hf-3",
                "title": "Third HF Paper",
                "abstract": "Third abstract.",
                "full_text": {
                    "section_name": ["Overview"],
                    "paragraphs": [["Third paper evidence."]],
                },
                "qas": {
                    "question": ["What is included?"],
                    "question_id": ["hf-q3"],
                    "answers": [
                        {
                            "answer": [
                                {
                                    "unanswerable": False,
                                    "extractive_spans": ["Third paper evidence"],
                                    "free_form_answer": "",
                                    "yes_no": None,
                                    "evidence": ["Third paper evidence."],
                                }
                            ]
                        }
                    ],
                },
            },
        ]

    def test_converts_qasper_raw_json_to_unified_dataset(self):
        from Scripts.preprocess.qasper_evibridge import convert_qasper_to_unified

        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "qasper.json"
            output_path = Path(tmp) / "unified.json"
            raw_path.write_text(json.dumps(self._raw_qasper()), encoding="utf-8")

            rows = convert_qasper_to_unified(
                raw_path=str(raw_path),
                output_path=str(output_path),
                pdf_dir=str(Path(tmp) / "pdfs"),
            )
            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(len(rows), 1)
        self.assertEqual(saved[0]["doc_uuid"], "paper-1")
        self.assertTrue(saved[0]["doc_path"].endswith("paper-1.pdf"))
        self.assertEqual(saved[0]["qasper_question_id"], "q1")
        self.assertEqual(saved[0]["answer"][0]["evidence_block_ids"], [4])

    def test_converts_huggingface_rows_to_unified_dataset(self):
        from Scripts.preprocess.qasper_evibridge import convert_qasper_to_unified

        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "qasper_validation.json"
            output_path = Path(tmp) / "qasper_validation_unified.json"
            raw_path.write_text(json.dumps(self._hf_qasper_rows()), encoding="utf-8")

            rows = convert_qasper_to_unified(
                raw_path=str(raw_path),
                output_path=str(output_path),
            )

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["doc_uuid"], "paper-hf-1")
        self.assertEqual(rows[0]["doc_path"], "qasper://paper-hf-1")
        self.assertEqual(rows[0]["qasper_question_id"], "hf-q1")
        self.assertEqual(rows[0]["answer"][0]["evidence_block_ids"], [4])
        self.assertEqual(rows[0]["evidence_block_ids"], [4])

    def test_qasper_tree_includes_abstract_and_can_be_saved(self):
        from Core.Index.Tree import DocumentTree
        from Scripts.preprocess.qasper_evibridge import qasper_paper_to_tree

        with tempfile.TemporaryDirectory() as tmp:
            tree, paragraph_lookup = qasper_paper_to_tree(
                paper_id="paper-hf-1",
                paper=self._hf_qasper_rows()[0],
                save_dir=tmp,
                doc_path="qasper://paper-hf-1",
            )
            tree.save_to_file()
            loaded_tree = DocumentTree.load_from_file(str(Path(tmp) / "tree.pkl"))

            node_contents = [
                node.meta_info.content for node in loaded_tree.nodes if node.meta_info.content
            ]
            self.assertIn("Abstract", node_contents)
            self.assertIn("This abstract introduces bridge retrieval.", node_contents)
            self.assertEqual(
                paragraph_lookup["hf evidence paragraph supports the answer."],
                4,
            )
            self.assertTrue((Path(tmp) / "tree.json").exists())

    def test_prepare_qasper_sample_writes_trees_and_configs(self):
        from Scripts.preprocess.qasper_evibridge import prepare_qasper_sample

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "qasper_validation.json"
            output_path = tmp_path / "processed" / "qasper_validation_sample_3docs.json"
            work_dir = tmp_path / "work"
            config_dir = tmp_path / "config"
            raw_path.write_text(json.dumps(self._hf_qasper_rows()), encoding="utf-8")

            summary = prepare_qasper_sample(
                raw_path=str(raw_path),
                output_path=str(output_path),
                working_dir=str(work_dir),
                sample_docs=3,
                dataset_config_path=str(config_dir / "qasper_validation_sample.yaml"),
                system_config_path=str(config_dir / "evibridge_qasper.yaml"),
            )

            saved_rows = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["paper_count"], 3)
            self.assertEqual(summary["question_count"], 3)
            self.assertEqual(len(saved_rows), 3)
            self.assertTrue((work_dir / "paper-hf-1" / "tree.pkl").exists())
            dataset_cfg = (config_dir / "qasper_validation_sample.yaml").read_text(
                encoding="utf-8"
            )
            self.assertIn("dataset_name: qasper", dataset_cfg)
            self.assertIn("qasper_validation_sample_3docs.json", dataset_cfg)
            system_cfg = (config_dir / "evibridge_qasper.yaml").read_text(encoding="utf-8")
            self.assertIn("strategy: evibridge", system_cfg)
            self.assertIn("enable_vector_recall: false", system_cfg)

    def test_builds_evibridge_index_from_qasper_paper(self):
        from Scripts.preprocess.qasper_evibridge import build_qasper_evibridge_index_from_paper

        with tempfile.TemporaryDirectory() as tmp:
            index = build_qasper_evibridge_index_from_paper(
                paper_id="paper-1",
                paper=self._raw_qasper()["paper-1"],
                save_dir=tmp,
                doc_path=str(Path(tmp) / "paper-1.pdf"),
            )

        block_types = {block.block_type for block in index.blocks.values()}
        self.assertIn("paragraph", block_types)
        self.assertIn("summary", block_types)
        self.assertIn("entity", block_types)
        self.assertTrue(
            any(
                bridge.bridge_type == "semantic"
                and bridge.relation_type in {"mentions_entity", "shared_terms"}
                for bridge in index.bridges
            )
        )


if __name__ == "__main__":
    unittest.main()
