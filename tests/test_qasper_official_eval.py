import json
import tempfile
import unittest
from pathlib import Path


class QasperOfficialEvalTests(unittest.TestCase):
    def test_evidence_recall_at_ks_uses_one_annotation_chosen_at_largest_k(self):
        from Scripts.eval.qasper_official import evidence_recall_at_ks

        recalls = evidence_recall_at_ks(
            ranked_evidence=["a", "x", "b", "c"],
            references=[
                {"evidence": ["a", "z"]},
                {"evidence": ["b", "c"]},
            ],
            ks=(1, 2, 4),
        )

        self.assertEqual(recalls, {1: 0.0, 2: 0.0, 4: 1.0})

    def test_official_eval_reports_recall_at_k_when_ranked_evidence_is_present(self):
        from Scripts.eval.qasper_official import evaluate_qasper_official

        gold = {
            "q1": [
                {
                    "answer": "answer",
                    "type": "extractive",
                    "evidence": ["p1", "p2"],
                }
            ]
        }
        predicted = {
            "q1": {
                "answer": "answer",
                "evidence": ["p1"],
                "ranked_evidence": ["p1", "noise", "p2"],
            }
        }

        scores = evaluate_qasper_official(gold, predicted)

        self.assertEqual(scores["Evidence Recall@5"], 1.0)
        self.assertEqual(scores["Evidence Recall@10"], 1.0)
        self.assertEqual(scores["Evidence Recall@20"], 1.0)

    def test_official_eval_reports_precision_and_recall_from_first_best_f1_reference(self):
        from Scripts.eval.qasper_official import evaluate_qasper_official

        gold = {
            "q1": [
                {
                    "answer": "answer",
                    "type": "extractive",
                    "evidence": ["p1"],
                },
                {
                    "answer": "answer",
                    "type": "extractive",
                    "evidence": ["p1", "p2", "p3", "p4"],
                },
            ]
        }
        predicted = {
            "q1": {
                "answer": "answer",
                "evidence": ["p1", "p2"],
            }
        }

        scores = evaluate_qasper_official(gold, predicted)

        self.assertEqual(scores["Evidence Precision"], 0.5)
        self.assertEqual(scores["Evidence Recall"], 1.0)
        self.assertEqual(scores["Evidence F1"], 0.666667)

    def test_official_eval_details_expose_per_question_prf_for_bootstrap(self):
        from Scripts.eval.qasper_official import evaluate_qasper_official_details

        gold = {
            "q1": [
                {
                    "answer": "answer",
                    "type": "extractive",
                    "evidence": ["p1"],
                }
            ]
        }
        predicted = {"q1": {"answer": "answer", "evidence": ["p1", "p2"]}}

        details = evaluate_qasper_official_details(gold, predicted)

        self.assertEqual(
            details,
            [
                {
                    "question_id": "q1",
                    "answer_f1": 1.0,
                    "answer_type": "extractive",
                    "evidence_precision": 0.5,
                    "evidence_recall": 1.0,
                    "evidence_f1": 0.666667,
                    "missing_prediction": False,
                }
            ],
        )

    def _write_sample_outputs(self, tmp: str):
        root = Path(tmp)
        dataset_path = root / "dataset.json"
        working_dir = root / "work"
        result_dir = working_dir / "paper-1" / "eval_qasper_evibridge"
        query_dir = result_dir / "query_001"
        query_dir.mkdir(parents=True)

        rows = [
            {
                "question": "What is the answer?",
                "answer": [
                    {
                        "unanswerable": False,
                        "extractive_spans": ["The answer"],
                        "free_form_answer": "",
                        "yes_no": None,
                        "evidence": ["Gold evidence paragraph."],
                    }
                ],
                "doc_uuid": "paper-1",
                "doc_path": "qasper://paper-1",
                "qasper_question_id": "q1",
            }
        ]
        dataset_path.write_text(json.dumps(rows), encoding="utf-8")
        (result_dir / "final_results.json").write_text(
            json.dumps(
                [
                    {
                        "question": "What is the answer?",
                        "qasper_question_id": "q1",
                        "output": "The answer with a long explanation that should not be used when answer_short exists.",
                        "answer_short": "The answer",
                    }
                ]
            ),
            encoding="utf-8",
        )
        (query_dir / "retrieval_res.json").write_text(
            json.dumps(
                {
                    "selected": [
                        {
                            "block_id": 9,
                            "block_type": "summary",
                            "text": "Summary text should not be exported by default.",
                            "selection_rank": 1,
                        },
                        {
                            "block_id": 7,
                            "block_type": "paragraph",
                            "text": "Gold evidence paragraph.",
                            "selection_rank": 2,
                        },
                        {
                            "block_id": 8,
                            "block_type": "paragraph",
                            "text": "Later paragraph.",
                            "selection_rank": 3,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        (query_dir / "evidence_chain.json").write_text(
            json.dumps(
                {
                    "evidence_chain": [
                        {
                            "block_id": 7,
                            "block_type": "paragraph",
                            "text": "Gold evidence paragraph.",
                        },
                        {
                            "block_id": 8,
                            "block_type": "patch",
                            "text": "Patch text should not be exported by default.",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        return dataset_path, working_dir

    def test_exports_predictions_jsonl_in_official_qasper_format(self):
        from Scripts.eval.qasper_official import export_predictions

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path, working_dir = self._write_sample_outputs(tmp)
            output_path = Path(tmp) / "official" / "predictions.jsonl"

            predictions = export_predictions(
                dataset_path=str(dataset_path),
                working_dir=str(working_dir),
                dataset_name="qasper",
                method="evibridge",
                output_path=str(output_path),
            )
            saved = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            summary = json.loads((output_path.parent / "export_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(predictions, saved)
        self.assertEqual(saved[0]["question_id"], "q1")
        self.assertEqual(saved[0]["predicted_answer"], "The answer")
        self.assertEqual(saved[0]["predicted_evidence"], ["Gold evidence paragraph.", "Later paragraph."])
        self.assertEqual(summary["num_predictions"], 1)
        self.assertEqual(summary["evidence_counts"], [2])
        self.assertFalse(summary["dynamic_evidence_topk"])

    def test_export_predictions_rejects_partial_run_before_writing_official_file(self):
        from Scripts.eval.qasper_official import export_predictions
        from Scripts.eval.qasper_run_validator import QasperRunValidationError

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path, working_dir = self._write_sample_outputs(tmp)
            rows = json.loads(dataset_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "question": "Missing question?",
                    "answer": [],
                    "doc_uuid": "paper-1",
                    "doc_path": "qasper://paper-1",
                    "qasper_question_id": "q2",
                }
            )
            dataset_path.write_text(json.dumps(rows), encoding="utf-8")
            output_path = Path(tmp) / "official" / "predictions.jsonl"

            with self.assertRaises(QasperRunValidationError):
                export_predictions(
                    dataset_path=str(dataset_path),
                    working_dir=str(working_dir),
                    dataset_name="qasper",
                    method="evibridge",
                    output_path=str(output_path),
                )

        self.assertFalse(output_path.exists())

    def test_export_predictions_canonicalizes_unanswerable_aliases(self):
        from Scripts.eval.qasper_official import export_predictions

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path, working_dir = self._write_sample_outputs(tmp)
            result_dir = working_dir / "paper-1" / "eval_qasper_evibridge"
            (result_dir / "final_results.json").write_text(
                json.dumps(
                    [
                        {
                            "question": "What is not answered?",
                            "qasper_question_id": "q1",
                            "output": "Not answerable",
                            "answer_short": "Not answerable",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            predictions = export_predictions(
                dataset_path=str(dataset_path),
                working_dir=str(working_dir),
                dataset_name="qasper",
                method="evibridge",
                output_path=str(Path(tmp) / "official" / "predictions.jsonl"),
            )

        self.assertEqual(predictions[0]["predicted_answer"], "Unanswerable")

    def test_export_predictions_writes_complete_coverage_manifest(self):
        from Scripts.eval.qasper_official import export_predictions

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path, working_dir = self._write_sample_outputs(tmp)
            output_path = Path(tmp) / "official" / "predictions.jsonl"

            export_predictions(
                dataset_path=str(dataset_path),
                working_dir=str(working_dir),
                dataset_name="qasper",
                method="evibridge",
                output_path=str(output_path),
            )
            manifest = json.loads(
                (output_path.parent / "coverage_manifest.json").read_text(encoding="utf-8")
            )

        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["expected_questions"], 1)
        self.assertEqual(manifest["final_results_questions"], 1)
        self.assertEqual(manifest["predictions_questions"], 1)
        self.assertEqual(manifest["documents"], 1)
        self.assertEqual(len(manifest["dataset_sha256"]), 64)

    def test_top_k_evidence_uses_selector_order(self):
        from Scripts.eval.qasper_official import export_predictions

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path, working_dir = self._write_sample_outputs(tmp)
            output_path = Path(tmp) / "official" / "predictions.jsonl"

            predictions = export_predictions(
                dataset_path=str(dataset_path),
                working_dir=str(working_dir),
                dataset_name="qasper",
                method="evibridge",
                output_path=str(output_path),
                top_k_evidence=1,
            )

        self.assertEqual(predictions[0]["predicted_evidence"], ["Gold evidence paragraph."])

    def test_dynamic_evidence_topk_uses_prediction_and_demand_without_gold_answer_type(self):
        from Scripts.eval.qasper_official import export_predictions

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path, working_dir = self._write_sample_outputs(tmp)
            result_dir = working_dir / "paper-1" / "eval_qasper_evibridge"
            query_dir = result_dir / "query_001"
            (result_dir / "final_results.json").write_text(
                json.dumps(
                    [
                        {
                            "question": "Is the model realistic?",
                            "qasper_question_id": "q1",
                            "output": "Yes",
                            "answer_short": "Yes",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (query_dir / "retrieval_res.json").write_text(
                json.dumps(
                    {
                        "demand": {"intent": "boolean"},
                        "supporting_evidence": [
                            {"block_type": "paragraph", "text": "p1", "supporting_rank": 1},
                            {"block_type": "paragraph", "text": "p2", "supporting_rank": 2},
                            {"block_type": "paragraph", "text": "p3", "supporting_rank": 3},
                            {"block_type": "paragraph", "text": "p4", "supporting_rank": 4},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            predictions = export_predictions(
                dataset_path=str(dataset_path),
                working_dir=str(working_dir),
                dataset_name="qasper",
                method="evibridge",
                output_path=str(Path(tmp) / "official" / "predictions.jsonl"),
                dynamic_evidence_topk=True,
            )

        self.assertEqual(predictions[0]["predicted_evidence"], ["p1", "p2", "p3"])

    def test_dynamic_evidence_topk_preserves_explicit_controller_evidence(self):
        from Scripts.eval.qasper_official import export_predictions

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path, working_dir = self._write_sample_outputs(tmp)
            result_dir = working_dir / "paper-1" / "eval_qasper_evibridge"
            query_dir = result_dir / "query_001"
            (result_dir / "final_results.json").write_text(
                json.dumps(
                    [
                        {
                            "question": "Is the model realistic?",
                            "qasper_question_id": "q1",
                            "output": "Yes",
                            "answer_short": "Yes",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (query_dir / "retrieval_res.json").write_text(
                json.dumps(
                    {
                        "demand": {"intent": "boolean"},
                        "support_controller": {"enabled": True, "triggered": False},
                        "supporting_block_ids": [1, 2, 3, 4],
                        "supporting_evidence": [
                            {
                                "block_id": block_id,
                                "block_type": "paragraph",
                                "text": f"p{block_id}",
                                "supporting_rank": block_id,
                            }
                            for block_id in range(1, 5)
                        ],
                    }
                ),
                encoding="utf-8",
            )

            predictions = export_predictions(
                dataset_path=str(dataset_path),
                working_dir=str(working_dir),
                dataset_name="qasper",
                method="evibridge",
                output_path=str(Path(tmp) / "official" / "predictions.jsonl"),
                dynamic_evidence_topk=True,
            )

        self.assertEqual(
            predictions[0]["predicted_evidence"],
            ["p1", "p2", "p3", "p4"],
        )

    def test_dynamic_evidence_topk_exports_empty_evidence_for_predicted_unanswerable(self):
        from Scripts.eval.qasper_official import export_predictions

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path, working_dir = self._write_sample_outputs(tmp)
            result_dir = working_dir / "paper-1" / "eval_qasper_evibridge"
            query_dir = result_dir / "query_001"
            (result_dir / "final_results.json").write_text(
                json.dumps(
                    [
                        {
                            "question": "What is not answered?",
                            "qasper_question_id": "q1",
                            "output": "Unanswerable",
                            "answer_short": "Unanswerable",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (query_dir / "retrieval_res.json").write_text(
                json.dumps(
                    {
                        "demand": {"intent": "fact"},
                        "supporting_evidence": [
                            {"block_type": "paragraph", "text": "p1", "supporting_rank": 1},
                            {"block_type": "paragraph", "text": "p2", "supporting_rank": 2},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            predictions = export_predictions(
                dataset_path=str(dataset_path),
                working_dir=str(working_dir),
                dataset_name="qasper",
                method="evibridge",
                output_path=str(Path(tmp) / "official" / "predictions.jsonl"),
                dynamic_evidence_topk=True,
            )

        self.assertEqual(predictions[0]["predicted_evidence"], [])

    def test_fixed_top_k_evidence_caps_dynamic_evidence_policy(self):
        from Scripts.eval.qasper_official import export_predictions

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path, working_dir = self._write_sample_outputs(tmp)
            query_dir = working_dir / "paper-1" / "eval_qasper_evibridge" / "query_001"
            (query_dir / "retrieval_res.json").write_text(
                json.dumps(
                    {
                        "demand": {"intent": "boolean"},
                        "supporting_evidence": [
                            {"block_type": "paragraph", "text": "p1", "supporting_rank": 1},
                            {"block_type": "paragraph", "text": "p2", "supporting_rank": 2},
                            {"block_type": "paragraph", "text": "p3", "supporting_rank": 3},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            predictions = export_predictions(
                dataset_path=str(dataset_path),
                working_dir=str(working_dir),
                dataset_name="qasper",
                method="evibridge",
                output_path=str(Path(tmp) / "official" / "predictions.jsonl"),
                top_k_evidence=1,
                dynamic_evidence_topk=True,
            )

        self.assertEqual(predictions[0]["predicted_evidence"], ["p1"])

    def test_supporting_evidence_takes_precedence_over_selected_for_official_export(self):
        from Scripts.eval.qasper_official import export_predictions

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path, working_dir = self._write_sample_outputs(tmp)
            query_dir = working_dir / "paper-1" / "eval_qasper_evibridge" / "query_001"
            (query_dir / "retrieval_res.json").write_text(
                json.dumps(
                    {
                        "supporting_evidence": [
                            {
                                "block_id": 3,
                                "block_type": "summary",
                                "text": "Summary should not be exported.",
                                "supporting_rank": 1,
                            },
                            {
                                "block_id": 7,
                                "block_type": "paragraph",
                                "text": "Gold evidence paragraph.",
                                "supporting_rank": 2,
                            },
                        ],
                        "selected": [
                            {
                                "block_id": 8,
                                "block_type": "paragraph",
                                "text": "Selected fallback should not be used.",
                                "selection_rank": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output_path = Path(tmp) / "official" / "predictions.jsonl"

            predictions = export_predictions(
                dataset_path=str(dataset_path),
                working_dir=str(working_dir),
                dataset_name="qasper",
                method="evibridge",
                output_path=str(output_path),
            )

        self.assertEqual(predictions[0]["predicted_evidence"], ["Gold evidence paragraph."])

    def test_exports_bm25_ranked_results_as_evidence(self):
        from Scripts.eval.qasper_official import export_predictions

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path, working_dir = self._write_sample_outputs(tmp)
            evibridge_dir = working_dir / "paper-1" / "eval_qasper_evibridge"
            bm25_dir = working_dir / "paper-1" / "eval_qasper_bm25"
            bm25_query_dir = bm25_dir / "query_001"
            bm25_query_dir.mkdir(parents=True)
            (bm25_dir / "final_results.json").write_text(
                (evibridge_dir / "final_results.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (bm25_query_dir / "retrieval_res.json").write_text(
                json.dumps(
                    {
                        "ranked_results": [
                            {
                                "id": 1,
                                "content": "Gold evidence paragraph.",
                                "rank": 1,
                            },
                            {
                                "id": 2,
                                "content": "BM25 second paragraph.",
                                "rank": 2,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output_path = Path(tmp) / "official" / "bm25_predictions.jsonl"

            predictions = export_predictions(
                dataset_path=str(dataset_path),
                working_dir=str(working_dir),
                dataset_name="qasper",
                method="bm25",
                output_path=str(output_path),
            )

        self.assertEqual(
            predictions[0]["predicted_evidence"],
            ["Gold evidence paragraph.", "BM25 second paragraph."],
        )

    def test_exports_raptor_summary_as_child_paragraph_evidence(self):
        from Core.Index.Tree import DocumentTree, NodeType, TreeNode
        from Scripts.eval.qasper_official import export_predictions

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path, working_dir = self._write_sample_outputs(tmp)
            evibridge_dir = working_dir / "paper-1" / "eval_qasper_evibridge"
            raptor_dir = working_dir / "paper-1" / "eval_qasper_raptor"
            raptor_query_dir = raptor_dir / "query_001"
            raptor_query_dir.mkdir(parents=True)
            tree = DocumentTree(
                meta_dict={"content": "paper-1", "file_name": "paper-1"},
                cfg=type("Cfg", (), {"save_path": str(working_dir / "paper-1")})(),
            )
            section = TreeNode({"content": "Section", "pdf_id": 1})
            section.type = NodeType.TITLE
            section.outline_node = True
            tree.root_node.add_child(section)
            tree.add_node(section)
            paragraph = TreeNode({"content": "Gold evidence paragraph.", "pdf_id": 2})
            paragraph.type = NodeType.TEXT
            section.add_child(paragraph)
            tree.add_node(paragraph)
            tree.save_to_file()
            (raptor_dir / "final_results.json").write_text(
                (evibridge_dir / "final_results.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (raptor_query_dir / "retrieval_res.json").write_text(
                json.dumps(
                    {
                        "ranked_results": [
                            {
                                "id": 10,
                                "content": "Summary should map back to its paragraph.",
                                "rank": 1,
                                "block_type": "summary",
                                "source": "raptor_summary",
                                "child_source_node_ids": str(paragraph.index_id),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            predictions = export_predictions(
                dataset_path=str(dataset_path),
                working_dir=str(working_dir),
                dataset_name="qasper",
                method="raptor",
                output_path=str(Path(tmp) / "official" / "raptor_predictions.jsonl"),
            )

        self.assertEqual(predictions[0]["predicted_evidence"], ["Gold evidence paragraph."])

    def test_evaluates_predictions_with_official_qasper_metrics(self):
        from Scripts.eval.qasper_official import (
            evaluate_predictions_file,
            export_predictions,
        )

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path, working_dir = self._write_sample_outputs(tmp)
            predictions_path = Path(tmp) / "official" / "predictions.jsonl"
            detail_path = Path(tmp) / "official" / "official_eval_detail.json"
            export_predictions(
                dataset_path=str(dataset_path),
                working_dir=str(working_dir),
                dataset_name="qasper",
                method="evibridge",
                output_path=str(predictions_path),
                top_k_evidence=1,
            )

            scores = evaluate_predictions_file(
                dataset_path=str(dataset_path),
                predictions_path=str(predictions_path),
                output_path=str(Path(tmp) / "official" / "official_eval.json"),
                detail_output_path=str(detail_path),
                text_evidence_only=True,
            )
            details = json.loads(detail_path.read_text(encoding="utf-8"))

        self.assertEqual(scores["Answer F1"], 1.0)
        self.assertEqual(scores["Evidence F1"], 1.0)
        self.assertEqual(scores["Missing predictions"], 0)
        self.assertEqual(scores["Answer F1 by type"]["extractive"], 1.0)
        self.assertEqual(details[0]["question_id"], "q1")
        self.assertEqual(details[0]["evidence_precision"], 1.0)

    def test_exports_gold_json_for_official_evaluator(self):
        from Scripts.eval.qasper_official import export_gold_for_official_evaluator

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path, _ = self._write_sample_outputs(tmp)
            output_path = Path(tmp) / "official" / "gold.json"

            gold = export_gold_for_official_evaluator(
                dataset_path=str(dataset_path),
                output_path=str(output_path),
            )
            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(gold, saved)
        self.assertIn("paper-1", saved)
        self.assertEqual(saved["paper-1"]["qas"][0]["question_id"], "q1")
        self.assertEqual(
            saved["paper-1"]["qas"][0]["answers"][0]["answer"]["extractive_spans"],
            ["The answer"],
        )

    def test_calls_external_official_evaluator_script(self):
        from Scripts.eval.qasper_official import run_external_official_evaluator

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            evaluator = tmp_path / "qasper_evaluator.py"
            predictions = tmp_path / "predictions.jsonl"
            gold = tmp_path / "gold.json"
            output_path = tmp_path / "official_stdout.json"
            evaluator.write_text(
                "import argparse, json\n"
                "p=argparse.ArgumentParser()\n"
                "p.add_argument('--predictions')\n"
                "p.add_argument('--gold')\n"
                "p.add_argument('--text_evidence_only', action='store_true')\n"
                "args=p.parse_args()\n"
                "open(args.predictions).read()\n"
                "print(json.dumps({'Answer F1': 1.0, 'Evidence F1': 1.0, 'Missing predictions': 0}))\n",
                encoding="utf-8",
            )
            predictions.write_text(
                json.dumps(
                    {
                        "question_id": "q1",
                        "predicted_answer": "The answer with “unicode quotes”",
                        "predicted_evidence": ["Gold evidence paragraph."],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            gold.write_text(json.dumps({"paper-1": {"qas": []}}), encoding="utf-8")

            scores = run_external_official_evaluator(
                evaluator_path=str(evaluator),
                predictions_path=str(predictions),
                gold_path=str(gold),
                output_path=str(output_path),
                text_evidence_only=True,
            )

            self.assertEqual(scores["Answer F1"], 1.0)
            self.assertEqual(scores["Evidence F1"], 1.0)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), scores)


if __name__ == "__main__":
    unittest.main()
