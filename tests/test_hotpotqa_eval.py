import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


class HotpotQAEvalTests(unittest.TestCase):
    def test_answer_and_supporting_fact_metrics(self):
        from Eval.utils.hotpotqa_eval import evaluate_hotpotqa_predictions

        gold_rows = [
            {
                "hotpotqa_question_id": "q1",
                "answer": "Chief of Protocol",
                "hotpot_supporting_facts": [["Kiss and Tell", 0], ["Shirley Temple", 1]],
            }
        ]
        predictions = {
            "q1": {
                "answer": "Chief Protocol",
                "sp": [["Kiss and Tell", 0], ["Wrong", 2]],
            }
        }

        scores = evaluate_hotpotqa_predictions(gold_rows, predictions)

        self.assertEqual(scores["missing_predictions"], 0)
        self.assertEqual(scores["answer_em"], 0.0)
        self.assertGreater(scores["answer_f1"], 0.0)
        self.assertEqual(scores["sp_em"], 0.0)
        self.assertEqual(scores["sp_recall"], 0.5)
        self.assertGreater(scores["joint_f1"], 0.0)

    def test_eval_hotpotqa_reads_final_results_and_retrieval_supporting_facts(self):
        from Eval.utils.hotpotqa_eval import eval_hotpotqa

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "hotpotqa.json"
            working_dir = root / "work"
            dataset_rows = [
                {
                    "question": "What position?",
                    "answer": "Chief of Protocol",
                    "doc_uuid": "hotpot-1",
                    "doc_path": "hotpotqa://distractor/validation/hotpot-1",
                    "hotpotqa_question_id": "hotpot-1",
                    "hotpot_supporting_facts": [["Kiss and Tell", 0], ["Shirley Temple", 1]],
                }
            ]
            dataset_path.write_text(json.dumps(dataset_rows), encoding="utf-8")
            result_dir = working_dir / "hotpot-1" / "eval_hotpotqa_evibridge"
            query_dir = result_dir / "query_001"
            query_dir.mkdir(parents=True)
            (result_dir / "final_results.json").write_text(
                json.dumps(
                    [
                        {
                            **dataset_rows[0],
                            "output": "Chief of Protocol",
                            "answer_short": "Chief of Protocol",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (query_dir / "retrieval_res.json").write_text(
                json.dumps(
                    {
                        "supporting_evidence": [
                            {"title": "Kiss and Tell", "sent_id": 0},
                            {"title": "Shirley Temple", "sent_id": 1},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            data_cfg = SimpleNamespace(
                dataset_path=str(dataset_path),
                working_dir=str(working_dir),
                dataset_name="hotpotqa",
            )

            scores = eval_hotpotqa(dataset_rows, data_cfg, method="evibridge")
            saved_score = json.loads(
                (working_dir / "0_results" / "final_eval_hotpotqa_evibridge.score.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(scores["answer_em"], 1.0)
        self.assertEqual(scores["sp_em"], 1.0)
        self.assertEqual(saved_score["joint_em"], 1.0)


if __name__ == "__main__":
    unittest.main()
