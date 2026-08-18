import json
import tempfile
import unittest
from pathlib import Path


class HotpotQALeadOnlySummaryTests(unittest.TestCase):
    def test_build_summary_validates_provenance_and_averages_cost(self):
        from Scripts.analysis.hotpotqa_lead_only_summary import build_summary

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work_root = root / "work"
            run_root = root / "run"
            result_root = work_root / "0_results"
            eval_root = work_root / "question-1" / "eval_hotpotqa_lead_only"
            query_root = eval_root / "query_001"
            query_root.mkdir(parents=True)
            result_root.mkdir(parents=True)
            run_root.mkdir(parents=True)

            score = {
                "answer_em": 0.5,
                "answer_f1": 0.6,
                "sp_precision": 0.7,
                "sp_recall": 0.4,
                "sp_f1": 0.5,
                "joint_f1": 0.3,
                "missing_predictions": 0,
                "total_samples": 1,
            }
            (result_root / "final_eval_hotpotqa_lead_only.score.json").write_text(
                json.dumps(score), encoding="utf-8"
            )
            (run_root / "audit_lead_only.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "completed_questions": 1,
                        "completed_documents": 1,
                    }
                ),
                encoding="utf-8",
            )
            (run_root / "bootstrap.json").write_text("{}", encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps({"unified_sha256": "dataset-sha"}), encoding="utf-8"
            )
            result = {
                "hotpot_node_facts": {"2": ["Article A", 0]},
                "retrieved_node_ids": [2],
                "supporting_block_ids": [2],
            }
            retrieval = {
                "ranked_results": [
                    {"id": 2, "hotpot_title": "Article A", "hotpot_sent_id": 0}
                ],
                "supporting_evidence": [
                    {"id": 2, "hotpot_title": "Article A", "hotpot_sent_id": 0}
                ],
            }
            (query_root / "result.json").write_text(
                json.dumps(result), encoding="utf-8"
            )
            (query_root / "retrieval_res.json").write_text(
                json.dumps(retrieval), encoding="utf-8"
            )
            (eval_root / "token_cost.json").write_text(
                json.dumps(
                    {"rag_cost": {"total_tokens": 750}, "time": 2.5}
                ),
                encoding="utf-8",
            )

            summary = build_summary(
                work_root=work_root,
                run_root=run_root,
                manifest_path=manifest_path,
                config_path=Path("config.yaml"),
                expected_questions=1,
                bootstrap_path=run_root / "bootstrap.json",
            )

        self.assertEqual(summary["coverage"]["questions"], 1)
        self.assertEqual(summary["metrics"]["answer_f1"], 0.6)
        self.assertEqual(summary["efficiency"]["tokens_per_question"], 750.0)
        self.assertEqual(summary["efficiency"]["seconds_per_question"], 2.5)
        self.assertEqual(summary["provenance"]["validated_ranked_leads"], 1)
        self.assertEqual(summary["dataset_sha256"], "dataset-sha")


if __name__ == "__main__":
    unittest.main()
