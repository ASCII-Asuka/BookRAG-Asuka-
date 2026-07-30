import json
import tempfile
import unittest
from pathlib import Path

from Scripts.eval.react_diagnostics import (
    aggregate_react_diagnostics,
    validate_expected_queries,
)


class ReactDiagnosticsTests(unittest.TestCase):
    def test_aggregates_steps_calls_actions_and_termination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, payload in enumerate(
                [
                    {
                        "react_steps": [
                            {"valid_action": True},
                            {"valid_action": True},
                        ],
                        "react_num_calls": 2,
                        "react_num_bad_calls": 0,
                        "react_search_count": 1,
                        "react_lookup_count": 0,
                        "react_termination_reason": "finish",
                    },
                    {
                        "react_steps": [
                            {"valid_action": False}
                        ]
                        * 8,
                        "react_num_calls": 8,
                        "react_num_bad_calls": 1,
                        "react_search_count": 0,
                        "react_lookup_count": 1,
                        "react_termination_reason": "step_limit",
                    },
                ]
            ):
                query_dir = root / f"query_{index:03d}"
                query_dir.mkdir()
                (query_dir / "retrieval_res.json").write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )

            report = aggregate_react_diagnostics(root)

        self.assertEqual(report["queries"], 2)
        self.assertEqual(report["mean_steps"], 5.0)
        self.assertEqual(report["mean_calls"], 5.0)
        self.assertEqual(report["step_limit_rate"], 0.5)
        self.assertEqual(report["invalid_action_rate"], 0.8)

    def test_aggregates_multiple_directories_and_token_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = []
            for index in range(2):
                eval_dir = Path(tmp) / f"doc-{index}"
                query_dir = eval_dir / "query_001"
                query_dir.mkdir(parents=True)
                (query_dir / "retrieval_res.json").write_text(
                    json.dumps(
                        {
                            "react_steps": [],
                            "react_num_calls": index + 1,
                        }
                    ),
                    encoding="utf-8",
                )
                (eval_dir / "token_cost.json").write_text(
                    json.dumps(
                        {
                            "rag_cost": {
                                "prompt_tokens": 10,
                                "completion_tokens": 2,
                                "total_tokens": 12,
                            },
                            "time": 1.5,
                        }
                    ),
                    encoding="utf-8",
                )
                roots.append(eval_dir)

            report = aggregate_react_diagnostics(roots)

        self.assertEqual(report["documents"], 2)
        self.assertEqual(report["queries"], 2)
        self.assertEqual(report["elapsed_seconds"], 3.0)
        self.assertEqual(report["token_totals"]["total_tokens"], 24)

    def test_expected_query_gate_rejects_partial_results(self):
        with self.assertRaises(ValueError) as context:
            validate_expected_queries({"queries": 9}, expected_queries=10)

        self.assertIn("9", str(context.exception))
        self.assertIn("10", str(context.exception))
