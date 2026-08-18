import unittest

from Scripts.analysis.closed_loop_analysis import (
    aggregate_dataset,
    build_query_log,
    choose_median_gain_case,
    choose_successful_qasper_case,
    evidence_prf,
    qasper_answer_f1,
    validate_query_logs,
)


def _block(block_id, text, block_type="paragraph", **metadata):
    return {
        "block_id": block_id,
        "block_type": block_type,
        "text": text,
        "section_id": metadata.pop("section_id", "S"),
        "metadata": metadata,
    }


class ClosedLoopAnalysisTests(unittest.TestCase):
 def test_qasper_query_log_reconstructs_rounds_and_best_gold_group(self):
    result = {
        "doc_uuid": "paper-1",
        "qasper_question_id": "q-1",
        "question": "Which evidence is needed?",
        "answer": [
            {"evidence_paragraph_ids": [2, 3]},
            {"evidence_paragraph_ids": [2, 4]},
        ],
    }
    retrieval = {
        "iterations": [
            {
                "iteration": 1,
                "selected_block_ids": [2, 10],
                "candidate_scores": {"2": 1.0, "10": 0.7},
                "verification": {
                    "sufficient": False,
                    "missing_types": ["paragraph"],
                    "missing_bridge_types": ["semantic"],
                    "next_action": "expand",
                },
                "refinement": {"query": "bridge query"},
            },
            {
                "iteration": 2,
                "selected_block_ids": [2, 3, 11],
                "candidate_scores": {"2": 1.0, "3": 0.9, "11": 0.6},
                "verification": {
                    "sufficient": True,
                    "missing_types": [],
                    "missing_bridge_types": [],
                    "next_action": "accept",
                },
            },
        ],
        "stopping_reason": "sufficient",
    }
    blocks = {
        2: _block(2, "alpha beta"),
        3: _block(3, "gamma delta epsilon"),
        10: _block(10, "auxiliary patch", "patch"),
        11: _block(11, "entity node", "entity"),
    }

    log = build_query_log(
        "qasper", result, retrieval, blocks, token_counter=lambda text: len(text.split())
    )

    self.assertEqual(log["round_count"], 2)
    self.assertFalse(log["y_1"])
    self.assertTrue(log["y_2"])
    self.assertTrue(log["y_final"])
    self.assertFalse(log["first_round_accepted"])
    self.assertTrue(log["second_round_triggered"])
    self.assertEqual(log["rounds"][0]["selected_atomic_ids"], [2])
    self.assertEqual(log["rounds"][1]["selected_atomic_ids"], [2, 3])
    self.assertEqual(log["added_atomic_ids"], [3])
    self.assertEqual(log["added_context_tokens"], 3)
    self.assertEqual(log["first_atomic_block_count"], 1)
    self.assertEqual(log["final_atomic_block_count"], 2)
    self.assertEqual(log["first_auxiliary_node_count"], 1)
    self.assertEqual(log["final_auxiliary_node_count"], 1)
    self.assertEqual(log["first_context_tokens"], 2)
    self.assertEqual(log["final_context_tokens"], 5)
    self.assertAlmostEqual(log["rounds"][0]["evidence_f1"], 2 / 3)
    self.assertAlmostEqual(log["rounds"][1]["evidence_f1"], 1.0)
    self.assertAlmostEqual(log["evidence_gain"], 1 / 3)
    self.assertEqual(log["refinement_query"], "bridge query")


 def test_hotpot_supporting_fact_mapping_uses_title_sentence_pairs(self):
    result = {
        "doc_uuid": "hp-1",
        "hotpot_id": "hp-1",
        "hotpot_type": "bridge",
        "question": "Who?",
        "hotpot_supporting_facts": [["A", 0], ["B", 2]],
        "hotpot_node_facts": {"5": ["A", 0], "6": ["B", 2], "7": ["C", 1]},
    }
    retrieval = {
        "iterations": [
            {
                "iteration": 1,
                "selected_block_ids": [5, 7],
                "candidate_scores": {"5": 1.0, "7": 0.5},
                "verification": {"sufficient": False, "next_action": "expand"},
                "refinement": {"query": "B relation"},
            },
            {
                "iteration": 2,
                "selected_block_ids": [5, 6],
                "candidate_scores": {"5": 1.0, "6": 0.9},
                "verification": {"sufficient": True, "next_action": "accept"},
            },
        ],
        "stopping_reason": "sufficient",
    }
    blocks = {i: _block(i, f"sentence {i}") for i in (5, 6, 7)}

    log = build_query_log(
        "hotpotqa", result, retrieval, blocks, token_counter=lambda _: 4
    )

    self.assertEqual(log["rounds"][0]["predicted_evidence"], [["A", 0], ["C", 1]])
    self.assertAlmostEqual(log["rounds"][0]["evidence_f1"], 0.5)
    self.assertAlmostEqual(log["rounds"][1]["evidence_f1"], 1.0)
    self.assertEqual(log["added_context_tokens"], 4)


 def test_evidence_prf_handles_empty_sets_and_duplicate_predictions(self):
    self.assertEqual(evidence_prf([], []), (1.0, 1.0, 1.0))
    precision, recall, f1 = evidence_prf(["a", "a", "b"], ["a", "c"])
    self.assertAlmostEqual(precision, 0.5)
    self.assertAlmostEqual(recall, 0.5)
    self.assertAlmostEqual(f1, 0.5)


 def test_validation_rejects_duplicates_and_round_two_non_candidate_additions(self):
    valid = {
        "dataset": "qasper",
        "question_id": "q1",
        "round_count": 2,
        "second_round_triggered": True,
        "added_atomic_ids": [3],
        "rounds": [
            {"selected_atomic_ids": [2], "candidate_ids": [2]},
            {"selected_atomic_ids": [2, 3], "candidate_ids": [2, 3]},
        ],
    }
    validate_query_logs([valid], expected_total=1)

    with self.assertRaisesRegex(ValueError, "duplicate question_id"):
        validate_query_logs([valid, dict(valid)], expected_total=2)

    broken = {**valid, "added_atomic_ids": [4]}
    with self.assertRaisesRegex(ValueError, "second-round candidates"):
        validate_query_logs([broken], expected_total=1)


 def test_aggregate_and_median_gain_case_are_deterministic(self):
    logs = [
        {
            "dataset": "qasper",
            "question_id": "q1",
            "round_count": 1,
            "first_round_accepted": True,
            "second_round_triggered": False,
            "final_sufficient": True,
            "evidence_gain": 0.0,
            "added_context_tokens": 0,
        },
        {
            "dataset": "qasper",
            "question_id": "q2",
            "round_count": 2,
            "first_round_accepted": False,
            "second_round_triggered": True,
            "final_sufficient": True,
            "evidence_gain": 0.2,
            "added_context_tokens": 20,
        },
        {
            "dataset": "qasper",
            "question_id": "q3",
            "round_count": 2,
            "first_round_accepted": False,
            "second_round_triggered": True,
            "final_sufficient": False,
            "evidence_gain": 0.6,
            "added_context_tokens": 40,
        },
    ]
    summary = aggregate_dataset(logs)
    self.assertEqual(summary["total_questions"], 3)
    self.assertAlmostEqual(summary["first_round_accept_rate"], 1 / 3)
    self.assertAlmostEqual(summary["second_round_trigger_rate"], 2 / 3)
    self.assertAlmostEqual(summary["final_insufficient_rate"], 1 / 3)
    self.assertAlmostEqual(summary["triggered_mean_evidence_gain"], 0.4)
    self.assertAlmostEqual(summary["triggered_mean_added_context_tokens"], 30)

    candidates = [
        {"question_id": "b", "evidence_gain": 0.6},
        {"question_id": "a", "evidence_gain": 0.2},
        {"question_id": "c", "evidence_gain": 0.4},
    ]
    self.assertEqual(choose_median_gain_case(candidates)["question_id"], "c")


 def test_qasper_success_case_uses_original_median_and_requires_cited_added_gold(self):
    strict_candidates = [
        {
            "question_id": "wrong-answer",
            "evidence_gain": 1 / 6,
            "answer_f1": 0.0,
            "added_gold_support_ids": [51],
        },
        {
            "question_id": "approved",
            "evidence_gain": 1 / 6,
            "answer_f1": 16 / 17,
            "added_gold_support_ids": [25],
        },
        {
            "question_id": "not-cited",
            "evidence_gain": 1 / 6,
            "answer_f1": 1.0,
            "added_gold_support_ids": [],
        },
        {
            "question_id": "extreme-gain",
            "evidence_gain": 5 / 8,
            "answer_f1": 8 / 9,
            "added_gold_support_ids": [19, 27],
        },
    ]

    selected, audit = choose_successful_qasper_case(strict_candidates)

    self.assertEqual(selected["question_id"], "approved")
    self.assertEqual(audit["strict_candidate_count"], 4)
    self.assertEqual(audit["successful_candidate_count"], 2)
    self.assertAlmostEqual(audit["candidate_median_evidence_gain"], 1 / 6)
    self.assertEqual(audit["selected_added_gold_support_ids"], [25])


 def test_qasper_answer_f1_uses_best_reference(self):
    result = {
        "answer_short": "IITB English-Hindi parallel corpus and ILCI English-Hindi parallel corpus",
        "answer": [
            {"free_form_answer": "irrelevant reference"},
            {
                "extractive_spans": [
                    "IITB English-Hindi parallel corpus",
                    "ILCI English-Hindi parallel corpus",
                ]
            },
        ],
    }

    self.assertAlmostEqual(qasper_answer_f1(result), 16 / 17)


if __name__ == "__main__":
    unittest.main()
