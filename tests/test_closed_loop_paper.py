import unittest

from Scripts.analysis.render_closed_loop_paper import (
    _round_two_cell,
    insert_addition,
    render_addition,
    upsert_addition,
)


class ClosedLoopPaperTests(unittest.TestCase):
    def test_render_contains_required_sections_labels_and_real_rates(self):
        summaries = {
            "qasper": {
                "total_questions": 1005,
                "first_round_accept_count": 902,
                "first_round_accept_rate": 902 / 1005,
                "second_round_trigger_count": 52,
                "second_round_trigger_rate": 52 / 1005,
                "unaccepted_without_second_round_count": 51,
                "expanded_accept_count": 0,
                "expanded_accept_denominator": 52,
                "expanded_accept_rate": 0.0,
                "final_insufficient_count": 103,
                "final_insufficient_rate": 103 / 1005,
                "mean_round_count": 1.052,
                "triggered_mean_added_atomic_blocks": 1.90,
                "triggered_first_evidence_metric": 0.1227,
                "triggered_final_evidence_metric": 0.1417,
                "triggered_mean_evidence_gain": 0.0190,
                "triggered_mean_added_context_tokens": 220.2,
            },
            "hotpotqa": {
                "total_questions": 1000,
                "first_round_accept_count": 990,
                "first_round_accept_rate": 0.99,
                "second_round_trigger_count": 4,
                "second_round_trigger_rate": 0.004,
                "unaccepted_without_second_round_count": 6,
                "expanded_accept_count": 0,
                "expanded_accept_denominator": 4,
                "expanded_accept_rate": 0.0,
                "final_insufficient_count": 10,
                "final_insufficient_rate": 0.01,
                "mean_round_count": 1.004,
                "triggered_mean_added_atomic_blocks": 1.25,
                "triggered_first_evidence_metric": 0.3654,
                "triggered_final_evidence_metric": 0.3929,
                "triggered_mean_evidence_gain": 0.0275,
                "triggered_mean_added_context_tokens": 44.8,
            },
        }
        cases = {"cases": {"qasper": _case("q1", "Q question"), "hotpotqa": _case("h1", "H question")}}
        addition = render_addition(summaries, cases)
        self.assertIn(r"\subsection{闭环检索行为分析}", addition)
        self.assertIn(r"\label{tab:closed_loop_behavior}", addition)
        self.assertIn(r"89.8\% (902/1005)", addition)
        self.assertIn(r"\subsection{检索轨迹案例分析}", addition)
        self.assertIn(r"\label{tab:retrieval_trajectory_cases}", addition)
        self.assertIn(r"答案 F1=94.12\%", addition)
        self.assertIn("新增且命中标准证据的原子块进入最终支持集", addition)
        self.assertNotIn("最终生成答案仍偏离标准答案", addition)

    def test_insert_changes_only_before_unique_parameter_anchor(self):
        source = "before\n\\subsection{参数敏感性分析}\nafter\n"
        addition = "ADDED\n"
        rendered = insert_addition(source, addition)
        self.assertEqual(rendered, "before\nADDED\n\n\\subsection{参数敏感性分析}\nafter\n")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            insert_addition(source + r"\subsection{参数敏感性分析}", addition)

    def test_upsert_replaces_only_existing_analysis_block(self):
        source = (
            "prefix\n"
            "\\subsection{闭环检索行为分析}\n"
            "OLD LOOP\n"
            "\\subsection{检索轨迹案例分析}\n"
            "OLD CASE\n"
            "\\subsection{参数敏感性分析}\n"
            "suffix\n"
        )
        addition = (
            "\\subsection{闭环检索行为分析}\nNEW LOOP\n"
            "\\subsection{检索轨迹案例分析}\nNEW CASE\n"
        )

        rendered = upsert_addition(source, addition)

        self.assertEqual(
            rendered,
            "prefix\n" + addition.rstrip() + "\n\n\\subsection{参数敏感性分析}\nsuffix\n",
        )
        self.assertEqual(rendered.count(r"\subsection{闭环检索行为分析}"), 1)
        self.assertEqual(rendered.count(r"\subsection{检索轨迹案例分析}"), 1)

    def test_round_two_prioritizes_added_gold_block_used_by_final_answer(self):
        case = {
            "dataset": "qasper",
            "round_2_added_gold_support_ids": [25],
            "round_2_added_evidence": [
                {"block_id": 27, "section_id": "Network", "text": "model details"},
                {"block_id": 25, "section_id": "Datasets", "text": "IITB and ILCI corpora"},
            ],
            "round_2_selected_bridges": [],
        }

        rendered = _round_two_cell(case)

        self.assertLess(rendered.index(r"\texttt{[25]}"), rendered.index(r"\texttt{[27]}"))

    def test_round_two_excerpt_keeps_both_dataset_names(self):
        case = {
            "dataset": "qasper",
            "round_2_added_gold_support_ids": [25],
            "round_2_added_evidence": [
                {
                    "block_id": 25,
                    "section_id": "Datasets",
                    "text": "IITB English-Hindi parallel corpus " + "x" * 115 + " ILCI English-Hindi parallel corpus",
                }
            ],
            "round_2_selected_bridges": [],
        }

        self.assertIn("ILCI", _round_two_cell(case))


def _case(qid, question):
    return {
        "question_id": qid,
        "question": question,
        "gold_answer": ["gold"],
        "gold_evidence": [[1, 2]],
        "bm25_reranker": {"answer": "base", "evidence_f1": 0.2, "supporting_evidence": []},
        "round_1": {
            "evidence_f1": 0.3,
            "selected_atomic_blocks": [
                {"block_id": 1, "section_id": "S", "text": "first evidence", "metadata": {}}
            ],
        },
        "diagnosis": {
            "missing_demand": ["missing"],
            "suggested_bridge_types": ["semantic"],
            "controlled_action": "expand",
            "refinement_query": question,
        },
        "round_2_added_evidence": [
            {"block_id": 2, "section_id": "T", "text": "added evidence", "metadata": {}}
        ],
        "round_2_selected_bridges": [
            {"bridge_type": "semantic", "relation_type": "shared_terms"}
        ],
        "final_output": {
            "answer": "final",
            "answer_f1": 16 / 17,
            "supporting_block_ids": [1],
            "supporting_blocks": [],
            "evidence_f1": 0.5,
        },
        "selection": {
            "candidate_count": 3,
            "candidate_median_evidence_gain": 0.2,
            "selected_evidence_gain": 0.2,
        },
    }


if __name__ == "__main__":
    unittest.main()
