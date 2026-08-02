import json
import math
import unittest


def _candidate(
    block_id, text, score, block_type="paragraph", section_id="section"
):
    return {
        "block_id": block_id,
        "block_type": block_type,
        "text": text,
        "section_id": section_id,
        "score_parts": {"answer_conditioned_rerank_score": score},
    }


class CoverageAwareSupportPrunerTests(unittest.TestCase):
    def _prune(self, **overrides):
        from Core.rag.evibridge_support_pruner import prune_supporting_evidence

        kwargs = {
            "question": "Which model is best?",
            "draft_answer": "Model B is best.",
            "intent": "fact",
            "subqueries": [],
            "candidates": [
                _candidate(1, "Model B is best.", 0.9),
                _candidate(2, "Unrelated training details.", 0.8),
            ],
            "anchor_ids": [],
            "allowed_types": {"paragraph", "table", "caption", "figure"},
            "max_items": 2,
            "min_normalized_relevance": 0.5,
            "redundancy_overlap_threshold": 0.75,
        }
        kwargs.update(overrides)
        return prune_supporting_evidence(**kwargs)

    def test_retains_legal_anchor_without_padding(self):
        result = self._prune(anchor_ids=[1])

        self.assertEqual(result["final_ids"], [1])
        self.assertEqual(result["stopping_reason"], "answer_complete_anchor")

    def test_comparison_adds_distinct_facets_without_filling_with_noise(self):
        result = self._prune(
            question="Compare Method A and Method B.",
            draft_answer="Method B performs better.",
            intent="comparison",
            subqueries=["Method A score", "Method B score"],
            candidates=[
                _candidate(1, "Method A score is 80.", 0.95),
                _candidate(2, "Method B score is 84.", 0.90),
                _candidate(3, "Unrelated setup details.", 0.85),
            ],
            max_items=3,
            min_normalized_relevance=0.4,
        )

        self.assertEqual(result["final_ids"], [1, 2])
        self.assertEqual(result["stopping_reason"], "no_marginal_coverage")

    def test_comparison_keeps_one_high_relevance_candidate_after_coverage_saturates(self):
        result = self._prune(
            question="What additional features and context are proposed?",
            draft_answer="The extension adds syntax cues.",
            intent="comparison",
            candidates=[
                _candidate(
                    2,
                    "The proposal adds syntax cues.",
                    0.9,
                    section_id="Abstract",
                ),
                _candidate(38, "Discourse-aware representations are introduced.", 0.6),
                _candidate(25, "Unrelated experimental details.", 0.1),
            ],
            anchor_ids=[2],
            max_items=3,
            min_normalized_relevance=0.35,
        )

        self.assertEqual(result["final_ids"], [2, 38])
        diagnostic = next(
            item for item in result["candidate_diagnostics"] if item["block_id"] == 38
        )
        self.assertEqual(diagnostic["reason"], "high_relevance_fallback")

    def test_top_ranked_body_anchor_is_exported_without_supplement(self):
        result = self._prune(
            question="Which sentiment class is most accurately predicted?",
            draft_answer="Negative.",
            intent="comparison",
            candidates=[
                _candidate(
                    20,
                    "Negative sentiment is predicted most accurately.",
                    0.9,
                    section_id="Results and Discussion",
                ),
                _candidate(
                    2,
                    "The sentiment system predicts several classes.",
                    0.8,
                    section_id="Introduction",
                ),
            ],
            anchor_ids=[20],
            max_items=3,
            min_normalized_relevance=0.35,
        )

        self.assertEqual(result["final_ids"], [20])
        self.assertEqual(result["stopping_reason"], "answer_complete_anchor")

    def test_top_non_anchor_beats_lower_ranked_generic_coverage(self):
        result = self._prune(
            question="What discourse relations does it work best and worst for?",
            draft_answer="It works best for Comparison and Temporal relations.",
            intent="comparison",
            subqueries=["best relation", "worst relation"],
            candidates=[
                _candidate(40, "The labels are Comparison and Temporal.", 0.97),
                _candidate(
                    64,
                    "It works best for Comparison and Temporal relations.",
                    0.92,
                    section_id="Experimental Results",
                ),
                _candidate(
                    36,
                    "Best relation and worst relation results.",
                    0.90,
                ),
            ],
            anchor_ids=[64],
            max_items=3,
            min_normalized_relevance=0.35,
        )

        self.assertEqual(result["final_ids"], [64, 40])
        diagnostic = next(
            item for item in result["candidate_diagnostics"] if item["block_id"] == 40
        )
        self.assertEqual(diagnostic["reason"], "top_relevance_supplement")

    def test_boolean_answer_does_not_use_high_relevance_coverage_fallback(self):
        result = self._prune(
            question="Does the proposed method outperform the baseline?",
            draft_answer="No.",
            intent="comparison",
            candidates=[
                _candidate(1, "The proposed method does not outperform the baseline.", 0.9),
                _candidate(2, "Training uses a larger batch.", 0.8),
                _candidate(3, "Unrelated implementation details.", 0.1),
            ],
            anchor_ids=[1],
            max_items=3,
            min_normalized_relevance=0.35,
        )

        self.assertEqual(result["final_ids"], [1])
        reasons = {
            item["block_id"]: item["reason"]
            for item in result["candidate_diagnostics"]
        }
        self.assertNotEqual(reasons[2], "high_relevance_fallback")

    def test_relevance_dominates_generic_lexical_coverage(self):
        result = self._prune(
            question="Which discourse relations does it work best and worst for?",
            draft_answer="It works best for Expansion and worst for Comparison.",
            intent="comparison",
            candidates=[
                _candidate(40, "Expansion Comparison.", 0.973),
                _candidate(64, "Results appear in Table 4.", 0.92),
                _candidate(36, "Discourse best Expansion Comparison.", 0.8757),
                _candidate(7, "Unrelated appendix material.", 0.0),
            ],
            anchor_ids=[64],
            max_items=2,
            min_normalized_relevance=0.35,
        )

        self.assertEqual(result["final_ids"], [64, 40])

    def test_rejects_bridge_auxiliary_type(self):
        result = self._prune(
            candidates=[
                _candidate(8, "Model B is best.", 0.99, block_type="entity"),
                _candidate(1, "Model B is best.", 0.9),
            ]
        )

        self.assertEqual(result["final_ids"], [1])
        diagnostic = next(
            item for item in result["candidate_diagnostics"] if item["block_id"] == 8
        )
        self.assertEqual(diagnostic["decision"], "rejected")
        self.assertEqual(diagnostic["reason"], "ineligible_type")

    def test_rejects_redundant_contained_paragraph(self):
        result = self._prune(
            question="What accuracy does Model B achieve?",
            draft_answer="Model B achieves 84 percent accuracy.",
            candidates=[
                _candidate(1, "Model B achieves 84 percent accuracy on the test set.", 0.9),
                _candidate(2, "Model B achieves 84 percent accuracy.", 0.89),
            ],
            min_normalized_relevance=0.0,
        )

        self.assertEqual(result["final_ids"], [1])
        diagnostic = next(
            item for item in result["candidate_diagnostics"] if item["block_id"] == 2
        )
        self.assertEqual(diagnostic["reason"], "redundant")
        self.assertEqual(diagnostic["overlap_block_id"], 1)

    def test_deduplicates_anchor_ids_in_first_seen_order(self):
        result = self._prune(
            candidates=[
                _candidate(1, "Model A result.", 0.8),
                _candidate(2, "Model B is best.", 0.9),
            ],
            anchor_ids=[2, 1, 2],
        )

        self.assertEqual(result["final_ids"], [2, 1])

    def test_unanswerable_short_circuits_to_empty(self):
        result = self._prune(anchor_ids=[1], unanswerable=True)

        self.assertEqual(result["final_ids"], [])
        self.assertEqual(result["stopping_reason"], "unanswerable")

    def test_non_finite_scores_are_rejected(self):
        result = self._prune(
            candidates=[
                _candidate(1, "Model B is best.", float("nan")),
                _candidate(2, "Model B is best.", float("inf")),
                _candidate(3, "Model B is best.", 0.5),
            ]
        )

        self.assertEqual(result["final_ids"], [3])
        reasons = {
            item["block_id"]: item["reason"]
            for item in result["candidate_diagnostics"]
        }
        self.assertEqual(reasons[1], "non_finite_score")
        self.assertEqual(reasons[2], "non_finite_score")
        json.dumps(result, allow_nan=False)

    def test_tied_scores_have_deterministic_rank_normalization(self):
        candidates = [
            _candidate(2, "Model B is best.", 0.5),
            _candidate(1, "Model A is second.", 0.5),
            _candidate(3, "Unrelated details.", 0.5),
        ]

        first = self._prune(candidates=candidates, min_normalized_relevance=0.0)
        second = self._prune(candidates=candidates, min_normalized_relevance=0.0)

        self.assertEqual(first, second)
        normalized = [
            item["normalized_relevance"]
            for item in first["candidate_diagnostics"]
            if math.isfinite(item.get("raw_relevance", float("nan")))
        ]
        self.assertEqual(normalized, [1.0, 0.5, 0.0])

    def test_no_anchor_uses_top_eligible_candidate_as_safe_fallback(self):
        result = self._prune(
            question="Which model is best?",
            draft_answer="Unknown.",
            candidates=[
                _candidate(4, "Completely unrelated material.", 0.9),
                _candidate(5, "More unrelated material.", 0.8),
            ],
            min_normalized_relevance=0.95,
        )

        self.assertEqual(result["final_ids"], [4])
        self.assertEqual(result["stopping_reason"], "no_marginal_coverage")

    def test_every_candidate_has_accept_or_reject_reason(self):
        candidates = [
            _candidate(1, "Model B is best.", 0.9),
            _candidate(2, "Unrelated material.", 0.8),
            _candidate(3, "Bridge node.", 0.7, block_type="summary"),
        ]

        result = self._prune(candidates=candidates)

        diagnostics = result["candidate_diagnostics"]
        self.assertEqual([item["block_id"] for item in diagnostics], [1, 2, 3])
        self.assertTrue(all(item["decision"] for item in diagnostics))
        self.assertTrue(all(item["reason"] for item in diagnostics))


if __name__ == "__main__":
    unittest.main()
