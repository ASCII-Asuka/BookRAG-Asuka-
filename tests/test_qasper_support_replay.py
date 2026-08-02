import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _candidate(block_id, text, block_type="paragraph"):
    return {
        "block_id": block_id,
        "block_type": block_type,
        "text": text,
        "section_id": "section",
        "score_parts": {},
        "evidence_role": "answer_evidence",
    }


class FakeReranker:
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = []

    def rerank(self, query, documents, batch_size=4):
        self.calls.append({"query": query, "documents": list(documents)})
        return list(self.scores)


class QasperSupportReplayTests(unittest.TestCase):
    def _case(self):
        return {
            "question_id": "q1",
            "question": "Compare Method A and Method B.",
            "draft_answer": "Method B performs better.",
            "intent": "comparison",
            "subqueries": ["Method A score", "Method B score"],
            "candidates": [
                _candidate(1, "Method A score is 80."),
                _candidate(2, "Method B score is 84."),
                _candidate(3, "Unrelated setup material."),
            ],
            "anchor_ids": [1],
            "max_items": 3,
            "allowed_types": ["paragraph", "table", "caption", "figure"],
            "unanswerable": False,
        }

    def test_reconstruct_support_candidates_matches_runtime_order(self):
        from Scripts.eval.qasper_support_replay import reconstruct_support_candidates

        retrieval = {
            "selected": [
                {
                    "block_id": 2,
                    "block_type": "paragraph",
                    "text": "Selected block.",
                    "evidence_role": "answer_evidence",
                    "score_parts": {"rerank_rank": 3},
                }
            ],
            "typed_ppr_scores": {"1": 0.8, "2": 0.9, "3": 0.7, "4": 1.0},
            "typed_ppr_score_parts": {
                "1": {"rerank_rank": 1},
                "2": {"rerank_rank": 3},
                "3": {"rerank_rank": 2},
                "4": {"rerank_rank": 0},
            },
        }
        index_payload = {
            "blocks": [
                {"block_id": 1, "block_type": "paragraph", "text": "Block one."},
                {"block_id": 2, "block_type": "paragraph", "text": "Block two."},
                {"block_id": 3, "block_type": "table", "text": "Block three."},
                {"block_id": 4, "block_type": "entity", "text": "Bridge."},
            ]
        }

        candidates = reconstruct_support_candidates(
            retrieval=retrieval,
            index_payload=index_payload,
            topk=20,
            allowed_types={"paragraph", "table", "caption", "figure"},
        )

        self.assertEqual([item["block_id"] for item in candidates], [2, 1, 3])
        self.assertEqual(candidates[0]["text"], "Selected block.")
        self.assertEqual(candidates[1]["text"], "Block one.")

    def test_fingerprint_changes_for_every_semantic_input_and_excludes_secret(self):
        from Scripts.eval.qasper_support_replay import support_rerank_fingerprint

        case = self._case()
        provider = {
            "model_name": "BAAI/bge-reranker-v2-m3",
            "api_base": "https://api.siliconflow.cn/v1",
            "api_key": "secret-token",
        }
        config = {"answer_conditioned_support_topk": 20}
        baseline = support_rerank_fingerprint(
            question=case["question"],
            draft_answer=case["draft_answer"],
            candidate_payload=case["candidates"],
            provider_identity=provider,
            controller_config=config,
        )
        changed_question = support_rerank_fingerprint(
            question=case["question"] + " changed",
            draft_answer=case["draft_answer"],
            candidate_payload=case["candidates"],
            provider_identity=provider,
            controller_config=config,
        )
        reversed_candidates = support_rerank_fingerprint(
            question=case["question"],
            draft_answer=case["draft_answer"],
            candidate_payload=list(reversed(case["candidates"])),
            provider_identity=provider,
            controller_config=config,
        )

        self.assertNotEqual(baseline, changed_question)
        self.assertNotEqual(baseline, reversed_candidates)
        self.assertNotIn("secret-token", json.dumps(baseline))
        self.assertNotIn("api_key", json.dumps(baseline))

    def test_score_cache_calls_reranker_once_then_reuses_validated_entry(self):
        from Scripts.eval.qasper_support_replay import get_or_build_score_cache

        reranker = FakeReranker([0.7, 0.9, 0.1])
        provider = {
            "model_name": "BAAI/bge-reranker-v2-m3",
            "api_base": "https://api.siliconflow.cn/v1",
            "api_key": "secret-token",
        }
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "q1.json"
            first, first_hit = get_or_build_score_cache(
                cache_path=cache_path,
                reranker=reranker,
                replay_case=self._case(),
                provider_identity=provider,
                controller_config={"answer_conditioned_support_topk": 20},
            )
            second, second_hit = get_or_build_score_cache(
                cache_path=cache_path,
                reranker=reranker,
                replay_case=self._case(),
                provider_identity=provider,
                controller_config={"answer_conditioned_support_topk": 20},
            )
            serialized = cache_path.read_text(encoding="utf-8")

        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertEqual(first, second)
        self.assertEqual(len(reranker.calls), 1)
        self.assertNotIn("secret-token", serialized)
        self.assertEqual(
            [item["block_id"] for item in first["scores"]], [1, 2, 3]
        )

    def test_cache_fingerprint_mismatch_fails_instead_of_reranking(self):
        from Scripts.eval.qasper_support_replay import load_validated_score_cache

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            path.write_text(
                json.dumps({"fingerprint": "old", "scores": []}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_validated_score_cache(path, expected_fingerprint="new")

    def test_replay_uses_same_scores_and_preserves_draft_answer(self):
        from Scripts.eval.qasper_support_replay import replay_controller

        case = self._case()
        score_entry = {
            "scores": [
                {"block_id": 1, "score": 0.7, "rank": 2},
                {"block_id": 2, "score": 0.9, "rank": 1},
                {"block_id": 3, "score": 0.1, "rank": 3},
            ]
        }

        legacy = replay_controller(
            replay_case=case,
            score_entry=score_entry,
            policy="fill_budget",
            min_normalized_relevance=0.5,
            redundancy_overlap_threshold=0.75,
        )
        pruned = replay_controller(
            replay_case=case,
            score_entry=score_entry,
            policy="coverage_prune",
            min_normalized_relevance=0.5,
            redundancy_overlap_threshold=0.75,
        )

        self.assertEqual(legacy["final_ids"], [1, 2, 3])
        self.assertEqual(pruned["final_ids"], [1, 2])
        self.assertEqual(legacy["answer"], case["draft_answer"])
        self.assertEqual(pruned["answer"], case["draft_answer"])
        self.assertEqual(legacy["score_fingerprint"], pruned["score_fingerprint"])

    def test_replay_rejects_incomplete_scores(self):
        from Scripts.eval.qasper_support_replay import replay_controller

        with self.assertRaisesRegex(ValueError, "score IDs"):
            replay_controller(
                replay_case=self._case(),
                score_entry={"scores": [{"block_id": 1, "score": 0.7, "rank": 1}]},
                policy="coverage_prune",
                min_normalized_relevance=0.5,
                redundancy_overlap_threshold=0.75,
            )

    def test_replay_grid_selects_a_configuration_only_when_all_gates_pass(self):
        from Scripts.eval.qasper_support_replay import evaluate_replay_grid

        case = self._case()
        case.update(
            {
                "triggered": True,
                "current_final_ids": [1, 2, 3],
                "text_by_id": {
                    1: "Method A score is 80.",
                    2: "Method B score is 84.",
                    3: "Unrelated setup material.",
                },
            }
        )
        gold = {
            "q1": [
                {
                    "answer": "Method B performs better.",
                    "type": "abstractive",
                    "evidence": [
                        "Method A score is 80.",
                        "Method B score is 84.",
                    ],
                }
            ]
        }
        score_entries = {
            "q1": {
                "fingerprint": "shared-score-cache",
                "scores": [
                    {"block_id": 1, "score": 0.7, "rank": 2},
                    {"block_id": 2, "score": 0.9, "rank": 1},
                    {"block_id": 3, "score": 0.1, "rank": 3},
                ],
            }
        }

        report = evaluate_replay_grid(
            replay_cases=[case],
            gold=gold,
            score_entries=score_entries,
            min_relevance_grid=(0.5,),
            redundancy_grid=(0.75,),
            bootstrap_resamples=100,
        )

        self.assertEqual(report["baseline"]["Evidence F1"], 0.8)
        self.assertEqual(report["selected"]["Evidence F1"], 1.0)
        self.assertGreaterEqual(report["selected"]["delta"]["Evidence F1"], 0.02)
        self.assertTrue(report["selected"]["passes_all_gates"])
        self.assertEqual(report["selected"]["thresholds"]["min_relevance"], 0.5)

    def test_build_replay_cases_requires_complete_matching_outputs(self):
        from Scripts.eval.qasper_support_replay import build_replay_cases

        rows = [
            {
                "question": "Compare A and B.",
                "answer": [],
                "doc_uuid": "paper-1",
                "qasper_question_id": "q1",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc_dir = root / "paper-1"
            query_dir = doc_dir / "eval_qasper_evibridge_support_controller" / "query_001"
            query_dir.mkdir(parents=True)
            (doc_dir / "evibridge_index.json").write_text(
                json.dumps(
                    {
                        "blocks": [
                            {
                                "block_id": 1,
                                "block_type": "paragraph",
                                "text": "A scores 80.",
                            },
                            {
                                "block_id": 2,
                                "block_type": "paragraph",
                                "text": "B scores 84.",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (query_dir / "result.json").write_text(
                json.dumps(
                    {
                        "qasper_question_id": "q1",
                        "question": "Compare A and B.",
                        "answer_short": "B performs better.",
                        "supporting_block_ids": [1, 2],
                    }
                ),
                encoding="utf-8",
            )
            (query_dir / "retrieval_res.json").write_text(
                json.dumps(
                    {
                        "demand": {
                            "intent": "comparison",
                            "subqueries": ["A score", "B score"],
                        },
                        "selected": [
                            {
                                "block_id": 1,
                                "block_type": "paragraph",
                                "text": "A scores 80.",
                                "score_parts": {"rerank_rank": 1},
                                "evidence_role": "answer_evidence",
                            }
                        ],
                        "typed_ppr_scores": {"1": 0.9, "2": 0.8},
                        "typed_ppr_score_parts": {
                            "1": {"rerank_rank": 1},
                            "2": {"rerank_rank": 2},
                        },
                        "supporting_block_ids": [1, 2],
                        "support_controller": {
                            "triggered": True,
                            "anchored_block_ids": [1],
                        },
                        "verification": {
                            "sufficient": True,
                            "missing": [],
                            "missing_types": [],
                            "missing_bridge_types": [],
                            "next_action": "accept",
                        },
                    }
                ),
                encoding="utf-8",
            )

            cases = build_replay_cases(
                dataset_rows=rows,
                retrieval_root=root,
                method="evibridge_support_controller",
                topk=20,
            )

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["question_id"], "q1")
        self.assertEqual(cases[0]["anchor_ids"], [1])
        self.assertEqual(cases[0]["current_final_ids"], [1, 2])
        self.assertEqual(cases[0]["max_items"], 3)
        self.assertFalse(cases[0]["verifier_state_contradiction"])

    def test_run_replay_experiment_writes_cache_report_and_decision(self):
        from Scripts.eval.qasper_support_replay import run_replay_experiment

        case = self._case()
        case.update(
            {
                "triggered": True,
                "current_final_ids": [1, 2, 3],
                "text_by_id": {
                    1: "Method A score is 80.",
                    2: "Method B score is 84.",
                    3: "Unrelated setup material.",
                },
                "verifier_state_contradiction": False,
            }
        )
        rows = [
            {
                "question": case["question"],
                "qasper_question_id": "q1",
                "doc_uuid": "paper-1",
                "answer": [
                    {
                        "unanswerable": False,
                        "extractive_spans": [],
                        "free_form_answer": case["draft_answer"],
                        "yes_no": None,
                        "evidence": [
                            "Method A score is 80.",
                            "Method B score is 84.",
                        ],
                    }
                ],
            }
        ]
        reranker = FakeReranker([0.7, 0.9, 0.1])
        with tempfile.TemporaryDirectory() as tmp, patch(
            "Scripts.eval.qasper_support_replay.build_replay_cases",
            return_value=[case],
        ):
            root = Path(tmp)
            report = run_replay_experiment(
                dataset_rows=rows,
                retrieval_root=root / "unused",
                method="evibridge_support_controller",
                cache_dir=root / "cache",
                output_dir=root / "output",
                reranker=reranker,
                provider_identity={
                    "model_name": "BAAI/bge-reranker-v2-m3",
                    "api_base": "https://api.siliconflow.cn/v1",
                    "api_key": "secret-token",
                },
                controller_config={
                    "answer_conditioned_support_topk": 20,
                    "rerank_batch_size": 50,
                },
                min_relevance_grid=(0.5,),
                redundancy_grid=(0.75,),
                bootstrap_resamples=100,
                decision_provenance={
                    "source_commit": "abc123",
                    "config_sha256": "config-hash",
                    "manifest_sha256": "manifest-hash",
                    "dataset_sha256": "dataset-hash",
                },
            )
            report_payload = json.loads(
                (root / "output" / "replay_report.json").read_text(encoding="utf-8")
            )
            decision_payload = json.loads(
                (root / "output" / "replay_decision.json").read_text(
                    encoding="utf-8"
                )
            )
            cache_payload = (root / "cache" / "q1.json").read_text(
                encoding="utf-8"
            )

        self.assertTrue(report["selected"]["passes_all_gates"])
        self.assertEqual(report_payload["cache_misses"], 1)
        self.assertEqual(decision_payload["question_count"], 1)
        self.assertEqual(decision_payload["provenance"]["source_commit"], "abc123")
        self.assertEqual(
            decision_payload["provenance"]["score_cache_sha256"],
            report_payload["score_cache_sha256"],
        )
        self.assertNotIn("secret-token", cache_payload)
        self.assertEqual(len(reranker.calls), 1)


if __name__ == "__main__":
    unittest.main()
