import copy
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from tests import test_evibridge_modules as fixtures
FakeReranker = fixtures.FakeReranker
_stub_rag_provider_imports = fixtures._stub_rag_provider_imports
_stub_rag_provider_imports()
from Core.configs.rag.evibridge_config import EviBridgeRAGConfig
from Core.rag.evibridge_demand import EvidenceDemand
from Core.rag.evibridge_rag import EviBridgeRAG
from Core.rag.evibridge_selector import SelectedEvidence
from Core.rag.evibridge_verifier import SufficiencyVerdict

QUERY = "Compare Method A and Method B."
DRAFT = json.dumps({"answer_short": "Method B performs better than Method A.", "supporting_block_ids": [1]})
FINAL = json.dumps({"answer_short": "Method B scores 84 while Method A scores 80.", "supporting_block_ids": [3]})


class RecordingLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get_completion(self, prompt, json_response=False):
        self.calls.append({"prompt": prompt, "ids": [int(v) for v in re.findall(r"\[block_id=(\d+)\]", prompt)]})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class GenerationContextTests(unittest.TestCase):
    def make_rag(self, overrides=None, responses=None, sufficient=True):
        index = fixtures.EviBridgeModuleTests()._build_index()
        config_args = dict(support_context_policy="strict", support_completion_policy="weak_only",
                           support_selection_policy="fill_budget", enable_answer_conditioned_support_rerank=True,
                           regenerate_on_support_expansion=True, enable_llm_verifier=False,
                           enable_long_context_fallback=False, enable_short_answer_extraction=False)
        config_args.update(overrides or {})
        llm = RecordingLLM(responses if responses is not None else [DRAFT, FINAL])
        rag = EviBridgeRAG(config=EviBridgeRAGConfig(**config_args), llm=llm, bm25=None,
                           evibridge_index=index, reranker=FakeReranker({"Table 1. Accuracy": 0.99,
                                             "Method B compares": 0.8, "Method A uses": 0.2}))
        selected = [dict(block_id=i, block_type=index.blocks[i].block_type,
                         page=index.blocks[i].page, section_id="Methods", selection_rank=rank,
                         text=index.blocks[i].text, evidence_role="answer_evidence",
                         score_parts={"rerank_rank": rank}) for rank, i in enumerate([1, 2], 1)]
        info = dict(query=QUERY, demand=EvidenceDemand(intent="comparison", subqueries=["Method A score", "Method B score"]),
                    seed_results=[], typed_ppr_scores={1: 0.9, 2: 0.8, 3: 0.7},
                    typed_ppr_score_parts={1: {"rerank_rank": 1}, 2: {"rerank_rank": 2}, 3: {"rerank_rank": 3}},
                    connector_paths=[], connector_edges=[], selected_payload=selected, selected_block_ids=[1, 2],
                    selected_bridges=[], retrieved_block_ids=[1, 2], supporting_evidence=[], supporting_block_ids=[],
                    evidence_chain=rag._answer_context_chain(selected), iterations=[],
                    verification=SufficiencyVerdict(sufficient=sufficient), stopping_reason="sufficient")
        info["evidence_chain"] = rag._build_evidence_chain(
            [SelectedEvidence(block=index.blocks[item["block_id"]], score=1.0,
                              score_parts=item["score_parts"], selection_rank=item["selection_rank"])
             for item in selected], info["demand"], info["verification"])
        rag._retrieve = lambda query: copy.deepcopy(info)
        return rag, llm, selected

    def generate(self, rag):
        with tempfile.TemporaryDirectory() as td:
            answer, retrieved = rag.generation(QUERY, td)
            saved = json.loads((Path(td)/"retrieval_res.json").read_text(encoding="utf-8"))
            chain = json.loads((Path(td)/"evidence_chain.json").read_text(encoding="utf-8"))
        return answer, retrieved, saved, chain

    def assert_context_trace(self, saved, llm, retained_stage):
        self.assertIn("generation_provenance", saved)
        p = saved["generation_provenance"]
        self.assertEqual(p["retained_stage"], retained_stage)
        calls = p["calls"]
        self.assertEqual(len(calls), len(llm.calls))
        for call, actual in zip(calls, llm.calls):
            self.assertEqual(call["prompt_block_ids"], actual["ids"])
            self.assertEqual(call["prompt_sha256"], hashlib.sha256(actual["prompt"].encode("utf-8")).hexdigest())
            self.assertNotIn("prompt", call)
        self.assertEqual(saved["answer_context_block_ids"], p["prompt_block_ids"])
        self.assertLessEqual(set(saved["supporting_block_ids"]), set(p["prompt_block_ids"]))
        self.assertTrue(saved["support_context_validation"]["passed"])

    def test_coverage_prune_disabled_keeps_external_support_posthoc(self):
        rag, llm, _ = self.make_rag(dict(support_selection_policy="coverage_prune", regenerate_on_support_expansion=False,
                                       support_min_normalized_relevance=0.5))
        _, _, saved, chain = self.generate(rag)
        self.assertEqual(saved["supporting_block_ids"], [1])
        self.assertEqual([x["block_id"] for x in saved["posthoc_supporting_evidence"]], [3])
        self.assertEqual(len(llm.calls), 1)
        self.assert_context_trace(saved, llm, "initial")
        self.assertEqual(chain["generation_provenance"], saved["generation_provenance"])

    def test_coverage_prune_enabled_regenerates_external_support(self):
        rag, llm, _ = self.make_rag(dict(support_selection_policy="coverage_prune", support_min_normalized_relevance=0.5))
        _, _, saved, _ = self.generate(rag)
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(saved["supporting_block_ids"], [3])
        self.assertTrue(saved["answer_regeneration"]["used"])
        self.assert_context_trace(saved, llm, "regeneration")

    def test_disabled_regeneration_does_not_save_planned_context_as_actual(self):
        rag, llm, _ = self.make_rag(dict(regenerate_on_support_expansion=False))
        _, _, saved, _ = self.generate(rag)
        self.assertLessEqual(set(saved["supporting_block_ids"]), {1, 2})
        self.assertNotIn(3, saved["answer_context_block_ids"])
        self.assertIn(3, saved["planned_answer_context_block_ids"])
        self.assert_context_trace(saved, llm, "initial")

    def test_regeneration_failure_retains_draft_context_and_redacts_error(self):
        rag, llm, _ = self.make_rag(responses=[DRAFT, RuntimeError("secret=fake-token endpoint=http://private.invalid")])
        answer, _, saved, _ = self.generate(rag)
        self.assertEqual(answer, DRAFT)
        self.assertLessEqual(set(saved["supporting_block_ids"]), {1, 2})
        self.assertFalse(saved["answer_regeneration"]["used"])
        self.assert_context_trace(saved, llm, "initial")
        trace_text = json.dumps(saved["generation_provenance"])
        self.assertNotIn("fake-token", trace_text)
        self.assertEqual(saved["generation_provenance"]["calls"][-1]["error_type"], "RuntimeError")
        self.assertFalse(saved["generation_provenance"]["calls"][-1]["success"])

    def test_budget_clipping_and_invalid_final_citations_remove_unseen_support(self):
        invalid = json.dumps({"answer_short": "Method B has higher accuracy.", "supporting_block_ids": [999]})
        rag, llm, _ = self.make_rag(dict(max_context_blocks=2), responses=[DRAFT, invalid])
        _, _, saved, _ = self.generate(rag)
        self.assertEqual(llm.calls[-1]["ids"], [1, 3])
        self.assertNotIn(2, saved["supporting_block_ids"])
        self.assertIn(2, saved["support_context_validation"]["removed_ids"])
        self.assert_context_trace(saved, llm, "regeneration")

    def test_nonjson_final_response_is_still_checked_against_actual_context(self):
        rag, llm, _ = self.make_rag(dict(max_context_blocks=2), responses=[DRAFT, "Method B has higher accuracy."])
        _, _, saved, _ = self.generate(rag)
        self.assertNotIn(2, saved["supporting_block_ids"])
        self.assert_context_trace(saved, llm, "regeneration")

    def test_fallback_uses_retained_prompt_instead_of_selected_union(self):
        response = json.dumps({"answer_short": "Method A uses retrieval.", "supporting_block_ids": [1, 2]})
        rag, llm, selected = self.make_rag(dict(enable_long_context_fallback=True, support_completion_policy="none"),
                                           responses=[DRAFT, response], sufficient=False)
        rag._build_long_context_fallback = lambda **kwargs: [selected[0]]
        _, _, saved, _ = self.generate(rag)
        self.assertEqual(saved["supporting_block_ids"], [1])
        self.assertEqual(saved["citation_validation"]["invalid_ids"], [2])
        self.assert_context_trace(saved, llm, "fallback")

    def test_fallback_failure_keeps_initial_response(self):
        rag, llm, selected = self.make_rag(dict(enable_long_context_fallback=True, support_completion_policy="none"),
                                           responses=[DRAFT, RuntimeError("fake failure")], sufficient=False)
        rag._build_long_context_fallback = lambda **kwargs: [selected[0]]
        answer, _, saved, _ = self.generate(rag)
        self.assertEqual(answer, DRAFT)
        self.assertFalse(saved["fallback"]["used"])
        self.assert_context_trace(saved, llm, "initial")

    def test_legacy_output_is_preserved_but_membership_violation_is_visible(self):
        rag, llm, _ = self.make_rag(dict(support_context_policy="legacy", support_selection_policy="coverage_prune",
                                       regenerate_on_support_expansion=False, support_min_normalized_relevance=0.5))
        _, _, saved, _ = self.generate(rag)
        self.assertEqual(saved["supporting_block_ids"], [1, 3])
        self.assertIn("support_context_validation", saved)
        self.assertFalse(saved["support_context_validation"]["passed"])
        self.assertEqual(saved["support_context_validation"]["outside_context_ids"], [3])
        self.assertEqual(len(llm.calls), 1)

    def test_trace_is_reset_for_each_question(self):
        rag, llm, _ = self.make_rag(dict(support_completion_policy="none"), responses=[DRAFT, DRAFT])
        _, _, first, _ = self.generate(rag)
        _, _, second, _ = self.generate(rag)
        self.assertIn("generation_provenance", second)
        self.assertEqual(len(first["generation_provenance"]["calls"]), 1)
        self.assertEqual(len(second["generation_provenance"]["calls"]), 1)
        self.assertEqual(rag.last_generation_provenance, second["generation_provenance"])
        self.assertEqual(rag.last_support_context_validation, second["support_context_validation"])

    def test_rule_only_mode_is_wired_to_rag_verifier(self):
        rag, _, _ = self.make_rag(dict(verifier_acceptance_mode="rule_only"))
        self.assertEqual(getattr(rag.verifier, "acceptance_mode", None), "rule_only")

    def test_actual_initial_chain_preserves_exact_support_source_mapping(self):
        from Eval.utils.hotpotqa_eval import _facts_from_payload
        for completion in ("none", "weak_only"):
            with self.subTest(completion=completion):
                rag, llm, _ = self.make_rag(dict(support_completion_policy=completion))
                info = rag._retrieve(QUERY)
                selected = []
                for rank, block_id in enumerate([1, 2], 1):
                    block = rag.evibridge_index.blocks[block_id]
                    block.metadata = {"hotpot_title": "Document" + str(block_id),
                                      "hotpot_sent_id": block_id,
                                      "paragraph_source_ids": [block_id * 10]}
                    selected.append(SelectedEvidence(block=block, score=1.0,
                                                     selection_rank=rank))
                info["demand"] = EvidenceDemand(intent="fact")
                info["selected_payload"] = rag._selected_payload(selected, {})
                info["evidence_chain"] = rag._build_evidence_chain(
                    selected, info["demand"], info["verification"])
                rag._retrieve = lambda query: copy.deepcopy(info)
                _, _, saved, _ = self.generate(rag)
                self.assertEqual(saved["supporting_block_ids"], [1])
                self.assertEqual(_facts_from_payload(saved), [["Document1", 1]])
                self.assertEqual(saved["supporting_evidence"][0]["metadata"]["paragraph_source_ids"], [10])
                self.assert_context_trace(saved, llm, "initial")


if __name__ == "__main__":
    unittest.main()
