import inspect
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from Core.Index.EvidenceBridgeIndex import EvidenceBlock
from Core.configs.rag.evibridge_config import EviBridgeRAGConfig
from Core.configs.system_config import load_system_config
from Core.rag.evibridge_demand import EvidenceDemand
from Core.rag.evibridge_verifier import (
    EvidenceSufficiencyVerifier,
    RuleBasedSufficiencyVerifier,
)


class VerifierLLM:
    def __init__(self, payload=None, error=None):
        self.payload = payload or {"sufficient": True}
        self.error = error
        self.calls = 0

    def get_json_completion(self, prompt, schema):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return schema(**self.payload)


class EviBridgeVerifierModeTests(unittest.TestCase):
    def setUp(self):
        self.query = "alpha beta"
        self.evidence = [
            EvidenceBlock(block_id=1, block_type="paragraph", text="alpha"),
            EvidenceBlock(block_id=2, block_type="paragraph", text="beta"),
        ]
        self.demand = EvidenceDemand(intent="multi-hop")

    def _verifier(self, **options):
        self.assertIn("acceptance_mode", inspect.signature(EvidenceSufficiencyVerifier).parameters)
        return EvidenceSufficiencyVerifier(**options)

    def _verify(self, verifier, demand=None, evidence=None):
        return verifier.verify(
            self.query,
            self.demand if demand is None else demand,
            self.evidence if evidence is None else evidence,
            [],
        )

    def test_rule_only_minimal_llm_accept_keeps_rule_missing_and_repair(self):
        verifier = self._verifier(
            llm=VerifierLLM(), enable_llm=True, acceptance_mode="rule_only"
        )
        rule = RuleBasedSufficiencyVerifier().verify(
            self.query, self.demand, self.evidence, []
        )

        final = self._verify(verifier)

        self.assertFalse(final.sufficient)
        self.assertEqual(final.missing, rule.missing)
        self.assertEqual(final.missing_bridge_types, rule.missing_bridge_types)
        self.assertEqual(final.next_bridge, rule.next_bridge)
        self.assertEqual(final.next_action, rule.next_action)
        self.assertEqual(final.reason, rule.reason)
        self.assertTrue(verifier.last_trace["llm_verdict"]["sufficient"])
        self.assertFalse(verifier.last_trace["final_verdict"]["sufficient"])
        self.assertEqual(verifier.last_trace["decision_source"], "rule")

    def test_rule_only_minimal_llm_accept_keeps_combined_missing(self):
        demand = EvidenceDemand(
            intent="multi-hop", scope="document", granularity="summary"
        )
        verifier = self._verifier(
            llm=VerifierLLM(), enable_llm=True, acceptance_mode="rule_only"
        )

        final = self._verify(verifier, demand=demand)

        self.assertFalse(final.sufficient)
        self.assertEqual(final.missing, ["semantic_link", "hierarchy_context"])
        self.assertEqual(final.missing_types, ["summary"])
        self.assertEqual(final.missing_bridge_types, ["semantic", "hierarchy"])
        self.assertEqual(final.next_action, "expand_semantic_bridge")

    def test_rule_only_llm_can_revise_diagnosis_but_not_rule_metrics(self):
        llm = VerifierLLM(
            {
                "sufficient": True,
                "missing": ["causal_link"],
                "missing_types": ["ENTITY", "invalid"],
                "missing_bridge_types": ["hierarchy", "invalid"],
                "next_bridge": ["hierarchy"],
                "next_action": "expand_hierarchy_context",
                "reason": "Follow the source hierarchy",
                "relevance": 0.12,
                "connectivity": 0.98,
                "coverage": 0.31,
                "specificity": 0.27,
                "noise": 0.89,
                "noise_warning": True,
            }
        )
        verifier = self._verifier(
            llm=llm, enable_llm=True, acceptance_mode="rule_only"
        )
        rule = RuleBasedSufficiencyVerifier().verify(
            self.query, self.demand, self.evidence, []
        )

        final = self._verify(verifier)

        self.assertFalse(final.sufficient)
        self.assertEqual(final.missing, ["causal_link"])
        self.assertEqual(final.missing_types, ["entity"])
        self.assertEqual(final.missing_bridge_types, ["hierarchy"])
        self.assertEqual(final.next_bridge, ["hierarchy"])
        self.assertEqual(final.next_action, "expand_hierarchy_context")
        self.assertEqual(final.reason, "Follow the source hierarchy")
        for name in (
            "relevance", "connectivity", "coverage", "specificity", "noise",
            "noise_warning",
        ):
            with self.subTest(field=name):
                self.assertEqual(getattr(final, name), getattr(rule, name))
        self.assertEqual(verifier.last_trace["llm_status"], "success")
        self.assertEqual(verifier.last_trace["llm_verdict"]["missing_types"], ["ENTITY", "invalid"])
        self.assertEqual(verifier.last_trace["rule_verdict"], rule.model_dump(mode="json"))
        self.assertEqual(verifier.last_trace["final_verdict"], final.model_dump(mode="json"))

    def test_legacy_hybrid_still_allows_llm_soft_accept(self):
        for options in ({}, {"acceptance_mode": "legacy_hybrid"}):
            with self.subTest(options=options):
                verifier = self._verifier(
                    llm=VerifierLLM(), enable_llm=True, **options
                )
                final = self._verify(verifier)
                self.assertTrue(final.sufficient)
                self.assertEqual(final.missing, [])
                self.assertEqual(final.next_action, "accept")
                self.assertEqual(verifier.last_trace["acceptance_mode"], "legacy_hybrid")
                self.assertEqual(verifier.last_trace["decision_source"], "legacy_hybrid")

    def test_gates_do_not_call_llm_and_record_trace(self):
        scenarios = (
            ("rule_sufficient", True, True, EvidenceDemand(), [
                EvidenceBlock(block_id=1, block_type="paragraph", text=self.query)
            ]),
            ("hard_rule_missing", True, True, self.demand, []),
            ("llm_disabled", False, True, self.demand, self.evidence),
            ("llm_unavailable", True, False, self.demand, self.evidence),
        )
        for mode in ("legacy_hybrid", "rule_only"):
            for reason, enabled, available, demand, evidence in scenarios:
                with self.subTest(mode=mode, skip_reason=reason):
                    llm = VerifierLLM()
                    verifier = self._verifier(
                        llm=llm if available else None,
                        enable_llm=enabled,
                        acceptance_mode=mode,
                    )
                    final = self._verify(verifier, demand=demand, evidence=evidence)
                    self.assertEqual(llm.calls, 0)
                    trace = verifier.last_trace
                    self.assertEqual(trace["schema_version"], 1)
                    self.assertEqual(trace["acceptance_mode"], mode)
                    self.assertEqual(trace["llm_status"], "skipped")
                    self.assertEqual(trace["skip_reason"], reason)
                    self.assertIsNone(trace["llm_verdict"])
                    self.assertIsNone(trace["error_type"])
                    self.assertIs(trace.get("llm_call_attempted"), False)
                    self.assertIsNone(trace.get("error_stage"))
                    self.assertEqual(trace["decision_source"], "rule")
                    self.assertEqual(trace["rule_verdict"], final.model_dump(mode="json"))
                    self.assertEqual(trace["final_verdict"], final.model_dump(mode="json"))
                    json.dumps(trace)

    def test_exception_falls_back_to_rule_and_records_only_error_class(self):
        private_message = "private service request and key must not be logged"
        for mode in ("legacy_hybrid", "rule_only"):
            with self.subTest(mode=mode):
                verifier = self._verifier(
                    llm=VerifierLLM(error=RuntimeError(private_message)),
                    enable_llm=True,
                    acceptance_mode=mode,
                )
                final = self._verify(verifier)
                trace = verifier.last_trace
                self.assertFalse(final.sufficient)
                self.assertEqual(trace["llm_status"], "error")
                self.assertEqual(trace["error_type"], "RuntimeError")
                self.assertIs(trace.get("llm_call_attempted"), True)
                self.assertEqual(trace.get("error_stage"), "llm")
                self.assertIsNone(trace["skip_reason"])
                self.assertIsNone(trace["llm_verdict"])
                self.assertEqual(trace["decision_source"], "rule")
                self.assertEqual(trace["rule_verdict"], trace["final_verdict"])
                self.assertNotIn(private_message, json.dumps(trace))

    def test_prompt_error_records_no_llm_call_and_falls_back_to_rule(self):
        for mode in ("legacy_hybrid", "rule_only"):
            with self.subTest(mode=mode):
                llm = VerifierLLM()
                verifier = self._verifier(llm=llm, enable_llm=True, acceptance_mode=mode)
                with patch.object(verifier, "_prompt", side_effect=ValueError("private prompt input")):
                    final = self._verify(verifier)
                trace = verifier.last_trace
                self.assertFalse(final.sufficient)
                self.assertEqual(llm.calls, 0)
                self.assertIs(trace.get("llm_call_attempted"), False)
                self.assertEqual(trace.get("error_stage"), "prompt")
                self.assertEqual(trace["error_type"], "ValueError")
                self.assertEqual(trace["llm_status"], "error")
                self.assertIsNone(trace["llm_verdict"])
                self.assertEqual(trace["rule_verdict"], trace["final_verdict"])
                self.assertNotIn("private prompt input", json.dumps(trace))

    def test_sanitize_error_records_attempt_and_preserves_parsed_llm_verdict(self):
        for mode in ("legacy_hybrid", "rule_only"):
            with self.subTest(mode=mode):
                llm = VerifierLLM()
                verifier = self._verifier(llm=llm, enable_llm=True, acceptance_mode=mode)
                with patch.object(EvidenceSufficiencyVerifier, "_sanitize_llm_verdict", side_effect=TypeError("private verdict")):
                    final = self._verify(verifier)
                trace = verifier.last_trace
                self.assertFalse(final.sufficient)
                self.assertEqual(llm.calls, 1)
                self.assertIs(trace.get("llm_call_attempted"), True)
                self.assertEqual(trace.get("error_stage"), "sanitize")
                self.assertEqual(trace["error_type"], "TypeError")
                self.assertEqual(trace["llm_status"], "error")
                self.assertTrue(trace["llm_verdict"]["sufficient"])
                self.assertEqual(trace["rule_verdict"], trace["final_verdict"])
                self.assertNotIn("private verdict", json.dumps(trace))

    def test_rule_error_is_recorded_then_propagates_and_next_call_resets(self):
        llm = VerifierLLM()
        verifier = self._verifier(llm=llm, enable_llm=True, acceptance_mode="rule_only")
        with patch.object(verifier.rule_verifier, "verify", side_effect=LookupError("private rule input")):
            with self.assertRaises(LookupError):
                self._verify(verifier)
        failed_trace = verifier.last_trace
        self.assertEqual(llm.calls, 0)
        self.assertIs(failed_trace.get("llm_call_attempted"), False)
        self.assertEqual(failed_trace.get("error_stage"), "rule")
        self.assertEqual(failed_trace["error_type"], "LookupError")
        self.assertEqual(failed_trace["skip_reason"], "rule_error")
        self.assertEqual(failed_trace["llm_status"], "skipped")
        self.assertIsNone(failed_trace["rule_verdict"])
        self.assertIsNone(failed_trace["final_verdict"])
        self.assertNotIn("private rule input", json.dumps(failed_trace))

        self._verify(verifier)

        self.assertIsNot(verifier.last_trace, failed_trace)
        self.assertIs(verifier.last_trace["llm_call_attempted"], True)
        self.assertIsNone(verifier.last_trace["error_stage"])
        self.assertIsNone(verifier.last_trace["error_type"])
        self.assertIsNone(verifier.last_trace["skip_reason"])
        self.assertEqual(verifier.last_trace["llm_status"], "success")

    def test_trace_is_fresh_and_detached_for_each_verify_call(self):
        verifier = self._verifier(
            llm=VerifierLLM(), enable_llm=True, acceptance_mode="rule_only"
        )
        final = self._verify(verifier)
        first_trace = verifier.last_trace
        self.assertIs(first_trace.get("llm_call_attempted"), True)
        self.assertIsNone(first_trace.get("error_stage"))
        first_snapshot = json.dumps(first_trace, sort_keys=True)
        final.missing.append("mutated_by_caller")
        self.assertEqual(json.dumps(first_trace, sort_keys=True), first_snapshot)

        self._verify(verifier, evidence=[])

        self.assertIsNot(verifier.last_trace, first_trace)
        self.assertEqual(json.dumps(first_trace, sort_keys=True), first_snapshot)
        self.assertEqual(verifier.last_trace["llm_status"], "skipped")
        self.assertIsNone(verifier.last_trace["llm_verdict"])
        self.assertIsNone(verifier.last_trace["error_type"])
        self.assertIs(verifier.last_trace.get("llm_call_attempted"), False)
        self.assertIsNone(verifier.last_trace.get("error_stage"))


class EviBridgeVerifierModeConfigTests(unittest.TestCase):
    def test_omitted_modes_keep_legacy_defaults_and_suffixes(self):
        legacy_options = (
            ({}, "evibridge"),
            ({"support_completion_policy": "weak_only"}, "evibridge_support_controller"),
            ({"support_selection_policy": "coverage_prune"}, "evibridge_support_pruned"),
            ({"support_selection_policy": "coverage_prune", "enable_long_context_fallback": True},
             "evibridge_fallback_support_pruned"),
            ({"ablation_variant": "wo_typed_weights", "enable_long_context_fallback": True},
             "evibridge_wo_typed_weights_fallback"),
        )
        for options, expected in legacy_options:
            with self.subTest(options=options):
                cfg = EviBridgeRAGConfig(**options)
                self.assertIn("verifier_acceptance_mode", EviBridgeRAGConfig.model_fields)
                self.assertIn("support_context_policy", EviBridgeRAGConfig.model_fields)
                self.assertEqual(cfg.verifier_acceptance_mode, "legacy_hybrid")
                self.assertEqual(cfg.support_context_policy, "legacy")
                self.assertEqual(cfg.method_suffix, expected)

    def test_modes_append_suffix_after_existing_suffix_parts(self):
        cfg = EviBridgeRAGConfig(
            support_selection_policy="coverage_prune",
            enable_long_context_fallback=True,
            verifier_acceptance_mode="rule_only",
            support_context_policy="strict",
        )
        self.assertEqual(
            cfg.method_suffix,
            "evibridge_fallback_support_pruned_rule_only_strict_context",
        )
        self.assertEqual(EviBridgeRAGConfig(verifier_acceptance_mode="rule_only").method_suffix,
                         "evibridge_rule_only")
        self.assertEqual(EviBridgeRAGConfig(support_context_policy="strict").method_suffix,
                         "evibridge_strict_context")

    def test_invalid_modes_are_rejected(self):
        for field in ("verifier_acceptance_mode", "support_context_policy"):
            with self.subTest(field=field):
                with self.assertRaises(ValidationError):
                    EviBridgeRAGConfig(**{field: "invalid"})

    def test_auditable_template_inherits_budgets_and_has_distinct_suffix(self):
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / "config" / "evibridge_auditable.yaml").is_file())
        cfg = load_system_config(str(root / "config" / "evibridge_auditable.yaml"))
        legacy = load_system_config(str(root / "config" / "evibridge_support_pruned.yaml"))
        rag = cfg.rag.strategy_config
        self.assertEqual(rag.verifier_acceptance_mode, "rule_only")
        self.assertEqual(rag.support_context_policy, "strict")
        self.assertFalse(rag.regenerate_on_support_expansion)
        self.assertEqual(rag.support_selection_policy, "coverage_prune")
        self.assertEqual(rag.method_suffix, "evibridge_support_pruned_rule_only_strict_context")
        for field in (
            "bm25_topk", "embedding_topk", "patch_topk", "entity_topk",
            "candidate_rerank_topk", "ppr_topk", "max_context_blocks",
            "max_context_tokens", "max_iterations", "supporting_evidence_topk",
        ):
            with self.subTest(field=field):
                self.assertEqual(getattr(rag, field), getattr(legacy.rag.strategy_config, field))


if __name__ == "__main__":
    unittest.main()
