import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from Core.Index.EvidenceBridgeIndex import EvidenceBlock, EvidenceBridge, EvidenceBridgeIndex
from Core.configs.rag_config import RAGConfig


def _stub_rag_provider_imports():
    class FakeOpenAIClient:
        def __init__(self, *args, **kwargs):
            pass

    openai_module = types.ModuleType("openai")
    openai_module.OpenAI = FakeOpenAIClient
    tiktoken_module = types.ModuleType("tiktoken")
    tiktoken_module.Encoding = object
    tiktoken_module.get_encoding = lambda name: types.SimpleNamespace(
        encode=lambda text: str(text).split()
    )
    json_repair_module = types.ModuleType("json_repair")
    json_repair_module.repair_json = lambda json_str, return_objects=False: json_str
    sys.modules.setdefault("openai", openai_module)
    sys.modules.setdefault("ollama", types.ModuleType("ollama"))
    sys.modules.setdefault("tiktoken", tiktoken_module)
    sys.modules.setdefault("json_repair", json_repair_module)


class FakeLLM:
    def __init__(self):
        self.prompt = ""

    def get_completion(self, prompt, json_response=False):
        self.prompt = prompt
        return "Answer based on bridged evidence."

    def get_json_completion(self, prompt, schema, images=None, think_mode=False):
        return schema(
            sufficient=True,
            missing=[],
            relevance=0.9,
            connectivity=0.8,
            coverage=0.8,
            specificity=0.8,
            noise=0.1,
            next_bridge=[],
            reason="enough evidence",
        )


class FakeReranker:
    def __init__(self, scores_by_text=None):
        self.scores_by_text = scores_by_text or {}
        self.calls = []

    def rerank(self, query, documents, instruction=None, batch_size=4):
        self.calls.append(
            {
                "query": query,
                "documents": documents,
                "instruction": instruction,
                "batch_size": batch_size,
            }
        )
        return [
            float(
                max(
                    (
                        score
                        for marker, score in self.scores_by_text.items()
                        if marker in document
                    ),
                    default=0.1,
                )
            )
            for document in documents
        ]


class EviBridgeModuleTests(unittest.TestCase):
    def _build_index(self):
        blocks = {
            1: EvidenceBlock(
                block_id=1,
                block_type="paragraph",
                text="Method A uses retrieval augmented generation.",
                title_path=["Methods"],
                page=1,
            ),
            2: EvidenceBlock(
                block_id=2,
                block_type="paragraph",
                text="Method B compares retrieval with graph reasoning.",
                title_path=["Methods"],
                page=2,
            ),
            3: EvidenceBlock(
                block_id=3,
                block_type="table",
                text="Table 1. Accuracy results\nMethod A | 80\nMethod B | 84",
                title_path=["Experiments"],
                page=3,
            ),
            4: EvidenceBlock(
                block_id=4,
                block_type="summary",
                text="The paper evaluates retrieval and graph reasoning methods.",
                title_path=["Document Summary"],
                page=None,
            ),
            5: EvidenceBlock(
                block_id=5,
                block_type="entity",
                text="retrieval",
                title_path=["Entities"],
                page=None,
                metadata={"entity": "retrieval"},
            ),
            6: EvidenceBlock(
                block_id=6,
                block_type="patch",
                text="Evidence patch for Methods: Method A uses retrieval augmented generation. Method B compares retrieval with graph reasoning.",
                title_path=["Methods"],
                page=None,
                metadata={"generated_type": "section_patch", "contains_block_ids": [1, 2]},
            ),
        }
        bridges = [
            EvidenceBridge(source_id=1, target_id=2, bridge_type="semantic", relation_type="shared_terms", weight=0.8),
            EvidenceBridge(source_id=2, target_id=3, bridge_type="context", relation_type="reference_to_table", weight=0.9),
            EvidenceBridge(source_id=1, target_id=4, bridge_type="hierarchy", relation_type="belongs_to_summary", weight=0.7),
            EvidenceBridge(source_id=4, target_id=3, bridge_type="hierarchy", relation_type="summary_to_block", weight=0.7),
            EvidenceBridge(source_id=1, target_id=5, bridge_type="semantic", relation_type="mentions_entity", weight=0.75),
            EvidenceBridge(source_id=5, target_id=2, bridge_type="semantic", relation_type="mentioned_by", weight=0.55),
            EvidenceBridge(source_id=6, target_id=1, bridge_type="context", relation_type="patch_contains", weight=0.72),
            EvidenceBridge(source_id=6, target_id=2, bridge_type="context", relation_type="patch_contains", weight=0.72),
        ]
        return EvidenceBridgeIndex(save_dir="", blocks=blocks, bridges=bridges)

    def test_demand_parser_returns_controlled_vector_and_weights(self):
        _stub_rag_provider_imports()
        from Core.rag.evibridge_demand import DemandParser, EvidenceDemand, sanitize_demand, weights_for_demand

        parser = DemandParser(llm=None, mode="rule")

        table_demand = parser.parse("Which table reports the accuracy results?")
        compare_demand = parser.parse("Compare Method A and Method B using supporting evidence.")
        global_demand = parser.parse("Summarize the whole document.")
        fact_demand = parser.parse("Which datasets did they experiment with?")
        dirty_demand = EvidenceDemand(
            intent="fact",
            scope="document",
            modality=["text"],
            granularity="entity",
            bridge_need=["yes", "language_pairs"],
        )
        fallback = EvidenceDemand(bridge_need=["context", "semantic"])
        sanitized = sanitize_demand(dirty_demand, fallback=fallback)

        self.assertEqual(table_demand.intent, "table-figure")
        self.assertIn("table", table_demand.modality)
        self.assertEqual(compare_demand.intent, "comparison")
        self.assertIn("semantic", compare_demand.bridge_need)
        self.assertEqual(global_demand.scope, "document")
        self.assertEqual(fact_demand.granularity, "block")
        self.assertEqual(sanitized.bridge_need, ["context", "semantic"])
        self.assertEqual(weights_for_demand(table_demand), {"context": 0.65, "semantic": 0.2, "hierarchy": 0.15})

    def test_qasper_conservative_demand_avoids_fact_question_overrouting(self):
        _stub_rag_provider_imports()
        from Core.rag.evibridge_demand import DemandParser

        parser = DemandParser(
            llm=None,
            mode="rule",
            qasper_demand_mode="conservative",
            multi_hop_requires_explicit_bridge=True,
        )

        numeric = parser.parse("How many participants were trying this communication game?")
        boolean = parser.parse("Is the template-based model realistic?")
        collected = parser.parse("How was this data collected?")
        comparison = parser.parse("Which methods are compared with the proposed system?")

        self.assertEqual(numeric.intent, "fact")
        self.assertEqual(numeric.granularity, "block")
        self.assertEqual(boolean.intent, "boolean")
        self.assertEqual(boolean.granularity, "block")
        self.assertEqual(collected.intent, "fact")
        self.assertEqual(collected.granularity, "block")
        self.assertEqual(comparison.intent, "comparison")

    def test_entity_extraction_filters_question_words(self):
        entities = EvidenceBridgeIndex._extract_entities(
            "Which models did they use for the Stanford NER and MNMT experiments?"
        )

        self.assertNotIn("which", entities)
        self.assertNotIn("they", entities)
        self.assertIn("stanford ner", entities)
        self.assertIn("mnmt", entities)

    def test_typed_ppr_uses_query_adaptive_bridge_weights(self):
        _stub_rag_provider_imports()
        from Core.rag.evibridge_demand import EvidenceDemand
        from Core.rag.evibridge_ppr import run_typed_ppr, run_typed_ppr_with_details

        index = self._build_index()
        demand = EvidenceDemand(
            intent="table-figure",
            scope="local",
            modality=["text", "table"],
            granularity="block",
            bridge_need=["context"],
        )

        scores = run_typed_ppr(index=index, seed_scores={2: 1.0}, demand=demand, top_k=4)

        self.assertGreater(scores[3], scores[1])
        self.assertIn(2, scores)
        self.assertIn(3, scores)

        detailed = run_typed_ppr_with_details(index=index, seed_scores={2: 1.0}, demand=demand, top_k=4)
        self.assertIn(3, detailed.scores)
        self.assertIn("context", detailed.score_parts[3])
        self.assertGreater(detailed.score_parts[3]["context"], detailed.score_parts[3]["hierarchy"])

    def test_selector_balances_relevance_coverage_connectivity_and_redundancy(self):
        _stub_rag_provider_imports()
        from Core.rag.evibridge_demand import EvidenceDemand
        from Core.rag.evibridge_selector import select_budgeted_evidence

        index = self._build_index()
        demand = EvidenceDemand(
            intent="table-figure",
            scope="local",
            modality=["text", "table"],
            granularity="block",
            bridge_need=["context", "semantic"],
        )

        selected = select_budgeted_evidence(
            index=index,
            query="Which table compares the accuracy results?",
            candidate_scores={1: 0.4, 2: 0.7, 3: 0.8, 4: 0.3},
            demand=demand,
            max_blocks=3,
            max_tokens=120,
        )

        selected_ids = [item.block.block_id for item in selected]
        self.assertIn(3, selected_ids)
        self.assertIn(2, selected_ids)
        self.assertLessEqual(len(selected), 3)
        self.assertTrue(all(item.score_parts["cost"] >= 0 for item in selected))

    def test_selector_prioritizes_paragraph_answer_evidence_over_auxiliary_blocks(self):
        _stub_rag_provider_imports()
        from Core.rag.evibridge_demand import EvidenceDemand
        from Core.rag.evibridge_selector import select_budgeted_evidence

        index = self._build_index()
        demand = EvidenceDemand(
            intent="fact",
            scope="document",
            modality=["text"],
            granularity="block",
            bridge_need=["context", "semantic"],
        )

        selected = select_budgeted_evidence(
            index=index,
            query="Which retrieval method is evaluated?",
            candidate_scores={1: 0.3, 2: 0.25, 4: 0.95, 5: 0.9},
            demand=demand,
            max_blocks=3,
            max_tokens=200,
            final_evidence_types=["paragraph"],
            bridge_auxiliary_types=["summary", "entity", "patch", "title"],
            paragraph_quota=2,
            auxiliary_quota=1,
        )

        selected_ids = [item.block.block_id for item in selected]
        selected_roles = [item.evidence_role for item in selected]
        self.assertIn(1, selected_ids)
        self.assertIn(2, selected_ids)
        self.assertLessEqual(selected_roles.count("bridge_auxiliary"), 1)

    def test_rule_verifier_detects_missing_table_and_disconnected_evidence(self):
        _stub_rag_provider_imports()
        from Core.rag.evibridge_demand import EvidenceDemand
        from Core.rag.evibridge_verifier import RuleBasedSufficiencyVerifier

        index = self._build_index()
        verifier = RuleBasedSufficiencyVerifier()
        table_demand = EvidenceDemand(
            intent="table-figure",
            scope="local",
            modality=["table"],
            granularity="block",
            bridge_need=["context"],
        )
        compare_demand = EvidenceDemand(
            intent="multi-hop",
            scope="multi-document",
            modality=["text"],
            granularity="block",
            bridge_need=["semantic"],
        )

        missing_table = verifier.verify("Which table reports accuracy?", table_demand, [index.blocks[1]], [])
        disconnected = verifier.verify(
            "How are Method A and results connected?",
            compare_demand,
            [index.blocks[1], index.blocks[3]],
            [],
        )

        self.assertFalse(missing_table.sufficient)
        self.assertIn("table_or_caption_context", missing_table.missing)
        self.assertIn("table", missing_table.missing_types)
        self.assertEqual(missing_table.next_action, "expand_table_caption")
        self.assertFalse(disconnected.sufficient)
        self.assertIn("semantic_link", disconnected.missing)
        self.assertIn("semantic", disconnected.missing_bridge_types)
        self.assertEqual(disconnected.next_action, "expand_semantic_bridge")

    def test_llm_verifier_sanitizes_invalid_fields_and_preserves_hard_rule_failures(self):
        _stub_rag_provider_imports()
        from Core.rag.evibridge_demand import EvidenceDemand
        from Core.rag.evibridge_verifier import EvidenceSufficiencyVerifier, SufficiencyVerdict

        class DirtyVerifierLLM:
            def get_json_completion(self, prompt, schema, images=None, think_mode=False):
                return SufficiencyVerdict(
                    sufficient=True,
                    missing=[],
                    missing_types=["sufficient", "noise", "next_action", "table"],
                    missing_bridge_types=["yes", "semantic"],
                    relevance=1.0,
                    connectivity=1.0,
                    coverage=1.0,
                    specificity=1.0,
                    noise=0.0,
                    next_bridge=["yes", "context"],
                    next_action="nonsense",
                    reason="llm accepted",
                )

        index = self._build_index()
        verifier = EvidenceSufficiencyVerifier(llm=DirtyVerifierLLM(), enable_llm=True)
        table_demand = EvidenceDemand(
            intent="table-figure",
            scope="local",
            modality=["table"],
            granularity="block",
            bridge_need=["context"],
        )

        verdict = verifier.verify("Which table reports accuracy?", table_demand, [index.blocks[1]], [])

        self.assertFalse(verdict.sufficient)
        self.assertIn("table_or_caption_context", verdict.missing)
        self.assertIn("table", verdict.missing_types)
        self.assertNotIn("sufficient", verdict.missing_types)
        self.assertNotIn("next_action", verdict.missing_types)
        self.assertEqual(verdict.missing_bridge_types, ["context"])
        self.assertEqual(verdict.next_action, "expand_table_caption")

    def test_shortest_path_connector_returns_nodes_edges_and_paths(self):
        _stub_rag_provider_imports()
        from Core.rag.evibridge_ppr import shortest_path_connector_with_paths

        index = self._build_index()

        connector = shortest_path_connector_with_paths(
            index=index,
            seed_ids=[1],
            candidate_ids=[3],
            max_paths=2,
        )

        self.assertTrue(connector.paths)
        self.assertTrue(connector.edges)
        self.assertTrue(set(connector.paths[0]["nodes"][1:-1]) & set(connector.scores))
        self.assertEqual(connector.paths[0]["nodes"][0], 1)
        self.assertEqual(connector.paths[0]["nodes"][-1], 3)

    def test_evibridge_rag_uses_multigranularity_seeds_unless_ablated(self):
        _stub_rag_provider_imports()
        from Core.configs.rag.evibridge_config import EviBridgeRAGConfig
        from Core.rag.evibridge_demand import EvidenceDemand
        from Core.rag.evibridge_rag import EviBridgeRAG

        with tempfile.TemporaryDirectory() as tmp:
            index = self._build_index()
            index.save_dir = tmp
            bm25 = index.build_bm25()
            demand = EvidenceDemand(
                intent="global-summary",
                scope="document",
                modality=["text"],
                granularity="summary",
                bridge_need=["hierarchy", "context"],
            )
            rag = EviBridgeRAG(
                config=EviBridgeRAGConfig(
                    bm25_topk=1,
                    patch_topk=2,
                    entity_topk=2,
                    enable_llm_verifier=False,
                ),
                llm=FakeLLM(),
                evibridge_index=index,
                bm25=bm25,
            )
            seeds = rag._hybrid_seed_retrieval("Summarize the whole document retrieval reasoning.", demand)

            ablated = EviBridgeRAG(
                config=EviBridgeRAGConfig(
                    bm25_topk=1,
                    patch_topk=2,
                    entity_topk=2,
                    ablation_variant="wo_multi_granularity_seeds",
                    enable_llm_verifier=False,
                ),
                llm=FakeLLM(),
                evibridge_index=index,
                bm25=bm25,
            )
            ablated_seeds = ablated._hybrid_seed_retrieval(
                "Summarize the whole document retrieval reasoning.",
                demand,
            )

        self.assertTrue(any(item.get("seed_family") == "summary" for item in seeds))
        self.assertTrue(any(item.get("seed_family") == "entity" for item in seeds))
        self.assertFalse(any(item.get("seed_family") in {"summary", "entity"} for item in ablated_seeds))

    def test_evibridge_rag_preserves_direct_paragraph_seed_after_ppr(self):
        _stub_rag_provider_imports()
        from Core.configs.rag.evibridge_config import EviBridgeRAGConfig
        from Core.rag.evibridge_demand import EvidenceDemand
        from Core.rag.evibridge_rag import EviBridgeRAG

        with tempfile.TemporaryDirectory() as tmp:
            index = self._build_index()
            index.save_dir = tmp
            bm25 = index.build_bm25()
            rag = EviBridgeRAG(
                config=EviBridgeRAGConfig(
                    bm25_topk=5,
                    ppr_topk=1,
                    preserve_seed_topk=2,
                    max_context_blocks=3,
                    paragraph_quota=2,
                    enable_llm_verifier=False,
                ),
                llm=FakeLLM(),
                evibridge_index=index,
                bm25=bm25,
            )

            demand = EvidenceDemand(
                intent="fact",
                scope="document",
                modality=["text"],
                granularity="block",
                bridge_need=["context", "semantic"],
            )
            retrieval_info = rag._retrieve_with_demand(
                "retrieval graph reasoning",
                demand,
            )

        selected_ids = retrieval_info["selected_block_ids"]
        selected_payload = retrieval_info["selected_payload"]
        self.assertTrue({1, 2}.issubset(set(selected_ids)))
        self.assertTrue(all("direct_seed_rank" in item for item in selected_payload))
        self.assertTrue(all("selection_rank" in item for item in selected_payload))

    def test_evibridge_rag_splits_generation_context_from_supporting_evidence(self):
        _stub_rag_provider_imports()
        from Core.configs.rag.evibridge_config import EviBridgeRAGConfig
        from Core.rag.evibridge_demand import EvidenceDemand
        from Core.rag.evibridge_rag import EviBridgeRAG

        with tempfile.TemporaryDirectory() as tmp:
            index = self._build_index()
            index.save_dir = tmp
            bm25 = index.build_bm25()
            rag = EviBridgeRAG(
                config=EviBridgeRAGConfig(
                    bm25_topk=6,
                    max_context_blocks=5,
                    supporting_evidence_topk=2,
                    paragraph_quota=2,
                    auxiliary_quota=1,
                    enable_llm_verifier=False,
                ),
                llm=FakeLLM(),
                evibridge_index=index,
                bm25=bm25,
            )
            demand = EvidenceDemand(
                intent="fact",
                scope="local",
                modality=["text"],
                granularity="block",
                bridge_need=["context", "semantic"],
            )

            retrieval_info = rag._retrieve_with_demand("retrieval graph reasoning", demand)

        self.assertLessEqual(len(retrieval_info["supporting_evidence"]), 2)
        self.assertGreater(len(retrieval_info["retrieved_block_ids"]), len(retrieval_info["supporting_block_ids"]))
        self.assertTrue(
            all(item["block_type"] == "paragraph" for item in retrieval_info["supporting_evidence"])
        )

    def test_evibridge_candidate_rerank_boosts_answer_evidence_and_records_scores(self):
        _stub_rag_provider_imports()
        from Core.configs.rag.evibridge_config import EviBridgeRAGConfig
        from Core.rag.evibridge_rag import EviBridgeRAG

        index = self._build_index()
        reranker = FakeReranker(
            {
                "Method B compares": 0.95,
                "Method A uses": 0.2,
            }
        )
        rag = EviBridgeRAG(
            config=EviBridgeRAGConfig(
                enable_candidate_rerank=True,
                candidate_rerank_topk=2,
                candidate_rerank_weight=1.0,
                rerank_batch_size=8,
                enable_llm_verifier=False,
            ),
            llm=FakeLLM(),
            evibridge_index=index,
            bm25=None,
            reranker=reranker,
        )
        candidate_scores = {1: 0.8, 2: 0.3, 5: 0.9}
        score_parts = {1: {}, 2: {}, 5: {}}

        reranked = rag._rerank_candidate_scores(
            query="Which method compares retrieval with graph reasoning?",
            candidate_scores=candidate_scores,
            candidate_score_parts=score_parts,
        )

        self.assertGreater(reranked[2], reranked[1])
        self.assertNotIn("rerank_score", score_parts[5])
        self.assertEqual(score_parts[2]["rerank_rank"], 1)
        self.assertGreater(score_parts[2]["rerank_score"], score_parts[1]["rerank_score"])
        self.assertEqual(reranker.calls[0]["batch_size"], 8)

    def test_evibridge_supporting_evidence_prefers_rerank_order(self):
        _stub_rag_provider_imports()
        from Core.configs.rag.evibridge_config import EviBridgeRAGConfig
        from Core.rag.evibridge_rag import EviBridgeRAG

        rag = EviBridgeRAG(
            config=EviBridgeRAGConfig(
                supporting_evidence_topk=2,
                enable_supporting_rerank=True,
                enable_llm_verifier=False,
            ),
            llm=FakeLLM(),
            evibridge_index=self._build_index(),
            bm25=None,
        )
        selected_payload = [
            {
                "block_id": 1,
                "block_type": "paragraph",
                "selection_rank": 1,
                "evidence_role": "answer_evidence",
                "score_parts": {"rerank_score": 0.1, "rerank_rank": 3},
            },
            {
                "block_id": 2,
                "block_type": "paragraph",
                "selection_rank": 2,
                "evidence_role": "answer_evidence",
                "score_parts": {"rerank_score": 0.95, "rerank_rank": 1},
            },
            {
                "block_id": 3,
                "block_type": "table",
                "selection_rank": 3,
                "evidence_role": "answer_evidence",
                "score_parts": {"rerank_score": 0.7, "rerank_rank": 2},
            },
        ]

        supporting = rag._supporting_evidence_payload(selected_payload)

        self.assertEqual([item["block_id"] for item in supporting], [2, 3])
        self.assertEqual([item["supporting_rank"] for item in supporting], [1, 2])

    def test_evibridge_can_ignore_llm_supporting_ids_for_reranked_supporting_evidence(self):
        _stub_rag_provider_imports()
        from Core.configs.rag.evibridge_config import EviBridgeRAGConfig
        from Core.rag.evibridge_rag import EviBridgeRAG

        rag = EviBridgeRAG(
            config=EviBridgeRAGConfig(
                supporting_evidence_topk=1,
                enable_supporting_rerank=True,
                trust_answer_supporting_ids=False,
                enable_llm_verifier=False,
            ),
            llm=FakeLLM(),
            evibridge_index=self._build_index(),
            bm25=None,
        )
        retrieval_info = {
            "selected_payload": [
                {
                    "block_id": 1,
                    "block_type": "paragraph",
                    "selection_rank": 1,
                    "evidence_role": "answer_evidence",
                    "score_parts": {"rerank_score": 0.1, "rerank_rank": 2},
                },
                {
                    "block_id": 2,
                    "block_type": "paragraph",
                    "selection_rank": 2,
                    "evidence_role": "answer_evidence",
                    "score_parts": {"rerank_score": 0.95, "rerank_rank": 1},
                },
            ]
        }

        rag._apply_answer_supporting_ids(retrieval_info, supporting_ids=[1])

        self.assertEqual(retrieval_info["supporting_block_ids"], [2])

    def test_evibridge_answer_payload_parses_supporting_ids_and_normalizes_boolean(self):
        _stub_rag_provider_imports()
        from Core.configs.rag.evibridge_config import EviBridgeRAGConfig
        from Core.rag.evibridge_demand import EvidenceDemand
        from Core.rag.evibridge_rag import EviBridgeRAG

        rag = EviBridgeRAG(
            config=EviBridgeRAGConfig(enable_llm_verifier=False),
            llm=FakeLLM(),
            evibridge_index=self._build_index(),
            bm25=None,
        )
        answer_short, rationale, supporting_ids = rag._parse_answer_payload(
            '{"answer_short": "yes, because the model is realistic", '
            '"answer_rationale": "The evidence says it is realistic.", '
            '"supporting_block_ids": [2, "3", "bad"]}'
        )
        normalized = rag._normalize_answer_short(
            answer_short,
            EvidenceDemand(intent="boolean", scope="local", modality=["text"], granularity="block"),
        )

        self.assertEqual(normalized, "Yes")
        self.assertEqual(rationale, "The evidence says it is realistic.")
        self.assertEqual(supporting_ids, [2, 3])

    def test_evibridge_config_is_part_of_rag_discriminator(self):
        parsed = RAGConfig(strategy_config={"strategy": "evibridge"})

        self.assertEqual(parsed.strategy_config.strategy, "evibridge")
        self.assertEqual(parsed.strategy_config.max_iterations, 2)

    def test_evibridge_rag_writes_retrieval_and_evidence_chain_outputs(self):
        _stub_rag_provider_imports()
        from Core.configs.rag.evibridge_config import EviBridgeRAGConfig
        from Core.rag.evibridge_rag import EviBridgeRAG

        with tempfile.TemporaryDirectory() as tmp:
            index = self._build_index()
            index.save_dir = tmp
            bm25 = index.build_bm25()
            rag = EviBridgeRAG(
                config=EviBridgeRAGConfig(
                    bm25_topk=3,
                    max_context_blocks=3,
                    max_iterations=1,
                    enable_llm_verifier=False,
                ),
                llm=FakeLLM(),
                evibridge_index=index,
                bm25=bm25,
            )

            answer, retrieved_ids = rag.generation("Which table reports accuracy results?", tmp)

            retrieval_path = Path(tmp) / "retrieval_res.json"
            chain_path = Path(tmp) / "evidence_chain.json"
            retrieval_payload = json.loads(retrieval_path.read_text(encoding="utf-8"))
            chain_payload = json.loads(chain_path.read_text(encoding="utf-8"))

            self.assertEqual(answer, "Answer based on bridged evidence.")
            self.assertTrue(retrieved_ids)
            self.assertIn("supporting_evidence", retrieval_payload)
            self.assertIn("supporting_block_ids", retrieval_payload)
            self.assertIn("retrieved_block_ids", retrieval_payload)
            self.assertIn("typed_ppr_scores", retrieval_payload)
            self.assertIn("typed_ppr_score_parts", retrieval_payload)
            self.assertIn("connector_paths", retrieval_payload)
            self.assertIn("verification", retrieval_payload)
            self.assertTrue(chain_payload["evidence_chain"])
            self.assertIn("iterations", chain_payload)


if __name__ == "__main__":
    unittest.main()
