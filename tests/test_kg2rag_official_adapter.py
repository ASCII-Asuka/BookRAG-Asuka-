import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml


class KG2RAGOfficialAdapterTests(unittest.TestCase):
    def test_load_json_accepts_utf8_bom_manifests(self):
        from Scripts.baselines.kg2rag.common import load_json

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            manifest_path.write_text('{"scope_count": 2}', encoding="utf-8-sig")

            self.assertEqual(load_json(manifest_path), {"scope_count": 2})

    def test_source_lock_and_public_config_are_pinned(self):
        lock = json.loads(
            Path("Scripts/baselines/kg2rag/official-source-lock.json").read_text(
                encoding="utf-8"
            )
        )
        config_path = Path("config/kg2rag_official.yaml")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            lock["repository"], "https://github.com/nju-websoft/KG2RAG.git"
        )
        self.assertEqual(
            lock["revision"], "7d626c77b7af30b55aa3f960cde755b9549a0616"
        )
        self.assertEqual(lock["license"], "GPL-3.0")
        self.assertEqual(config["method_suffix"], "kg2rag_official")
        self.assertEqual(config["seed_topk"], 10)
        self.assertEqual(config["expansion_hops"], 1)
        self.assertEqual(config["retrieval_topk"], 20)
        self.assertEqual(config["generation_max_blocks"], 10)
        self.assertEqual(config["generation_max_tokens"], 4000)

        serialized = config_path.read_text(encoding="utf-8").lower()
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("siliconflow", serialized)

    def test_entity_normalization_is_deterministic_and_non_generative(self):
        from Scripts.baselines.kg2rag.common import normalize_entity

        self.assertEqual(normalize_entity("  Café—Model. "), "café model")
        self.assertEqual(normalize_entity("Café Model"), "café model")

    def test_cache_identity_changes_with_material_inputs(self):
        from Scripts.baselines.kg2rag.common import build_cache_identity

        base = build_cache_identity("c", "p", "r", "openie", "embed", "rerank")
        self.assertNotEqual(
            base,
            build_cache_identity("c2", "p", "r", "openie", "embed", "rerank"),
        )
        self.assertNotEqual(
            base,
            build_cache_identity("c", "p2", "r", "openie", "embed", "rerank"),
        )

    def test_generation_budget_stops_before_oversized_chunk(self):
        from Scripts.baselines.kg2rag.common import apply_generation_budget

        ranked = [
            {"source_id": "a", "content": "1234"},
            {"source_id": "b", "content": "12"},
        ]
        selected = apply_generation_budget(ranked, 10, 3, len)
        self.assertEqual(selected, [])

    def test_qasper_citation_fallback_uses_top_ranked_legal_paragraph(self):
        from Scripts.baselines.kg2rag.common import validate_qasper_citations

        ranked = [
            {
                "source_id": "p7",
                "content": "Indexed text",
                "metadata": {
                    "paragraph_id": 7,
                    "qasper_evidence_text": "Original paragraph.",
                },
            }
        ]
        evidence, audit = validate_qasper_citations([999], ranked, max_support=4)
        self.assertEqual(evidence[0]["paragraph_id"], 7)
        self.assertEqual(evidence[0]["content"], "Original paragraph.")
        self.assertTrue(audit["fallback_used"])

    def test_qasper_citation_accepts_exact_evidence_label_wrapper(self):
        from Scripts.baselines.kg2rag.common import validate_qasper_citations

        ranked = [
            {
                "source_id": "7",
                "content": "Indexed text",
                "metadata": {
                    "paragraph_id": 7,
                    "qasper_evidence_text": "Original paragraph.",
                },
            }
        ]
        evidence, audit = validate_qasper_citations(
            ["Evidence 7"], ranked, max_support=4
        )

        self.assertEqual(evidence[0]["paragraph_id"], 7)
        self.assertFalse(audit["fallback_used"])
        self.assertEqual(audit["rejected"], [])

    def test_hotpot_citations_reject_out_of_scope_sentence(self):
        from Scripts.baselines.kg2rag.common import validate_hotpot_citations

        ranked = [
            {
                "source_id": "title-0",
                "content": "Title: A",
                "metadata": {
                    "hotpot_title": "A",
                    "hotpot_sentences": [
                        {"sent_id": 0, "text": "Born in Paris."}
                    ],
                },
            }
        ]
        evidence, audit = validate_hotpot_citations(
            [["A", 9]], ranked, max_support=4, question="Where was the person born?"
        )
        self.assertEqual(evidence[0]["hotpot_title"], "A")
        self.assertEqual(evidence[0]["hotpot_sent_id"], 0)
        self.assertTrue(audit["fallback_used"])

    def test_online_cost_excludes_offline_openie(self):
        from Scripts.baselines.kg2rag.common import combine_online_cost

        cost = combine_online_cost(
            retrieval={"prompt_tokens": 5, "completion_tokens": 1, "time": 0.4},
            generation={"prompt_tokens": 7, "completion_tokens": 2, "time": 0.6},
        )
        self.assertEqual(cost["rag_cost"]["total_tokens"], 15)
        self.assertEqual(cost["time"], 1.0)

    def test_qasper_chunks_preserve_paragraph_id_and_original_text(self):
        from Scripts.baselines.kg2rag.prepare import qasper_chunks_from_tree

        title = SimpleNamespace(
            type="title",
            meta_info=SimpleNamespace(content="Methods"),
            parent=None,
        )
        paragraph = SimpleNamespace(
            type="text",
            index_id=7,
            meta_info=SimpleNamespace(content="Original paragraph."),
            parent=title,
        )
        tree = SimpleNamespace(get_nodes=lambda hasRoot=False: [title, paragraph])

        chunks = qasper_chunks_from_tree(tree, "paper-1")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["source_id"], "7")
        self.assertEqual(chunks[0]["metadata"]["paragraph_id"], 7)
        self.assertEqual(
            chunks[0]["metadata"]["qasper_evidence_text"], "Original paragraph."
        )
        self.assertTrue(chunks[0]["content"].startswith("[Section: Methods]"))

    def test_hotpot_chunks_preserve_title_sentence_mapping(self):
        from Scripts.baselines.kg2rag.prepare import hotpot_chunks_from_tree

        def sentence(title, sent_id, text, index_id):
            parent = SimpleNamespace(
                type="title",
                meta_info=SimpleNamespace(content=title),
                parent=None,
            )
            metadata = SimpleNamespace(
                content=text,
                pdf_para_block={
                    "hotpot_title": title,
                    "hotpot_sent_id": sent_id,
                },
            )
            return parent, SimpleNamespace(
                type="text", index_id=index_id, meta_info=metadata, parent=parent
            )

        title_a, a0 = sentence("A", 0, "Alpha.", "a0")
        title_b, b0 = sentence("B", 0, "Beta.", "b0")
        tree = SimpleNamespace(
            get_nodes=lambda hasRoot=False: [title_a, a0, title_b, b0]
        )

        chunks = hotpot_chunks_from_tree(tree, "question-1")

        self.assertEqual(
            {chunk["metadata"]["hotpot_title"] for chunk in chunks}, {"A", "B"}
        )
        self.assertEqual(chunks[0]["metadata"]["hotpot_sentences"][0]["sent_id"], 0)
        self.assertEqual(chunks[0]["metadata"]["hotpot_sentences"][0]["text"], "Alpha.")

    def test_prepare_rejects_duplicate_question_ids(self):
        from Scripts.baselines.kg2rag.prepare import _group_rows

        rows = [
            {"doc_uuid": "p", "question_id": "q"},
            {"doc_uuid": "p", "question_id": "q"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _group_rows(rows)

    def test_model_clients_keep_tls_verification_enabled(self):
        from Scripts.baselines.kg2rag.model_clients import OpenAICompatibleClient

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"data": [{"embedding": [0.25, 0.75]}]}
        with patch("requests.Session.post", return_value=response) as post:
            client = OpenAICompatibleClient(
                api_key="secret",
                llm_base_url="https://chat.example/v1",
                embedding_base_url="https://embed.example/v1",
                reranker_base_url="https://rerank.example/v1",
                timeout=30,
            )
            vectors, _ = client.embed("embed-model", ["test passage"])

        self.assertEqual(vectors.shape, (1, 2))
        _, kwargs = post.call_args
        self.assertTrue(kwargs["verify"])
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret")

    def test_chat_json_retries_truncated_content_and_counts_all_usage(self):
        from Scripts.baselines.kg2rag.model_clients import OpenAICompatibleClient

        truncated = Mock()
        truncated.raise_for_status.return_value = None
        truncated.json.return_value = {
            "choices": [
                {"message": {"content": '{"answer_short":"unfinished'}}
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
        }
        valid = Mock()
        valid.raise_for_status.return_value = None
        valid.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"answer_short":"Paris",'
                        '"answer_rationale":"Evidence supports it",'
                        '"supporting_block_ids":["7"]}'
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 6,
                "total_tokens": 16,
            },
        }

        with patch(
            "requests.Session.post", side_effect=[truncated, valid]
        ) as post, patch("Scripts.baselines.kg2rag.model_clients.time.sleep"):
            client = OpenAICompatibleClient(
                api_key="secret",
                llm_base_url="https://chat.example/v1",
                embedding_base_url="https://embed.example/v1",
                reranker_base_url="https://rerank.example/v1",
                max_attempts=3,
                retry_backoff_seconds=0,
            )
            payload, usage = client.chat_json(
                "chat-model",
                [{"role": "user", "content": "Return JSON."}],
                temperature=0.1,
                max_tokens=512,
            )

        self.assertEqual(post.call_count, 2)
        self.assertEqual(payload["answer_short"], "Paris")
        self.assertEqual(usage["prompt_tokens"], 20)
        self.assertEqual(usage["completion_tokens"], 10)
        self.assertEqual(usage["total_tokens"], 30)

    def test_chat_json_retries_after_transport_attempts_are_exhausted(self):
        from Scripts.baselines.kg2rag.model_clients import OpenAICompatibleClient

        client = OpenAICompatibleClient(
            api_key="secret",
            llm_base_url="https://chat.example/v1",
            embedding_base_url="https://embed.example/v1",
            reranker_base_url="https://rerank.example/v1",
            max_attempts=3,
            retry_backoff_seconds=0,
        )
        valid = {
            "choices": [
                {"message": {"content": '{"answer_short":"Paris"}'}}
            ],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 2,
                "total_tokens": 10,
            },
        }
        with patch.object(
            client,
            "_post",
            side_effect=[RuntimeError("transport retries exhausted"), valid],
        ) as post, patch("Scripts.baselines.kg2rag.model_clients.time.sleep"):
            payload, usage = client.chat_json(
                "chat-model",
                [{"role": "user", "content": "Return JSON."}],
                temperature=0.1,
                max_tokens=512,
            )

        self.assertEqual(post.call_count, 2)
        self.assertEqual(payload["answer_short"], "Paris")
        self.assertEqual(usage["total_tokens"], 10)

    def test_chat_json_uses_separate_bounded_content_retry_limit(self):
        from Scripts.baselines.kg2rag.model_clients import OpenAICompatibleClient

        client = OpenAICompatibleClient(
            api_key="secret",
            llm_base_url="https://chat.example/v1",
            embedding_base_url="https://embed.example/v1",
            reranker_base_url="https://rerank.example/v1",
            max_attempts=3,
            json_max_attempts=6,
            retry_backoff_seconds=0,
        )
        truncated = {
            "choices": [{"message": {"content": '{"answer_short":"cut'}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2},
        }
        valid = {
            "choices": [
                {"message": {"content": '{"answer_short":"Paris"}'}}
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2},
        }
        with patch.object(
            client, "_post", side_effect=[truncated, truncated, truncated, valid]
        ) as post, patch("Scripts.baselines.kg2rag.model_clients.time.sleep"):
            payload, usage = client.chat_json(
                "chat-model",
                [{"role": "user", "content": "Return JSON."}],
                temperature=0.1,
                max_tokens=512,
            )

        self.assertEqual(post.call_count, 4)
        self.assertEqual(payload["answer_short"], "Paris")
        self.assertEqual(usage["total_tokens"], 40)

    def test_openie_accepts_only_nonempty_string_triples(self):
        from Scripts.baselines.kg2rag.index import parse_triples

        response = {
            "triples": [
                ["A", "rel", "B"],
                ["A", "", "C"],
                ["A", "rel", True],
                ["only", "two"],
            ]
        }
        self.assertEqual(parse_triples(response), [["A", "rel", "B"]])

    def test_openie_payload_contains_no_question_or_gold_fields(self):
        from Scripts.baselines.kg2rag.index import openie_input

        chunk = {
            "source_id": "p1",
            "content": "Alpha was born in Paris.",
            "metadata": {"paragraph_id": 1},
        }
        payload = openie_input(chunk)
        lowered = json.dumps(payload, ensure_ascii=False).lower()
        self.assertNotIn("question", lowered)
        self.assertNotIn("answer", lowered)
        self.assertNotIn("supporting", lowered)

    def test_openie_recovery_prompt_compacts_latex_and_bounds_output(self):
        from Scripts.baselines.kg2rag.index import openie_input

        chunk = {
            "source_id": "27",
            "content": (
                "The tokenizer $\\hat{\\mathcal {T}}_\\mathrm {LM}$ "
                "improves precision."
            ),
        }
        payload = openie_input(chunk, recovery=True)

        self.assertIn("at most 12", payload[0]["content"])
        self.assertIn("at most 160 characters", payload[0]["content"])
        self.assertNotIn("\\mathcal", payload[1]["content"])
        self.assertIn("[MATHEMATICAL NOTATION]", payload[1]["content"])

    def test_scope_index_uses_stricter_recovery_prompt_after_parse_failure(self):
        from Scripts.baselines.kg2rag.index import index_scope

        class RecoveringClients:
            def __init__(self):
                self.messages = []

            def chat_json(self, model, messages, temperature, max_tokens):
                self.messages.append(messages)
                if len(self.messages) == 1:
                    raise ValueError("unterminated JSON")
                return (
                    {"triples": [["tokenizer", "improves", "precision"]]},
                    {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
                )

            def embed(self, model, texts):
                return (
                    np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32),
                    {"prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2},
                )

        scope = {
            "dataset": "qasper",
            "scope_id": "paper-recovery",
            "corpus_sha256": "sha",
            "chunks": [
                {
                    "source_id": "27",
                    "content": "Tokenizer $\\hat{T}$ improves precision.",
                    "metadata": {"paragraph_id": 27},
                }
            ],
            "queries": [],
        }
        config = {
            "openie": {
                "model_name": "llm",
                "temperature": 0.0,
                "max_output_tokens": 1024,
                "max_attempts": 3,
            },
            "embedding": {"model_name": "embed", "batch_size": 16},
        }
        clients = RecoveringClients()

        with tempfile.TemporaryDirectory() as tmp:
            result = index_scope(scope, config, clients, Path(tmp))

        self.assertEqual(len(clients.messages), 2)
        self.assertNotIn("at most 12", clients.messages[0][0]["content"])
        self.assertIn("at most 12", clients.messages[1][0]["content"])
        self.assertNotIn("\\hat", clients.messages[1][1]["content"])
        self.assertEqual(result["index_stats"]["openie_recovered_chunks"], 1)

    def test_scope_index_preserves_triple_source_and_offline_cost(self):
        from Scripts.baselines.kg2rag.index import index_scope

        class FakeClients:
            def chat_json(self, model, messages, temperature, max_tokens):
                return (
                    {"triples": [["Alpha", "born in", "Paris"]]},
                    {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
                )

            def embed(self, model, texts):
                return (
                    np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32),
                    {"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3},
                )

        scope = {
            "dataset": "qasper",
            "scope_id": "paper-1",
            "corpus_sha256": "corpus-sha",
            "chunks": [
                {
                    "source_id": "p1",
                    "content": "Alpha was born in Paris.",
                    "metadata": {"paragraph_id": 1},
                }
            ],
            "queries": [
                {
                    "question_id": "q1",
                    "question": "Where was Alpha born?",
                    "dataset_row": {
                        "answer": "Paris",
                        "evidence_paragraph_ids": [1],
                    },
                }
            ],
        }
        config = {
            "source_revision": "rev",
            "openie": {
                "model_name": "llm",
                "temperature": 0.0,
                "max_output_tokens": 256,
                "max_attempts": 3,
            },
            "embedding": {"model_name": "embed", "batch_size": 16},
            "reranker": {"model_name": "rerank"},
        }

        with tempfile.TemporaryDirectory() as tmp:
            result = index_scope(scope, config, FakeClients(), Path(tmp))
            embeddings = np.load(Path(tmp) / result["chunk_embeddings"]["vectors_file"])

        self.assertEqual(result["triples"][0]["source_id"], "p1")
        self.assertEqual(result["triples"][0]["head_key"], "alpha")
        self.assertEqual(result["triples"][0]["tail_key"], "paris")
        self.assertEqual(result["index_stats"]["offline_tokens"], 18)
        self.assertEqual(embeddings.shape, (1, 2))

    def test_one_hop_expansion_does_not_leak_into_second_hop(self):
        from Scripts.baselines.kg2rag.retrieval import expand_one_hop

        triples = [
            {"head_key": "a", "tail_key": "b", "source_id": "seed"},
            {"head_key": "b", "tail_key": "c", "source_id": "bridge"},
            {"head_key": "c", "tail_key": "d", "source_id": "two-hop"},
        ]
        result = expand_one_hop(["seed"], triples)

        self.assertEqual(result["source_ids"], {"seed", "bridge"})
        self.assertNotIn("two-hop", result["source_ids"])

    def test_mst_keeps_maximum_weight_edges_and_dfs_is_deterministic(self):
        from Scripts.baselines.kg2rag.retrieval import organize_components

        triples = [
            {
                "head_key": "a",
                "tail_key": "b",
                "head": "A",
                "tail": "B",
                "relation": "ab",
                "source_id": "p1",
            },
            {
                "head_key": "b",
                "tail_key": "c",
                "head": "B",
                "tail": "C",
                "relation": "bc",
                "source_id": "p2",
            },
            {
                "head_key": "a",
                "tail_key": "c",
                "head": "A",
                "tail": "C",
                "relation": "ac",
                "source_id": "p3",
            },
        ]
        components = organize_components(
            triples, {"p1": 0.9, "p2": 0.8, "p3": 0.1}
        )

        self.assertEqual(
            {edge["source_id"] for edge in components[0]["mst_edges"]},
            {"p1", "p2"},
        )
        self.assertEqual(components[0]["ordered_source_ids"], ["p1", "p2"])

    def test_retrieval_returns_unique_top20_ranking_and_cost(self):
        from Scripts.baselines.kg2rag.retrieval import retrieve

        class FakeClients:
            def embed(self, model, texts):
                self.embed_texts = list(texts)
                return (
                    np.asarray([[1.0, 0.0]], dtype=np.float32),
                    {"prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2},
                )

            def rerank(self, model, query, documents):
                return (
                    [float(len(documents) - index) for index in range(len(documents))],
                    {"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3},
                )

        index = {
            "chunks": [
                {"source_id": "p1", "content": "A relates to B.", "metadata": {}},
                {"source_id": "p2", "content": "B relates to C.", "metadata": {}},
                {"source_id": "p3", "content": "Unrelated.", "metadata": {}},
            ],
            "triples": [
                {
                    "head_key": "a",
                    "tail_key": "b",
                    "head": "A",
                    "tail": "B",
                    "relation": "relates to",
                    "source_id": "p1",
                },
                {
                    "head_key": "b",
                    "tail_key": "c",
                    "head": "B",
                    "tail": "C",
                    "relation": "relates to",
                    "source_id": "p2",
                },
            ],
            "embedding_matrix": np.asarray(
                [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32
            ),
        }
        clients = FakeClients()

        result = retrieve(
            "How is A related to C?",
            index,
            clients,
            embedding_model="embed",
            reranker_model="rerank",
            seed_topk=1,
            retrieval_topk=20,
            expansion_hops=1,
        )

        ids = [item["source_id"] for item in result["ranked_results"]]
        self.assertEqual(ids[:2], ["p1", "p2"])
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            [item["rank"] for item in result["ranked_results"]],
            list(range(1, len(ids) + 1)),
        )
        self.assertEqual(result["retrieval_token_cost"]["total_tokens"], 5)

    def test_runner_reuses_matching_cache_and_separates_changed_config(self):
        from Scripts.baselines.kg2rag.common import write_json
        from Scripts.baselines.kg2rag.run import run_scopes

        class FakeClients:
            def chat_json(self, model, messages, temperature, max_tokens):
                return (
                    {"triples": [["A", "relates to", "B"]]},
                    {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                )

            def embed(self, model, texts):
                vectors = [
                    [1.0, 0.0] if index == 0 else [0.8, 0.2]
                    for index, _ in enumerate(texts)
                ]
                return np.asarray(vectors, dtype=np.float32), {
                    "prompt_tokens": len(texts),
                    "completion_tokens": 0,
                    "total_tokens": len(texts),
                }

            def rerank(self, model, query, documents):
                return [1.0 for _ in documents], {
                    "prompt_tokens": len(documents),
                    "completion_tokens": 0,
                    "total_tokens": len(documents),
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "prepared"
            output = root / "retrieval"
            scope = {
                "dataset": "qasper",
                "scope_id": "paper-1",
                "corpus_sha256": "corpus-a",
                "chunks": [
                    {
                        "source_id": "p1",
                        "content": "A relates to B.",
                        "metadata": {"paragraph_id": 1},
                    }
                ],
                "queries": [
                    {
                        "question_id": "q1",
                        "question": "How are A and B related?",
                        "dataset_row": {"doc_uuid": "paper-1"},
                    }
                ],
            }
            write_json(prepared / "scopes" / "paper.json", scope)
            write_json(
                prepared / "manifest.json",
                {
                    "dataset": "qasper",
                    "working_dir": str(root / "work"),
                    "scope_count": 1,
                    "query_count": 1,
                    "scopes": [
                        {
                            "scope_id": "paper-1",
                            "relative_path": "scopes/paper.json",
                            "corpus_sha256": "corpus-a",
                        }
                    ],
                },
            )
            config = root / "config.yaml"
            config.write_text(
                "\n".join(
                    [
                        "method_suffix: kg2rag_official",
                        "source_revision: rev-a",
                        "seed_topk: 10",
                        "retrieval_topk: 20",
                        "expansion_hops: 1",
                        "openie:",
                        "  model_name: llm",
                        "  max_output_tokens: 256",
                        "  max_attempts: 3",
                        "embedding:",
                        "  model_name: embed",
                        "  batch_size: 16",
                        "reranker:",
                        "  model_name: rerank",
                    ]
                ),
                encoding="utf-8",
            )

            first = run_scopes(prepared, config, output, clients=FakeClients())
            second = run_scopes(prepared, config, output, clients=FakeClients())
            changed = run_scopes(
                prepared,
                config,
                output,
                clients=FakeClients(),
                config_override={"seed_topk": 5},
            )
            status_text = (output / "status.json").read_text(encoding="utf-8")

        self.assertFalse(first["scopes"][0]["reused"])
        self.assertTrue(second["scopes"][0]["reused"])
        self.assertNotEqual(first["run_id"], changed["run_id"])
        self.assertNotIn("secret", status_text.lower())

    def test_finalize_writes_evaluator_compatible_qasper_outputs(self):
        from Scripts.baselines.kg2rag.common import write_json
        from Scripts.baselines.kg2rag.finalize import finalize_runs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "prepared"
            run_directory = root / "retrieval" / "runs" / "run-a"
            work = root / "work"
            scope = {
                "dataset": "qasper",
                "scope_id": "paper-1",
                "corpus_sha256": "corpus-a",
                "chunks": [],
                "queries": [
                    {
                        "question_id": "q1",
                        "question": "What is the answer?",
                        "dataset_row": {
                            "doc_uuid": "paper-1",
                            "qasper_question_id": "q1",
                            "question": "What is the answer?",
                            "answer": {"answer": "alpha", "evidence": []},
                        },
                    }
                ],
            }
            write_json(prepared / "scopes" / "paper.json", scope)
            write_json(
                prepared / "manifest.json",
                {
                    "dataset": "qasper",
                    "working_dir": str(work),
                    "scope_count": 1,
                    "query_count": 1,
                    "scopes": [
                        {
                            "scope_id": "paper-1",
                            "relative_path": "scopes/paper.json",
                        }
                    ],
                },
            )
            ranked = [
                {
                    "rank": 1,
                    "score": 0.9,
                    "source_id": "p7",
                    "content": "[Section: Results]\nAnswer is alpha.",
                    "metadata": {
                        "paragraph_id": 7,
                        "qasper_evidence_text": "Answer is alpha.",
                        "section": "Results",
                    },
                }
            ]
            write_json(
                run_directory / "scopes" / "paper" / "scope_result.json",
                {
                    "dataset": "qasper",
                    "scope_id": "paper-1",
                    "cache_identity": "cache-a",
                    "index_stats": {"triples": 1},
                    "queries": [
                        {
                            "question_id": "q1",
                            "question": "What is the answer?",
                            "ranked_results": ranked,
                            "semantic_seeds": [{"source_id": "p7", "score": 0.9}],
                            "expanded_source_ids": ["p7"],
                            "expansion_edges": [],
                            "components": [],
                            "retrieval_time": 0.2,
                            "retrieval_token_cost": {
                                "prompt_tokens": 5,
                                "completion_tokens": 1,
                                "total_tokens": 6,
                                "time": 0.2,
                            },
                            "dense_fallback": True,
                            "dense_fallback_reason": "no_graph_component",
                        }
                    ],
                },
            )
            retrieval_manifest = root / "retrieval" / "manifest.json"
            write_json(
                retrieval_manifest,
                {
                    "status": "complete",
                    "run_id": "run-a",
                    "dataset": "qasper",
                    "working_dir": str(work),
                    "prepared_manifest_path": str(prepared / "manifest.json"),
                    "run_directory": str(run_directory),
                    "scope_count": 1,
                    "query_count": 1,
                    "scopes": [
                        {
                            "scope_id": "paper-1",
                            "relative_path": "scopes/paper/scope_result.json",
                        }
                    ],
                },
            )
            config = root / "config.yaml"
            config.write_text(
                "\n".join(
                    [
                        "method_suffix: kg2rag_official",
                        "generation_max_blocks: 10",
                        "generation_max_tokens: 4000",
                        "supporting_evidence_topk: 4",
                        "generation:",
                        "  model_name: generator",
                        "  temperature: 0.1",
                    ]
                ),
                encoding="utf-8",
            )

            summary = finalize_runs(
                retrieval_manifest,
                config,
                generator=lambda dataset, question, selected: {
                    "answer_short": "alpha",
                    "answer_rationale": "supported",
                    "supporting_block_ids": [999],
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 2,
                        "total_tokens": 9,
                    },
                    "time": 0.3,
                },
                token_counter=len,
            )
            result_dir = work / "paper-1" / "eval_qasper_kg2rag_official"
            final_results = json.loads(
                (result_dir / "final_results.json").read_text(encoding="utf-8")
            )
            retrieval = json.loads(
                (result_dir / "query_001" / "retrieval_res.json").read_text(
                    encoding="utf-8"
                )
            )
            cost = json.loads(
                (result_dir / "token_cost.json").read_text(encoding="utf-8")
            )

        self.assertEqual(summary["question_count"], 1)
        self.assertEqual(final_results[0]["answer_short"], "alpha")
        self.assertEqual(retrieval["supporting_block_ids"], [7])
        self.assertTrue(retrieval["citation_validation"]["fallback_used"])
        self.assertEqual(cost["rag_cost"]["total_tokens"], 15)

    def test_full_launcher_is_single_process_and_covers_both_datasets(self):
        launcher = Path(
            "Scripts/baselines/kg2rag/run_full_experiments.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("Qasper.yaml", launcher)
        self.assertIn("HotpotQA.yaml", launcher)
        self.assertIn("full_experiment_status.json", launcher)
        self.assertNotIn("Start-Process", launcher)
        self.assertNotIn("OPENAI_API_KEY", launcher)


if __name__ == "__main__":
    unittest.main()
