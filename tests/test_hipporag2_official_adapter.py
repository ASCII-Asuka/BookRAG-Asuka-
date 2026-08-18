import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


class HippoRAG2OfficialAdapterTests(unittest.TestCase):
    @staticmethod
    def _write_dataset_config(path: Path, dataset_path: Path, working_dir: Path, name: str) -> None:
        path.write_text(
            "\n".join(
                [
                    f"dataset_path: {dataset_path.as_posix()}",
                    f"working_dir: {working_dir.as_posix()}",
                    f"dataset_name: {name}",
                ]
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _save_qasper_tree(save_dir: Path) -> None:
        from Core.Index.Tree import DocumentTree, NodeType, TreeNode

        cfg = SimpleNamespace(save_path=str(save_dir))
        tree = DocumentTree(
            meta_dict={"file_name": "paper-1.pdf", "file_path": "qasper://paper-1"},
            cfg=cfg,
        )
        title = TreeNode({"content": "Methods", "page_idx": 0, "pdf_id": 1})
        title.type = NodeType.TITLE
        title.outline_node = True
        tree.add_node(title)
        tree.root_node.add_child(title)
        for text in ("First original paragraph.", "Second original paragraph."):
            paragraph = TreeNode(
                {"content": text, "page_idx": 0, "pdf_id": len(tree.nodes)}
            )
            paragraph.type = NodeType.TEXT
            tree.add_node(paragraph)
            title.add_child(paragraph)
        save_dir.mkdir(parents=True, exist_ok=True)
        tree.save_to_file()

    @staticmethod
    def _save_hotpot_tree(save_dir: Path) -> None:
        from Core.Index.Tree import DocumentTree, NodeType, TreeNode

        cfg = SimpleNamespace(save_path=str(save_dir))
        tree = DocumentTree(
            meta_dict={"file_name": "hotpot-1.json", "file_path": "hotpotqa://hotpot-1"},
            cfg=cfg,
        )
        title = TreeNode({"content": "Article A", "page_idx": 0, "pdf_id": 1})
        title.type = NodeType.TITLE
        title.outline_node = True
        tree.add_node(title)
        tree.root_node.add_child(title)
        for sent_id, text in enumerate(("Alpha is a scientist.", "Alpha was born in Paris.")):
            sentence = TreeNode(
                {
                    "content": text,
                    "page_idx": 0,
                    "pdf_id": len(tree.nodes),
                    "pdf_para_block": {
                        "source": "hotpotqa_sentence",
                        "hotpot_title": "Article A",
                        "hotpot_sent_id": sent_id,
                    },
                }
            )
            sentence.type = NodeType.TEXT
            tree.add_node(sentence)
            title.add_child(sentence)
        save_dir.mkdir(parents=True, exist_ok=True)
        tree.save_to_file()

    def test_prepare_qasper_groups_questions_and_preserves_paragraphs(self):
        from Scripts.baselines.hipporag2.prepare import prepare_scopes

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "qasper.json"
            working_dir = root / "work"
            config_path = root / "qasper.yaml"
            output_root = root / "prepared"
            rows = [
                {
                    "doc_uuid": "paper-1",
                    "doc_path": "qasper://paper-1",
                    "qasper_question_id": "q-1",
                    "question": "What is first?",
                    "answer": [{"free_form_answer": "first"}],
                },
                {
                    "doc_uuid": "paper-1",
                    "doc_path": "qasper://paper-1",
                    "qasper_question_id": "q-2",
                    "question": "What is second?",
                    "answer": [{"free_form_answer": "second"}],
                },
            ]
            dataset_path.write_text(json.dumps(rows), encoding="utf-8")
            self._save_qasper_tree(working_dir / "paper-1")
            self._write_dataset_config(config_path, dataset_path, working_dir, "qasper")

            manifest = prepare_scopes(str(config_path), str(output_root))
            scope = json.loads(
                (output_root / manifest["scopes"][0]["relative_path"]).read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["scope_count"], 1)
        self.assertEqual(manifest["query_count"], 2)
        self.assertEqual([c["source_id"] for c in scope["chunks"]], ["2", "3"])
        self.assertEqual(scope["chunks"][0]["metadata"]["qasper_evidence_text"], "First original paragraph.")
        self.assertEqual(scope["chunks"][0]["metadata"]["section"], "Methods")
        self.assertTrue(scope["chunks"][0]["content"].startswith("[Section: Methods]"))

    def test_prepare_hotpot_uses_one_scope_and_one_passage_per_title(self):
        from Scripts.baselines.hipporag2.prepare import prepare_scopes

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "hotpot.json"
            working_dir = root / "work"
            config_path = root / "hotpot.yaml"
            output_root = root / "prepared"
            row = {
                "doc_uuid": "hotpot-1",
                "hotpotqa_question_id": "hotpot-1",
                "question": "Where was Alpha born?",
                "answer": "Paris",
                "hotpot_type": "bridge",
            }
            dataset_path.write_text(json.dumps([row]), encoding="utf-8")
            self._save_hotpot_tree(working_dir / "hotpot-1")
            self._write_dataset_config(config_path, dataset_path, working_dir, "hotpotqa")

            manifest = prepare_scopes(str(config_path), str(output_root))
            scope = json.loads(
                (output_root / manifest["scopes"][0]["relative_path"]).read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["scope_count"], 1)
        self.assertEqual(len(scope["chunks"]), 1)
        metadata = scope["chunks"][0]["metadata"]
        self.assertEqual(metadata["hotpot_title"], "Article A")
        self.assertEqual(metadata["hotpot_sentences"][1]["sent_id"], 1)
        self.assertIn("Alpha was born in Paris.", scope["chunks"][0]["content"])

    def test_ranked_results_require_unique_known_source_ids(self):
        from Scripts.baselines.hipporag2.common import normalize_ranked_results

        scope = {
            "chunks": [
                {"source_id": "p1", "content": "one", "metadata": {"paragraph_id": 1}},
                {"source_id": "p2", "content": "two", "metadata": {"paragraph_id": 2}},
            ]
        }
        ranked = normalize_ranked_results(
            scope,
            docs=["two", "one"],
            scores=[0.9, 0.8],
            metadatas=[{"source_id": "p2"}, {"source_id": "p1"}],
            graph_seeds=[["alpha", 0.7]],
        )
        self.assertEqual([item["source_id"] for item in ranked], ["p2", "p1"])
        self.assertEqual(ranked[0]["metadata"]["paragraph_id"], 2)
        with self.assertRaises(ValueError):
            normalize_ranked_results(scope, ["x"], [0.1], [{"source_id": "missing"}], [])
        with self.assertRaises(ValueError):
            normalize_ranked_results(scope, ["one", "one"], [0.2, 0.1], [{"source_id": "p1"}, {"source_id": "p1"}], [])

    def test_generation_budget_never_splits_a_block(self):
        from Scripts.baselines.hipporag2.common import apply_generation_budget

        ranked = [
            {"source_id": "p1", "content": "a" * 10},
            {"source_id": "p2", "content": "b" * 20},
            {"source_id": "p3", "content": "c" * 5},
        ]
        selected = apply_generation_budget(
            ranked,
            max_blocks=10,
            max_tokens=15,
            token_counter=len,
        )
        self.assertEqual([item["source_id"] for item in selected], ["p1"])

    def test_citation_validation_rejects_invalid_ids_and_uses_audited_fallback(self):
        from Scripts.baselines.hipporag2.common import validate_qasper_citations

        ranked = [
            {"source_id": "1", "metadata": {"paragraph_id": 1}, "content": "one"},
            {"source_id": "2", "metadata": {"paragraph_id": 2}, "content": "two"},
        ]
        evidence, audit = validate_qasper_citations([999], ranked, max_support=1)
        self.assertEqual(evidence[0]["paragraph_id"], 1)
        self.assertTrue(audit["fallback_used"])
        self.assertEqual(audit["rejected"], ["999"])
        self.assertNotIn("gold", json.dumps(audit).lower())

    def test_hotpot_citations_must_be_legal_title_sentence_pairs(self):
        from Scripts.baselines.hipporag2.common import validate_hotpot_citations

        ranked = [
            {
                "source_id": "title-0",
                "content": "Article A",
                "metadata": {
                    "hotpot_title": "Article A",
                    "hotpot_sentences": [
                        {"sent_id": 0, "text": "Alpha is a scientist."},
                        {"sent_id": 1, "text": "Alpha was born in Paris."},
                    ],
                },
            }
        ]
        evidence, audit = validate_hotpot_citations(
            [{"title": "Article A", "sent_id": 1}, {"title": "Article A", "sent_id": 9}],
            ranked,
            max_support=4,
        )
        self.assertEqual(evidence, [{"hotpot_title": "Article A", "hotpot_sent_id": 1, "text": "Alpha was born in Paris."}])
        self.assertEqual(len(audit["rejected"]), 1)
        with self.assertRaises(ValueError):
            validate_hotpot_citations([], [], max_support=4)

    def test_cache_identity_and_cost_accounting_are_separated(self):
        from Scripts.baselines.hipporag2.common import build_cache_identity, combine_online_cost

        first = build_cache_identity("corpus-a", "config-a", "commit-a", "llm-a", "embed-a")
        second = build_cache_identity("corpus-a", "config-b", "commit-a", "llm-a", "embed-a")
        self.assertNotEqual(first, second)
        cost = combine_online_cost(
            {"prompt_tokens": 11, "completion_tokens": 3},
            {"prompt_tokens": 7, "completion_tokens": 2},
            retrieval_seconds=1.25,
            generation_seconds=0.75,
        )
        self.assertEqual(cost["rag_cost"]["total_tokens"], 23)
        self.assertEqual(cost["time"], 2.0)
        self.assertEqual(cost["stages"]["retrieval"]["total_tokens"], 14)

    def test_official_runner_preserves_metadata_and_reuses_only_matching_cache(self):
        from Scripts.baselines.hipporag2.common import write_json
        from Scripts.baselines.hipporag2.run_official import run_official_scopes

        class FakeHippoRAG:
            index_calls = 0

            def __init__(self, save_dir):
                self.save_dir = save_dir
                self.chunks = []
                self.graph = SimpleNamespace(vcount=lambda: 5, ecount=lambda: 7)
                self.ent_node_to_chunk_ids = {"e": ["p1"]}
                self.proc_triples_to_docs = {"f": ["p1"]}

            def index(self, chunks):
                type(self).index_calls += 1
                self.chunks = list(chunks)

            def retrieve(self, queries, num_to_retrieve=None):
                ordered = list(reversed(self.chunks))[:num_to_retrieve]
                return [
                    SimpleNamespace(
                        docs=[chunk.content for chunk in ordered],
                        doc_scores=[0.9 - index * 0.1 for index in range(len(ordered))],
                        doc_metadata=[
                            {**chunk.metadata, "source_id": chunk.source_id}
                            for chunk in ordered
                        ],
                        graph_seeds=[["fact", 0.8]],
                    )
                ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "prepared"
            scope = {
                "dataset": "qasper",
                "scope_id": "paper-1",
                "corpus_sha256": "corpus-a",
                "chunks": [
                    {"source_id": "p1", "content": "one", "metadata": {"paragraph_id": 1}},
                    {"source_id": "p2", "content": "two", "metadata": {"paragraph_id": 2}},
                ],
                "queries": [{"question_id": "q1", "question": "Which?", "answer": "two", "dataset_row": {"doc_uuid": "paper-1"}}],
            }
            write_json(prepared / "scopes" / "paper.json", scope)
            write_json(
                prepared / "manifest.json",
                {
                    "dataset": "qasper",
                    "working_dir": str(root / "work"),
                    "scope_count": 1,
                    "query_count": 1,
                    "scopes": [{"scope_id": "paper-1", "relative_path": "scopes/paper.json", "corpus_sha256": "corpus-a"}],
                },
            )
            config_path = root / "hipporag2.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "method_suffix: hipporag2_official",
                        "source_revision: commit-a",
                        "retrieval_topk: 20",
                        "hipporag:",
                        "  llm_model_name: llm-a",
                        "  embedding_model_name: VLLM/embed-a",
                    ]
                ),
                encoding="utf-8",
            )

            def factory(settings, save_dir):
                return FakeHippoRAG(save_dir)

            chunk_factory = lambda content, source_id, metadata: SimpleNamespace(
                content=content, source_id=source_id, metadata=metadata
            )
            first = run_official_scopes(
                str(prepared), str(config_path), str(root / "raw"), factory, chunk_factory
            )
            second = run_official_scopes(
                str(prepared), str(config_path), str(root / "raw"), factory, chunk_factory
            )
            raw_result = json.loads(
                (root / "raw" / first["scopes"][0]["relative_path"]).read_text(encoding="utf-8")
            )
            config_path.write_text(config_path.read_text(encoding="utf-8") + "\nretrieval_topk: 10\n", encoding="utf-8")
            third = run_official_scopes(
                str(prepared), str(config_path), str(root / "raw"), factory, chunk_factory
            )

        self.assertEqual(FakeHippoRAG.index_calls, 2)
        self.assertTrue(second["scopes"][0]["reused"])
        self.assertNotEqual(first["run_id"], third["run_id"])
        self.assertEqual(raw_result["queries"][0]["ranked_results"][0]["source_id"], "p2")
        self.assertEqual(raw_result["queries"][0]["ranked_results"][0]["metadata"]["paragraph_id"], 2)

    def test_finalize_writes_evaluator_compatible_qasper_outputs(self):
        from Scripts.baselines.hipporag2.common import write_json
        from Scripts.baselines.hipporag2.finalize import finalize_runs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "prepared"
            raw = root / "raw"
            work = root / "work"
            scope = {
                "dataset": "qasper",
                "scope_id": "paper-1",
                "corpus_sha256": "corpus-a",
                "chunks": [{"source_id": "2", "content": "[Section: Results]\nAnswer is alpha.", "metadata": {"paragraph_id": 2, "qasper_evidence_text": "Answer is alpha.", "section": "Results"}}],
                "queries": [{"question_id": "q1", "question": "What is the answer?", "answer": "alpha", "dataset_row": {"doc_uuid": "paper-1", "qasper_question_id": "q1", "question": "What is the answer?", "answer": "alpha"}}],
            }
            write_json(prepared / "scopes" / "paper.json", scope)
            write_json(prepared / "manifest.json", {"dataset": "qasper", "working_dir": str(work), "scopes": [{"scope_id": "paper-1", "relative_path": "scopes/paper.json"}]})
            raw_scope = {
                "scope_id": "paper-1",
                "cache_identity": "cache-a",
                "index_stats": {"passages": 1},
                "queries": [{"question_id": "q1", "question": "What is the answer?", "ranked_results": [{"rank": 1, "score": 0.9, "source_id": "2", "content": "[Section: Results]\nAnswer is alpha.", "metadata": {"source_id": "2", "paragraph_id": 2, "qasper_evidence_text": "Answer is alpha.", "section": "Results"}}], "graph_seeds": [["alpha", 0.9]], "retrieval_time": 0.2, "retrieval_token_cost": {"prompt_tokens": 5, "completion_tokens": 1}, "dense_fallback": False, "dense_fallback_reason": None}],
            }
            write_json(raw / "scope-result.json", raw_scope)
            write_json(raw / "manifest.json", {"dataset": "qasper", "prepared_manifest_path": str(prepared / "manifest.json"), "scopes": [{"scope_id": "paper-1", "relative_path": "scope-result.json"}]})
            config_path = root / "hipporag2.yaml"
            config_path.write_text("\n".join(["method_suffix: hipporag2_official", "generation_max_blocks: 10", "generation_max_tokens: 4000", "supporting_evidence_topk: 4", "generation:", "  model_name: model-a", "  temperature: 0.1"]), encoding="utf-8")

            generation_calls = []

            def generator(dataset, question, selected):
                generation_calls.append(question)
                return {
                    "answer_short": "alpha",
                    "answer_rationale": "supported",
                    "supporting_block_ids": [999],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2},
                    "time": 0.3,
                }

            summary = finalize_runs(str(raw / "manifest.json"), str(config_path), generator=generator, token_counter=len)
            finalize_runs(str(raw / "manifest.json"), str(config_path), generator=generator, token_counter=len)
            result_dir = work / "paper-1" / "eval_qasper_hipporag2_official"
            final_results = json.loads((result_dir / "final_results.json").read_text(encoding="utf-8"))
            retrieval = json.loads((result_dir / "query_001" / "retrieval_res.json").read_text(encoding="utf-8"))
            cost = json.loads((result_dir / "token_cost.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["question_count"], 1)
        self.assertEqual(generation_calls, ["What is the answer?"])
        self.assertEqual(final_results[0]["answer_short"], "alpha")
        self.assertEqual(retrieval["supporting_block_ids"], [2])
        self.assertTrue(retrieval["citation_validation"]["fallback_used"])
        self.assertEqual(cost["rag_cost"]["total_tokens"], 15)

    def test_finalize_rejects_illegal_hotpot_fact_and_falls_back_legally(self):
        from Scripts.baselines.hipporag2.common import write_json
        from Scripts.baselines.hipporag2.finalize import finalize_runs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "prepared"
            raw = root / "raw"
            work = root / "work"
            scope = {
                "dataset": "hotpotqa",
                "scope_id": "h1",
                "corpus_sha256": "c",
                "chunks": [],
                "queries": [{"question_id": "h1", "question": "Where?", "answer": "Paris", "dataset_row": {"doc_uuid": "h1", "hotpotqa_question_id": "h1", "question": "Where?", "answer": "Paris"}}],
            }
            write_json(prepared / "scopes" / "h1.json", scope)
            write_json(prepared / "manifest.json", {"dataset": "hotpotqa", "working_dir": str(work), "scopes": [{"scope_id": "h1", "relative_path": "scopes/h1.json"}]})
            ranked = [{"rank": 1, "score": 1.0, "source_id": "title-0", "content": "Title: A", "metadata": {"source_id": "title-0", "hotpot_title": "A", "hotpot_sentences": [{"sent_id": 0, "text": "Born in Paris."}]}}]
            write_json(raw / "h1.json", {"scope_id": "h1", "cache_identity": "x", "index_stats": {}, "queries": [{"question_id": "h1", "question": "Where?", "ranked_results": ranked, "graph_seeds": [], "retrieval_time": 0.1, "retrieval_token_cost": {}, "dense_fallback": True, "dense_fallback_reason": "no_graph_facts"}]})
            write_json(raw / "manifest.json", {"dataset": "hotpotqa", "prepared_manifest_path": str(prepared / "manifest.json"), "scopes": [{"scope_id": "h1", "relative_path": "h1.json"}]})
            config_path = root / "cfg.yaml"
            config_path.write_text("method_suffix: hipporag2_official\ngeneration_max_blocks: 10\ngeneration_max_tokens: 4000\nsupporting_evidence_topk: 4\n", encoding="utf-8")

            finalize_runs(
                str(raw / "manifest.json"),
                str(config_path),
                generator=lambda dataset, question, selected: {"answer_short": "Paris", "supporting_facts": [["A", 9]], "usage": {}, "time": 0.0},
                token_counter=len,
            )
            retrieval = json.loads((work / "h1" / "eval_hotpotqa_hipporag2_official" / "query_001" / "retrieval_res.json").read_text(encoding="utf-8"))

        self.assertEqual(retrieval["supporting_evidence"][0]["hotpot_title"], "A")
        self.assertEqual(retrieval["supporting_evidence"][0]["hotpot_sent_id"], 0)
        self.assertTrue(retrieval["citation_validation"]["fallback_used"])

    def test_authenticated_vllm_embedding_adds_bearer_header(self):
        from unittest.mock import Mock, patch

        from Scripts.baselines.hipporag2.run_official import _attach_vllm_bearer_auth

        embedding_model = SimpleNamespace(
            model_id="Qwen/Qwen3-Embedding-0.6B",
            base_url="https://api.example/v1/embeddings",
        )
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"data": [{"embedding": [0.25, 0.75]}]}

        with patch(
            "Scripts.baselines.hipporag2.run_official.requests.post",
            return_value=response,
        ) as post:
            _attach_vllm_bearer_auth(embedding_model, "secret-key")
            vectors = embedding_model.call_model(["test passage"])

        self.assertEqual(vectors.shape, (1, 2))
        _, kwargs = post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret-key")
        self.assertEqual(kwargs["json"]["model"], "Qwen/Qwen3-Embedding-0.6B")

    def test_authenticated_vllm_embedding_retries_transient_ssl_failure(self):
        from unittest.mock import Mock, patch

        from requests.exceptions import SSLError

        from Scripts.baselines.hipporag2.run_official import _attach_vllm_bearer_auth

        embedding_model = SimpleNamespace(
            model_id="Qwen/Qwen3-Embedding-0.6B",
            base_url="https://api.example/v1/embeddings",
        )
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"data": [{"embedding": [0.25, 0.75]}]}

        with (
            patch(
                "Scripts.baselines.hipporag2.run_official.requests.post",
                side_effect=[SSLError("remote closed TLS connection"), response],
            ) as post,
            patch("Scripts.baselines.hipporag2.run_official.time.sleep") as sleep,
        ):
            _attach_vllm_bearer_auth(
                embedding_model,
                "secret-key",
                max_attempts=3,
                retry_backoff_seconds=0.25,
            )
            vectors = embedding_model.call_model(["test passage"])

        self.assertEqual(vectors.shape, (1, 2))
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(0.25)

    def test_openie_executor_caps_default_and_excessive_worker_counts(self):
        from concurrent.futures import ThreadPoolExecutor

        from Scripts.baselines.hipporag2.run_official import _attach_openie_worker_limit

        openie_module = SimpleNamespace(ThreadPoolExecutor=ThreadPoolExecutor)
        _attach_openie_worker_limit(openie_module, max_workers=4)

        with openie_module.ThreadPoolExecutor() as default_executor:
            self.assertEqual(default_executor._max_workers, 4)
        with openie_module.ThreadPoolExecutor(max_workers=2) as smaller_executor:
            self.assertEqual(smaller_executor._max_workers, 2)
        with openie_module.ThreadPoolExecutor(max_workers=12) as capped_executor:
            self.assertEqual(capped_executor._max_workers, 4)

    def test_json_safe_openie_recovers_valid_json_rejected_by_upstream_eval(self):
        from Scripts.baselines.hipporag2.run_official import _attach_json_safe_openie

        class FakeOpenIE:
            def triple_extraction(self, chunk_key, passage, named_entities):
                return SimpleNamespace(
                    chunk_id=chunk_key,
                    response='{"triples":[["corpus","is annotated",true],["corpus","comes from","Wikipedia"]]}',
                    metadata={"error": "name 'true' is not defined"},
                    triples=[],
                )

            def batch_openie(self, chunks):
                result = self.triple_extraction("chunk-1", "passage", [])
                return {}, {"chunk-1": result}

        openie = FakeOpenIE()
        _attach_json_safe_openie(openie)
        _, triple_results = openie.batch_openie({"chunk-1": {"content": "passage"}})

        result = triple_results["chunk-1"]
        self.assertNotIn("error", result.metadata)
        self.assertTrue(result.metadata["adapter_json_recovered"])
        self.assertEqual(len(result.triples), 2)

    def test_json_safe_openie_repairs_latex_unicode_escape_only(self):
        from Scripts.baselines.hipporag2.run_official import _triples_from_json_response

        response = (
            '{"triples":['
            '["ISA$^\\uparrow$ questions","about","trousers"],'
            '["tau","equals","\\u03c4"]'
            ']}'
        )

        triples = _triples_from_json_response(response)

        self.assertEqual(
            triples,
            [["ISA$^\\uparrow$ questions", "about", "trousers"], ["tau", "equals", "τ"]],
        )

    def test_json_safe_openie_repairs_unterminated_triple_string_at_line_end(self):
        from Scripts.baselines.hipporag2.run_official import _triples_from_json_response

        response = """{
  "triples": [
    ["emoji examples", "include", ":D"],
    ["emoji examples", "include", ":(],
    ["emoji examples", "include", "<3"]
  ]
}"""

        triples = _triples_from_json_response(response)

        self.assertEqual(
            triples,
            [
                ["emoji examples", "include", ":D"],
                ["emoji examples", "include", ":("],
                ["emoji examples", "include", "<3"],
            ],
        )

    def test_json_safe_openie_escapes_unquoted_inch_mark_inside_triple(self):
        from Scripts.baselines.hipporag2.run_official import _triples_from_json_response

        response = r'''{
  "triples": [
    ["Mile Ilić", "is", "2.15 m tall (7'1")"],
    ["Mile Ilić", "played for", "New Jersey Nets"]
  ]
}'''

        triples = _triples_from_json_response(response)

        self.assertEqual(
            triples,
            [
                ["Mile Ilić", "is", "2.15 m tall (7'1\")"],
                ["Mile Ilić", "played for", "New Jersey Nets"],
            ],
        )

    def test_json_safe_openie_retries_twice_then_fails_scope(self):
        from Scripts.baselines.hipporag2.run_official import _attach_json_safe_openie

        class FakeOpenIE:
            def __init__(self):
                self.calls = []

            def triple_extraction(self, chunk_key, passage, named_entities):
                self.calls.append(passage)
                return SimpleNamespace(
                    chunk_id=chunk_key,
                    response="not-json",
                    metadata={"error": "parse failure"},
                    triples=[],
                )

            def batch_openie(self, chunks):
                result = self.triple_extraction("chunk-1", "passage", [])
                return {}, {"chunk-1": result}

        openie = FakeOpenIE()
        _attach_json_safe_openie(openie)

        with self.assertRaisesRegex(RuntimeError, "chunk-1"):
            openie.batch_openie({"chunk-1": {"content": "passage"}})

        self.assertEqual(openie.calls, ["passage", "passage\n", "passage\n\n"])


if __name__ == "__main__":
    unittest.main()
