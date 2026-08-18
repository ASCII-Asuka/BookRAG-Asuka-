from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import yaml


REPO = Path(__file__).resolve().parents[1]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class ReactOfficialReproTests(unittest.TestCase):
    def test_source_lock_and_public_config_are_fixed_and_secret_free(self):
        lock_path = (
            REPO
            / "Scripts"
            / "baselines"
            / "react_official"
            / "official-source-lock.json"
        )
        config_path = REPO / "config" / "react_official_repro.yaml"

        lock = _load_json(lock_path)
        config_text = config_path.read_text(encoding="utf-8")
        config = yaml.safe_load(config_text)

        self.assertEqual(lock["repository"], "https://github.com/ysymyth/ReAct")
        self.assertEqual(
            lock["commit"], "6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9"
        )
        self.assertEqual(lock["license"], "MIT")
        self.assertEqual(config["method_suffix"], "react_official_repro")
        self.assertEqual(config["react"]["temperature"], 0.0)
        self.assertEqual(config["react"]["max_output_tokens"], 100)
        self.assertEqual(config["react"]["max_steps"], 7)
        lowered = config_text.lower()
        self.assertNotIn("api_key", lowered)
        self.assertNotIn("api_base", lowered)
        self.assertNotIn("base_url", lowered)

    def test_run_identity_changes_when_corpus_or_config_changes(self):
        from Scripts.baselines.react_official.common import build_run_identity

        first = build_run_identity("corpus-a", "config-a", "commit", "model")
        self.assertEqual(
            first, build_run_identity("corpus-a", "config-a", "commit", "model")
        )
        self.assertNotEqual(
            first, build_run_identity("corpus-b", "config-a", "commit", "model")
        )
        self.assertNotEqual(
            first, build_run_identity("corpus-a", "config-b", "commit", "model")
        )

    def test_adapter_code_hash_changes_with_implementation_files(self):
        from Scripts.baselines.react_official.common import adapter_code_sha256

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "a.py"
            second = root / "b.py"
            first.write_text("one", encoding="utf-8")
            second.write_text("two", encoding="utf-8")
            original = adapter_code_sha256([first, second])
            second.write_text("changed", encoding="utf-8")

            self.assertNotEqual(original, adapter_code_sha256([first, second]))

    def test_completion_uses_official_parameters_tls_and_client_stop(self):
        from Scripts.baselines.react_official.model_client import CompletionClient

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [
                        {
                            "text": (
                                "Thought 1: inspect\nAction 1: Search[Alpha]"
                                "\nObservation 1: leaked"
                            )
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 8,
                        "total_tokens": 20,
                    },
                }

        class FakeSession:
            def __init__(self):
                self.last_url = None
                self.last_json = None
                self.last_verify = None

            def post(self, url, **kwargs):
                self.last_url = url
                self.last_json = kwargs["json"]
                self.last_verify = kwargs["verify"]
                return FakeResponse()

        session = FakeSession()
        client = CompletionClient(
            api_key="secret",
            base_url="https://service.example/v1",
            timeout=30,
            max_attempts=2,
            session=session,
        )
        text, usage = client.complete(
            "Qwen/Qwen3-VL-8B-Instruct",
            "prompt",
            stop="\nObservation 1:",
        )

        self.assertEqual(
            text, "Thought 1: inspect\nAction 1: Search[Alpha]"
        )
        self.assertEqual(session.last_url, "https://service.example/v1/completions")
        self.assertEqual(session.last_json["temperature"], 0.0)
        self.assertEqual(session.last_json["max_tokens"], 100)
        self.assertEqual(session.last_json["stop"], ["\nObservation 1:"])
        self.assertTrue(session.last_verify)
        self.assertEqual(usage["total_tokens"], 20)

    def test_completion_retries_network_failures_with_a_bound(self):
        from Scripts.baselines.react_official.model_client import CompletionClient

        class RecoveringSession:
            def __init__(self):
                self.calls = 0

            def post(self, _url, **_kwargs):
                import requests

                self.calls += 1
                if self.calls == 1:
                    raise requests.Timeout("temporary")
                return SimpleNamespace(
                    raise_for_status=lambda: None,
                    json=lambda: {"choices": [{"text": "Finish[ok]"}]},
                )

        session = RecoveringSession()
        client = CompletionClient(
            api_key="secret",
            base_url="https://service.example/v1",
            max_attempts=2,
            retry_backoff_seconds=0,
            session=session,
        )
        with patch("Scripts.baselines.react_official.model_client.time.sleep"):
            text, _usage = client.complete("model", "prompt", stop="\n")

        self.assertEqual(text, "Finish[ok]")
        self.assertEqual(session.calls, 2)

    def test_chat_completion_wrapper_emulates_text_continuation(self):
        from Scripts.baselines.react_official.model_client import CompletionClient

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "Search[Alpha]"}}]}

        class FakeSession:
            def post(self, _url, **kwargs):
                self.payload = kwargs["json"]
                return FakeResponse()

        session = FakeSession()
        client = CompletionClient(
            "secret",
            "https://service.example/v1/chat/completions",
            session=session,
        )
        text, _usage = client.complete("model", "prompt ending in Action 1:", "\n")

        self.assertEqual(text, "Search[Alpha]")
        self.assertEqual(session.payload["messages"][-1]["content"], "prompt ending in Action 1:")
        self.assertIn("text-completion engine", session.payload["messages"][0]["content"])
        self.assertIn("Do not repeat", session.payload["messages"][0]["content"])

    def test_hotpot_search_and_lookup_never_leave_current_scope(self):
        from Scripts.baselines.react_official.environment import HotpotEnvironment

        chunks = [
            {
                "source_id": "alpha",
                "content": "Alpha\nAlpha was founded in 1900. Its founder was Ada.",
                "metadata": {
                    "hotpot_title": "Alpha",
                    "hotpot_sentences": [
                        {"sent_id": 0, "text": "Alpha was founded in 1900."},
                        {"sent_id": 1, "text": "Its founder was Ada."},
                        {"sent_id": 2, "text": "The founder retired in 1930."},
                    ],
                },
            },
            {
                "source_id": "beta",
                "content": "Beta\nBeta is unrelated.",
                "metadata": {
                    "hotpot_title": "Beta",
                    "hotpot_sentences": [
                        {"sent_id": 0, "text": "Beta is unrelated."},
                    ],
                },
            },
        ]
        env = HotpotEnvironment.from_chunks(chunks)

        search = env.search("alpha")
        self.assertTrue(search.observation.startswith("Alpha"))
        self.assertEqual(
            {item["hotpot_title"] for item in search.evidence}, {"Alpha"}
        )
        first = env.lookup("founder")
        second = env.lookup("founder")
        self.assertEqual(first.evidence[0]["hotpot_sent_id"], 1)
        self.assertEqual(second.evidence[0]["hotpot_sent_id"], 2)
        self.assertNotIn("Outside Scope", search.observation)

    def test_hotpot_distinguishes_titles_that_differ_only_by_symbols(self):
        from Scripts.baselines.react_official.environment import HotpotEnvironment

        chunks = [
            {
                "source_id": "plus",
                "metadata": {
                    "hotpot_title": "Romeo + Juliet",
                    "hotpot_sentences": [
                        {"sent_id": 0, "text": "The plus-title page."}
                    ],
                },
            },
            {
                "source_id": "times",
                "metadata": {
                    "hotpot_title": "Romeo × Juliet",
                    "hotpot_sentences": [
                        {"sent_id": 0, "text": "The times-title page."}
                    ],
                },
            },
        ]

        env = HotpotEnvironment.from_chunks(chunks)

        self.assertEqual(env.search("Romeo + Juliet").evidence[0]["source_id"], "plus")
        self.assertEqual(env.search("Romeo × Juliet").evidence[0]["source_id"], "times")

    def test_hotpot_distinguishes_titles_that_differ_by_case(self):
        from Scripts.baselines.react_official.environment import HotpotEnvironment

        chunks = [
            {
                "source_id": "magazine",
                "metadata": {
                    "hotpot_title": "Popular Science",
                    "hotpot_sentences": [
                        {"sent_id": 0, "text": "Popular Science is a magazine."}
                    ],
                },
            },
            {
                "source_id": "concept",
                "metadata": {
                    "hotpot_title": "Popular science",
                    "hotpot_sentences": [
                        {"sent_id": 0, "text": "Popular science is written for general audiences."}
                    ],
                },
            },
        ]

        env = HotpotEnvironment.from_chunks(chunks)

        self.assertEqual(env.search("Popular Science").evidence[0]["source_id"], "magazine")
        self.assertEqual(env.search("Popular science").evidence[0]["source_id"], "concept")

    def test_qasper_search_keeps_original_paragraph_ids_and_text(self):
        from Scripts.baselines.react_official.environment import QasperEnvironment

        chunks = [
            {
                "source_id": "p-1",
                "content": "[Section: Methods]\noriginal paragraph",
                "metadata": {
                    "paragraph_id": "p-1",
                    "qasper_evidence_text": "original paragraph",
                    "section": "Methods",
                    "title_path": "Paper > Methods",
                },
            },
            {
                "source_id": "p-2",
                "content": "[Section: Results]\nresult paragraph",
                "metadata": {
                    "paragraph_id": "p-2",
                    "qasper_evidence_text": "result paragraph",
                    "section": "Results",
                    "title_path": "Paper > Results",
                },
            },
        ]
        env = QasperEnvironment.from_chunks(chunks)

        result = env.search("Methods")
        self.assertEqual(result.evidence[0]["paragraph_id"], "p-1")
        self.assertEqual(result.evidence[0]["content"], "original paragraph")
        self.assertNotIn("[Section: Methods]", result.evidence[0]["content"])

    def test_search_fallback_is_deterministic_and_scope_local(self):
        from Scripts.baselines.react_official.environment import HotpotEnvironment

        chunks = [
            {
                "source_id": "a",
                "content": "A title\nA telescope observes planets.",
                "metadata": {
                    "hotpot_title": "A title",
                    "hotpot_sentences": [
                        {"sent_id": 0, "text": "A telescope observes planets."}
                    ],
                },
            },
            {
                "source_id": "b",
                "content": "B title\nA violin produces music.",
                "metadata": {
                    "hotpot_title": "B title",
                    "hotpot_sentences": [
                        {"sent_id": 0, "text": "A violin produces music."}
                    ],
                },
            },
        ]
        env = HotpotEnvironment.from_chunks(chunks)

        self.assertTrue(env.search("planets telescope").observation.startswith("A title"))

    def test_prompt_sources_have_six_examples_and_official_action_contract(self):
        from Scripts.baselines.react_official.prompts import (
            REACT_INSTRUCTION,
            get_react_examples,
        )

        hotpot = get_react_examples("hotpotqa", REPO / "runs/third_party/ReAct")
        qasper = get_react_examples("qasper", REPO / "runs/third_party/ReAct")
        self.assertEqual(hotpot.count("Question:"), 6)
        self.assertEqual(qasper.count("Question:"), 6)
        self.assertIn("Search[entity]", REACT_INSTRUCTION)
        self.assertIn("Lookup[keyword]", REACT_INSTRUCTION)
        self.assertIn("Finish[answer]", REACT_INSTRUCTION)

    def test_parse_action_accepts_only_three_official_actions(self):
        from Scripts.baselines.react_official.runner import parse_action

        self.assertEqual(parse_action("Search[Alpha]").name, "Search")
        self.assertEqual(parse_action("lookup[founder]").name, "Lookup")
        self.assertEqual(parse_action("Finish[yes]").argument, "yes")
        with self.assertRaises(ValueError):
            parse_action("Browse[Alpha]")

    def test_runner_repairs_once_and_stops_on_finish(self):
        from Scripts.baselines.react_official.environment import ObservationResult
        from Scripts.baselines.react_official.runner import ReActRunner

        class FakeModel:
            def __init__(self):
                self.responses = [
                    "inspect",
                    "Search[Alpha]",
                    "done\nAction 2: Finish[answer]",
                ]
                self.stops = []

            def complete(self, _model, _prompt, stop, **_kwargs):
                self.stops.append(stop)
                return self.responses.pop(0), {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                }

        class FakeEnvironment:
            def execute(self, action, argument):
                self.last = (action, argument)
                return ObservationResult(
                    action,
                    argument,
                    "Alpha is evidence.",
                    [{"source_id": "alpha", "content": "Alpha is evidence."}],
                )

        model = FakeModel()
        result = ReActRunner(model, model_name="model", max_steps=7).run(
            "question", FakeEnvironment(), examples="Question: demo\nAction 1: Finish[x]"
        )

        self.assertEqual(result.answer, "answer")
        self.assertEqual(result.repair_calls, 1)
        self.assertEqual(
            model.stops, ["\nObservation 1:", "\n", "\nObservation 2:"]
        )
        self.assertEqual(result.termination_reason, "finish")
        self.assertEqual(result.model_calls, 3)

    def test_runner_repairs_action_marker_with_empty_body(self):
        from Scripts.baselines.react_official.runner import ReActRunner

        class FakeModel:
            def __init__(self):
                self.responses = [
                    "inspect the evidence\nAction 1:",
                    "Finish[answer]",
                ]

            def complete(self, _model, _prompt, _stop, **_kwargs):
                return self.responses.pop(0), {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                }

        result = ReActRunner(FakeModel(), model_name="model", max_steps=7).run(
            "question", object(), examples="Question: demo"
        )

        self.assertEqual(result.answer, "answer")
        self.assertEqual(result.repair_calls, 1)
        self.assertEqual(result.model_calls, 2)
        self.assertTrue(result.steps[0].repaired)

    def test_runner_records_invalid_action_when_repair_response_is_empty(self):
        from Scripts.baselines.react_official.runner import ReActRunner

        class FakeModel:
            def __init__(self):
                self.responses = [
                    "inspect the evidence",
                    "",
                    "conclude\nAction 2: Finish[answer]",
                ]

            def complete(self, _model, _prompt, _stop, **_kwargs):
                return self.responses.pop(0), {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                }

        result = ReActRunner(FakeModel(), model_name="model", max_steps=7).run(
            "question", object(), examples="Question: demo"
        )

        self.assertEqual(result.answer, "answer")
        self.assertEqual(result.repair_calls, 1)
        self.assertEqual(result.invalid_actions, 1)
        self.assertEqual(result.steps[0].action, "")
        self.assertFalse(result.steps[0].valid_action)

    def test_runner_forces_empty_finish_after_seven_steps(self):
        from Scripts.baselines.react_official.environment import ObservationResult
        from Scripts.baselines.react_official.runner import ReActRunner

        class FakeModel:
            def __init__(self):
                self.step = 0

            def complete(self, _model, _prompt, _stop, **_kwargs):
                self.step += 1
                return (
                    f"continue\nAction {self.step}: Search[x]",
                    {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                )

        class FakeEnvironment:
            def execute(self, action, argument):
                return ObservationResult(action, argument, "keep going", [])

        result = ReActRunner(FakeModel(), model_name="model", max_steps=7).run(
            "question", FakeEnvironment(), examples="Question: demo"
        )

        self.assertEqual(result.answer, "")
        self.assertEqual(result.termination_reason, "step_limit")
        self.assertEqual(len(result.steps), 7)
        self.assertEqual(result.model_calls, 7)

    def test_run_reuses_only_complete_matching_identity(self):
        from Scripts.baselines.react_official.common import scope_corpus_sha256
        from Scripts.baselines.react_official.run import run_prepared

        class FinishingModel:
            def __init__(self):
                self.calls = 0

            def complete(self, _model, _prompt, _stop, **_kwargs):
                self.calls += 1
                return (
                    "done\nAction 1: Finish[yes]",
                    {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                )

        class FailIfCalledModel:
            def complete(self, *_args, **_kwargs):
                raise AssertionError("valid cached query must not call the model")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            (prepared / "scopes").mkdir(parents=True)
            scope = {
                "scope_id": "paper-1",
                "chunks": [
                    {
                        "source_id": "p-1",
                        "content": "[Section: Methods]\noriginal paragraph",
                        "metadata": {
                            "paragraph_id": "p-1",
                            "qasper_evidence_text": "original paragraph",
                            "section": "Methods",
                            "title_path": "Paper > Methods",
                        },
                    }
                ],
                "queries": [
                    {
                        "question_id": "q-1",
                        "question": "Is this a test?",
                        "answer": "yes",
                        "dataset_row": {"doc_uuid": "paper-1"},
                    }
                ],
            }
            scope["corpus_sha256"] = scope_corpus_sha256(scope["chunks"])
            (prepared / "scopes" / "paper.json").write_text(
                json.dumps(scope), encoding="utf-8"
            )
            (prepared / "manifest.json").write_text(
                json.dumps(
                    {
                        "dataset": "qasper",
                        "working_dir": str(root / "work"),
                        "scope_count": 1,
                        "query_count": 1,
                        "scopes": [
                            {
                                "scope_id": "paper-1",
                                "relative_path": "scopes/paper.json",
                                "corpus_sha256": scope["corpus_sha256"],
                                "query_count": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "config.yaml"
            config_path.write_text(
                (REPO / "config/react_official_repro.yaml").read_text("utf-8"),
                encoding="utf-8",
            )
            model = FinishingModel()
            first = run_prepared(
                prepared,
                config_path,
                root / "out",
                completion_model=model,
                source_root=REPO / "runs/third_party/ReAct",
            )
            second = run_prepared(
                prepared,
                config_path,
                root / "out",
                completion_model=FailIfCalledModel(),
                source_root=REPO / "runs/third_party/ReAct",
            )

            self.assertEqual(first["question_count"], 1)
            self.assertEqual(second["question_count"], 1)
            self.assertEqual(second["cached_question_count"], 1)
            self.assertEqual(model.calls, 1)
            raw_manifest = _load_json(Path(first["manifest_path"]))
            self.assertEqual(raw_manifest["status"], "complete")

    def test_run_rejects_duplicate_question_ids(self):
        from Scripts.baselines.react_official.common import scope_corpus_sha256
        from Scripts.baselines.react_official.run import run_prepared

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            (prepared / "scopes").mkdir(parents=True)
            scope = {
                "scope_id": "paper-1",
                "chunks": [
                    {
                        "source_id": "p-1",
                        "content": "text",
                        "metadata": {
                            "paragraph_id": "p-1",
                            "qasper_evidence_text": "text",
                            "section": "S",
                            "title_path": "S",
                        },
                    }
                ],
                "queries": [
                    {"question_id": "duplicate", "question": "one"},
                    {"question_id": "duplicate", "question": "two"},
                ],
            }
            scope["corpus_sha256"] = scope_corpus_sha256(scope["chunks"])
            (prepared / "scopes/scope.json").write_text(
                json.dumps(scope), encoding="utf-8"
            )
            (prepared / "manifest.json").write_text(
                json.dumps(
                    {
                        "dataset": "qasper",
                        "working_dir": str(root / "work"),
                        "scope_count": 1,
                        "query_count": 2,
                        "scopes": [
                            {
                                "scope_id": "paper-1",
                                "relative_path": "scopes/scope.json",
                                "corpus_sha256": scope["corpus_sha256"],
                                "query_count": 2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "config.yaml"
            config_path.write_text(
                (REPO / "config/react_official_repro.yaml").read_text("utf-8"),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate"):
                run_prepared(
                    prepared,
                    config_path,
                    root / "out",
                    completion_model=SimpleNamespace(),
                    source_root=REPO / "runs/third_party/ReAct",
                )

    def test_observed_evidence_orders_lookup_before_search_and_deduplicates(self):
        from Scripts.baselines.react_official.finalize import rank_observed_evidence

        steps = [
            {
                "action_name": "Search",
                "evidence": [
                    {"source_id": "shared", "content": "shared"},
                    {"source_id": "search-only", "content": "search"},
                ],
            },
            {
                "action_name": "Lookup",
                "evidence": [
                    {"source_id": "shared", "content": "shared"},
                    {"source_id": "lookup-only", "content": "lookup"},
                ],
            },
        ]

        ranked = rank_observed_evidence(steps, dataset="qasper")

        self.assertEqual(
            [item["source_id"] for item in ranked],
            ["shared", "lookup-only", "search-only"],
        )
        self.assertEqual([item["supporting_rank"] for item in ranked], [1, 2, 3])

    def test_empty_trace_submits_empty_evidence_without_gold_fallback(self):
        from Scripts.baselines.react_official.finalize import rank_observed_evidence

        self.assertEqual(rank_observed_evidence([], dataset="qasper"), [])
        self.assertEqual(rank_observed_evidence([], dataset="hotpotqa"), [])

    def test_finalize_writes_qasper_and_hotpot_legal_trace_evidence(self):
        from Scripts.baselines.react_official.common import scope_corpus_sha256
        from Scripts.baselines.react_official.finalize import finalize_runs

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.yaml"
            config_path.write_text(
                (REPO / "config/react_official_repro.yaml").read_text("utf-8"),
                encoding="utf-8",
            )
            for dataset in ("qasper", "hotpotqa"):
                dataset_root = root / dataset
                prepared = dataset_root / "prepared"
                raw = dataset_root / "raw"
                (prepared / "scopes").mkdir(parents=True)
                (raw / "scopes").mkdir(parents=True)
                scope_id = "paper" if dataset == "qasper" else "hotpot-id"
                question_id = "q-1" if dataset == "qasper" else "hotpot-id"
                if dataset == "qasper":
                    chunks = [
                        {
                            "source_id": "paragraph-1",
                            "content": "[Section: S]\nsource text",
                            "metadata": {
                                "paragraph_id": "paragraph-1",
                                "qasper_evidence_text": "source text",
                                "section": "S",
                                "title_path": "S",
                            },
                        }
                    ]
                    evidence = {
                        "source_id": "paragraph-1",
                        "paragraph_id": "paragraph-1",
                        "content": "source text",
                    }
                    dataset_row = {
                        "doc_uuid": scope_id,
                        "qasper_question_id": question_id,
                    }
                else:
                    chunks = [
                        {
                            "source_id": "title-0",
                            "content": "Title\nsource sentence",
                            "metadata": {
                                "hotpot_title": "Title",
                                "hotpot_sentences": [
                                    {"sent_id": 0, "text": "source sentence"}
                                ],
                            },
                        }
                    ]
                    evidence = {
                        "source_id": "title-0",
                        "hotpot_title": "Title",
                        "hotpot_sent_id": 0,
                        "content": "source sentence",
                    }
                    dataset_row = {
                        "doc_uuid": scope_id,
                        "hotpotqa_question_id": question_id,
                    }
                scope = {
                    "scope_id": scope_id,
                    "chunks": chunks,
                    "queries": [
                        {
                            "question_id": question_id,
                            "question": "question",
                            "answer": "gold must not be used",
                            "dataset_row": dataset_row,
                        }
                    ],
                }
                scope["corpus_sha256"] = scope_corpus_sha256(chunks)
                (prepared / "scopes/scope.json").write_text(
                    json.dumps(scope), encoding="utf-8"
                )
                (prepared / "manifest.json").write_text(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "working_dir": str(dataset_root / "work"),
                            "scope_count": 1,
                            "query_count": 1,
                            "scopes": [
                                {
                                    "scope_id": scope_id,
                                    "relative_path": "scopes/scope.json",
                                    "query_count": 1,
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                raw_scope = {
                    "status": "complete",
                    "dataset": dataset,
                    "scope_id": scope_id,
                    "cache_identity": "cache",
                    "queries": [
                        {
                            "question_id": question_id,
                            "question": "question",
                            "answer": "predicted",
                            "trajectory": "trajectory",
                            "termination_reason": "finish",
                            "model_calls": 1,
                            "repair_calls": 0,
                            "invalid_actions": 0,
                            "search_calls": 1,
                            "lookup_calls": 0,
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 2,
                                "total_tokens": 12,
                            },
                            "elapsed_seconds": 0.5,
                            "steps": [
                                {
                                    "step": 1,
                                    "action_name": "Search",
                                    "evidence": [evidence],
                                }
                            ],
                        }
                    ],
                }
                (raw / "scopes/scope.json").write_text(
                    json.dumps(raw_scope), encoding="utf-8"
                )
                (raw / "manifest.json").write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "dataset": dataset,
                            "run_id": "run",
                            "run_directory": str(raw),
                            "prepared_manifest_path": str(prepared / "manifest.json"),
                            "scope_count": 1,
                            "query_count": 1,
                            "scopes": [
                                {
                                    "scope_id": scope_id,
                                    "relative_path": "scopes/scope.json",
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

                summary = finalize_runs(raw / "manifest.json", config_path)
                result_directory = (
                    dataset_root
                    / "work"
                    / scope_id
                    / f"eval_{dataset}_react_official_repro"
                )
                result = _load_json(result_directory / "query_001/result.json")
                retrieval = _load_json(
                    result_directory / "query_001/retrieval_res.json"
                )
                self.assertTrue(summary["complete"])
                self.assertEqual(result["answer_short"], "predicted")
                self.assertFalse(retrieval["citation_validation"]["gold_used"])
                if dataset == "qasper":
                    self.assertEqual(result["supporting_block_ids"], ["paragraph-1"])
                else:
                    self.assertEqual(result["supporting_facts"], [["Title", 0]])

    def test_diagnostics_reports_official_failure_rates(self):
        from Scripts.baselines.react_official.diagnostics import summarize_trajectories

        trajectories = [
            {
                "steps": [{}],
                "model_calls": 1,
                "repair_calls": 0,
                "invalid_actions": 0,
                "search_calls": 1,
                "lookup_calls": 0,
                "termination_reason": "finish",
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                "elapsed_seconds": 1.0,
            },
            {
                "steps": [{}],
                "model_calls": 2,
                "repair_calls": 1,
                "invalid_actions": 0,
                "search_calls": 0,
                "lookup_calls": 0,
                "termination_reason": "finish",
                "usage": {"prompt_tokens": 20, "completion_tokens": 4},
                "elapsed_seconds": 2.0,
            },
            {
                "steps": [{}],
                "model_calls": 1,
                "repair_calls": 0,
                "invalid_actions": 1,
                "search_calls": 0,
                "lookup_calls": 0,
                "termination_reason": "finish",
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                "elapsed_seconds": 1.0,
            },
            {
                "steps": [{}, {}, {}, {}, {}, {}, {}],
                "model_calls": 7,
                "repair_calls": 0,
                "invalid_actions": 0,
                "search_calls": 7,
                "lookup_calls": 0,
                "termination_reason": "step_limit",
                "usage": {"prompt_tokens": 70, "completion_tokens": 14},
                "elapsed_seconds": 7.0,
            },
        ]

        report = summarize_trajectories(trajectories)

        self.assertEqual(report["query_count"], 4)
        self.assertAlmostEqual(report["repair_call_rate"], 1 / 11)
        self.assertAlmostEqual(report["repair_query_rate"], 0.25)
        self.assertAlmostEqual(report["invalid_action_rate"], 0.1)
        self.assertAlmostEqual(report["invalid_action_query_rate"], 0.25)
        self.assertAlmostEqual(report["step_limit_rate"], 0.25)
        self.assertAlmostEqual(report["mean_total_tokens"], 33.0)

    def test_private_config_loader_resolves_extends_without_exposing_it(self):
        from Scripts.baselines.react_official.full_experiment import (
            load_private_credentials,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "parent.yaml").write_text(
                """
llm:
  model_name: model
  api_key: secret
  api_base: https://service.example/v1
  temperature: 0.1
""".strip(),
                encoding="utf-8",
            )
            (root / "child.yaml").write_text(
                """
extends: parent.yaml
llm:
  temperature: 0.0
""".strip(),
                encoding="utf-8",
            )

            credentials = load_private_credentials(root / "child.yaml")

            self.assertEqual(credentials.model_name, "model")
            self.assertEqual(credentials.api_key, "secret")
            self.assertEqual(
                credentials.completion_url,
                "https://service.example/v1/chat/completions",
            )

    def test_orchestration_uses_locked_source_and_contains_no_secrets(self):
        script_path = (
            REPO
            / "Scripts/baselines/react_official/run_full_experiments.ps1"
        )
        script = script_path.read_text(encoding="utf-8")

        self.assertIn("6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9", script)
        self.assertIn("react_official_repro", script)
        self.assertIn("full_experiment", script)
        self.assertIn("--private-config", script)
        self.assertNotIn("OPENAI_API_KEY=", script)
        self.assertNotIn("Bearer ", script)


if __name__ == "__main__":
    unittest.main()
