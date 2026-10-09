import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class FakeRAG:
    name = "fake"
    last_generation_provenance = {"stage": "initial", "prompt_block_ids": [7]}
    last_support_context_validation = {"passed": True}

    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def generation(self, query, query_output_dir):
        self.calls += 1
        self.last_generation_provenance = {"stage": "initial", "prompt_block_ids": [7]}
        self.last_support_context_validation = {"passed": True}
        if self.fail:
            raise RuntimeError("sk-secret http://10.1.2.3/v1 C:\\private\\model")
        (query_output_dir / "retrieval_res.json").write_text("{}", encoding="utf-8")
        return "answer", [7]

    def close(self):
        pass


class RunProvenanceTests(unittest.TestCase):
    def _module(self):
        module_path = Path(__file__).resolve().parents[1] / "Core/utils/run_provenance.py"
        self.assertTrue(module_path.exists(), "run provenance helper is missing")
        return importlib.import_module("Core.utils.run_provenance")

    def _run_rag(self, root, data, rag=None, **kwargs):
        # Reuse the existing SDK fixture instead of installing model dependencies.
        from tests.test_evibridge_wiring import _stub_runtime_imports
        _stub_runtime_imports()
        from Core.inference import run_rag

        dataset_path = root / "dataset.json"
        dataset_path.write_text(json.dumps(data), encoding="utf-8")
        run_rag(rag or FakeRAG(), root / "out", dataset_path=str(dataset_path), **kwargs)
        return root / "out"

    def _manifest(self, out):
        files = list((out / ".runs").glob("*/manifest.json"))
        self.assertEqual(len(files), 1, "each invocation needs its own manifest")
        return files[0], json.loads(files[0].read_text(encoding="utf-8"))

    def _events(self, manifest):
        return [json.loads(line) for line in manifest.with_name("events.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_runtime_config_redacts_nested_credentials_urls_and_local_paths(self):
        module = self._module()
        config = {
            "strategy": "evibridge", "top_k": 4,
            "llm": {"api_key": "sk-private-value", "api_base": "http://10.1.2.3/v1", "model_name": "C:\\private\\model"},
            "nested": [{"password": "unusual-credential", "note": "service at https://secret.example/api", "save_path": "/home/private/output"}],
        }
        safe = module.safe_runtime_config(config)
        serialized = json.dumps(safe)
        for value in ("sk-private-value", "10.1.2.3", "C:\\private", "unusual-credential", "secret.example", "/home/private"):
            self.assertNotIn(value, serialized)
        self.assertEqual(safe["top_k"], 4)
        self.assertEqual(safe["strategy"], "evibridge")

    def test_invocations_get_distinct_exclusive_manifests(self):
        module = self._module()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            first = module.RunProvenance(out, [{"question": "q"}], repo_root=Path(tmp))
            before = first.manifest_path.read_bytes()
            second = module.RunProvenance(out, [{"question": "q"}], repo_root=Path(tmp))
            self.assertNotEqual(first.run_id, second.run_id)
            self.assertEqual(first.manifest_path.read_bytes(), before)
            with self.assertRaises(FileExistsError):
                with first.manifest_path.open("x", encoding="utf-8"):
                    pass

    def test_unknown_git_revision_is_explicit(self):
        module = self._module()
        with tempfile.TemporaryDirectory() as tmp, patch.object(module.subprocess, "run", side_effect=OSError("private path")):
            run = module.RunProvenance(Path(tmp), [], repo_root=Path(tmp))
            manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
        self.assertIsNone(manifest["code"]["git_commit"])
        self.assertIsNone(manifest["code"]["git_dirty"])
        self.assertEqual(manifest["code"]["git_status"], "unknown")

    def test_dirty_source_hash_detects_uncommitted_code(self):
        module = self._module()
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "Core").mkdir()
            source = repo / "Core" / "example.py"
            source.write_text("v = 1\n", encoding="utf-8")
            def git_reply(args, **kwargs):
                output = "a" * 40 + "\n" if "rev-parse" in args else " M Core/example.py\n"
                return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")
            with patch.object(module.subprocess, "run", side_effect=git_reply):
                first = module.RunProvenance(repo / "out", [], repo_root=repo)
                source.write_text("v = 2\n", encoding="utf-8")
                second = module.RunProvenance(repo / "out", [], repo_root=repo)
            a = json.loads(first.manifest_path.read_text(encoding="utf-8"))["code"]
            b = json.loads(second.manifest_path.read_text(encoding="utf-8"))["code"]
        self.assertTrue(a["git_dirty"])
        self.assertNotEqual(a["source_sha256"], b["source_sha256"])
        self.assertNotEqual(a["source_files"]["Core/example.py"]["sha256"], b["source_files"]["Core/example.py"]["sha256"])

    def test_dataset_hashes_change_with_question_identity_and_order(self):
        module = self._module()
        first_data = [{"qasper_question_id": "q1", "question": "one", "answer": "gold"}, {"qasper_question_id": "q2", "question": "two"}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = module.RunProvenance(root, first_data, repo_root=root)
            reordered = module.RunProvenance(root, first_data[::-1], repo_root=root)
            changed = module.RunProvenance(root, [{**first_data[0], "qasper_question_id": "q3"}, first_data[1]], repo_root=root)
            manifests = [json.loads(run.manifest_path.read_text(encoding="utf-8")) for run in (first, reordered, changed)]
            for manifest in manifests:
                text = json.dumps(manifest)
                self.assertNotIn('"answer"', text)
                self.assertNotIn('"question"', text)
                self.assertNotIn('"gold"', text)
        for field in ("input_sha256", "ordered_qid_sha256"):
            self.assertNotEqual(manifests[0]["dataset"][field], manifests[1]["dataset"][field])
            self.assertNotEqual(manifests[0]["dataset"][field], manifests[2]["dataset"][field])

    def test_index_is_hashed_without_local_path(self):
        module = self._module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = root / "private" / "evibridge_index.json"
            index.parent.mkdir()
            index.write_bytes(b"index-content")
            run = module.RunProvenance(root / "out", [], repo_root=root, index_path=index.parent)
            manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
            self.assertNotIn(str(root), json.dumps(manifest))
            self.assertEqual(manifest["indexes"]["files"]["evibridge_index.json"]["sha256"], hashlib.sha256(b"index-content").hexdigest())

    def test_configured_custom_store_content_changes_index_fingerprint(self):
        module = self._module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_root = root / "indexes"
            index_root.mkdir()
            (index_root / "tree.pkl").write_bytes(b"tree")
            custom = index_root / "custom_evibridge_vdb" / "chroma.sqlite3"
            custom.parent.mkdir()
            custom.write_bytes(b"vector-v1")
            config = {"rag": {"strategy_config": {"strategy": "evibridge", "enable_vector_recall": True, "evibridge_vdb_config": {"vdb_dir_name": "custom_evibridge_vdb"}}}}
            first = module.RunProvenance(root / "out", [], index_path=index_root, runtime_config=config, repo_root=root)
            custom.write_bytes(b"vector-v2")
            second = module.RunProvenance(root / "out", [], index_path=index_root, runtime_config=config, repo_root=root)
            a = json.loads(first.manifest_path.read_text(encoding="utf-8"))["indexes"]
            b = json.loads(second.manifest_path.read_text(encoding="utf-8"))["indexes"]
        self.assertNotEqual(a["sha256"], b["sha256"])
        self.assertTrue(any(name.endswith("/chroma.sqlite3") for name in a["files"]))
        self.assertNotEqual(a["status"], "known")

    def test_configured_external_store_uses_safe_stable_label(self):
        module = self._module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            indexes = root / "indexes"
            indexes.mkdir()
            (indexes / "tree.pkl").write_bytes(b"tree")
            private = root / "private-external-store"
            private.mkdir()
            (private / "chroma.sqlite3").write_bytes(b"external")
            config = {"rag": {"strategy_config": {"strategy": "vanilla", "vdb_config": {"vdb_dir_name": str(private)}}}}
            a = module.RunProvenance(root / "out", [], index_path=indexes, runtime_config=config, repo_root=root)
            b = module.RunProvenance(root / "out", [], index_path=indexes, runtime_config=config, repo_root=root)
            manifests = [json.loads(run.manifest_path.read_text(encoding="utf-8")) for run in (a, b)]
            for manifest in manifests:
                text = json.dumps(manifest)
                self.assertNotIn(str(private), text)
                self.assertNotIn("private-external-store", text)
                stores = manifest["indexes"].get("configured_stores", {})
                self.assertTrue(stores, "Configured external storage must be fingerprinted")
                self.assertTrue(any(name.endswith("/chroma.sqlite3") for name in manifest["indexes"]["files"]))
            self.assertEqual(set(manifests[0]["indexes"]["configured_stores"]), set(manifests[1]["indexes"]["configured_stores"]))

    def test_missing_configured_store_does_not_claim_complete_index(self):
        module = self._module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tree.pkl").write_bytes(b"tree")
            config = {"rag": {"strategy_config": {"strategy": "vanilla", "vdb_config": {"vdb_dir_name": "missing_dense_paragraph_vdb"}}}}
            run = module.RunProvenance(root / "out", [], index_path=root, runtime_config=config, repo_root=root)
            indexes = json.loads(run.manifest_path.read_text(encoding="utf-8"))["indexes"]
        self.assertIn(indexes["status"], ("partial", "unknown"))
        self.assertTrue(any(store["status"] == "unknown" for store in indexes["configured_stores"].values()))

    def _assert_index_follows_actual_loader(self, raw_index_path, raw_store_path):
        from types import SimpleNamespace
        from tests.test_evibridge_wiring import _stub_runtime_imports
        _stub_runtime_imports()
        from Core.Index.EvidenceBridgeIndex import EvidenceBridgeIndex
        from Core.configs.rag.evibridge_config import EviBridgeRAGConfig
        from Core.utils.resource_loader import prepare_rag_dependencies

        module = self._module()
        observed = {}
        embedding_module = types.ModuleType("Core.provider.embedding")
        embedding_module.TextEmbeddingProvider = lambda **kwargs: "fixture-embedding"
        vector_module = types.ModuleType("Core.provider.vdb")

        def vector_store(**kwargs):
            observed["loader_path"] = kwargs["db_path"]
            return "fixture-vector-store"

        vector_module.VectorStore = vector_store
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            try:
                os.chdir(root)
                strategy = EviBridgeRAGConfig(enable_vector_recall=True)
                strategy.evibridge_vdb_config.vdb_dir_name = raw_store_path
                cfg = SimpleNamespace(save_path=raw_index_path, rag=SimpleNamespace(strategy_config=strategy))
                with patch.object(EvidenceBridgeIndex, "load_from_dir", return_value=None), patch.object(EvidenceBridgeIndex, "load_bm25", return_value=None), patch.dict(sys.modules, {"Core.provider.embedding": embedding_module, "Core.provider.vdb": vector_module}):
                    prepare_rag_dependencies(cfg)
                actual = Path(observed["loader_path"]) / "chroma.sqlite3"
                normalized_shortcut = Path(raw_store_path) / "chroma.sqlite3"
                self.assertNotEqual(actual.resolve(), normalized_shortcut.resolve())
                for file in (actual, normalized_shortcut):
                    file.parent.mkdir(parents=True, exist_ok=True)
                actual.write_bytes(b"actual-loader-v1")
                normalized_shortcut.write_bytes(b"unused-shortcut-v1")
                runtime_config = {"save_path": raw_index_path, "rag": {"strategy_config": strategy.model_dump()}}

                def indexes():
                    run = module.RunProvenance(root / "out", [], index_path=raw_index_path, runtime_config=runtime_config, repo_root=root)
                    return json.loads(run.manifest_path.read_text(encoding="utf-8"))["indexes"]

                first = indexes()
                actual.write_bytes(b"actual-loader-v2")
                changed_actual = indexes()
                self.assertNotEqual(first["sha256"], changed_actual["sha256"], "Index hash must follow the loader's actual path")
                normalized_shortcut.write_bytes(b"unused-shortcut-v2")
                changed_unused = indexes()
                self.assertEqual(changed_actual["sha256"], changed_unused["sha256"], "Unused normalized shortcut must not affect the configured dependency hash")
                store = next(iter(changed_actual["configured_stores"].values()))
                self.assertEqual(store["status"], "known")
                hash_values = {item["sha256"] for name, item in changed_actual["files"].items() if name.endswith("/chroma.sqlite3")}
                self.assertEqual(hash_values, {hashlib.sha256(b"actual-loader-v2").hexdigest()})
            finally:
                os.chdir(previous_cwd)

    def test_dot_relative_index_path_preserves_loader_string_rules(self):
        self._assert_index_follows_actual_loader("./runs/fixture/document", "runs/fixture/document/custom_vdb")

    @unittest.skipUnless(os.name == "nt", "Mixed slash loader case is Windows-specific")
    def test_windows_mixed_slashes_preserve_loader_string_rules(self):
        self._assert_index_follows_actual_loader(".\\runs/fixture\\document", "runs\\fixture/document\\custom_vdb")

    def test_fresh_result_links_to_manifest_and_preserves_generation_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._run_rag(Path(tmp), [{"question": "q", "qasper_question_id": "q1"}])
            manifest_path, manifest = self._manifest(out)
            result_path = out / "query_001/result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            provenance = result["run_provenance"]
            self.assertEqual(provenance["run_id"], manifest["run_id"])
            self.assertEqual((result_path.parent / provenance["manifest_path"]).resolve(), manifest_path.resolve())
            self.assertEqual(result["generation_provenance"], FakeRAG.last_generation_provenance)
            self.assertEqual(result["support_context_validation"], FakeRAG.last_support_context_validation)
            event = self._events(manifest_path)[0]
            self.assertEqual(event["event"], "generated")
            self.assertEqual(event["origin_run_id"], manifest["run_id"])
            self.assertEqual(event["files"]["result.json"]["sha256"], hashlib.sha256(result_path.read_bytes()).hexdigest())

    def test_cached_result_bytes_and_unknown_origin_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "out/query_001/result.json"
            result_path.parent.mkdir(parents=True)
            original = b'{ "output": "cached", "retrieved_node_ids": [9] }\n'
            result_path.write_bytes(original)
            rag = FakeRAG()
            out = self._run_rag(root, [{"question": "q", "hotpotqa_question_id": "h1"}], rag=rag)
            manifest_path, _ = self._manifest(out)
            self.assertEqual(result_path.read_bytes(), original)
            self.assertEqual(rag.calls, 0)
            event = self._events(manifest_path)[0]
        self.assertEqual(event["event"], "reused")
        self.assertIsNone(event["origin_run_id"])
        self.assertEqual(event["origin_status"], "unknown")
        self.assertEqual(event.get("cache_input_status"), "unknown")
        self.assertIsNone(event["cached_qid"])

    def test_reused_result_keeps_original_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = self._run_rag(root, [{"question": "q"}])
            old_result = (out / "query_001/result.json").read_bytes()
            self.assertIn("run_provenance", json.loads(old_result))
            old_run = json.loads(old_result)["run_provenance"]["run_id"]
            self._run_rag(root, [{"question": "q"}])
            self.assertEqual((out / "query_001/result.json").read_bytes(), old_result)
            manifests = list((out / ".runs").glob("*/manifest.json"))
            self.assertEqual(len(manifests), 2)
            events = [self._events(manifest)[0] for manifest in manifests]
            reused = next(event for event in events if event["event"] == "reused")
            self.assertEqual(reused["origin_run_id"], old_run)

    def test_failure_records_exception_type_without_sensitive_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(RuntimeError):
                self._run_rag(root, [{"question": "q"}], rag=FakeRAG(fail=True))
            manifest_path, _ = self._manifest(root / "out")
            event = self._events(manifest_path)[0]
            text = manifest_path.with_name("events.jsonl").read_text(encoding="utf-8")
            for value in ("sk-secret", "10.1.2.3", "private", "model"):
                self.assertNotIn(value, text)
        self.assertEqual(event["event"], "failed")
        self.assertEqual(event["error_type"], "RuntimeError")

    def test_missing_question_has_skipped_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._run_rag(Path(tmp), [{"qasper_question_id": "q1"}])
            manifest_path, manifest = self._manifest(out)
            self.assertEqual(manifest["config"]["status"], "unknown")
            self.assertIsNone(manifest["config"]["sha256"])
            event = self._events(manifest_path)[0]
        self.assertEqual(event["event"], "skipped")
        self.assertEqual(event["reason"], "missing_question")

    def test_inference_records_actual_config_models_and_output_suffix(self):
        from types import SimpleNamespace
        from tests.test_evibridge_wiring import _stub_runtime_imports
        _stub_runtime_imports()
        import pandas as pd
        from Core.inference import inference

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            strategy = SimpleNamespace(strategy="evibridge", method_suffix="evibridge_custom_support_pruned")
            actual = {"rag": {"strategy_config": {"strategy": "evibridge", "top_k": 19}}, "llm": {"model_name": "actual-model", "api_key": "private-key", "api_base": "http://10.2.3.4/v1"}}
            cfg = SimpleNamespace(save_path=str(root), rag=SimpleNamespace(strategy_config=strategy), llm=actual["llm"], vlm={}, rag_force_reprocess=False, model_dump=lambda: actual)
            with patch("Core.inference.prepare_rag_dependencies", return_value={}), patch("Core.inference.create_rag_agent", return_value=FakeRAG()):
                inference(cfg, pd.DataFrame([{"question": "q"}]), "actual_dataset")
            _, manifest = self._manifest(root / "eval_actual_dataset_evibridge_custom_support_pruned")
        self.assertEqual(manifest["method_suffix"], strategy.method_suffix)
        self.assertEqual(manifest["config"]["safe_snapshot"]["rag"]["strategy_config"]["top_k"], 19)
        self.assertEqual(manifest["metadata"]["models"]["llm"]["model_name"], "actual-model")
        self.assertIsNone(manifest["metadata"]["models"]["llm"]["revision"])
        self.assertEqual(manifest["metadata"]["models"]["llm"]["revision_status"], "unknown")

    def test_failure_event_keeps_only_current_generation_trace(self):
        class TraceRAG(FakeRAG):
            last_generation_provenance = {"stage": "old_question"}

            def generation(self, query, query_output_dir):
                self.last_generation_provenance = {"calls": [{"stage": "initial", "success": False, "error_type": "RuntimeError", "prompt_sha256": "a" * 64}]}
                raise RuntimeError("private message")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(RuntimeError):
                self._run_rag(root, [{"question": "q"}], rag=TraceRAG())
            manifest_path, _ = self._manifest(root / "out")
            event = self._events(manifest_path)[0]
        self.assertEqual(event["generation_provenance"]["calls"][0]["stage"], "initial")
        self.assertNotIn("old_question", json.dumps(event))

    def test_token_budget_remains_visible_and_affects_config_fingerprint(self):
        module = self._module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = module.RunProvenance(root, [], runtime_config={"token_budget": 100, "accessToken": "private-credential"}, repo_root=root)
            b = module.RunProvenance(root, [], runtime_config={"token_budget": 200, "accessToken": "other-credential"}, repo_root=root)
            first = json.loads(a.manifest_path.read_text(encoding="utf-8"))["config"]
            second = json.loads(b.manifest_path.read_text(encoding="utf-8"))["config"]
        self.assertEqual(first["safe_snapshot"]["token_budget"], 100)
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertNotIn("private-credential", json.dumps(first))

    def test_relative_path_objects_and_exception_messages_are_private(self):
        module = self._module()
        safe = module.safe_runtime_config({"model_name": Path("private/model"), "nested": {"exception_message": "credential without a recognizable pattern"}})
        text = json.dumps(safe)
        self.assertNotIn("private", text)
        self.assertNotIn("credential without", text)

    def test_concurrent_invocations_do_not_share_a_manifest(self):
        from concurrent.futures import ThreadPoolExecutor
        module = self._module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with ThreadPoolExecutor(max_workers=2) as pool:
                runs = list(pool.map(lambda _: module.RunProvenance(root, [], repo_root=root), range(2)))
            self.assertEqual(len({run.run_id for run in runs}), 2)
            self.assertTrue(all(run.manifest_path.is_file() for run in runs))

    def test_finalization_failure_is_recorded_and_reraised(self):
        from tests.test_evibridge_wiring import _stub_runtime_imports
        _stub_runtime_imports()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("Core.inference.TokenTracker.record_stage", side_effect=RuntimeError("private-service-url")):
                with self.assertRaises(RuntimeError):
                    self._run_rag(root, [{"question": "q"}])
            manifest_path, _ = self._manifest(root / "out")
            events = self._events(manifest_path)
            self.assertEqual(events[-1]["event"], "failed")
            self.assertEqual(events[-1]["reason"], "finalize")
            summary = json.loads(manifest_path.with_name("summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "failed")

    def test_non_mapping_cache_is_reprocessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "out/query_001/result.json"
            result_path.parent.mkdir(parents=True)
            result_path.write_text("[]", encoding="utf-8")
            rag = FakeRAG()
            try:
                out = self._run_rag(root, [{"question": "q"}], rag=rag)
            except AttributeError:
                self.fail("Non-mapping cache must be treated as corrupt and reprocessed")
            manifest_path, _ = self._manifest(out)
            self.assertEqual(rag.calls, 1)
            self.assertEqual(self._events(manifest_path)[0]["event"], "generated")
            self.assertEqual(json.loads(result_path.read_text(encoding="utf-8"))["output"], "answer")

    def test_cache_read_oserror_is_recorded_without_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "out/query_001/result.json"
            result_path.parent.mkdir(parents=True)
            result_path.write_text('{"output": "cached"}', encoding="utf-8")
            # run_rag currently uses builtin open for the cache.
            import builtins
            original_builtin_open = builtins.open

            def builtin_read_error(file, mode="r", *args, **kwargs):
                if Path(file) == result_path and mode == "r":
                    raise PermissionError("sk-private http://10.2.3.4 C:\\private")
                return original_builtin_open(file, mode, *args, **kwargs)

            with patch("builtins.open", side_effect=builtin_read_error):
                with self.assertRaises(PermissionError):
                    self._run_rag(root, [{"question": "q"}])
            manifest_path, _ = self._manifest(root / "out")
            events = self._events(manifest_path)
            self.assertTrue(events, "Cache read errors must have a failed event")
            event = events[0]
            self.assertEqual(event["event"], "failed")
            self.assertEqual(event["reason"], "cache_read")
            self.assertEqual(event["error_type"], "PermissionError")
            self.assertNotIn("private", json.dumps(event))
            summary = json.loads(manifest_path.with_name("summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "failed")

    def test_cache_input_mismatch_stops_without_writing_or_model_call(self):
        first_input = {"question": "old question", "qasper_question_id": "old-qid", "doc_uuid": "doc-a", "doc_path": "document-a", "answer": "gold-a"}
        changes = [{"question": "new question", "qasper_question_id": "new-qid"}, {"doc_uuid": "doc-b"}, {"doc_path": "document-b"}, {"answer": "gold-b"}]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                out = self._run_rag(root, [first_input])
                old_manifest, old_identity = self._manifest(out)
                preserved = {path: path.read_bytes() for path in (out / "query_001/result.json", out / "query_001/retrieval_res.json", out / "final_results.json", old_manifest)}
                rag = FakeRAG()
                with self.assertRaises(RuntimeError):
                    self._run_rag(root, [{**first_input, **change}], rag=rag)
                self.assertEqual(rag.calls, 0)
                for path, original in preserved.items():
                    self.assertEqual(path.read_bytes(), original)
                new_manifest = next(path for path in (out / ".runs").glob("*/manifest.json") if path != old_manifest)
                events = self._events(new_manifest)
                self.assertEqual(len(events), 1)
                event = events[0]
                self.assertEqual(event["event"], "failed")
                self.assertEqual(event["reason"], "cache_input_mismatch")
                self.assertEqual(event["cache_input_status"], "mismatch")
                self.assertEqual(event["cached_qid"], "old-qid")
                self.assertEqual(event["origin_run_id"], old_identity["run_id"])
                self.assertNotIn("old question", json.dumps(event))
                self.assertNotIn("gold-a", json.dumps(event))
                summary = json.loads(new_manifest.with_name("summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["status"], "failed")

    def test_legacy_cache_with_missing_identity_fields_is_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "out/query_001/result.json"
            result_path.parent.mkdir(parents=True)
            original = b'{"question":"q","output":"cached","run_provenance":{"run_id":"known-old-run"}}'
            result_path.write_bytes(original)
            rag = FakeRAG()
            out = self._run_rag(root, [{"question": "q", "qasper_question_id": "q1", "doc_uuid": "doc-a"}], rag=rag)
            manifest_path, _ = self._manifest(out)
            self.assertEqual(result_path.read_bytes(), original)
            self.assertEqual(rag.calls, 0)
            event = self._events(manifest_path)[0]
        self.assertEqual(event.get("cache_input_status"), "partial")
        self.assertIsNone(event["cached_qid"])
        self.assertEqual(event["origin_run_id"], "known-old-run")
        self.assertIn("qasper_question_id", event["cache_input_validation"]["missing_cached_fields"])

    def test_matching_cache_normalizes_nan_and_sequences(self):
        from tests.test_evibridge_wiring import _stub_runtime_imports
        _stub_runtime_imports()
        import pandas as pd
        from Core.inference import run_rag

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "out/query_001/result.json"
            result_path.parent.mkdir(parents=True)
            cached = {"question": "q", "qasper_question_id": "q1", "answer": None, "extra": [1, 2], "output": "cached"}
            original = json.dumps(cached).encode("utf-8")
            result_path.write_bytes(original)
            rag = FakeRAG()
            run_rag(rag, root / "out", data_df=pd.DataFrame([{"question": "q", "qasper_question_id": "q1", "answer": float("nan"), "extra": (1, 2)}]))
            manifest_path, _ = self._manifest(root / "out")
            self.assertEqual(result_path.read_bytes(), original)
            self.assertEqual(rag.calls, 0)
            event = self._events(manifest_path)[0]
        self.assertEqual(event.get("cache_input_status"), "matched")
        self.assertEqual(event["cached_qid"], "q1")

    def test_query_directory_creation_failure_has_terminal_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            conflict = root / "out/query_001"
            conflict.parent.mkdir(parents=True)
            conflict.write_bytes(b"preserved-user-file")
            rag = FakeRAG()
            with self.assertRaises(FileExistsError):
                self._run_rag(root, [{"question": "q"}], rag=rag)
            manifest_path, _ = self._manifest(root / "out")
            events = self._events(manifest_path)
            self.assertTrue(events, "Query mkdir failure needs a failed event")
            self.assertEqual(events[0]["event"], "failed")
            self.assertEqual(events[0]["error_type"], "FileExistsError")
            self.assertNotIn("generation_provenance", events[0])
            self.assertEqual(conflict.read_bytes(), b"preserved-user-file")
            self.assertEqual(rag.calls, 0)
            summary = json.loads(manifest_path.with_name("summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "failed")


if __name__ == "__main__":
    unittest.main()
