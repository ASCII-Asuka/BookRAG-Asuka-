import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from Core.Index.Tree import DocumentTree, NodeType, TreeNode
from Core.pipelines.vdb_index import get_input_text


class JsonOutputSanitizationTests(unittest.TestCase):
    def test_run_rag_writes_standard_json_when_dataframe_contains_nan(self):
        import pandas as pd

        from Core.inference import run_rag

        class DummyRag:
            def generation(self, query, query_output_dir):
                return "answer", [1]

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "eval_dataset_hri"
            output_dir.mkdir()
            data_df = pd.DataFrame(
                [
                    {
                        "question": "Q1",
                        "gold_relations": math.nan,
                    }
                ]
            )

            run_rag(
                rag_agent=DummyRag(),
                output_dir=output_dir,
                force_reprocess=True,
                data_df=data_df,
            )

            final_text = (output_dir / "final_results.json").read_text(
                encoding="utf-8"
            )
            query_text = (output_dir / "query_001" / "result.json").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("NaN", final_text)
            self.assertNotIn("NaN", query_text)
            final_data = json.loads(final_text)
            self.assertIsNone(final_data[0]["gold_relations"])


class VanillaMarkdownInputTests(unittest.TestCase):
    def _build_tree(self, save_dir: Path) -> DocumentTree:
        cfg = SimpleNamespace(save_path=str(save_dir))
        tree = DocumentTree(
            meta_dict={"file_name": "water.pdf", "file_path": "water.pdf"},
            cfg=cfg,
        )

        def add_node(parent, node_type, content, page_idx=0):
            node = TreeNode(
                {
                    "content": content,
                    "page_idx": page_idx,
                    "pdf_id": len(tree.nodes),
                }
            )
            node.type = node_type
            node.outline_node = node_type == NodeType.TITLE
            tree.add_node(node)
            parent.add_child(node)
            return node

        chapter = add_node(tree.root_node, NodeType.TITLE, "5.2预警", page_idx=17)
        add_node(chapter, NodeType.TEXT, "发布预警后应开展工程巡查。", page_idx=21)
        tree.save_to_file()
        return tree

    def test_vanilla_chunks_prefer_tree_metadata_when_tree_exists(self):
        from Core.pipelines.vdb_index import get_all_chunks

        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp)
            pdf_path = save_dir / "water.pdf"
            pdf_path.write_bytes(b"%PDF-1.4")
            self._build_tree(save_dir)
            auto_dir = save_dir / "auto"
            auto_dir.mkdir()
            (auto_dir / "water.md").write_text("markdown fallback", encoding="utf-8")

            cfg = SimpleNamespace(
                index_type="vanilla",
                pdf_path=str(pdf_path),
                save_path=str(save_dir),
                mineru=SimpleNamespace(method="auto"),
            )

            chunks, metadatas = get_all_chunks(cfg)

            self.assertIn("发布预警后应开展工程巡查。", chunks)
            metadata = metadatas[chunks.index("发布预警后应开展工程巡查。")]
            self.assertEqual(metadata["source"], "tree")
            self.assertEqual(metadata["page"], 22)
            self.assertEqual(metadata["section_id"], "5.2预警")
            self.assertIn("5.2预警", metadata["title_path"])
            self.assertEqual(metadata["node_type"], "text")

    def test_get_input_text_reads_mineru_auto_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp)
            pdf_path = save_dir / "water.pdf"
            pdf_path.write_bytes(b"%PDF-1.4")
            auto_dir = save_dir / "auto"
            auto_dir.mkdir()
            expected = "water conservancy four predictions"
            (auto_dir / "water.md").write_text(expected, encoding="utf-8")

            cfg = SimpleNamespace(
                pdf_path=str(pdf_path),
                save_path=str(save_dir),
                mineru=SimpleNamespace(method="auto"),
            )

            self.assertEqual(get_input_text(cfg), expected)

    def test_get_input_text_error_lists_candidate_markdown_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp)
            pdf_path = save_dir / "missing.pdf"
            pdf_path.write_bytes(b"%PDF-1.4")
            cfg = SimpleNamespace(
                pdf_path=str(pdf_path),
                save_path=str(save_dir),
                mineru=SimpleNamespace(method="auto"),
            )

            with self.assertRaises(FileNotFoundError) as ctx:
                get_input_text(cfg)

            message = str(ctx.exception)
            self.assertIn("auto", message)
            self.assertIn("vlm", message)
            self.assertIn("ocr", message)
            self.assertIn("txt", message)

    def test_vanilla_retrieval_files_include_metadata(self):
        from Core.rag.vanilla_rag import VanillaRAG

        with tempfile.TemporaryDirectory() as tmp:
            query_dir = Path(tmp)
            rag = VanillaRAG.__new__(VanillaRAG)
            retrieved_ids = rag._save_retrieval_res(
                [
                    {
                        "id": "vdb-1",
                        "content": "发布预警后应开展工程巡查。",
                        "metadata": {
                            "node_id": 42,
                            "page": 22,
                            "section_id": "5.2预警",
                            "title_path": "水利部文件 > 5.2预警",
                            "node_type": "text",
                        },
                    }
                ],
                query_dir,
            )

            self.assertEqual(retrieved_ids, [42])
            payload = json.loads((query_dir / "42.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["page"], 22)
            self.assertEqual(payload["section_id"], "5.2预警")
            self.assertEqual(payload["title_path"], "水利部文件 > 5.2预警")
            self.assertEqual(payload["node_type"], "text")

    def test_bm25_retrieval_returns_document_metadata(self):
        from Core.utils.bm25 import BM25

        bm25 = BM25(
            ["forecast warning", "hydro requirement"],
            metadatas=[
                {"page": 3, "section_id": "3.2技术框架"},
                {"page": 9, "section_id": "5.2预警"},
            ],
        )
        bm25.initialize()

        result = bm25.search("hydro", top_k=1)[0]

        self.assertEqual(result["content"], "hydro requirement")
        self.assertEqual(result["metadata"]["page"], 9)
        self.assertEqual(result["metadata"]["section_id"], "5.2预警")

    def test_bm25_retrieves_chinese_without_word_spaces(self):
        from Core.utils.bm25 import BM25

        bm25 = BM25(
            ["预报工作应满足时效要求", "预警发布应包含阈值和防御建议"],
            metadatas=[
                {"page": 4, "section_id": "4.1预报"},
                {"page": 9, "section_id": "5.2预警"},
            ],
        )
        bm25.initialize()

        result = bm25.search("预警发布要求", top_k=1)[0]

        self.assertEqual(result["content"], "预警发布应包含阈值和防御建议")
        self.assertEqual(result["metadata"]["section_id"], "5.2预警")

    def test_raptor_metadata_keeps_original_and_summary_traceability(self):
        from Core.utils.raptor_utils import build_summary_metadata, raptor_tree

        base_metadatas = [
            {
                "source": "tree",
                "chunk_id": 0,
                "source_node_id": 11,
                "page": 3,
                "section_id": "3.2技术框架",
                "title_path": "水利部文件 > 3.2技术框架",
                "node_type": "text",
            },
            {
                "source": "tree",
                "chunk_id": 1,
                "source_node_id": 22,
                "page": 9,
                "section_id": "5.2预警",
                "title_path": "水利部文件 > 5.2预警",
                "node_type": "text",
            },
        ]

        chunks, metadatas = raptor_tree(
            ["预报要求", "预警要求"],
            embedder=None,
            llm=None,
            max_depth=20,
            base_metadatas=base_metadatas,
        )
        summary_metadata = build_summary_metadata(
            child_metadatas=base_metadatas,
            depth=1,
            chunk_id=2,
        )

        self.assertEqual(chunks, ["预报要求", "预警要求"])
        self.assertEqual(metadatas[0]["page"], 3)
        self.assertEqual(metadatas[1]["section_id"], "5.2预警")
        self.assertEqual(summary_metadata["source"], "raptor_summary")
        self.assertEqual(summary_metadata["raptor_depth"], 1)
        self.assertEqual(summary_metadata["child_source_node_ids"], "11,22")
        self.assertEqual(summary_metadata["child_pages"], "3,9")
        self.assertIn("5.2预警", summary_metadata["child_sections"])


class HydroEvalMetricTests(unittest.TestCase):
    def test_hydro_deterministic_metrics_cover_answer_and_evidence(self):
        from Eval.utils.hydro_eval import evaluate_hydro_result_item

        item = {
            "question": "What are the four prediction functions?",
            "answer": "The four functions are forecast, warning, rehearsal, and plan.",
            "answer_keypoints": ["forecast", "warning", "rehearsal", "plan"],
            "gold_evidence_keywords": [["forecast"], ["warning"], ["rehearsal"], ["plan"]],
            "output": "The document covers forecast, warning, rehearsal, and plan.",
        }
        retrieval_texts = [
            "forecast and warning requirements",
            "rehearsal and plan requirements",
        ]

        evaluated = evaluate_hydro_result_item(
            item=item,
            retrieval_texts=retrieval_texts,
            llm_score=None,
            extracted_res=None,
        )

        self.assertEqual(evaluated["keypoint_recall"], 1.0)
        self.assertEqual(evaluated["evidence_keyword_recall"], 1.0)
        self.assertEqual(evaluated["evidence_recall@5"], 1.0)
        self.assertEqual(evaluated["mrr"], 1.0)
        self.assertNotIn("char_f1", evaluated)
        self.assertNotIn("rouge_l", evaluated)
        self.assertEqual(evaluated["retrieved_count"], 2)

    def test_hydro_chain_and_explainability_metrics(self):
        from Eval.utils.hydro_eval import evaluate_hydro_result_item

        item = {
            "question": "Which warning requirements apply when risk occurs?",
            "answer": "Use warning conditions, requirements, and tables.",
            "answer_keypoints": ["conditions", "requirements", "tables"],
            "gold_evidence_keywords": [["alpha"], ["beta"], ["gamma"]],
            "gold_chain_roles": ["condition", "requirement", "table"],
            "gold_evidence_pages": [2, 5],
            "gold_evidence_sections": ["5.2 warning", "appendix table"],
            "gold_relations": ["condition_of", "requires", "parameter_of"],
            "gold_path_relations": ["condition_of", "parameter_of"],
            "output": "The answer uses conditions, requirements, and tables.",
        }
        retrieval_items = [
            {
                "text": "condition alpha",
                "role": "condition",
                "page": 2,
                "section": "5.2 warning",
                "relations": [{"relation_type": "condition_of"}],
            },
            {
                "text": "requirement beta",
                "role": "requirement",
                "relations": [{"relation_type": "requires"}],
            },
            {
                "text": "table gamma",
                "role": "table",
                "page": 5,
                "section": "appendix table",
                "relations": [{"relation_type": "parameter_of"}],
            },
        ]

        evaluated = evaluate_hydro_result_item(
            item=item,
            retrieval_texts=[entry["text"] for entry in retrieval_items],
            retrieval_items=retrieval_items,
            llm_score=None,
            extracted_res=None,
        )

        self.assertEqual(evaluated["evidence_recall@5"], 1.0)
        self.assertEqual(evaluated["mrr"], 1.0)
        self.assertEqual(evaluated["evidence_chain_coverage"], 1.0)
        self.assertEqual(evaluated["page_hit_rate"], 1.0)
        self.assertEqual(evaluated["section_hit_rate"], 1.0)
        self.assertEqual(evaluated["relation_hit_rate"], 1.0)
        self.assertEqual(evaluated["path_completeness"], 1.0)
        self.assertEqual(evaluated["relation_support_rate"], 1.0)

    def test_hydro_eval_writes_summary_files_without_llm_judge(self):
        from Eval.utils.hydro_eval import eval_hydro

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc_uuid = "doc-1"
            method = "vanilla"
            dataset_name = "hri_four_predictions"
            res_dir = root / doc_uuid / f"eval_{dataset_name}_{method}"
            query_dir = res_dir / "query_001"
            query_dir.mkdir(parents=True)
            dataset_path = root / "dataset.json"

            dataset = [
                {
                    "question": "What are the four prediction functions?",
                    "answer": "forecast, warning, rehearsal, plan",
                    "answer_format": "KeyPoints",
                    "answer_keypoints": ["forecast", "warning", "rehearsal", "plan"],
                    "gold_evidence_keywords": [["forecast"], ["warning"], ["rehearsal"], ["plan"]],
                    "gold_chain_roles": ["article"],
                    "gold_evidence_pages": [1],
                    "gold_evidence_sections": ["scope"],
                    "doc_uuid": doc_uuid,
                    "doc_path": str(root / "doc.pdf"),
                }
            ]
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
            (root / "doc.pdf").write_bytes(b"%PDF-1.4")
            final_results = [
                {
                    **dataset[0],
                    "gold_relations": math.nan,
                    "output": "forecast, warning, rehearsal, and plan",
                    "retrieved_node_ids": [1],
                }
            ]
            (res_dir / "final_results.json").write_text(
                json.dumps(final_results), encoding="utf-8"
            )
            (query_dir / "1.json").write_text(
                json.dumps(
                    {
                        "content": "forecast warning rehearsal plan",
                        "page": 1,
                        "section": "scope",
                        "role": "article",
                    }
                ),
                encoding="utf-8",
            )
            (res_dir / "token_cost.json").write_text(
                json.dumps(
                    {
                        "rag_cost": {
                            "prompt_tokens": 3,
                            "completion_tokens": 4,
                            "total_tokens": 7,
                        },
                        "time": 1.5,
                    }
                ),
                encoding="utf-8",
            )
            data_cfg = SimpleNamespace(
                dataset_name=dataset_name,
                dataset_path=str(dataset_path),
                working_dir=str(root),
            )

            eval_hydro(
                data_df=__import__("pandas").read_json(dataset_path),
                data_cfg=data_cfg,
                method=method,
                max_workers=1,
                skip_llm_judge=True,
            )

            score_path = (
                root
                / "0_results"
                / f"final_eval_{dataset_name}_{method}.score.json"
            )
            detail_path = root / "0_results" / f"final_eval_{dataset_name}_{method}.json"
            self.assertTrue(score_path.exists())
            self.assertTrue(detail_path.exists())
            self.assertNotIn("NaN", detail_path.read_text(encoding="utf-8"))
            self.assertNotIn("NaN", score_path.read_text(encoding="utf-8"))
            score = json.loads(score_path.read_text(encoding="utf-8"))
            self.assertEqual(score["Total samples"], 1)
            self.assertEqual(score["Avg keypoint_recall"], 1.0)
            self.assertEqual(score["Avg evidence_keyword_recall"], 1.0)
            self.assertEqual(score["Avg Evidence Recall@5"], 1.0)
            self.assertEqual(score["Avg MRR"], 1.0)
            self.assertEqual(score["Avg Evidence Chain Coverage"], 1.0)
            self.assertEqual(score["Avg Page Hit Rate"], 1.0)
            self.assertEqual(score["Avg Section Hit Rate"], 1.0)
            self.assertNotIn("Avg char_f1", score)
            self.assertNotIn("Avg rouge_l", score)
            self.assertEqual(score["total_tokens"], 7)


class WindowsLoggingTests(unittest.TestCase):
    def test_inference_completion_logs_are_ascii_safe(self):
        source = Path("Core/inference.py").read_text(encoding="utf-8")

        self.assertNotIn("✅", source)

    def test_hri_answer_prompt_is_readable_chinese(self):
        source = Path("Core/prompts/hri_prompt.py").read_text(encoding="utf-8")

        self.assertIn("水利规范", source)
        self.assertIn("证据链", source)
        self.assertNotIn("浣犳槸", source)


class EvaluationCliTests(unittest.TestCase):
    def test_evaluation_script_help_runs_from_repo_root(self):
        result = subprocess.run(
            [sys.executable, "Eval/evaluation.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--method", result.stdout)


class HydroDatasetQualityTests(unittest.TestCase):
    def test_hri_four_predictions_annotations_have_no_ascii_question_placeholders(self):
        data_path = Path("Scripts/data/hri_four_predictions_questions.json")
        data = json.loads(data_path.read_text(encoding="utf-8"))

        checked_fields = [
            "gold_chain_role_keywords",
            "gold_evidence_sections",
        ]
        for idx, item in enumerate(data, start=1):
            for field in checked_fields:
                serialized = json.dumps(item.get(field, ""), ensure_ascii=False)
                self.assertNotIn(
                    "?",
                    serialized,
                    f"{data_path}:{idx} field {field} contains placeholder question marks",
                )


if __name__ == "__main__":
    unittest.main()
