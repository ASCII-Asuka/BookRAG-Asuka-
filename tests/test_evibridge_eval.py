import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd


def _stub_eval_imports():
    openai_module = types.ModuleType("openai")
    openai_module.OpenAI = object
    sys.modules.setdefault("openai", openai_module)


class FakeExtractor:
    def extract(self, question, output, correct_answer):
        return "The answer", "The answer", "Text", 1.0


class FakeOpenAIResponse:
    choices = [
        SimpleNamespace(
            message=SimpleNamespace(
                content="extracted results: The answer\nformat: extractive\nscore: 1.0"
            )
        )
    ]


class FakeCompletions:
    def __init__(self):
        self.parameters = None

    def create(self, **parameters):
        self.parameters = parameters
        return FakeOpenAIResponse()


class FakeClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


class EviBridgeEvalTests(unittest.TestCase):
    def test_qasper_eval_reads_evibridge_evidence_metrics(self):
        _stub_eval_imports()
        from Eval.utils.qasper_eval import eval_single_file

        with tempfile.TemporaryDirectory() as tmp:
            res_dir = Path(tmp)
            (res_dir / "query_001").mkdir()
            final_results = [
                {
                    "question": "What is the answer?",
                    "answer": [
                        {
                            "unanswerable": False,
                            "extractive_spans": ["The answer"],
                            "free_form_answer": "",
                            "yes_no": None,
                            "evidence": ["Gold evidence paragraph"],
                        }
                    ],
                    "output": "The answer",
                }
            ]
            (res_dir / "final_results.json").write_text(
                json.dumps(final_results),
                encoding="utf-8",
            )
            (res_dir / "query_001" / "evidence_chain.json").write_text(
                json.dumps(
                    {
                        "evidence_chain": [
                            {
                                "text": "Gold evidence paragraph",
                                "bridge_types": ["context", "semantic"],
                            }
                        ],
                        "verification": {
                            "connectivity": 0.75,
                            "noise": 0.1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = eval_single_file(str(res_dir), FakeExtractor())

        self.assertEqual(result[0]["evidence_f1"], 1.0)
        self.assertEqual(result[0]["evidence_precision"], 1.0)
        self.assertEqual(result[0]["evidence_recall"], 1.0)
        self.assertEqual(result[0]["path_connectivity"], 0.75)
        self.assertEqual(result[0]["noise_ratio"], 0.1)
        self.assertEqual(result[0]["bridge_coverage"], 0.666667)

    def test_qasper_eval_prefers_gold_evidence_ids_when_available(self):
        _stub_eval_imports()
        from Eval.utils.qasper_eval import eval_single_file

        with tempfile.TemporaryDirectory() as tmp:
            res_dir = Path(tmp)
            (res_dir / "query_001").mkdir()
            final_results = [
                {
                    "question": "What is the answer?",
                    "answer": [
                        {
                            "unanswerable": False,
                            "extractive_spans": ["The answer"],
                            "free_form_answer": "",
                            "yes_no": None,
                            "evidence": ["Gold evidence paragraph"],
                            "evidence_block_ids": [42],
                        }
                    ],
                    "output": "The answer",
                }
            ]
            (res_dir / "final_results.json").write_text(json.dumps(final_results), encoding="utf-8")
            (res_dir / "query_001" / "evidence_chain.json").write_text(
                json.dumps(
                    {
                        "evidence_chain": [
                            {
                                "block_id": 42,
                                "text": "A paraphrased evidence block that will not substring-match.",
                                "bridge_types": ["hierarchy"],
                            }
                        ],
                        "verification": {"connectivity": 1.0, "noise": 0.0},
                    }
                ),
                encoding="utf-8",
            )

            result = eval_single_file(str(res_dir), FakeExtractor())

        self.assertEqual(result[0]["evidence_f1"], 1.0)
        self.assertEqual(result[0]["evidence_precision"], 1.0)
        self.assertEqual(result[0]["evidence_recall"], 1.0)

    def test_qasper_doc_result_dir_casts_numeric_doc_uuid_to_string(self):
        _stub_eval_imports()
        from Eval.utils.qasper_eval import _doc_result_dir

        cfg = SimpleNamespace(working_dir="runs/qasper_evibridge/work", dataset_name="qasper")

        result = _doc_result_dir(cfg, np.float64(1609.00425), "evibridge")

        self.assertEqual(
            result,
            os.path.join("runs/qasper_evibridge/work", "1609.00425", "eval_qasper_evibridge"),
        )

    def test_get_all_cost_casts_numeric_doc_uuid_to_string(self):
        from Eval.utils.utils import get_all_cost

        with tempfile.TemporaryDirectory() as tmp:
            res_dir = Path(tmp) / "1609.00425" / "eval_qasper_evibridge"
            res_dir.mkdir(parents=True)
            (res_dir / "token_cost.json").write_text(
                json.dumps(
                    {
                        "rag_cost": {
                            "prompt_tokens": 2,
                            "completion_tokens": 3,
                            "total_tokens": 5,
                        },
                        "time": 1.25,
                    }
                ),
                encoding="utf-8",
            )
            data_df = pd.DataFrame(
                [{"doc_uuid": np.float64(1609.00425), "doc_path": "qasper://1609.00425"}]
            )
            cfg = SimpleNamespace(working_dir=tmp, dataset_name="qasper")

            score = get_all_cost(data_df, cfg, "evibridge")

        self.assertEqual(score["total_prompt_tokens"], 2)
        self.assertEqual(score["total_completion_tokens"], 3)
        self.assertEqual(score["total_tokens"], 5)
        self.assertEqual(score["total_time"], 1.25)

    def test_answer_extractor_does_not_send_enable_thinking_to_qwen_vl(self):
        from Eval.utils import extract_answer

        with tempfile.TemporaryDirectory() as tmp:
            api_config = Path(tmp) / "api.txt"
            api_config.write_text(
                "https://api.example.test/v1\nsk-test\nQwen/Qwen3-VL-8B-Instruct\n",
                encoding="utf-8",
            )
            with patch.object(extract_answer, "OpenAI", lambda **kwargs: SimpleNamespace()):
                extractor = extract_answer.AnswerExtractor(api_config_path=api_config)
                completions = FakeCompletions()
                extractor.client = FakeClient(completions)

                extractor.extract("question", "output", "The answer")

        self.assertNotIn("extra_body", completions.parameters)

    def test_qasper_eval_passes_custom_api_config_to_answer_extractor(self):
        _stub_eval_imports()
        from Eval.utils import qasper_eval

        captured = {}

        class CapturingExtractor:
            def __init__(self, api_config_path=None):
                captured["api_config_path"] = api_config_path

            def extract(self, question, output, correct_answer):
                return "The answer", "The answer", "Text", 1.0

        with tempfile.TemporaryDirectory() as tmp:
            res_dir = Path(tmp) / "paper-1" / "eval_qasper_evibridge"
            res_dir.mkdir(parents=True)
            (res_dir / "final_results.json").write_text(
                json.dumps(
                    [
                        {
                            "question": "What is the answer?",
                            "answer": [
                                {
                                    "unanswerable": False,
                                    "extractive_spans": ["The answer"],
                                    "free_form_answer": "",
                                    "yes_no": None,
                                    "evidence": [],
                                }
                            ],
                            "output": "The answer",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            data_df = pd.DataFrame(
                [{"doc_uuid": "paper-1", "doc_path": "qasper://paper-1"}]
            )
            cfg = SimpleNamespace(working_dir=tmp, dataset_name="qasper")

            with patch.object(qasper_eval, "AnswerExtractor", CapturingExtractor):
                qasper_eval.eval_qasper(
                    data_df,
                    cfg,
                    "evibridge",
                    max_workers=1,
                    api_config_path="custom-api.txt",
                )

        self.assertEqual(captured["api_config_path"], "custom-api.txt")


if __name__ == "__main__":
    unittest.main()
