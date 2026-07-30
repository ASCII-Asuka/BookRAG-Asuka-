import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


class IRCoTBaselineTests(unittest.TestCase):
    def test_config_accepts_ircot_retrieval_method(self):
        from Core.configs.rag.vanilla_config import VanillaConfig

        cfg = VanillaConfig(retrieval_method="ircot", ircot_max_steps=2)

        self.assertEqual(cfg.retrieval_method, "ircot")
        self.assertEqual(cfg.ircot_max_steps, 2)

    def test_ircot_generation_interleaves_thoughts_and_retrieval(self):
        from Core.rag.vanilla_rag import VanillaRAG

        class FakeBM25:
            def __init__(self):
                self.queries = []

            def search(self, query_text, top_k):
                self.queries.append(query_text)
                if len(self.queries) == 1:
                    return [
                        {
                            "id": 1,
                            "content": "Scott Derrickson is an American director.",
                            "score": 3.0,
                            "metadata": {
                                "node_id": 1,
                                "source_node_id": 1,
                                "qasper_evidence_text": "Scott Derrickson is an American director.",
                                "source": "qasper_paragraph",
                                "node_type": "text",
                                "hotpot_title": "Scott Derrickson",
                                "sent_id": 0,
                            },
                        }
                    ][:top_k]
                return [
                    {
                        "id": 2,
                        "content": "Ed Wood was an American filmmaker.",
                        "score": 2.0,
                        "metadata": {
                            "node_id": 2,
                            "source_node_id": 2,
                            "qasper_evidence_text": "Ed Wood was an American filmmaker.",
                            "source": "qasper_paragraph",
                            "node_type": "text",
                            "hotpot_title": "Ed Wood",
                            "sent_id": 0,
                        },
                    }
                ][:top_k]

        class FakeLLM:
            config = SimpleNamespace(max_tokens=2000)

            def __init__(self):
                self.prompts = []

            def get_completion(self, prompt, json_response=False):
                self.prompts.append(prompt)
                if "Generate the next reasoning step" in prompt:
                    step_no = len([p for p in self.prompts if "Generate the next reasoning step" in p])
                    return json.dumps(
                        {
                            "thought": "Find the second person's nationality." if step_no == 1 else "Both are American.",
                            "stop": step_no >= 2,
                        }
                    )
                return json.dumps(
                    {
                        "answer_short": "Yes",
                        "answer_rationale": "Both people are American.",
                        "supporting_block_ids": [1, 2],
                    }
                )

        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(
                retrieval_method="ircot",
                topk=2,
                answer_style="short",
                ircot_max_steps=2,
                ircot_step_topk=1,
                ircot_final_topk=2,
            )
            bm25 = FakeBM25()
            llm = FakeLLM()
            rag = VanillaRAG(config=cfg, llm=llm, bm25=bm25)

            answer, retrieval_ids = rag.generation(
                "Were Scott Derrickson and Ed Wood of the same nationality?",
                Path(tmp),
            )
            payload = json.loads((Path(tmp) / "retrieval_res.json").read_text(encoding="utf-8"))

        self.assertEqual(len(bm25.queries), 2)
        self.assertIn("Find the second person's nationality.", bm25.queries[1])
        self.assertEqual(retrieval_ids, [1, 2])
        self.assertEqual(rag.last_answer_short, "Yes")
        self.assertEqual(payload["strategy"], "ircot")
        self.assertEqual(len(payload["ircot_steps"]), 2)
        self.assertEqual(payload["supporting_block_ids"], [1, 2])
        self.assertEqual(payload["supporting_evidence"][1]["hotpot_title"], "Ed Wood")
        self.assertIn('"answer_short": "Yes"', answer)


if __name__ == "__main__":
    unittest.main()
