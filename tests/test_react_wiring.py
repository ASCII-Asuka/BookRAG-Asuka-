import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from Core.configs.rag.vanilla_config import VanillaConfig
from Core.rag.vanilla_rag import VanillaRAG
from Core.utils.bm25 import BM25


class ReactWiringTests(unittest.TestCase):
    def test_config_accepts_react_with_official_defaults(self):
        cfg = VanillaConfig(retrieval_method="react")

        self.assertEqual(cfg.react_max_steps, 7)
        self.assertEqual(cfg.react_search_topk, 1)
        self.assertEqual(cfg.react_page_observation_units, 5)
        self.assertEqual(cfg.react_prompt_file, "")
        self.assertEqual(cfg.react_dataset_name, "qasper")

    def test_resource_loader_loads_react_bm25(self):
        from Core.utils.resource_loader import prepare_rag_dependencies

        with tempfile.TemporaryDirectory() as tmp:
            bm25_dir = Path(tmp) / "bm25_vdb"
            bm25_dir.mkdir()
            bm25 = BM25(
                docs=["Evidence text."],
                metadatas=[{"source_node_id": 1}],
            )
            bm25.initialize()
            bm25.save(bm25_dir / "bm25_index.pkl")
            cfg = SimpleNamespace(
                save_path=tmp,
                rag=SimpleNamespace(
                    strategy_config=VanillaConfig(
                        retrieval_method="react",
                        bm25_vdb_dir_name="bm25_vdb",
                    )
                ),
            )

            dependencies = prepare_rag_dependencies(cfg)

        self.assertIn("bm25", dependencies)
        self.assertEqual(
            dependencies["bm25"].original_docs,
            ["Evidence text."],
        )

    def test_vanilla_rag_persists_react_outputs(self):
        class FakeLLM:
            config = SimpleNamespace(max_tokens=8000)

            def __init__(self):
                self.responses = [
                    " I should inspect Ada.\nAction 1: Search[Ada]",
                    " The answer is London.\nAction 2: Finish[London]",
                ]

            def get_completion(self, prompt, json_response=False):
                return self.responses.pop(0)

        bm25 = BM25(
            docs=[
                "Ada was born in London.",
                "London is in England.",
            ],
            metadatas=[
                {
                    "node_id": 1,
                    "source_node_id": 1,
                    "section_id": "Ada",
                    "section": "Ada",
                    "title_path": "Ada",
                    "qasper_evidence_text": "Ada was born in London.",
                    "source": "qasper_paragraph",
                    "node_type": "text",
                },
                {
                    "node_id": 2,
                    "source_node_id": 2,
                    "section_id": "London",
                    "section": "London",
                    "title_path": "London",
                    "qasper_evidence_text": "London is in England.",
                    "source": "qasper_paragraph",
                    "node_type": "text",
                },
            ],
        )
        bm25.initialize()
        rag = VanillaRAG(
            config=VanillaConfig(
                retrieval_method="react",
                react_dataset_name="qasper",
            ),
            llm=FakeLLM(),
            bm25=bm25,
        )

        with tempfile.TemporaryDirectory() as tmp:
            answer, retrieval_ids = rag.generation(
                "Where was Ada born?",
                Path(tmp),
            )
            payload = json.loads(
                (Path(tmp) / "retrieval_res.json").read_text(
                    encoding="utf-8",
                )
            )
            chain = json.loads(
                (Path(tmp) / "evidence_chain.json").read_text(
                    encoding="utf-8",
                )
            )

        self.assertEqual(payload["strategy"], "react")
        self.assertEqual(payload["react_termination_reason"], "finish")
        self.assertEqual(payload["supporting_block_ids"], [1])
        self.assertEqual(rag.last_answer_short, "London")
        self.assertEqual(rag.last_retrieved_block_ids, [1])
        self.assertEqual(retrieval_ids, [1])
        self.assertEqual(chain["supporting_block_ids"], [1])
        self.assertEqual(json.loads(answer)["answer_short"], "London")

    def test_hotpot_react_restores_title_and_sentence_ids_from_page_order(self):
        class FakeLLM:
            config = SimpleNamespace(max_tokens=8000)

            def __init__(self):
                self.responses = [
                    " I should inspect the film.\n"
                    "Action 1: Search[La Boheme]",
                    " The answer is known.\n"
                    "Action 2: Finish[1988]",
                ]

            def get_completion(self, prompt, json_response=False):
                return self.responses.pop(0)

        bm25 = BM25(
            docs=[
                "The film was released in 1988.",
                "It was directed by Luigi Comencini.",
            ],
            metadatas=[
                {
                    "node_id": 9,
                    "source_node_id": 9,
                    "section_id": "La Boheme",
                    "section": "La Boheme",
                    "title_path": "La Boheme",
                    "source": "qasper_paragraph",
                    "node_type": "text",
                },
                {
                    "node_id": 10,
                    "source_node_id": 10,
                    "section_id": "La Boheme",
                    "section": "La Boheme",
                    "title_path": "La Boheme",
                    "source": "qasper_paragraph",
                    "node_type": "text",
                },
            ],
        )
        bm25.initialize()
        rag = VanillaRAG(
            config=VanillaConfig(
                retrieval_method="react",
                react_dataset_name="hotpotqa",
            ),
            llm=FakeLLM(),
            bm25=bm25,
        )

        with tempfile.TemporaryDirectory() as tmp:
            rag.generation("When was the film released?", Path(tmp))
            payload = json.loads(
                (Path(tmp) / "retrieval_res.json").read_text(
                    encoding="utf-8",
                )
            )

        facts = [
            [item.get("hotpot_title"), item.get("hotpot_sent_id")]
            for item in payload["supporting_evidence"]
        ]
        self.assertEqual(
            facts,
            [["La Boheme", 0], ["La Boheme", 1]],
        )
