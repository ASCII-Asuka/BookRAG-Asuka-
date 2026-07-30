import unittest

from Core.rag.react_env import ReactLocalEnvironment
from Core.rag.react_runner import ReactRunner


def make_doc(node_id, content, section, score=1.0):
    return {
        "content": content,
        "score": score,
        "metadata": {
            "node_id": node_id,
            "source_node_id": node_id,
            "section_id": section,
            "section": section,
            "title_path": section,
            "qasper_evidence_text": content,
            "source": "qasper_paragraph",
        },
    }


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def get_completion(self, prompt, json_response=False):
        self.prompts.append(prompt)
        return self.responses.pop(0)


class ReactRunnerTests(unittest.TestCase):
    def test_runs_search_lookup_finish_trajectory(self):
        docs = [
            make_doc(
                1,
                "Ada was born in London. She was a mathematician.",
                "Ada",
            ),
            make_doc(2, "London is in England.", "London"),
        ]
        llm = FakeLLM(
            [
                " I need Ada's birthplace.\nAction 1: Search[Ada]",
                " The page names London.\nAction 2: Lookup[born]",
                " The answer is London.\nAction 3: Finish[London]",
            ]
        )
        runner = ReactRunner(
            llm=llm,
            environment=ReactLocalEnvironment(docs, bm25=None),
            dataset_name="hotpotqa",
            max_steps=7,
        )

        result = runner.run("Where was Ada born?")

        self.assertEqual(result.answer, "London")
        self.assertEqual(
            [step.action_type for step in result.steps],
            ["search", "lookup", "finish"],
        )
        self.assertEqual(result.num_calls, 3)
        self.assertEqual(result.num_bad_calls, 0)
        self.assertEqual(
            [unit.block_id for unit in result.observed_evidence],
            [1],
        )
        self.assertEqual(result.termination_reason, "finish")

    def test_repairs_action_once_when_combined_response_is_malformed(self):
        docs = [make_doc(1, "Ada was born in London.", "Ada")]
        llm = FakeLLM(
            [
                " I should search Ada.",
                "Search[Ada]",
                " The answer is London.\nAction 2: Finish[London]",
            ]
        )
        result = ReactRunner(
            llm=llm,
            environment=ReactLocalEnvironment(docs, bm25=None),
            dataset_name="hotpotqa",
            max_steps=7,
        ).run("Where was Ada born?")

        self.assertEqual(result.answer, "London")
        self.assertEqual(result.num_calls, 3)
        self.assertEqual(result.num_bad_calls, 1)
        self.assertEqual(result.steps[0].action, "Search[Ada]")

    def test_finishes_empty_after_seven_nonterminal_steps(self):
        docs = [make_doc(1, "Evidence.", "Page")]
        llm = FakeLLM(
            [
                f" Keep searching.\nAction {i}: Search[Page]"
                for i in range(1, 8)
            ]
        )
        result = ReactRunner(
            llm=llm,
            environment=ReactLocalEnvironment(docs, bm25=None),
            dataset_name="hotpotqa",
            max_steps=7,
        ).run("Question?")

        self.assertEqual(result.answer, "")
        self.assertEqual(len(result.steps), 8)
        self.assertEqual(result.steps[-1].action, "Finish[]")
        self.assertEqual(result.termination_reason, "step_limit")
