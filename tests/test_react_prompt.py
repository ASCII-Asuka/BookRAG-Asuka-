import tempfile
import unittest
from pathlib import Path

from Core.rag.react_prompt import (
    QASPER_DEMO_IDS,
    build_react_prompt,
    get_react_examples,
    parse_react_response,
)


class ReactPromptTests(unittest.TestCase):
    def test_parses_numbered_thought_and_action(self):
        parsed = parse_react_response(
            "Thought 2: I should inspect the results.\n"
            "Action 2: Search[Results]",
            step=2,
        )

        self.assertEqual(parsed.thought, "I should inspect the results.")
        self.assertEqual(parsed.action, "Search[Results]")
        self.assertTrue(parsed.complete)

    def test_finish_allows_brackets_inside_answer(self):
        parsed = parse_react_response(
            "Thought 1: The answer is known.\n"
            "Action 1: Finish[F1 [macro]]",
            step=1,
        )

        self.assertEqual(parsed.action, "Finish[F1 [macro]]")

    def test_malformed_response_keeps_first_thought_for_repair(self):
        parsed = parse_react_response("Need another section.", step=3)

        self.assertEqual(parsed.thought, "Need another section.")
        self.assertEqual(parsed.action, "")
        self.assertFalse(parsed.complete)

    def test_hotpot_and_qasper_each_have_six_fixed_examples(self):
        self.assertEqual(get_react_examples("hotpotqa").count("Question:"), 6)
        self.assertEqual(get_react_examples("qasper").count("Question:"), 6)

    def test_qasper_examples_are_from_fixed_train_ids(self):
        self.assertEqual(
            QASPER_DEMO_IDS,
            (
                "753990d0b621d390ed58f20c4d9e4f065f0dc672",
                "44c4bd6decc86f1091b5fc0728873d9324cdde4e",
                "003f884d3893532f8c302431c9f70be6f64d9be8",
                "938cf30c4f1d14fa182e82919e16072fdbcf2a82",
                "19c9cfbc4f29104200393e848b7b9be41913a7ac",
                "cd1034c183edf630018f47ff70b48d74d2bb1649",
            ),
        )

    def test_prompt_appends_question_and_existing_trajectory(self):
        prompt = build_react_prompt(
            dataset_name="qasper",
            question="What metric is reported?",
            trajectory=(
                "Thought 1: Inspect results.\n"
                "Action 1: Search[Results]\n"
                "Observation 1: F1 is reported.\n"
            ),
            step=2,
        )

        self.assertIn("Question: What metric is reported?", prompt)
        self.assertTrue(prompt.endswith("Thought 2:"))

    def test_prompt_file_overrides_builtin_examples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir) / "react_examples.txt"
            prompt_path.write_text(
                "Question: Custom?\nThought 1: Done.\n"
                "Action 1: Finish[Yes]\n",
                encoding="utf-8",
            )

            examples = get_react_examples(
                "qasper",
                prompt_file=str(prompt_path),
            )

        self.assertIn("Question: Custom?", examples)
