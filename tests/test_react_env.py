import unittest

from Core.rag.react_env import ReactLocalEnvironment


def make_doc(node_id, content, section, score=1.0, hotpot=False):
    metadata = {
        "node_id": node_id,
        "source_node_id": node_id,
        "section_id": section,
        "section": section,
        "title_path": section,
        "qasper_evidence_text": content,
        "source": "qasper_paragraph",
        "node_type": "text",
    }
    if hotpot:
        metadata["hotpot_title"] = section
        metadata["hotpot_sent_id"] = node_id
    return {"id": node_id, "content": content, "score": score, "metadata": metadata}


class ReactEnvironmentTests(unittest.TestCase):
    def test_groups_qasper_paragraphs_by_section(self):
        docs = [
            make_doc(1, "First introduction sentence.", "Introduction"),
            make_doc(2, "Second introduction sentence.", "Introduction"),
            make_doc(3, "A result sentence.", "Results"),
        ]
        env = ReactLocalEnvironment(docs=docs, bm25=None, page_observation_units=5)

        self.assertEqual(env.page_keys, ["Introduction", "Results"])
        self.assertEqual(
            [item.block_id for item in env.pages["Introduction"]],
            [1, 2],
        )

    def test_groups_hotpot_sentences_by_title(self):
        docs = [
            make_doc(4, "Alpha first.", "Alpha", hotpot=True),
            make_doc(5, "Alpha second.", "Alpha", hotpot=True),
            make_doc(6, "Beta first.", "Beta", hotpot=True),
        ]
        env = ReactLocalEnvironment(docs=docs, bm25=None, page_observation_units=5)

        self.assertEqual(env.page_keys, ["Alpha", "Beta"])
        self.assertEqual([item.block_id for item in env.pages["Alpha"]], [4, 5])

    def test_search_prefers_exact_page_and_returns_first_five_units(self):
        docs = [
            make_doc(i, f"Sentence {i}.", "Methods")
            for i in range(1, 7)
        ]
        env = ReactLocalEnvironment(docs=docs, bm25=None, page_observation_units=5)

        result = env.step("Search[Methods]")

        self.assertTrue(result.valid_action)
        self.assertEqual(env.active_page, "Methods")
        self.assertEqual(
            [item.block_id for item in result.evidence],
            [1, 2, 3, 4, 5],
        )

    def test_search_uses_bm25_result_page_when_title_does_not_match(self):
        class FakeBM25:
            def search(self, query_text, top_k):
                return [make_doc(8, "Target evidence.", "Results", score=9.0)]

        docs = [
            make_doc(7, "Other evidence.", "Introduction"),
            make_doc(8, "Target evidence.", "Results"),
        ]
        env = ReactLocalEnvironment(
            docs=docs,
            bm25=FakeBM25(),
            page_observation_units=5,
        )

        result = env.step("Search[target metric]")

        self.assertEqual(env.active_page, "Results")
        self.assertEqual([item.block_id for item in result.evidence], [8])

    def test_search_unknown_page_returns_deterministic_observation(self):
        env = ReactLocalEnvironment(
            docs=[make_doc(1, "Only evidence.", "Introduction")],
            bm25=None,
            page_observation_units=5,
        )

        result = env.step("Search[Missing]")

        self.assertFalse(result.valid_action)
        self.assertEqual(result.text, "Could not find Missing.")

    def test_lookup_advances_and_resets_per_keyword(self):
        docs = [
            make_doc(
                1,
                "The model uses BERT. BERT improves accuracy.",
                "Results",
            ),
            make_doc(2, "Accuracy reaches 91 percent.", "Results"),
        ]
        env = ReactLocalEnvironment(docs=docs, bm25=None, page_observation_units=5)
        env.step("Search[Results]")

        first = env.step("Lookup[BERT]")
        second = env.step("Lookup[BERT]")
        reset = env.step("Lookup[accuracy]")

        self.assertIn("(Result 1 / 2)", first.text)
        self.assertIn("(Result 2 / 2)", second.text)
        self.assertIn("(Result 1 / 2)", reset.text)

    def test_finish_preserves_empty_answer(self):
        env = ReactLocalEnvironment([], bm25=None, page_observation_units=5)

        result = env.step("Finish[]")

        self.assertTrue(result.done)
        self.assertEqual(result.answer, "")
        self.assertEqual(env.answer, "")

    def test_lookup_without_page_and_invalid_action_are_explicit(self):
        env = ReactLocalEnvironment([], bm25=None, page_observation_units=5)

        lookup = env.step("Lookup[key]")
        invalid = env.step("Open[url]")

        self.assertEqual(lookup.text, "No active page.")
        self.assertEqual(invalid.text, "Invalid action: Open[url]")
        self.assertFalse(invalid.valid_action)
