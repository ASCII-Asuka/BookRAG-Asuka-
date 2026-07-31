import unittest


def make_row(question_id: str, question_type: str, valid: bool = True) -> dict:
    return {
        "id": question_id,
        "question": f"Question {question_id}?",
        "answer": "answer",
        "type": question_type,
        "level": "hard",
        "supporting_facts": {
            "title": ["Article"],
            "sent_id": [0 if valid else 9],
        },
        "context": {
            "title": ["Article"],
            "sentences": [["Supporting sentence."]],
        },
    }


class HotpotQAFixedSampleTests(unittest.TestCase):
    def test_fixed_sample_uses_proportional_hamilton_quotas(self):
        from Scripts.preprocess.hotpotqa_fixed_sample import select_fixed_rows

        rows = [
            make_row(f"b-{index}", "bridge") for index in range(8)
        ] + [
            make_row(f"c-{index}", "comparison") for index in range(2)
        ]

        selected, audit = select_fixed_rows(rows, sample_size=5, seed=42)

        self.assertEqual(
            audit["selected_strata"],
            {
                "bridge|hard": 4,
                "comparison|hard": 1,
            },
        )
        self.assertEqual(len(selected), 5)

    def test_fixed_sample_excludes_and_audits_out_of_range_gold_fact(self):
        from Scripts.preprocess.hotpotqa_fixed_sample import select_fixed_rows

        rows = [
            make_row("good", "bridge"),
            make_row("bad", "bridge", valid=False),
        ]

        selected, audit = select_fixed_rows(rows, sample_size=1, seed=42)

        self.assertEqual([row["id"] for row in selected], ["good"])
        self.assertEqual(audit["excluded"][0]["question_id"], "bad")
        self.assertEqual(
            audit["excluded"][0]["reason"],
            "sentence_id_out_of_range",
        )

    def test_fixed_sample_selection_is_stable_under_input_reordering(self):
        from Scripts.preprocess.hotpotqa_fixed_sample import select_fixed_rows

        rows = [
            make_row(f"id-{index}", "bridge")
            for index in range(20)
        ]

        first, _ = select_fixed_rows(rows, sample_size=5, seed=42)
        second, _ = select_fixed_rows(
            list(reversed(rows)),
            sample_size=5,
            seed=42,
        )

        self.assertEqual(
            {row["id"] for row in first},
            {row["id"] for row in second},
        )


if __name__ == "__main__":
    unittest.main()
