import unittest


def _answer_payload(answer_type):
    if answer_type == "none":
        return [
            {
                "unanswerable": True,
                "extractive_spans": [],
                "free_form_answer": "",
                "yes_no": None,
            }
        ]
    if answer_type == "boolean":
        return [
            {
                "unanswerable": False,
                "extractive_spans": [],
                "free_form_answer": "",
                "yes_no": True,
            }
        ]
    if answer_type == "extractive":
        return [
            {
                "unanswerable": False,
                "extractive_spans": ["span"],
                "free_form_answer": "",
                "yes_no": None,
            }
        ]
    return [
        {
            "unanswerable": False,
            "extractive_spans": [],
            "free_form_answer": "free form",
            "yes_no": None,
        }
    ]


def _population():
    rows = []
    intents = {}
    counts = {
        "extractive": 55,
        "abstractive": 30,
        "boolean": 18,
        "none": 17,
    }
    index = 0
    for answer_type, count in counts.items():
        for local_index in range(count):
            question_id = f"q{index:03d}"
            if local_index < 3:
                intent = "multi-hop"
            elif local_index < 13:
                intent = "comparison"
            else:
                intent = "fact"
            rows.append(
                {
                    "question": f"Question {index}",
                    "answer": _answer_payload(answer_type),
                    "doc_uuid": f"paper-{index // 3}",
                    "doc_path": f"qasper://paper-{index // 3}",
                    "qasper_question_id": question_id,
                }
            )
            intents[question_id] = intent
            index += 1
    return rows, intents


class QasperSupportManifestTests(unittest.TestCase):
    def test_qasper_answer_type_uses_deterministic_priority(self):
        from Scripts.eval.qasper_support_manifest import qasper_answer_type

        for answer_type in ("extractive", "abstractive", "boolean", "none"):
            with self.subTest(answer_type=answer_type):
                self.assertEqual(
                    qasper_answer_type({"answer": _answer_payload(answer_type)}),
                    answer_type,
                )

    def test_build_split_meets_all_quotas_and_is_deterministic(self):
        from Scripts.eval.qasper_support_manifest import (
            build_support_optimization_split,
            qasper_answer_type,
        )

        rows, intents = _population()
        kwargs = {
            "rows": rows,
            "demand_intents": intents,
            "seed": 42,
            "target_size": 100,
            "answer_type_quotas": {
                "extractive": 45,
                "abstractive": 25,
                "boolean": 15,
                "none": 15,
            },
            "minimum_intent_counts": {"comparison": 30, "multi-hop": 8},
        }

        first = build_support_optimization_split(**kwargs)
        second = build_support_optimization_split(**kwargs)

        self.assertEqual(first, second)
        tuning = first["tuning_rows"]
        holdout = first["holdout_rows"]
        self.assertEqual(len(tuning), 100)
        self.assertEqual(len(holdout), 20)
        type_counts = {
            answer_type: sum(
                qasper_answer_type(row) == answer_type for row in tuning
            )
            for answer_type in ("extractive", "abstractive", "boolean", "none")
        }
        self.assertEqual(
            type_counts,
            {"extractive": 45, "abstractive": 25, "boolean": 15, "none": 15},
        )
        intent_counts = {
            intent: sum(
                intents[row["qasper_question_id"]] == intent for row in tuning
            )
            for intent in ("comparison", "multi-hop")
        }
        self.assertGreaterEqual(intent_counts["comparison"], 30)
        self.assertGreaterEqual(intent_counts["multi-hop"], 8)
        tuning_ids = {row["qasper_question_id"] for row in tuning}
        holdout_ids = {row["qasper_question_id"] for row in holdout}
        source_ids = {row["qasper_question_id"] for row in rows}
        self.assertFalse(tuning_ids & holdout_ids)
        self.assertEqual(tuning_ids | holdout_ids, source_ids)

    def test_build_split_rejects_impossible_intent_minimum(self):
        from Scripts.eval.qasper_support_manifest import build_support_optimization_split

        rows, intents = _population()

        with self.assertRaisesRegex(ValueError, "multi-hop"):
            build_support_optimization_split(
                rows=rows,
                demand_intents=intents,
                seed=42,
                target_size=100,
                answer_type_quotas={
                    "extractive": 45,
                    "abstractive": 25,
                    "boolean": 15,
                    "none": 15,
                },
                minimum_intent_counts={"comparison": 30, "multi-hop": 20},
            )

    def test_validate_split_rejects_overlap(self):
        from Scripts.eval.qasper_support_manifest import validate_support_split

        rows, intents = _population()

        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_support_split(
                source_rows=rows,
                tuning_rows=rows[:100],
                holdout_rows=[rows[0], *rows[100:]],
                demand_intents=intents,
                answer_type_quotas={
                    "extractive": 45,
                    "abstractive": 25,
                    "boolean": 15,
                    "none": 15,
                },
                minimum_intent_counts={"comparison": 30, "multi-hop": 8},
            )


if __name__ == "__main__":
    unittest.main()
