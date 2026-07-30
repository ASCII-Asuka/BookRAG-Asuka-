import unittest


class PairedBootstrapTests(unittest.TestCase):
    def test_reports_deterministic_paired_delta_and_confidence_interval(self):
        from Eval.utils.paired_bootstrap import paired_bootstrap_ci

        result = paired_bootstrap_ci(
            baseline={"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 0.0},
            candidate={"q1": 1.0, "q2": 1.0, "q3": 0.0, "q4": 1.0},
            n_resamples=2000,
            seed=42,
        )

        self.assertEqual(result["question_count"], 4)
        self.assertAlmostEqual(result["candidate_minus_baseline"], 0.75)
        self.assertGreaterEqual(result["ci_95"][0], 0.0)
        self.assertLessEqual(result["ci_95"][1], 1.0)
        self.assertEqual(result["seed"], 42)

    def test_rejects_unpaired_question_sets(self):
        from Eval.utils.paired_bootstrap import paired_bootstrap_ci

        with self.assertRaisesRegex(ValueError, "question IDs"):
            paired_bootstrap_ci(
                baseline={"q1": 0.0},
                candidate={"q2": 1.0},
                n_resamples=10,
            )


if __name__ == "__main__":
    unittest.main()
