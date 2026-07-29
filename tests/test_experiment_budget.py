import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiment_budget import make_token_budget_plan


class TokenBudgetPlanTest(unittest.TestCase):
    def test_builds_exact_nested_plan(self):
        plan = make_token_budget_plan(
            train_token_budget=268_435_456,
            tokens_per_iteration=65_536,
            data_unique_tokens=1_000_013_824,
            eval_at_tokens=[
                16_777_216,
                33_554_432,
                67_108_864,
                134_217_728,
                268_435_456,
            ],
            strict_sub_one_pass=True,
        )

        self.assertEqual(plan.iterations, 4096)
        self.assertEqual(plan.eval_at_steps, (256, 512, 1024, 2048, 4096))
        self.assertLess(plan.target_data_exposure, 1.0)

    def test_rejects_inexact_step_boundary(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            make_token_budget_plan(
                train_token_budget=1_000_000,
                tokens_per_iteration=65_536,
                data_unique_tokens=2_000_000,
                strict_sub_one_pass=True,
            )

    def test_rejects_more_than_one_strict_pass(self):
        with self.assertRaisesRegex(ValueError, "strict_sub_one_pass"):
            make_token_budget_plan(
                train_token_budget=131_072,
                tokens_per_iteration=65_536,
                data_unique_tokens=65_536,
                strict_sub_one_pass=True,
            )


if __name__ == "__main__":
    unittest.main()
