import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

RUNNER_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "sub_one_pass" / "run_all.py"
)
SPEC = importlib.util.spec_from_file_location("sub_one_pass_run_all", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class SubOnePassRunnerTest(unittest.TestCase):
    def test_registry_matches_repository_optimizer_surface(self):
        expected = {
            "adamw",
            "sgd",
            "muon",
            "soap",
            "ademamix",
            "lion",
            "sf-adamw",
            "sf-sgd",
            "signsgd",
            "signum",
            "prodigy",
            "sophiag",
            "adopt",
            "mars",
            "adafactor",
            "lamb",
            "scion",
            "scion-light",
            "d-muon",
            "muon-pytorch",
        }
        self.assertEqual(set(RUNNER.OPTIMIZER_ARGS), expected)
        self.assertEqual(set(RUNNER.LR_FLAGS), expected)

    def test_every_optimizer_configuration_is_flag_value_pairs(self):
        for optimizer, arguments in RUNNER.OPTIMIZER_ARGS.items():
            with self.subTest(optimizer=optimizer):
                self.assertEqual(len(arguments) % 2, 0)
                self.assertTrue(all(flag.startswith("--") for flag in arguments[::2]))

    def test_scales_all_optimizer_learning_rates(self):
        for optimizer in RUNNER.OPTIMIZER_ARGS:
            with self.subTest(optimizer=optimizer):
                original = RUNNER.scaled_optimizer_args(optimizer, 1.0)
                scaled = RUNNER.scaled_optimizer_args(optimizer, 3.0)
                for index, flag in enumerate(original[::2]):
                    value_index = index * 2 + 1
                    if flag in RUNNER.LR_FLAGS[optimizer]:
                        self.assertAlmostEqual(
                            float(scaled[value_index]),
                            3.0 * float(original[value_index]),
                        )
                    else:
                        self.assertEqual(scaled[value_index], original[value_index])

    def test_stage_budgets_are_exact_step_boundaries(self):
        tokens_per_step = 16 * 8 * 512
        for stage, boundaries in RUNNER.STAGE_EVAL_TOKENS.items():
            with self.subTest(stage=stage):
                self.assertEqual(boundaries, sorted(set(boundaries)))
                self.assertTrue(
                    all(tokens % tokens_per_step == 0 for tokens in boundaries)
                )

    def test_manifest_preserves_runs_across_invocations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest_tune.json"
            args = SimpleNamespace(stage="tune", marker="first")
            manifest = RUNNER.initialize_manifest(path, args, ["adamw"])
            RUNNER.upsert_run_record(
                manifest,
                {
                    "experiment_dir": "/tmp/adamw-scale1",
                    "optimizer": "adamw",
                    "status": "completed",
                },
            )
            RUNNER.write_manifest(path, manifest)

            args = SimpleNamespace(stage="tune", marker="second")
            restored = RUNNER.initialize_manifest(path, args, ["soap"])
            self.assertEqual(restored["optimizers"], ["adamw", "soap"])
            self.assertEqual(len(restored["invocations"]), 2)
            self.assertEqual(len(restored["runs"]), 1)

            record = RUNNER.upsert_run_record(
                restored,
                {
                    "experiment_dir": "/tmp/adamw-scale1",
                    "optimizer": "adamw",
                    "status": "skipped-completed",
                },
            )
            self.assertEqual(len(restored["runs"]), 1)
            self.assertEqual(record["status"], "skipped-completed")
            RUNNER.write_manifest(path, restored)
            self.assertEqual(
                json.loads(path.read_text())["runs"][0]["status"],
                "skipped-completed",
            )


if __name__ == "__main__":
    unittest.main()
