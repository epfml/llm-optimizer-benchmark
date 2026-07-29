import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optim.schedule import warmup_constant_schedule


class WarmupConstantScheduleTest(unittest.TestCase):
    def test_warms_up_then_stays_constant(self):
        schedule = warmup_constant_schedule(n_warmup=4, init_div_factor=100)
        self.assertAlmostEqual(schedule(0), 0.01)
        self.assertLess(schedule(1), schedule(2))
        self.assertEqual(schedule(4), 1.0)
        self.assertEqual(schedule(400), 1.0)

    def test_zero_warmup(self):
        schedule = warmup_constant_schedule(n_warmup=0)
        self.assertEqual(schedule(0), 1.0)


if __name__ == "__main__":
    unittest.main()
