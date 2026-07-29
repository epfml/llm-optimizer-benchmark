import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class DataReaderTest(unittest.TestCase):
    def test_fixed_boundaries_and_no_replacement(self):
        from data.reader import DataReader

        sequence_length = 8
        batch_size = 2
        data = np.arange(8 * 13 + 1, dtype=np.uint16)
        reader = DataReader(
            data,
            batch_size=batch_size,
            sequence_length=sequence_length,
            seed=7,
            fixed_data_boundaries=True,
            lazy_data_permutation=True,
        )

        starts = []
        for _ in range(reader.num_batches()):
            x, _ = reader.sample_batch()
            starts.extend(x[:, 0].tolist())

        self.assertEqual(reader.epoch_offset, 0)
        self.assertEqual(len(starts), len(set(starts)))
        self.assertIsNone(reader.order)
        self.assertTrue(all(value % sequence_length == 0 for value in starts))
        self.assertEqual(
            reader.unique_tokens_per_epoch,
            reader.num_batches() * batch_size * sequence_length,
        )


if __name__ == "__main__":
    unittest.main()
