import importlib.util
import unittest
from pathlib import Path

import numpy as np

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "sub_one_pass"
    / "prepare_fineweb_stream.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_fineweb_stream", SCRIPT_PATH)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class FakeTokenizer:
    eot_token = 99

    def encode_ordinary(self, text):
        return {
            "validation-one": [1, 2, 3, 4],
            "boundary-document": [5, 6, 7, 8],
            "training-one": [9, 10, 11],
        }[text]


class FineWebStreamTest(unittest.TestCase):
    def test_does_not_split_one_document_between_validation_and_train(self):
        stream = [
            {"text": "validation-one"},
            {"text": "boundary-document"},
            {"text": "training-one"},
        ]
        validation = np.zeros(7, dtype=np.uint16)
        train = np.zeros(4, dtype=np.uint16)

        documents, discarded = PREPARE.write_tokens(
            stream, FakeTokenizer(), validation, train
        )

        self.assertEqual(documents, 3)
        self.assertEqual(discarded, 3)
        self.assertEqual(validation.tolist(), [1, 2, 3, 4, 99, 5, 6])
        self.assertEqual(train.tolist(), [9, 10, 11, 99])


if __name__ == "__main__":
    unittest.main()
