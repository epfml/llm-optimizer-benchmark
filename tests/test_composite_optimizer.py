import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optim.composite import (
    CompositeOptimizer,
    CompositeScheduler,
    split_muon_param_groups,
)


class CompositeOptimizerTest(unittest.TestCase):
    def test_steps_and_restores_both_optimizers(self):
        first = torch.nn.Parameter(torch.tensor([1.0]))
        second = torch.nn.Parameter(torch.tensor([2.0]))
        first_opt = torch.optim.SGD([first], lr=0.1)
        second_opt = torch.optim.AdamW([second], lr=0.01)
        optimizer = CompositeOptimizer([first_opt, second_opt])
        scheduler = CompositeScheduler(
            [
                torch.optim.lr_scheduler.LambdaLR(first_opt, lambda _: 1.0),
                torch.optim.lr_scheduler.LambdaLR(second_opt, lambda _: 1.0),
            ]
        )

        first.grad = torch.ones_like(first)
        second.grad = torch.ones_like(second)
        optimizer.step()
        scheduler.step()

        self.assertLess(first.item(), 1.0)
        self.assertLess(second.item(), 2.0)
        self.assertEqual(len(optimizer.param_groups), 2)

        optimizer.load_state_dict(optimizer.state_dict())
        scheduler.load_state_dict(scheduler.state_dict())

    def test_partitions_embeddings_and_vectors_to_adamw(self):
        hidden = torch.nn.Parameter(torch.ones(4, 4))
        embedding = torch.nn.Parameter(torch.ones(8, 4))
        norm = torch.nn.Parameter(torch.ones(4))
        groups = [{"params": [hidden, embedding, norm], "weight_decay": 0.1}]

        muon_groups, adamw_groups = split_muon_param_groups(groups, {id(embedding)})

        self.assertEqual(muon_groups[0]["params"], [hidden])
        self.assertEqual(adamw_groups[0]["params"], [embedding, norm])


if __name__ == "__main__":
    unittest.main()
