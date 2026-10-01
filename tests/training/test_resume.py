import random

import numpy as np
import torch

from bfa.training.trainer import capture_training_state, restore_training_state


def test_checkpoint_restores_model_optimizer_scheduler_and_rng() -> None:
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    loss = model(torch.randn(2, 3)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    state = capture_training_state(model, optimizer, scheduler, epoch=1, step=1)
    expected = (torch.rand(3), np.random.rand(3), random.random())

    restored_model = torch.nn.Linear(3, 1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=9e-3)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        restored_optimizer, T_max=4
    )
    metadata = restore_training_state(
        state, restored_model, restored_optimizer, restored_scheduler
    )
    actual = (torch.rand(3), np.random.rand(3), random.random())

    assert metadata == {"epoch": 1, "step": 1}
    assert all(
        torch.equal(left, right)
        for left, right in zip(model.state_dict().values(), restored_model.state_dict().values())
    )
    assert torch.equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]
