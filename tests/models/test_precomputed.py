import torch

from bfa.models.precomputed import PrecomputedProjectionEncoder


def test_precomputed_projection_sequence_contract() -> None:
    model = PrecomputedProjectionEncoder(16)
    inputs = torch.randn(2, 31, 16, 16)
    output = model.forward_window_sequence(inputs)
    assert output.shape == (2, 31, 16, 128)
    output.mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
