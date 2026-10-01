from pathlib import Path

import torch

from bfa.models.singlem_adapter import SingLEMAdapter


def test_singlem_adapter_is_frozen_and_has_contract_shape() -> None:
    checkpoint = Path(
        "third_party/SingLEM/SingLEM/checkpoints/singlem_downstream_excluded.pt"
    )
    model = SingLEMAdapter(checkpoint).eval()
    inputs = torch.zeros(2, 16, 10, 128)
    output = model.forward_window(inputs)
    assert output.shape == (2, 16, 128)
    assert torch.isfinite(output).all()
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.projection.parameters())
