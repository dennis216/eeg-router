import pytest
import torch

from bfa.contracts import ModelOutput, WindowRef


def test_model_output_contract() -> None:
    ref = WindowRef("chb02", "chb02_01.edf", 0.0, 10.0)
    out = ModelOutput(
        probability=torch.tensor([0.7]),
        embedding=torch.zeros(1, 16, 128),
        channel_quality=torch.ones(1, 16),
        window_refs=(ref,),
    )
    out.validate()
    assert ref.available_at_s == 10.0


def test_model_output_rejects_wrong_embedding_shape() -> None:
    ref = WindowRef("chb02", "chb02_01.edf", 0.0, 10.0)
    out = ModelOutput(
        probability=torch.tensor([0.7]),
        embedding=torch.zeros(1, 16, 127),
        channel_quality=torch.ones(1, 16),
        window_refs=(ref,),
    )
    with pytest.raises(AssertionError):
        out.validate()
