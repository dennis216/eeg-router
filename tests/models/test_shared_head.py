import torch

from bfa.models.shared_head import CausalConv1d, SharedContextHead


def test_shared_head_shape_and_gradients() -> None:
    torch.manual_seed(17)
    model = SharedContextHead(dim=128, heads=4, blocks=4, dropout=0.0)
    inputs = torch.randn(2, 31, 16, 128, requires_grad=True)
    output = model(inputs)
    assert output.shape == (2,)
    output.sum().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_causal_conv_ignores_future_samples() -> None:
    torch.manual_seed(17)
    model = CausalConv1d(8, 8, 3, dilation=4).eval()
    inputs = torch.randn(2, 8, 31)
    first = model(inputs)
    changed = inputs.clone()
    changed[:, :, 20:] += 1000
    second = model(changed)
    assert torch.allclose(first[:, :, :20], second[:, :, :20], atol=1e-6)
