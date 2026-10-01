import torch

from bfa.models.tcn_gat import MultiScaleTCNGAT, build_bipolar_graph, has_edge
from bfa.preprocessing.channels import CHANNELS


def test_bipolar_graph_connectivity_and_self_loops() -> None:
    edge_index = build_bipolar_graph(CHANNELS)
    fp1_f7 = CHANNELS.index("FP1-F7")
    f7_t7 = CHANNELS.index("F7-T7")
    assert has_edge(edge_index, fp1_f7, f7_t7)
    assert has_edge(edge_index, 0, 0)


def test_tcn_gat_shape_gradients_and_stable_graph() -> None:
    torch.manual_seed(17)
    model = MultiScaleTCNGAT(dropout=0.0)
    inputs = torch.randn(2, 16, 2560)
    quality = torch.zeros(2, 16, 3)
    edge_before = model.edge_index.clone()
    output = model.forward_window(inputs, quality)
    assert output.shape == (2, 16, 128)

    permuted = model.forward_window(inputs.flip(0), quality.flip(0))
    assert torch.equal(model.edge_index, edge_before)
    assert torch.allclose(permuted.flip(0), output, atol=1e-5)

    (output.mean() + model.native_logits(output).mean()).backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_tcn_gat_parameter_snapshot() -> None:
    model = MultiScaleTCNGAT()
    assert sum(parameter.numel() for parameter in model.parameters()) == 119_361
