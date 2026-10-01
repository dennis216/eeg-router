from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bfa.models.cbramod_adapter import CBraModAdapter
from bfa.models.singlem_adapter import SingLEMAdapter
from bfa.models.tcn_gat import MultiScaleTCNGAT


def smoke_singlem() -> None:
    checkpoint = Path(
        "third_party/SingLEM/SingLEM/checkpoints/singlem_downstream_excluded.pt"
    )
    model = SingLEMAdapter(checkpoint).cuda().eval()
    inputs = torch.zeros(2, 16, 10, 128, device="cuda")
    with torch.inference_mode():
        output = model.forward_window(inputs)
    torch.cuda.synchronize()
    if output.shape != (2, 16, 128) or not torch.isfinite(output).all():
        raise AssertionError("SingLEM GPU smoke output failed its contract")
    allocated_mib = torch.cuda.max_memory_allocated() / 2**20
    print(
        f"SINGLEM_GPU_OK shape={tuple(output.shape)} dtype={output.dtype} "
        f"peak_allocated_mib={allocated_mib:.1f}"
    )


def smoke_cbramod() -> None:
    checkpoint = Path("third_party/CBraMod/pretrained_weights/pretrained_weights.pth")
    model = CBraModAdapter(checkpoint).cuda().eval()
    inputs = torch.zeros(2, 16, 10, 200, device="cuda")
    with torch.inference_mode():
        output = model.forward_window(inputs)
    torch.cuda.synchronize()
    if output.shape != (2, 16, 128) or not torch.isfinite(output).all():
        raise AssertionError("CBraMod GPU smoke output failed its contract")
    allocated_mib = torch.cuda.max_memory_allocated() / 2**20
    print(
        f"CBRAMOD_GPU_OK shape={tuple(output.shape)} dtype={output.dtype} "
        f"peak_allocated_mib={allocated_mib:.1f}"
    )


def smoke_tcn_gat() -> None:
    model = MultiScaleTCNGAT().cuda().eval()
    inputs = torch.zeros(2, 16, 2560, device="cuda")
    quality = torch.zeros(2, 16, 3, device="cuda")
    with torch.inference_mode():
        output = model.forward_window(inputs, quality)
    torch.cuda.synchronize()
    if output.shape != (2, 16, 128) or not torch.isfinite(output).all():
        raise AssertionError("TCN-GAT GPU smoke output failed its contract")
    allocated_mib = torch.cuda.max_memory_allocated() / 2**20
    print(
        f"TCN_GAT_GPU_OK shape={tuple(output.shape)} dtype={output.dtype} "
        f"peak_allocated_mib={allocated_mib:.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["singlem", "cbramod", "tcn_gat"], required=True)
    args = parser.parse_args()
    if args.model == "singlem":
        smoke_singlem()
    elif args.model == "cbramod":
        smoke_cbramod()
    elif args.model == "tcn_gat":
        smoke_tcn_gat()


if __name__ == "__main__":
    main()
