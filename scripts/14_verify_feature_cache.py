from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bfa.models.cbramod_adapter import CBraModAdapter
from bfa.models.precomputed import PrecomputedProjectionEncoder
from bfa.models.singlem_adapter import SingLEMAdapter


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["singlem", "cbramod"], required=True)
    args = parser.parse_args()
    torch.manual_seed(17)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    rate = 128 if args.model == "singlem" else 200
    dimension = 16 if args.model == "singlem" else 200
    extraction_batch = 256 if args.model == "singlem" else 32
    if args.model == "singlem":
        adapter = SingLEMAdapter(
            ROOT / "third_party/SingLEM/SingLEM/checkpoints/singlem_downstream_excluded.pt"
        )
    else:
        adapter = CBraModAdapter(
            ROOT / "third_party/CBraMod/pretrained_weights/pretrained_weights.pth"
        )
    projection = PrecomputedProjectionEncoder(dimension, adapter.projection)
    adapter = adapter.cuda().eval()
    projection = projection.cuda().eval()
    row = pd.read_parquet(ROOT / "manifests/recordings.parquet").iloc[0]
    relative = Path(row.relative_path)
    signal = np.load(
        Path("/mnt/d/EEGData/bfa_cache_bsn_pilot_v1")
        / args.model
        / relative.with_suffix(".npy"),
        mmap_mode="r",
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        signal, window_shape=10 * rate, axis=-1
    )[:, :: 2 * rate]
    inputs_np = np.ascontiguousarray(
        windows[:, :extraction_batch].transpose(1, 0, 2)
    ).reshape(
        extraction_batch, 16, 10, rate
    )
    cached = np.load(
        Path("/mnt/d/EEGData/bfa_features_bsn_pilot_v1")
        / args.model
        / relative.with_suffix(".npy"),
        mmap_mode="r",
    )[:extraction_batch].astype(np.float32)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        direct_features = adapter.backbone_features(torch.from_numpy(inputs_np).cuda())
        direct_output = adapter.projection(direct_features)
        cached_output = projection.forward_window(torch.from_numpy(cached).cuda())
    direct_features_np = direct_features.float().cpu().numpy()
    direct_output_np = direct_output.float().cpu().numpy()
    cached_output_np = cached_output.float().cpu().numpy()
    report = {
        "model": args.model,
        "recording_id": row.recording_id,
        "windows_checked": extraction_batch,
        "feature_max_abs_error": float(np.max(np.abs(direct_features_np - cached))),
        "feature_mean_abs_error": float(np.mean(np.abs(direct_features_np - cached))),
        "projected_max_abs_error": float(np.max(np.abs(direct_output_np - cached_output_np))),
        "projected_mean_abs_error": float(np.mean(np.abs(direct_output_np - cached_output_np))),
        "cached_dtype": "float32",
        "passed": True,
    }
    if report["feature_max_abs_error"] > 0.02 or report["projected_max_abs_error"] > 0.02:
        report["passed"] = False
        raise AssertionError(json.dumps(report, sort_keys=True))
    output = ROOT / "outputs/audits" / f"feature_cache_{args.model}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("FEATURE_CACHE_VERIFIED " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
