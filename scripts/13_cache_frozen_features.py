from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bfa.models.cbramod_adapter import CBraModAdapter
from bfa.models.singlem_adapter import SingLEMAdapter

ROOT = Path(__file__).resolve().parent.parent
INPUT_ROOT = Path("/mnt/d/EEGData/bfa_cache_bsn_pilot_v1")
OUTPUT_ROOT = Path("/mnt/d/EEGData/bfa_features_bsn_pilot_v1")


def _adapter(model: str) -> torch.nn.Module:
    if model == "singlem":
        return SingLEMAdapter(
            ROOT / "third_party/SingLEM/SingLEM/checkpoints/singlem_downstream_excluded.pt"
        )
    return CBraModAdapter(ROOT / "third_party/CBraMod/pretrained_weights/pretrained_weights.pth")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["singlem", "cbramod"], required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--profile", default="BSN-pilot-v1")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    rate = 128 if args.model == "singlem" else 200
    feature_dim = 16 if args.model == "singlem" else 200
    batch_size = args.batch_size or (256 if args.model == "singlem" else 32)
    adapter = _adapter(args.model).cuda().eval()
    adapter.projection.cpu()
    manifest = pd.read_parquet(ROOT / "manifests/recordings.parquet").sort_values(
        ["patient_id", "recording_id"], kind="stable"
    )
    config_sha256 = None
    if args.config is not None:
        config_sha256 = hashlib.sha256(args.config.read_bytes()).hexdigest()
    model_root = args.output_root / args.model
    metadata_root = args.output_root / "metadata" / args.model
    model_root.mkdir(parents=True, exist_ok=True)
    total_windows = 0
    started = time.monotonic()
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    with torch.inference_mode():
        for completed, row in enumerate(manifest.itertuples(index=False), start=1):
            relative = Path(row.relative_path)
            source = args.input_root / args.model / relative.with_suffix(".npy")
            target = model_root / relative.with_suffix(".npy")
            metadata_target = metadata_root / relative.with_suffix(".json")
            if target.exists() and metadata_target.exists():
                existing = json.loads(metadata_target.read_text())
                existing_profile = existing.get("profile", existing.get("pilot"))
                if (
                    existing_profile != args.profile
                    or existing.get("preprocess_config_sha256") != config_sha256
                ):
                    raise RuntimeError(
                        f"feature-cache provenance mismatch for {relative}; use a separate root"
                    )
                features = np.load(target, mmap_mode="r")
                total_windows += features.shape[0]
                continue
            signal = np.load(source, mmap_mode="r", allow_pickle=False)
            window_samples = 10 * rate
            stride_samples = 2 * rate
            windows = np.lib.stride_tricks.sliding_window_view(
                signal, window_shape=window_samples, axis=-1
            )[:, ::stride_samples]
            number_of_windows = windows.shape[1]
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
            output = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=np.float32,
                shape=(number_of_windows, 16, feature_dim),
            )
            recording_started = time.monotonic()
            for offset in range(0, number_of_windows, batch_size):
                batch = np.ascontiguousarray(
                    windows[:, offset : offset + batch_size].transpose(1, 0, 2)
                ).reshape(-1, 16, 10, rate)
                inputs = torch.from_numpy(batch).cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    features = adapter.backbone_features(inputs)
                output[offset : offset + len(batch)] = features.float().cpu().numpy()
            output.flush()
            del output
            os.replace(temporary, target)
            metadata_target.parent.mkdir(parents=True, exist_ok=True)
            metadata_target.write_text(
                json.dumps(
                    {
                        "profile": args.profile,
                        "preprocess_config_sha256": config_sha256,
                        "source_signal_cache": str(args.input_root),
                        "model": args.model,
                        "relative_path": str(relative),
                        "edf_sha256": row.sha256,
                        "windows": number_of_windows,
                        "channels": 16,
                        "feature_dim": feature_dim,
                        "dtype": "float32",
                        "stride_s": 2,
                        "window_s": 10,
                        "label_free": True,
                        "frozen_backbone": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            total_windows += number_of_windows
            print(
                f"FEATURE_CACHE_PROGRESS model={args.model} recordings={completed}/{len(manifest)} "
                f"recording_windows={number_of_windows} seconds={time.monotonic() - recording_started:.2f}",
                flush=True,
            )
    print(
        f"FEATURE_CACHE_COMPLETE model={args.model} windows={total_windows} "
        f"seconds={time.monotonic() - started:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
