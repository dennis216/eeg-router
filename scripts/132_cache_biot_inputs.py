from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from bfa.preprocessing.signal import load_canonical_raw_uv


ROOT = Path(__file__).resolve().parent.parent
PROFILE = "chb-mit-biot-router-minimal-v1"


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(tmp, path)


def cache_one(row: dict[str, object], data_root: str, output_root: str) -> dict[str, object]:
    started = time.monotonic()
    relative = Path(str(row["relative_path"]))
    root = Path(output_root)
    target = root / relative.with_suffix(".npy")
    metadata_target = root / "metadata" / relative.with_suffix(".json")
    if target.is_file() and metadata_target.is_file():
        metadata = json.loads(metadata_target.read_text())
        if metadata.get("profile") == PROFILE and metadata.get("edf_sha256") == row["sha256"]:
            return {"recording_id": row["recording_id"], "status": "cached", "seconds": 0.0}
        raise RuntimeError(f"provenance mismatch for existing BIOT cache {relative}")
    raw_uv, source_hz = load_canonical_raw_uv(Path(data_root) / relative)
    ratio = Fraction(200.0 / float(source_hz)).limit_denominator(10000)
    signal_200 = resample_poly(
        raw_uv.astype(np.float64), ratio.numerator, ratio.denominator, axis=-1
    ).astype(np.float32)
    if signal_200.shape[0] != 16 or not np.isfinite(signal_200).all():
        raise ValueError(f"invalid BIOT cache shape/values for {relative}: {signal_200.shape}")
    atomic_npy(target, signal_200)
    metadata_target.parent.mkdir(parents=True, exist_ok=True)
    metadata_target.write_text(
        json.dumps(
            {
                "profile": PROFILE,
                "recording_id": row["recording_id"],
                "relative_path": str(relative),
                "edf_sha256": row["sha256"],
                "source_sampling_hz": float(source_hz),
                "target_sampling_hz": 200.0,
                "resampling": "scipy.signal.resample_poly_antialias",
                "channels": [
                    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
                    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
                    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
                    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
                ],
                "dtype": "float32",
                "shape": list(signal_200.shape),
                "normalization_at_window_read": "abs_value_95th_quantile_per_channel",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {
        "recording_id": row["recording_id"],
        "status": "created",
        "seconds": time.monotonic() - started,
        "samples": int(signal_200.shape[-1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "manifests/recordings.parquet")
    ap.add_argument("--data-root", type=Path, default=Path("/mnt/d/EEGData/chbmit-1.0.0"))
    ap.add_argument("--output-root", type=Path, default=Path("/mnt/d/EEGData/bfa_biot_prest_input_v1"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    rows = pd.read_parquet(args.manifest).sort_values(["patient_id", "recording_id"], kind="stable")
    if args.limit is not None:
        rows = rows.head(args.limit)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results_path = ROOT / "outputs/reports/chb-mit-biot-router-minimal-v1/biot_cache_progress.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    payloads = rows.to_dict("records")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(cache_one, row, str(args.data_root), str(args.output_root)) for row in payloads]
        with results_path.open("a") as stream:
            for completed, future in enumerate(as_completed(futures), start=1):
                result = {"completed": completed, "total": len(futures), **future.result()}
                stream.write(json.dumps(result, sort_keys=True) + "\n")
                stream.flush()
                print(json.dumps(result, sort_keys=True), flush=True)
    summary = {"profile": PROFILE, "total": len(payloads), "completed": len(payloads), "status": "PASS"}
    (ROOT / "outputs/reports/chb-mit-biot-router-minimal-v1/biot_cache_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
