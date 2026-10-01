"""Build the missing TCN-GAT signal and quality cache for H4 revision v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from bfa.preprocessing.signal import load_canonical_raw_uv, model_view


ROOT = Path("/root/b_false_alarm_atlas")
DATA = Path("/mnt/d/EEGData/chbmit-1.0.0")
OUT = ROOT / "cache" / "h4-router-revision-v1-input"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)


def quality_features(raw_uv: np.ndarray, sampling_hz: float) -> np.ndarray:
    window_samples = int(round(10 * sampling_hz))
    stride_samples = int(round(2 * sampling_hz))
    starts = np.arange(0, raw_uv.shape[-1] - window_samples + 1, stride_samples)
    output = np.empty((len(starts), raw_uv.shape[0], 3), dtype=np.float32)
    frequencies = np.fft.rfftfreq(window_samples, d=1.0 / sampling_hz)
    peak = (frequencies >= 59) & (frequencies <= 61)
    shoulders = ((frequencies >= 55) & (frequencies < 59)) | ((frequencies > 61) & (frequencies <= 65))
    channel_extreme = np.quantile(np.abs(raw_uv), 0.999, axis=-1)
    for offset in range(0, len(starts), 128):
        batch_starts = starts[offset : offset + 128]
        windows = np.stack([raw_uv[:, start : start + window_samples] for start in batch_starts])
        differences = np.diff(windows, axis=-1)
        flat_fraction = np.mean(differences == 0, axis=-1)
        extreme = np.abs(windows[..., :-1]) >= channel_extreme[None, :, None]
        clipping_fraction = np.mean((differences == 0) & extreme, axis=-1)
        spectrum = np.abs(np.fft.rfft(windows, axis=-1)) ** 2
        line_noise = spectrum[..., peak].sum(axis=-1) / (
            spectrum[..., shoulders].sum(axis=-1) + np.finfo(np.float32).eps
        )
        output[offset : offset + len(batch_starts), :, 0] = line_noise
        output[offset : offset + len(batch_starts), :, 1] = clipping_fraction
        output[offset : offset + len(batch_starts), :, 2] = flat_fraction
    return output


def build_one(row: dict[str, object]) -> dict[str, object]:
    started = time.monotonic()
    relative = Path(str(row["relative_path"]))
    tcn_target = OUT / "tcn_gat" / relative.with_suffix(".npy")
    quality_target = OUT / "quality" / relative.with_suffix(".npy")
    metadata_target = OUT / "metadata" / relative.with_suffix(".json")
    if tcn_target.exists() and quality_target.exists() and metadata_target.exists():
        meta = json.loads(metadata_target.read_text())
        if meta.get("edf_sha256") != row["sha256"]:
            raise RuntimeError(f"EDF provenance mismatch: {relative}")
        return {"recording_id": row["recording_id"], "status": "cached", "seconds": 0.0}
    raw_uv, sampling_hz = load_canonical_raw_uv(DATA / relative)
    quality = quality_features(raw_uv, sampling_hz)
    notched = mne.filter.notch_filter(raw_uv.astype(np.float64), sampling_hz, freqs=[60.0], verbose="ERROR")
    filtered = mne.filter.filter_data(notched, sampling_hz, 0.5, 45.0, verbose="ERROR").astype(np.float32)
    tcn = model_view(filtered, sampling_hz, "tcn_gat")
    atomic_npy(tcn_target, tcn)
    atomic_npy(quality_target, quality)
    metadata_target.parent.mkdir(parents=True, exist_ok=True)
    metadata_target.write_text(
        json.dumps(
            {
                "analysis_id": "h4-router-revision-v1",
                "recording_id": row["recording_id"],
                "relative_path": str(relative),
                "edf_sha256": row["sha256"],
                "sampling_hz": sampling_hz,
                "tcn_gat": {"notch_hz": 60.0, "bandpass_hz": [0.5, 45.0], "sampling_hz": 256.0, "scale": 1.0},
                "quality_columns": ["line_noise_ratio", "clipping_fraction", "flat_fraction"],
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
    )
    return {"recording_id": row["recording_id"], "status": "created", "seconds": time.monotonic() - started}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    recordings_path = ROOT / "manifests" / "recordings.parquet"
    rows = pd.read_parquet(recordings_path).sort_values(["patient_id", "recording_id"], kind="stable")
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build_one, row) for row in rows.to_dict("records")]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            print(json.dumps({"completed": completed, "total": len(futures), **result}), flush=True)
    manifest = {
        "analysis_id": "h4-router-revision-v1-input-cache",
        "status": "complete",
        "recordings": len(rows),
        "source_recordings_sha256": sha256(recordings_path),
        "output_root": str(OUT),
        "contents": ["tcn_gat", "quality"],
    }
    (OUT / "cache_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("H4_REVISION_CACHE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
