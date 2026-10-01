from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from bfa.preprocessing.signal import load_canonical_raw_uv, model_view


def _quality_features(raw_uv: np.ndarray, sampling_hz: float) -> np.ndarray:
    window_samples = int(round(10 * sampling_hz))
    stride_samples = int(round(2 * sampling_hz))
    starts = np.arange(0, raw_uv.shape[-1] - window_samples + 1, stride_samples)
    output = np.empty((len(starts), raw_uv.shape[0], 3), dtype=np.float32)
    frequencies = np.fft.rfftfreq(window_samples, d=1.0 / sampling_hz)
    peak = (frequencies >= 59) & (frequencies <= 61)
    shoulders = ((frequencies >= 55) & (frequencies < 59)) | (
        (frequencies > 61) & (frequencies <= 65)
    )
    channel_extreme = np.quantile(np.abs(raw_uv), 0.999, axis=-1)
    for offset in range(0, len(starts), 128):
        batch_starts = starts[offset : offset + 128]
        windows = np.stack(
            [raw_uv[:, start : start + window_samples] for start in batch_starts]
        )
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


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)


def _cache_recording(row: dict[str, object], data_root: str, cache_root: str) -> dict[str, object]:
    started = time.monotonic()
    relative = Path(str(row["relative_path"]))
    cache = Path(cache_root)
    targets = {
        model: cache / model / relative.with_suffix(".npy")
        for model in ("singlem", "cbramod", "tcn_gat")
    }
    quality_target = cache / "quality" / relative.with_suffix(".npy")
    metadata_target = cache / "metadata" / relative.with_suffix(".json")
    if all(targets[model].exists() for model in ("cbramod", "tcn_gat")) and all(
        path.exists() for path in (quality_target, metadata_target)
    ):
        return {"recording_id": row["recording_id"], "status": "cached", "seconds": 0.0}
    raw_uv, sampling_hz = load_canonical_raw_uv(Path(data_root) / relative)
    quality = _quality_features(raw_uv, sampling_hz)
    import mne

    filtered = mne.filter.notch_filter(
        raw_uv.astype(np.float64), sampling_hz, freqs=[60.0], verbose="ERROR"
    )
    filtered = mne.filter.filter_data(
        filtered, sampling_hz, 0.5, 45.0, verbose="ERROR"
    ).astype(np.float32)
    for model in ("cbramod", "tcn_gat"):
        _atomic_npy(targets[model], model_view(filtered, sampling_hz, model))
    singlem_unclipped = resample_poly(filtered, 1, 2, axis=-1).astype(np.float32) * 0.01
    singlem_clip_fraction = float(np.mean(np.abs(singlem_unclipped) > 1.5))
    singlem_status = "protocol_blocked_clipping"
    if singlem_clip_fraction <= 0.01 and np.isfinite(singlem_unclipped).all():
        _atomic_npy(targets["singlem"], np.clip(singlem_unclipped, -1.5, 1.5))
        singlem_status = "cached"
    _atomic_npy(quality_target, quality)
    metadata = {
        "recording_id": row["recording_id"],
        "relative_path": str(relative),
        "edf_sha256": row["sha256"],
        "sampling_hz": sampling_hz,
        "singlem_clip_fraction": singlem_clip_fraction,
        "singlem_status": singlem_status,
        "quality_columns": ["line_noise_ratio", "clipping_fraction", "flat_fraction"],
    }
    metadata_target.parent.mkdir(parents=True, exist_ok=True)
    metadata_target.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {
        "recording_id": row["recording_id"],
        "status": "created" if singlem_status == "cached" else singlem_status,
        "seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/recordings.parquet"))
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/d/EEGData/chbmit-1.0.0"))
    parser.add_argument("--cache-root", type=Path, default=Path("/mnt/d/EEGData/bfa_cache_v3"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = pd.read_parquet(args.manifest).sort_values(
        ["patient_id", "recording_id"], kind="stable"
    )
    if args.limit is not None:
        rows = rows.head(args.limit)
    payloads = rows.to_dict("records")
    args.cache_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_cache_recording, row, str(args.data_root), str(args.cache_root))
            for row in payloads
        ]
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            print(
                json.dumps({"completed": completed, "total": len(payloads), **result}),
                flush=True,
            )


if __name__ == "__main__":
    main()
