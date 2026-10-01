"""Extract frozen signal/QC features for validation and test FP events.

Feature extraction is deterministic and does not fit a scaler, PCA, cluster, or
classifier. Those choices are made in the downstream H2/H3 script using
validation events only, then applied to test events.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = Path("/mnt/d/EEGData/bfa_cache_v3/tcn_gat")
QUALITY_ROOT = Path("/mnt/d/EEGData/bfa_cache_v3/quality")
EVENT_DIR = ROOT / "outputs" / "events"
REPORT_DIR = ROOT / "outputs" / "reports" / "v3-compact-extension-v1"
SAMPLE_HZ = 256.0
DECIMATED_HZ = 128.0
BATCH = 128


def _band_matrix() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    frequencies = np.fft.rfftfreq(256, d=1.0 / DECIMATED_HZ)
    valid = (frequencies >= 0.5) & (frequencies <= 45.0)
    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }
    masks = {name: valid & (frequencies >= low) & (frequencies < high) for name, (low, high) in bands.items()}
    return frequencies, masks


def _features_for_recording(frame: pd.DataFrame, relative_path: str) -> pd.DataFrame:
    raw = np.load(RAW_ROOT / Path(relative_path).with_suffix(".npy"), mmap_mode="r")
    quality = np.load(QUALITY_ROOT / Path(relative_path).with_suffix(".npy"), mmap_mode="r")
    frequencies, masks = _band_matrix()
    output: list[dict[str, object]] = []
    rows = frame.reset_index(drop=True)
    for start in range(0, len(rows), BATCH):
        batch = rows.iloc[start : start + BATCH]
        mid = ((batch.start_s.to_numpy(float) + batch.end_s.to_numpy(float)) / 2.0)
        centers = np.clip(np.rint(mid * SAMPLE_HZ).astype(np.int64), 256, raw.shape[1] - 256)
        offsets = np.arange(-256, 256, 2, dtype=np.int64)
        indices = centers[:, None] + offsets[None, :]
        # raw is [channel, sample]; advanced indexing produces [channel, event, sample].
        signal = np.asarray(raw[:, indices], dtype=np.float32).transpose(1, 0, 2)
        signal = signal - signal.mean(axis=2, keepdims=True)
        rms = np.sqrt(np.mean(signal * signal, axis=2) + 1e-8)
        rms_log = np.log1p(rms)
        spectrum = np.abs(np.fft.rfft(signal, axis=2)) ** 2
        total_power = spectrum[:, :, (frequencies >= 0.5) & (frequencies <= 45.0)].sum(axis=2) + 1e-8
        qidx = np.clip(np.rint((mid - 1.0) / 2.0).astype(np.int64), 0, quality.shape[0] - 1)
        q = np.asarray(quality[qidx], dtype=np.float32)
        for index, row in enumerate(batch.itertuples(index=False)):
            values: dict[str, object] = {
                "partition": row.partition,
                "split": int(row.split),
                "seed": int(row.seed),
                "model": row.model,
                "namespace": row.namespace,
                "patient": row.patient,
                "recording": row.recording,
                "event_type": row.event_type,
                "start_s": float(row.start_s),
                "end_s": float(row.end_s),
                "event_duration_s": float(row.end_s - row.start_s),
                "peak_probability": float(row.peak_probability),
                "rms_median_uv": float(np.median(rms[index])),
                "rms_iqr_uv": float(np.quantile(rms[index], 0.75) - np.quantile(rms[index], 0.25)),
                "spatial_rms_dispersion": float(np.std(rms_log[index])),
                "spatial_rms_max_min_log": float(np.max(rms_log[index]) - np.min(rms_log[index])),
                "line_noise_ratio": float(np.mean(q[index, :, 0])),
                "clipping_fraction": float(np.mean(q[index, :, 1])),
                "flat_fraction": float(np.mean(q[index, :, 2])),
            }
            for name, mask in masks.items():
                band_power = spectrum[index][:, mask].sum(axis=1)
                values[f"relative_{name}_power"] = float(np.mean(band_power / total_power[index]))
            values["high_low_ratio"] = float(
                (values["relative_beta_power"] + values["relative_gamma_power"])
                / (values["relative_delta_power"] + values["relative_theta_power"] + 1e-8)
            )
            output.append(values)
    return pd.DataFrame(output)


def main() -> None:
    windows = pd.read_parquet(ROOT / "manifests" / "windows.parquet")
    relative_by_recording = windows.drop_duplicates("recording").set_index("recording")["relative_path"].to_dict()
    validation_path = EVENT_DIR / "v3-compact-extension-v1_validation_events.parquet"
    test_path = EVENT_DIR / "v3-compact-extension-v1_test_events.parquet"
    validation = pd.read_parquet(validation_path)
    test = pd.read_parquet(test_path)
    validation["partition"] = "validation"
    test["partition"] = "test"
    frame = pd.concat([validation, test], ignore_index=True)
    frame = frame[frame.event_type == "fp"].copy()
    feature_frames: list[pd.DataFrame] = []
    for recording, group in frame.groupby("recording", sort=True):
        feature_frames.append(_features_for_recording(group, str(relative_by_recording[recording])))
    features = pd.concat(feature_frames, ignore_index=True)
    output_path = EVENT_DIR / "v3-compact-extension-v1_fp_features.parquet"
    features.to_parquet(output_path, index=False)
    summary = {
        "release_id": "v3-compact-extension-v1",
        "rows": int(len(features)),
        "validation_rows": int((features.partition == "validation").sum()),
        "test_rows": int((features.partition == "test").sum()),
        "recordings": int(features.recording.nunique()),
        "feature_columns": [column for column in features.columns if column not in {"partition", "split", "seed", "model", "namespace", "patient", "recording", "event_type"}],
        "fit_or_selection_performed": False,
        "source": "frozen tcn_gat signal cache and frozen quality cache",
        "event_tables": [str(validation_path), str(test_path)],
    }
    (REPORT_DIR / "fp_feature_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
