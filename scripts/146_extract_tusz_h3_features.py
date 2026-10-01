"""Extract the frozen CHB H3 event features from the external TUSZ FP events.

This script is deliberately feature-only: it never fits a scaler, selects a
model, changes a threshold, or uses TUSZ labels for selection.  It reconstructs
the same 15 event-level features used by the CHB GroupKFold H3 analysis from a
common TCN-GAT-style 16-channel, 256-Hz view.  Each EDF is loaded once and all
detector/checkpoint event rows for that recording are evaluated together.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bfa.preprocessing.tuh import (  # noqa: E402
    load_tuh_canonical_raw_uv,
    tuh_model_view,
    tuh_quality_features,
)


RUN_ID = "h3-external-tusz-v1"
SOURCE_RUN = "v3-groupkfold-confirmatory-v1"
EVENT_PATH = ROOT / "outputs" / "reports" / SOURCE_RUN / "tusz_groupkfold_external_v1_events.parquet"
COHORT_PATH = ROOT / "outputs" / "reports" / SOURCE_RUN / "tusz_groupkfold_external_manifest.json"
OUT_DIR = ROOT / "outputs" / "reports" / RUN_ID
if os.environ.get("BFA_OUTPUT_DIR"):
    OUT_DIR = Path(os.environ["BFA_OUTPUT_DIR"])
FEATURE_PATH = OUT_DIR / "tusz_h3_fp_features.parquet"
MANIFEST_PATH = OUT_DIR / "tusz_h3_fp_features_manifest.json"
SAMPLE_HZ = 256.0
DECIMATED_HZ = 128.0
FEATURES = (
    "event_duration_s",
    "peak_probability",
    "rms_median_uv",
    "rms_iqr_uv",
    "spatial_rms_dispersion",
    "spatial_rms_max_min_log",
    "line_noise_ratio",
    "clipping_fraction",
    "flat_fraction",
    "relative_delta_power",
    "relative_theta_power",
    "relative_alpha_power",
    "relative_beta_power",
    "relative_gamma_power",
    "high_low_ratio",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _band_masks() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    frequencies = np.fft.rfftfreq(256, d=1.0 / DECIMATED_HZ)
    valid = (frequencies >= 0.5) & (frequencies <= 45.0)
    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }
    masks = {
        name: valid & (frequencies >= low) & (frequencies < high)
        for name, (low, high) in bands.items()
    }
    return frequencies, masks


def _recording_features(item: tuple[str, list[dict[str, Any]], str]) -> dict[str, Any]:
    """Load one EDF and return its event rows plus a compact error record."""
    recording, records, absolute_path = item
    try:
        raw_uv, sampling_hz, raw_names = load_tuh_canonical_raw_uv(Path(absolute_path))
        view = tuh_model_view(raw_uv, sampling_hz, "tcn_gat")
        quality = tuh_quality_features(raw_uv, sampling_hz)
        if view.shape[0] != 16:
            raise RuntimeError(f"canonical view has {view.shape[0]} channels, expected 16")

        # The CHB feature cache uses a 256-Hz common view and 2x decimation for
        # a 256-sample (2-s) spectral window at 128 Hz.
        pad = 0
        if view.shape[1] < 512:
            pad = 512 - view.shape[1]
            left = pad // 2
            right = pad - left
            view = np.pad(view, ((0, 0), (left, right)), mode="edge")
        frequencies, masks = _band_masks()
        output: list[dict[str, Any]] = []
        mids = np.asarray([(float(row["start_s"]) + float(row["end_s"])) / 2.0 for row in records])
        centers = np.clip(np.rint(mids * SAMPLE_HZ).astype(np.int64) + pad // 2, 256, view.shape[1] - 256)
        offsets = np.arange(-256, 256, 2, dtype=np.int64)
        indices = centers[:, None] + offsets[None, :]
        signal = np.asarray(view[:, indices], dtype=np.float32).transpose(1, 0, 2)
        signal = signal - signal.mean(axis=2, keepdims=True)
        rms = np.sqrt(np.mean(signal * signal, axis=2) + 1e-8)
        rms_log = np.log1p(rms)
        spectrum = np.abs(np.fft.rfft(signal, axis=2)) ** 2
        valid = (frequencies >= 0.5) & (frequencies <= 45.0)
        total_power = spectrum[:, :, valid].sum(axis=2) + 1e-8
        qidx = np.clip(np.rint((mids - 1.0) / 2.0).astype(np.int64), 0, max(quality.shape[0] - 1, 0))
        q = np.asarray(quality[qidx], dtype=np.float32) if quality.size else np.zeros((len(records), 16, 3), dtype=np.float32)

        for index, row in enumerate(records):
            values: dict[str, Any] = {
                "model": str(row["model"]),
                "fold": int(row["fold"]),
                "seed": int(row["seed"]),
                "patient_id": str(row["patient_id"]),
                "recording": recording,
                "event_index": int(row["event_index"]),
                "start_s": float(row["start_s"]),
                "end_s": float(row["end_s"]),
                "event_duration_s": float(row["end_s"]) - float(row["start_s"]),
                "peak_probability": float(row["peak_probability"]),
                "edf_sampling_hz": float(sampling_hz),
                "canonical_channels": int(view.shape[0]),
                "feature_source": "tusz_common_tcn_gat_view_256hz_decimated128",
            }
            values["rms_median_uv"] = float(np.median(rms[index]))
            values["rms_iqr_uv"] = float(np.quantile(rms[index], 0.75) - np.quantile(rms[index], 0.25))
            values["spatial_rms_dispersion"] = float(np.std(rms_log[index]))
            values["spatial_rms_max_min_log"] = float(np.max(rms_log[index]) - np.min(rms_log[index]))
            values["line_noise_ratio"] = float(np.mean(q[index, :, 0]))
            values["clipping_fraction"] = float(np.mean(q[index, :, 1]))
            values["flat_fraction"] = float(np.mean(q[index, :, 2]))
            for name, mask in masks.items():
                band_power = spectrum[index][:, mask].sum(axis=1)
                values[f"relative_{name}_power"] = float(np.mean(band_power / total_power[index]))
            values["high_low_ratio"] = float(
                (values["relative_beta_power"] + values["relative_gamma_power"])
                / (values["relative_delta_power"] + values["relative_theta_power"] + 1e-8)
            )
            output.append(values)
        return {"recording": recording, "rows": output, "status": "ok", "channels": raw_names}
    except Exception as exc:  # retain a reproducible failure record and continue
        return {
            "recording": recording,
            "rows": [],
            "status": "error",
            "error": repr(exc),
            "traceback": traceback.format_exc(limit=8),
        }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if FEATURE_PATH.exists() or MANIFEST_PATH.exists():
        raise SystemExit(f"refusing to overwrite existing H3 external output in {OUT_DIR}")
    if not EVENT_PATH.exists() or not COHORT_PATH.exists():
        raise FileNotFoundError(f"missing frozen external inputs: {EVENT_PATH} / {COHORT_PATH}")

    events = pd.read_parquet(EVENT_PATH)
    required = {"model", "fold", "seed", "patient_id", "recording", "event_index", "start_s", "end_s", "peak_probability", "is_true_positive"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"external event table missing columns: {missing}")
    fp = events[~events["is_true_positive"].astype(bool)].copy()
    cohort = json.loads(COHORT_PATH.read_text())
    dataset_root = Path(str(cohort["dataset_root"]))
    records = cohort["records"]
    path_info = {str(item["relative_edf"]): item for item in records}

    jobs: list[tuple[str, list[dict[str, Any]], str]] = []
    missing_paths: list[str] = []
    for recording, frame in fp.groupby("recording", sort=True):
        if recording not in path_info:
            missing_paths.append(str(recording))
            continue
        absolute = dataset_root / str(recording)
        if not absolute.exists():
            missing_paths.append(str(absolute))
            continue
        jobs.append((str(recording), frame.to_dict("records"), str(absolute)))
    limit = int(os.environ.get("BFA_LIMIT_RECORDINGS", "0"))
    if limit > 0:
        jobs = jobs[:limit]
        # A smoke run is intentionally not allowed to write into the formal
        # namespace; the caller should point BFA_ROOT at a temporary root.
    if missing_paths:
        raise FileNotFoundError(f"{len(missing_paths)} external recordings missing from frozen cohort/files; first={missing_paths[:3]}")

    workers = max(1, int(os.environ.get("BFA_WORKERS", "4")))
    print(json.dumps({"run_id": RUN_ID, "fp_rows": len(fp), "recordings": len(jobs), "workers": workers}, sort_keys=True), flush=True)
    results: list[dict[str, Any]] = []
    if workers == 1:
        for index, job in enumerate(jobs, start=1):
            results.append(_recording_features(job))
            if index % 25 == 0 or index == len(jobs):
                print(f"TUSZ_H3_FEATURES_PROGRESS recordings={index}/{len(jobs)}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_recording_features, job) for job in jobs]
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                results.append(future.result())
                if index % 25 == 0 or index == len(jobs):
                    print(f"TUSZ_H3_FEATURES_PROGRESS recordings={index}/{len(jobs)}", flush=True)

    rows = [row for result in results for row in result.get("rows", [])]
    failures = [result for result in results if result.get("status") != "ok"]
    if failures:
        failure_path = OUT_DIR / "feature_extraction_failures.json"
        failure_path.write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"feature extraction failed for {len(failures)} recordings; see {failure_path}")
    frame = pd.DataFrame(rows)
    limit = int(os.environ.get("BFA_LIMIT_RECORDINGS", "0"))
    expected_events = fp if limit <= 0 else fp[fp["recording"].isin([job[0] for job in jobs])]
    if len(frame) != len(expected_events):
        raise RuntimeError(f"feature row count {len(frame)} != expected FP event row count {len(expected_events)}")
    frame = frame.sort_values(["patient_id", "recording", "model", "fold", "seed", "event_index"], kind="mergesort").reset_index(drop=True)
    frame.to_parquet(FEATURE_PATH, index=False)
    manifest = {
        "analysis_id": "h3-external-tusz-v1",
        "source_run": SOURCE_RUN,
        "source_events": str(EVENT_PATH.relative_to(ROOT)),
        "source_events_sha256": sha256(EVENT_PATH),
        "cohort_manifest": str(COHORT_PATH.relative_to(ROOT)),
        "cohort_manifest_sha256": sha256(COHORT_PATH),
        "dataset_root": str(dataset_root),
        "rows": int(len(frame)),
        "recordings": int(frame["recording"].nunique()),
        "patients": int(frame["patient_id"].nunique()),
        "models": sorted(frame["model"].unique().tolist()),
        "folds": sorted(frame["fold"].unique().tolist()),
        "seeds": sorted(frame["seed"].unique().tolist()),
        "fp_only": True,
        "smoke_run": bool(limit > 0),
        "feature_names": list(FEATURES),
        "feature_source": "common TUSZ canonical bipolar view, TCN-GAT bandpass 0.5-45 Hz at 256 Hz, 2x decimated 2-s window for signal features; frozen TUH quality features for QC columns",
        "fit_or_selection_performed": False,
        "external_seizure_labels_used_for_selection": False,
        "output_sha256": sha256(FEATURE_PATH),
        "failed_recordings": 0,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
