"""Build same-window competence labels and EEG10 features for H6.

For every frozen H6 decision endpoint, the input is the completed causal
10-second EEG window ending at time t.  Each expert label records whether its
frozen thresholded decision at t agrees with seizure overlap in that same
10-second window.  This is distinct from the older future-60-second H6 label.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
NAMESPACE = "h6-eeg-conditionable-competence-probe-v1"
OUT = ROOT / "outputs/reports" / NAMESPACE / "same_window"
OLD_LABELS = ROOT / "outputs/reports/h6-causal-policy-router-v2-corrected/labels"
CACHE_ROOT = ROOT / "outputs/reports/h6-causal-policy-router-v2"
EXPERT_ROOT = ROOT / "outputs/tuh_external/h6_tusz_raw_pre_router_v1"
MANIFEST_ROOT = ROOT / "outputs/reports/h6-tusz-raw-pre-router-v1/manifests"
MODELS = ("singlem", "cbramod", "tcn_gat")
FEATURES = (
    "rms_median_uv", "rms_iqr_uv", "spatial_rms_dispersion",
    "spatial_rms_max_min_log", "relative_delta_power",
    "relative_theta_power", "relative_alpha_power", "relative_beta_power",
    "relative_gamma_power", "high_low_ratio",
)
FS = 128.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def eeg10_batch(raw: np.ndarray) -> np.ndarray:
    """Compute the frozen ten EEG features for N x 16 x 1280 arrays."""
    x = np.asarray(raw, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (16, 1280) or not np.isfinite(x).all():
        raise RuntimeError(f"invalid raw EEG batch {x.shape}")
    x = x - x.mean(axis=2, keepdims=True)
    rms = np.sqrt(np.mean(x * x, axis=2) + 1e-8)
    log_rms = np.log1p(rms)
    spectrum = np.abs(np.fft.rfft(x, axis=2)) ** 2
    frequencies = np.fft.rfftfreq(x.shape[2], d=1.0 / FS)
    valid = (frequencies >= 0.5) & (frequencies <= 45.0)
    total = spectrum[:, :, valid].sum(axis=2) + 1e-8

    values = [
        np.median(rms, axis=1),
        np.quantile(rms, 0.75, axis=1) - np.quantile(rms, 0.25, axis=1),
        np.std(log_rms, axis=1),
        np.max(log_rms, axis=1) - np.min(log_rms, axis=1),
    ]
    bands: list[np.ndarray] = []
    for low, high in ((0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)):
        mask = valid & (frequencies >= low) & (frequencies < high)
        bands.append(np.mean(spectrum[:, :, mask].sum(axis=2) / total, axis=1))
    values.extend(bands)
    values.append((bands[3] + bands[4]) / (bands[0] + bands[1] + 1e-8))
    result = np.stack(values, axis=1).astype(np.float32)
    if result.shape[1] != len(FEATURES) or not np.isfinite(result).all():
        raise RuntimeError("non-finite EEG10 output")
    return result


def overlaps_seizure(start: float, end: float, events: list[dict[str, object]]) -> bool:
    return any(float(event["end_s"]) > start and float(event["start_s"]) < end for event in events)


def probability_at_times(path: Path, times: np.ndarray) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, columns=["window_end_s", "probability"])
    lookup = {round(float(t), 3): float(p) for t, p in zip(frame.window_end_s, frame.probability)}
    values = np.asarray([lookup.get(round(float(t), 3), np.nan) for t in times], dtype=np.float32)
    if not np.isfinite(values).all():
        missing = times[~np.isfinite(values)].tolist()
        raise RuntimeError(f"missing probability endpoints in {path}: {missing[:10]}")
    return values


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    mask_counts = frame.correct_mask.value_counts().sort_index()
    seizure = frame.current_has_seizure.astype(bool)
    return {
        "rows": int(len(frame)),
        "patients": int(frame.patient_id.nunique()),
        "records": int(frame.recording.nunique()),
        "current_seizure_rows": int(seizure.sum()),
        "background_rows": int((~seizure).sum()),
        "actionable_rows": int(frame.actionable.sum()),
        "all_correct_rows": int(frame.all_correct.sum()),
        "all_wrong_rows": int(frame.all_wrong.sum()),
        "mask_counts": {str(key): int(value) for key, value in mask_counts.items()},
        "correct_prevalence": {
            model: float(frame[f"correct_{model}"].mean()) for model in MODELS
        },
        "positive_prediction_prevalence": {
            model: float(frame[f"prediction_{model}"].mean()) for model in MODELS
        },
    }


def build_partition(partition: str) -> tuple[pd.DataFrame, dict[str, object]]:
    old_path = OLD_LABELS / f"{partition}_causal_policy.parquet"
    manifest_path = MANIFEST_ROOT / f"{partition}_manifest.json"
    old = pd.read_parquet(old_path).sort_values(["recording", "decision_time_s"]).reset_index(drop=True)
    manifest = json.loads(manifest_path.read_text())
    record_map = {str(record["relative_edf"]): record for record in manifest["records"]}
    parts: list[pd.DataFrame] = []

    for record_index, (recording, group) in enumerate(old.groupby("recording", sort=False), start=1):
        if recording not in record_map:
            raise RuntimeError(f"recording absent from manifest: {recording}")
        group = group.copy().reset_index(drop=True)
        times = group.decision_time_s.to_numpy(dtype=float)
        context_start = times - 10.0
        events = list(record_map[recording].get("seizure_events", []))
        group["current_start_s"] = context_start
        group["current_end_s"] = times
        group["current_has_seizure"] = [
            int(overlaps_seizure(float(start), float(end), events))
            for start, end in zip(context_start, times)
        ]

        cache_path = CACHE_ROOT / str(group.raw_cache_path_v2.iloc[0])
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
        with np.load(cache_path, allow_pickle=False) as payload:
            indices = group.decision_index_v1.to_numpy(dtype=int)
            raw = payload["x"][indices].astype(np.float32)
        features = eeg10_batch(raw)
        for feature_index, feature_name in enumerate(FEATURES):
            group[feature_name] = features[:, feature_index]

        truth = group.current_has_seizure.to_numpy(dtype=int)
        for model in MODELS:
            probability_path = (
                EXPERT_ROOT / partition / model / "fold0_seed17" / "probabilities"
                / Path(recording).with_suffix(".parquet")
            )
            probability = probability_at_times(probability_path, times)
            threshold_values = group[f"threshold_{model}"].to_numpy(dtype=float)
            if not np.allclose(threshold_values, threshold_values[0]):
                raise RuntimeError(f"non-constant threshold for {model} in {recording}")
            prediction = (probability >= threshold_values).astype(np.int8)
            group[f"probability_{model}"] = probability
            group[f"prediction_{model}"] = prediction
            group[f"correct_{model}"] = (prediction == truth).astype(np.int8)

        correctness = group[[f"correct_{model}" for model in MODELS]].astype(int)
        group["correct_mask"] = correctness.astype(str).agg("".join, axis=1)
        group["correct_count"] = correctness.sum(axis=1).astype(np.int8)
        group["actionable"] = group.correct_count.between(1, 2)
        group["all_correct"] = group.correct_count.eq(3)
        group["all_wrong"] = group.correct_count.eq(0)
        group["label_contract"] = "same_completed_10s_window"
        parts.append(group)
        if record_index % 500 == 0:
            print(f"{partition}: {record_index}/{old.recording.nunique()} recordings", flush=True)

    result = pd.concat(parts, ignore_index=True)
    return result, {
        "source_old_label_sha256": sha256(old_path),
        "source_manifest_sha256": sha256(manifest_path),
        **summarize(result),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite completed output: {manifest_path}")
    partitions: dict[str, dict[str, object]] = {}
    output_hashes: dict[str, str] = {}
    for partition in ("train", "dev"):
        frame, audit = build_partition(partition)
        output = OUT / f"{partition}_same_window_competence_eeg10.parquet"
        frame.to_parquet(output, index=False)
        output_hashes[str(output.relative_to(ROOT))] = sha256(output)
        partitions[partition] = audit

    train = pd.read_parquet(OUT / "train_same_window_competence_eeg10.parquet", columns=["patient_id"])
    dev = pd.read_parquet(OUT / "dev_same_window_competence_eeg10.parquet", columns=["patient_id"])
    patient_overlap = sorted(set(train.patient_id.astype(str)) & set(dev.patient_id.astype(str)))
    payload = {
        "release_id": NAMESPACE,
        "stage": "same_window_label_and_eeg10_build",
        "status": "PASS" if not patient_overlap else "FAIL_PATIENT_OVERLAP",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "eval_read": False,
        "model_training": False,
        "expert_training": False,
        "threshold_condition": "existing_frozen_thresholds",
        "context": "16 channels x 1280 samples, 10 s completed causal EEG",
        "target": "thresholded expert decision correctness on the same completed 10-s window",
        "patient_overlap_train_dev": patient_overlap,
        "feature_names": list(FEATURES),
        "partitions": partitions,
        "output_hashes": output_hashes,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (OUT / "manifest.sha256").write_text(f"{sha256(manifest_path)}  manifest.json\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if patient_overlap:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
