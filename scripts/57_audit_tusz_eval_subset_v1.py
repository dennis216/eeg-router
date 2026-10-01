"""Audit patient waterfall, window calibration, and signal QC for frozen TUSZ subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from bfa.evaluation.eventize import eventize
from bfa.evaluation.match import match_events
from bfa.preprocessing.tuh import load_tuh_canonical_raw_uv, tuh_quality_features


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports" / "v3-compact-extension-v1"
EXTERNAL_ROOT = ROOT / "outputs" / "tuh_external" / "tusz_eval_subset_v1"
MANIFEST_PATH = REPORT_DIR / "tusz_eval_subset_v1_manifest.json"
MODELS = ("singlem", "cbramod", "tcn_gat")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def calibration_bins(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> list[dict[str, float | int]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result: list[dict[str, float | int]] = []
    for index in range(bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        if index == bins - 1:
            mask = (probability >= lower) & (probability <= upper)
        else:
            mask = (probability >= lower) & (probability < upper)
        if not np.any(mask):
            result.append({"bin": index, "lower": lower, "upper": upper, "count": 0, "mean_probability": math.nan, "empirical_rate": math.nan, "abs_gap": math.nan})
            continue
        mean_probability = float(np.mean(probability[mask]))
        empirical_rate = float(np.mean(y_true[mask]))
        result.append({"bin": index, "lower": lower, "upper": upper, "count": int(np.sum(mask)), "mean_probability": mean_probability, "empirical_rate": empirical_rate, "abs_gap": abs(mean_probability - empirical_rate)})
    return result


def window_labels(ends: np.ndarray, truth: list[tuple[float, float]]) -> np.ndarray:
    labels = np.zeros(len(ends), dtype=np.int8)
    for index, end in enumerate(ends):
        window_start = float(end) - 10.0
        window_end = float(end)
        labels[index] = int(any(min(window_end, truth_end) - max(window_start, truth_start) >= 1.0 for truth_start, truth_end in truth))
    return labels


def finite_median(values: list[float]) -> float:
    return float(np.median(values)) if values else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--external-root", type=Path, default=EXTERNAL_ROOT)
    parser.add_argument("--prefix", default="tusz_eval_subset_v1")
    args = parser.parse_args()
    started = time.monotonic()
    manifest_path = args.manifest.resolve()
    external_root = args.external_root.resolve()
    prefix = args.prefix
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = sha256(manifest_path)
    patient_rows: list[dict[str, object]] = []
    record_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    calibration_bin_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    qc_cache: dict[str, dict[str, object]] = {}

    for record_index, record in enumerate(manifest["records"], start=1):
        relative = Path(record["relative_edf"])
        raw_path = Path(manifest["dataset_root"]) / relative
        raw_uv, source_hz, _ = load_tuh_canonical_raw_uv(raw_path)
        abs_signal = np.abs(raw_uv)
        quality = tuh_quality_features(raw_uv, source_hz)
        qc = {
            "patient_id": record["patient_id"],
            "recording": relative.as_posix(),
            "duration_s": float(record["duration_s"]),
            "sampling_hz": float(source_hz),
            "p99_abs_uv": float(np.quantile(abs_signal, 0.99)),
            "p999_abs_uv": float(np.quantile(abs_signal, 0.999)),
            "max_abs_uv": float(np.max(abs_signal)),
            "fraction_abs_gt_100_uv": float(np.mean(abs_signal > 100.0)),
            "fraction_abs_gt_1000_uv": float(np.mean(abs_signal > 1000.0)),
            "mean_line_noise_ratio": float(np.mean(quality[:, :, 0])) if len(quality) else math.nan,
            "mean_clipping_fraction": float(np.mean(quality[:, :, 1])) if len(quality) else math.nan,
            "mean_flat_fraction": float(np.mean(quality[:, :, 2])) if len(quality) else math.nan,
            "quality_windows": int(len(quality)),
        }
        qc_rows.append(qc)
        qc_cache[relative.as_posix()] = qc
        del raw_uv, abs_signal, quality
        for model in MODELS:
            probability_path = external_root / model / "probabilities" / relative.with_suffix(".parquet")
            metadata_path = external_root / model / "metadata" / relative.with_suffix(".json")
            table = pd.read_parquet(probability_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            threshold = float(metadata["threshold"])
            first_prediction_end = float(table["window_end_s"].iloc[0]) if len(table) else min(70.0, float(record["duration_s"]))
            truth_all = [(float(event["start_s"]), float(event["end_s"])) for event in record["seizure_events"]]
            truth = [(max(first_prediction_end, start), end) for start, end in truth_all if end > first_prediction_end]
            predictions = eventize(table["window_end_s"].to_numpy(dtype=float), table["probability"].to_numpy(dtype=float), threshold=threshold) if len(table) >= 2 else []
            matching = match_events(predictions, truth)
            delays = [predictions[pair.prediction_index].start_s - truth[pair.truth_index][0] for pair in matching.pairs]
            seizure_seconds = sum(max(0.0, end - start) for start, end in truth)
            nonseizure_seconds = max(0.0, float(record["duration_s"]) - first_prediction_end - seizure_seconds)
            window_probability = table["probability"].to_numpy(dtype=float)
            window_end = table["window_end_s"].to_numpy(dtype=float)
            y_true = window_labels(window_end, truth)
            record_rows.append({
                "model": model,
                "patient_id": record["patient_id"],
                "recording": relative.as_posix(),
                "record_index": record_index,
                "threshold": threshold,
                "windows": int(len(table)),
                "eligible_truth_events": int(len(truth)),
                "excluded_truth_events_before_prediction_start": int(len(truth_all) - len(truth)),
                "tp": int(len(matching.pairs)),
                "fp": int(len(matching.unmatched_predictions)),
                "fn": int(len(matching.unmatched_truths)),
                "sensitivity": float(len(matching.pairs) / len(truth)) if truth else math.nan,
                "nonseizure_h": nonseizure_seconds / 3600.0,
                "fa_per_24h": float(len(matching.unmatched_predictions) / (nonseizure_seconds / 86400.0)) if nonseizure_seconds else math.nan,
                "median_delay_s": finite_median(delays),
                "window_positive_count": int(np.sum(y_true)),
                "probability_exact_zero_count": int(np.sum(window_probability == 0.0)),
                "probability_exact_one_count": int(np.sum(window_probability == 1.0)),
                "probability_p50": float(np.quantile(window_probability, 0.50)) if len(window_probability) else math.nan,
                "probability_p95": float(np.quantile(window_probability, 0.95)) if len(window_probability) else math.nan,
                "probability_p99": float(np.quantile(window_probability, 0.99)) if len(window_probability) else math.nan,
            })
            if len(np.unique(y_true)) == 2:
                auc = float(roc_auc_score(y_true, window_probability))
                average_precision = float(average_precision_score(y_true, window_probability))
                brier = float(brier_score_loss(y_true, window_probability))
                logloss = float(log_loss(y_true, window_probability, labels=[0, 1]))
            else:
                auc = math.nan
                average_precision = math.nan
                brier = float(brier_score_loss(y_true, window_probability)) if len(y_true) else math.nan
                logloss = math.nan
            bins = calibration_bins(y_true, window_probability)
            nonempty = [row for row in bins if int(row["count"]) > 0]
            ece = float(sum(int(row["count"]) * float(row["abs_gap"]) for row in nonempty) / len(y_true)) if len(y_true) else math.nan
            calibration_rows.append({
                "model": model,
                "patient_id": record["patient_id"],
                "recording": relative.as_posix(),
                "windows": int(len(y_true)),
                "window_positive_count": int(np.sum(y_true)),
                "window_positive_rate": float(np.mean(y_true)) if len(y_true) else math.nan,
                "window_auroc": auc,
                "window_average_precision": average_precision,
                "window_brier": brier,
                "window_log_loss": logloss,
                "window_ece_10bin": ece,
                "probability_mean": float(np.mean(window_probability)) if len(window_probability) else math.nan,
                "probability_std": float(np.std(window_probability)) if len(window_probability) else math.nan,
                "probability_exact_zero_count": int(np.sum(window_probability == 0.0)),
                "probability_exact_one_count": int(np.sum(window_probability == 1.0)),
            })
            for bin_row in bins:
                calibration_bin_rows.append({"model": model, "patient_id": record["patient_id"], "recording": relative.as_posix(), **bin_row})
        print(f"AUDIT_PROGRESS records={record_index}/{len(manifest['records'])} seconds={time.monotonic() - started:.1f}", flush=True)

    record_df = pd.DataFrame(record_rows)
    qc_df = pd.DataFrame(qc_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    calibration_bin_df = pd.DataFrame(calibration_bin_rows)
    patient_records = []
    for (model, patient_id), group in record_df.groupby(["model", "patient_id"], sort=True):
        truth = int(group["eligible_truth_events"].sum())
        tp = int(group["tp"].sum())
        fp = int(group["fp"].sum())
        nonseizure = float(group["nonseizure_h"].sum())
        delays = [float(x) for x in group["median_delay_s"] if np.isfinite(x)]
        patient_qc = qc_df[qc_df["patient_id"] == patient_id]
        patient_records.append({
            "model": model,
            "patient_id": patient_id,
            "records": int(group["recording"].nunique()),
            "eligible_truth_events": truth,
            "tp": tp,
            "fp": fp,
            "fn": int(group["fn"].sum()),
            "sensitivity": float(tp / truth) if truth else math.nan,
            "fa_per_24h": float(fp / (nonseizure / 24.0)) if nonseizure else math.nan,
            "median_record_delay_s": finite_median(delays),
            "recording_start_excluded_events": int(group["excluded_truth_events_before_prediction_start"].sum()),
            "mean_p99_abs_uv": float(patient_qc["p99_abs_uv"].mean()),
            "max_abs_uv": float(patient_qc["max_abs_uv"].max()),
            "fraction_records_p99_gt_1000_uv": float(np.mean(patient_qc["p99_abs_uv"] > 1000.0)),
            "fraction_records_max_gt_1000_uv": float(np.mean(patient_qc["max_abs_uv"] > 1000.0)),
            "mean_line_noise_ratio": float(patient_qc["mean_line_noise_ratio"].mean()),
            "mean_clipping_fraction": float(patient_qc["mean_clipping_fraction"].mean()),
            "mean_flat_fraction": float(patient_qc["mean_flat_fraction"].mean()),
        })
    patient_df = pd.DataFrame(patient_records)

    pooled_calibration = []
    for model, group in calibration_df.groupby("model", sort=True):
        # Reconstruct pooled arrays from per-record parquet files, so metrics are window-weighted.
        y_all, p_all = [], []
        for record in manifest["records"]:
            relative = Path(record["relative_edf"])
            table = pd.read_parquet(external_root / model / "probabilities" / relative.with_suffix(".parquet"))
            if len(table) == 0:
                continue
            metadata = json.loads((external_root / model / "metadata" / relative.with_suffix(".json")).read_text())
            first_end = float(table["window_end_s"].iloc[0])
            truth = [(max(first_end, float(e["start_s"])), float(e["end_s"])) for e in record["seizure_events"] if float(e["end_s"]) > first_end]
            p_arr = table["probability"].to_numpy(dtype=float)
            y_all.append(window_labels(table["window_end_s"].to_numpy(dtype=float), truth))
            p_all.append(p_arr)
        y = np.concatenate(y_all) if y_all else np.empty(0, dtype=np.int8)
        probs = np.concatenate(p_all) if p_all else np.empty(0, dtype=float)
        bins = calibration_bins(y, probs)
        nonempty = [row for row in bins if int(row["count"]) > 0]
        pooled_calibration.append({
            "model": model,
            "windows": int(len(y)),
            "positive_windows": int(np.sum(y)),
            "positive_rate": float(np.mean(y)) if len(y) else math.nan,
            "window_auroc": float(roc_auc_score(y, probs)) if len(np.unique(y)) == 2 else math.nan,
            "window_average_precision": float(average_precision_score(y, probs)) if len(np.unique(y)) == 2 else math.nan,
            "window_brier": float(brier_score_loss(y, probs)) if len(y) else math.nan,
            "window_log_loss": float(log_loss(y, probs, labels=[0, 1])) if len(np.unique(y)) == 2 else math.nan,
            "window_ece_10bin": float(sum(int(row["count"]) * float(row["abs_gap"]) for row in nonempty) / len(y)) if len(y) else math.nan,
            "probability_exact_zero_count": int(np.sum(probs == 0.0)),
            "probability_exact_one_count": int(np.sum(probs == 1.0)),
            "probability_p50": float(np.quantile(probs, .50)) if len(probs) else math.nan,
            "probability_p95": float(np.quantile(probs, .95)) if len(probs) else math.nan,
            "probability_p99": float(np.quantile(probs, .99)) if len(probs) else math.nan,
        })

    patient_path = REPORT_DIR / f"{prefix}_patient_waterfall.csv"
    record_path = REPORT_DIR / f"{prefix}_record_performance.csv"
    qc_path = REPORT_DIR / f"{prefix}_signal_qc.csv"
    calibration_path = REPORT_DIR / f"{prefix}_calibration_summary.csv"
    calibration_bins_path = REPORT_DIR / f"{prefix}_calibration_bins.csv"
    patient_df.to_csv(patient_path, index=False)
    record_df.to_csv(record_path, index=False)
    qc_df.to_csv(qc_path, index=False)
    pd.DataFrame(pooled_calibration).to_csv(calibration_path, index=False)
    calibration_bin_df.to_csv(calibration_bins_path, index=False)

    result_manifest = {
        "release_id": "v3-compact-extension-v1",
        "cohort_id": manifest["cohort_id"],
        "cohort_manifest_sha256": manifest_hash,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "patient_waterfall_and_calibration_audit_complete",
        "model_training": False,
        "threshold_retuning": False,
        "audit_definitions": {
            "window_label": "positive when the 10-second causal window ending at window_end_s overlaps a seizure annotation by at least 1 second",
            "calibration": "window-weighted Brier, log loss, AUROC, average precision and 10-bin ECE; no recalibration",
            "patient_waterfall": "all selected records aggregated within patient; eligible truth begins at first prediction time (usually 70 seconds)",
            "signal_qc": "full-record canonical bipolar absolute-amplitude quantiles plus frozen line-noise, clipping and flatness features",
        },
        "pooled_calibration": pooled_calibration,
        "patient_waterfall_rows": len(patient_df),
        "record_rows": len(record_df),
        "qc_rows": len(qc_df),
        "audit_files": {
            "patient_waterfall": {"path": patient_path.name, "sha256": sha256(patient_path)},
            "record_performance": {"path": record_path.name, "sha256": sha256(record_path)},
            "signal_qc": {"path": qc_path.name, "sha256": sha256(qc_path)},
            "calibration_summary": {"path": calibration_path.name, "sha256": sha256(calibration_path)},
            "calibration_bins": {"path": calibration_bins_path.name, "sha256": sha256(calibration_bins_path)},
        },
        "decision_notes": {
            "external_subset_is_not_full_eval": True,
            "next_gate": "decide whether to expand to all 43 eval patients after reviewing patient-level heterogeneity and calibration",
        },
    }
    manifest_path_out = REPORT_DIR / f"{prefix}_audit_manifest.json"
    manifest_path_out.write_text(json.dumps(result_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest_path_out),
        "patient_rows": len(patient_df),
        "record_rows": len(record_df),
        "qc_rows": len(qc_df),
        "pooled_calibration": pooled_calibration,
        "runtime_s": time.monotonic() - started,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
