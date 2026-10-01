from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from bfa.preprocessing.channels import CHANNELS


QUANTILES = (0.001, 0.01, 0.50, 0.99, 0.999)
QUANTILE_LABELS = ("0p1", "1", "50", "99", "99p9")


def _audit_recording(row: dict[str, object], data_root: str, cache_root: str) -> dict[str, object]:
    relative = Path(str(row["relative_path"]))
    singlem_path = Path(cache_root) / "singlem" / relative.with_suffix(".npy")
    if singlem_path.exists():
        scaled = np.asarray(np.load(singlem_path, mmap_mode="r"), dtype=np.float32)
        singlem_uv = scaled / 0.01
    else:
        filtered_uv_256 = np.load(
            Path(cache_root) / "tcn_gat" / relative.with_suffix(".npy"), mmap_mode="r"
        )
        singlem_uv = resample_poly(filtered_uv_256, 1, 2, axis=-1).astype(np.float32)
        scaled = singlem_uv * 0.01
    outlier = np.abs(scaled) > 1.5
    sampled_uv = np.asarray(singlem_uv[:, ::128], dtype=np.float32)
    raw = mne.io.read_raw_edf(Path(data_root) / relative, preload=False, verbose="ERROR")
    original_units = sorted(set(str(value) for value in raw._orig_units.values()))
    return {
        "patient_id": str(row["patient_id"]),
        "recording_id": str(row["recording_id"]),
        "relative_path": str(relative),
        "original_units": original_units,
        "sample_count": int(scaled.size),
        "outlier_count": int(outlier.sum()),
        "channel_sample_count": np.full(16, scaled.shape[-1], dtype=np.int64),
        "channel_outlier_count": outlier.sum(axis=-1).astype(np.int64),
        "sampled_uv": sampled_uv,
    }


def _quantile_columns(values_uv: np.ndarray) -> dict[str, float]:
    scaled = values_uv * 0.01
    output: dict[str, float] = {}
    for label, value in zip(QUANTILE_LABELS, np.quantile(values_uv, QUANTILES), strict=True):
        output[f"uv_p{label}"] = float(value)
    for label, value in zip(QUANTILE_LABELS, np.quantile(scaled, QUANTILES), strict=True):
        output[f"scaled_p{label}"] = float(value)
    output["uv_abs_p99p9"] = float(np.quantile(np.abs(values_uv), 0.999))
    output["scaled_abs_p99p9"] = output["uv_abs_p99p9"] * 0.01
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/recordings.parquet"))
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/d/EEGData/chbmit-1.0.0"))
    parser.add_argument("--cache-root", type=Path, default=Path("/mnt/d/EEGData/bfa_cache_v3"))
    parser.add_argument("--output", type=Path, default=Path("outputs/audits/singlem_units"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    rows = pd.read_parquet(args.manifest).sort_values(
        ["patient_id", "recording_id"], kind="stable"
    )
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_audit_recording, row, str(args.data_root), str(args.cache_root))
            for row in rows.to_dict("records")
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(f"SINGLEM_AUDIT_PROGRESS {completed}/{len(futures)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    patient_samples: dict[str, list[np.ndarray]] = {}
    recording_rows = []
    channel_rows = []
    for result in results:
        patient = str(result["patient_id"])
        sample_count = int(result["sample_count"])
        outlier_count = int(result["outlier_count"])
        sampled_uv = np.asarray(result["sampled_uv"])
        patient_samples.setdefault(patient, []).append(sampled_uv.reshape(-1))
        recording_rows.append(
            {
                "patient_id": patient,
                "recording_id": result["recording_id"],
                "relative_path": result["relative_path"],
                "edf_original_units": ";".join(result["original_units"]),
                "mne_get_data_unit": "V",
                "converted_model_input_unit": "uV",
                "sample_count": sample_count,
                "outlier_count": outlier_count,
                "outlier_fraction": outlier_count / sample_count,
                **_quantile_columns(sampled_uv.reshape(-1)),
            }
        )
        for channel_index, channel in enumerate(CHANNELS):
            channel_samples = int(result["channel_sample_count"][channel_index])
            channel_outliers = int(result["channel_outlier_count"][channel_index])
            channel_rows.append(
                {
                    "patient_id": patient,
                    "recording_id": result["recording_id"],
                    "channel": channel,
                    "sample_count": channel_samples,
                    "outlier_count": channel_outliers,
                    "outlier_fraction": channel_outliers / channel_samples,
                }
            )

    recording_table = pd.DataFrame(recording_rows).sort_values(
        ["patient_id", "recording_id"], kind="stable"
    )
    channel_recording = pd.DataFrame(channel_rows)
    patient_rows = []
    for patient, arrays in sorted(patient_samples.items()):
        values = np.concatenate(arrays)
        subset = recording_table[recording_table.patient_id == patient]
        patient_rows.append(
            {
                "patient_id": patient,
                "recordings": len(subset),
                "sample_count": int(subset.sample_count.sum()),
                "outlier_count": int(subset.outlier_count.sum()),
                "outlier_fraction": float(subset.outlier_count.sum() / subset.sample_count.sum()),
                **_quantile_columns(values),
            }
        )
    patient_table = pd.DataFrame(patient_rows)
    channel_table = (
        channel_recording.groupby("channel", sort=False)[["sample_count", "outlier_count"]]
        .sum()
        .reset_index()
    )
    channel_table["outlier_fraction"] = (
        channel_table.outlier_count / channel_table.sample_count
    )
    recording_table.to_parquet(args.output / "recording_audit.parquet", index=False)
    patient_table.to_csv(args.output / "patient_quantiles.csv", index=False)
    channel_table.to_csv(args.output / "channel_outlier_fraction.csv", index=False)
    channel_recording.to_parquet(args.output / "channel_recording_audit.parquet", index=False)

    total_outliers = int(recording_table.outlier_count.sum())
    top_ten = int(recording_table.nlargest(10, "outlier_count").outlier_count.sum())
    summary = {
        "unit_chain": [
            "EDF physical unit recorded as uV",
            "MNE raw.get_data returns V",
            "pipeline multiplies by 1e6 to uV",
            "filter/resample in uV",
            "official SingLEM profile multiplies by 0.01",
        ],
        "official_code": {
            "profile": "third_party/SingLEM/preprocessing/build_trials/profiles/profiles.py:18",
            "mne_uv_conversion": "third_party/SingLEM/preprocessing/build_trials/profiles/transforms.py:25-28",
            "scaling": "third_party/SingLEM/preprocessing/build_trials/profiles/transforms.py:54-58",
        },
        "quantile_method": "deterministic 1 Hz sample after 128 Hz resampling; clip fractions are exact",
        "recordings": int(len(recording_table)),
        "patients": int(len(patient_table)),
        "recordings_over_one_percent": int((recording_table.outlier_fraction > 0.01).sum()),
        "recordings_over_one_percent_fraction": float(
            (recording_table.outlier_fraction > 0.01).mean()
        ),
        "total_sample_count": int(recording_table.sample_count.sum()),
        "total_outlier_count": total_outliers,
        "total_outlier_fraction": float(
            recording_table.outlier_count.sum() / recording_table.sample_count.sum()
        ),
        "top_ten_recordings_share_of_outliers": top_ten / max(total_outliers, 1),
        "hard_clipping_applied": False,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
