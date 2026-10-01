"""Freeze the H7 patient-local detector fine-tuning window labels.

This stage is deliberately data-only.  It preserves the already exposed H7
60/20/20 patient-local chronological split, excludes ambiguous windows, and
does not read detector probabilities or create any model checkpoint.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/b_false_alarm_atlas")
SOURCE = ROOT / "outputs/reports/h6-tusz-raw-pre-router-v1"
ELIGIBILITY = ROOT / "outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-fit-only-eligibility-v2/patient_detector_fit_only_eligibility.csv"
OUT = ROOT / "outputs/reports/h7-patient-finetune-comparison-v1/labels"
PARTITIONS = ("train", "dev", "eval")
WINDOW_SECONDS = 10.0
STRIDE_SECONDS = 2.0
FIRST_CONTEXT_END_SECONDS = 70.0
SEIZURE_MARGIN_SECONDS = 30.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def phase(global_center: float, patient_duration: float) -> str:
    if global_center < 0.60 * patient_duration:
        return "fit"
    if global_center < 0.80 * patient_duration:
        return "dev"
    return "holdout"


def label(center: float, seizures: list[tuple[float, float]]) -> str:
    if any(start <= center <= end for start, end in seizures):
        return "positive"
    if not seizures:
        return "negative"
    nearest_boundary = min(min(abs(center - start), abs(center - end)) for start, end in seizures)
    return "negative" if nearest_boundary >= SEIZURE_MARGIN_SECONDS else "ambiguous"


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite frozen label namespace: {OUT}")
    OUT.mkdir(parents=True, exist_ok=False)
    eligible = pd.read_csv(ELIGIBILITY)
    eligible = eligible[eligible["fit_eligible_standard"].astype(bool)][
        ["partition", "patient_id", "model"]
    ].drop_duplicates()
    if len(eligible) != 72:
        raise RuntimeError(f"expected 72 patient-detector units, found {len(eligible)}")

    records: list[dict[str, object]] = []
    source_manifests: dict[str, str] = {}
    for partition in PARTITIONS:
        path = SOURCE / "manifests" / f"{partition}_manifest.json"
        source_manifests[str(path.relative_to(ROOT))] = sha256(path)
        records.extend(json.loads(path.read_text())["records"])

    by_patient: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in records:
        relative = str(row["relative_edf"])
        partition = relative.split("/", maxsplit=2)[1]
        key = (partition, str(row["patient_id"]))
        if ((eligible.partition == key[0]) & (eligible.patient_id == key[1])).any():
            by_patient.setdefault(key, []).append(row)

    window_rows: list[dict[str, object]] = []
    edf_phase_rows: list[dict[str, object]] = []
    for (partition, patient), patient_records in sorted(by_patient.items()):
        total = float(sum(float(row["duration_s"]) for row in patient_records))
        offset = 0.0
        for row in sorted(patient_records, key=lambda item: str(item["relative_edf"])):
            duration = float(row["duration_s"])
            seizure_intervals = [
                (float(event["start_s"]), float(event["end_s"]))
                for event in row.get("seizure_events", [])
            ]
            recording = str(row["relative_edf"])
            phases: set[str] = set()
            for context_end in np.arange(
                FIRST_CONTEXT_END_SECONDS, duration + 1e-9, STRIDE_SECONDS
            ):
                center = float(context_end - WINDOW_SECONDS / 2.0)
                split_phase = phase(offset + center, total)
                window_label = label(center, seizure_intervals)
                phases.add(split_phase)
                window_id = hashlib.sha256(
                    f"{partition}|{patient}|{recording}|{context_end:.3f}".encode()
                ).hexdigest()[:24]
                window_rows.append(
                    {
                        "partition": partition,
                        "patient_id": patient,
                        "recording": recording,
                        "context_end_s": float(context_end),
                        "window_start_s": float(context_end - WINDOW_SECONDS),
                        "window_center_s": center,
                        "phase": split_phase,
                        "label_status": window_label,
                        "label": 1 if window_label == "positive" else 0 if window_label == "negative" else pd.NA,
                        "seizure_event_ids": "|".join(
                            str(index)
                            for index, (start, end) in enumerate(seizure_intervals)
                            if start <= center <= end
                        ),
                        "window_id": window_id,
                        "patient_total_duration_s": total,
                        "patient_recording_offset_s": offset,
                        "recording_duration_s": duration,
                        "edf_sha256": str(row["edf_sha256"]),
                        "csv_bi_sha256": str(row["csv_bi_sha256"]),
                    }
                )
            edf_phase_rows.append(
                {
                    "partition": partition,
                    "patient_id": patient,
                    "recording": recording,
                    "phase_count": len(phases),
                    "phases": "|".join(sorted(phases)),
                    "cross_phase": len(phases) > 1,
                }
            )
            offset += duration

    windows = pd.DataFrame(window_rows)
    windows["label"] = windows["label"].astype("Int8")
    phases = pd.DataFrame(edf_phase_rows)
    units = eligible.merge(windows, on=["partition", "patient_id"], how="inner")

    unit_summary = (
        units.groupby(["partition", "patient_id", "model", "phase", "label_status"])
        .size().rename("windows").reset_index()
        .pivot_table(
            index=["partition", "patient_id", "model", "phase"],
            columns="label_status", values="windows", fill_value=0,
        ).reset_index()
    )
    unit_summary.columns.name = None
    for name in ("positive", "negative", "ambiguous"):
        if name not in unit_summary:
            unit_summary[name] = 0
        unit_summary[name] = unit_summary[name].astype(int)
    unit_summary["has_both_train_classes"] = (
        (unit_summary.positive > 0) & (unit_summary.negative > 0)
    )

    detector_summary = (
        units.groupby(["model", "phase", "label_status"]).size().rename("windows").reset_index()
    )
    windows_path = OUT / "patient_window_labels.parquet"
    units_path = OUT / "patient_detector_window_labels.parquet"
    unit_summary_path = OUT / "patient_detector_phase_label_counts.csv"
    detector_summary_path = OUT / "detector_phase_label_counts.csv"
    phases_path = OUT / "edf_phase_audit.csv"
    windows.to_parquet(windows_path, index=False)
    units.to_parquet(units_path, index=False)
    unit_summary.to_csv(unit_summary_path, index=False)
    detector_summary.to_csv(detector_summary_path, index=False)
    phases.to_csv(phases_path, index=False)

    fit_summary = unit_summary[unit_summary.phase == "fit"]
    if len(fit_summary) != 72 or not fit_summary.has_both_train_classes.all():
        raise RuntimeError("all 72 fit units must contain both positive and negative labels")
    cross_phase = int(phases.cross_phase.sum())
    if cross_phase != 71:
        raise RuntimeError(f"legacy split audit changed: expected 71 cross-phase EDFs, got {cross_phase}")

    manifest = {
        "release_id": "h7-patient-finetune-comparison-v1",
        "stage": "frozen_window_label_manifest",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "TUSZ v2.0.6",
        "exploratory": True,
        "exploratory_reason": "legacy H7 chronological holdout was previously viewed and 71 EDFs cross phase boundaries",
        "split": "unchanged legacy patient-local cumulative-time 60% fit / 20% dev / 20% holdout",
        "input_contract": {
            "window_seconds": WINDOW_SECONDS,
            "stride_seconds": STRIDE_SECONDS,
            "causal_history_seconds": 60.0,
            "first_context_end_seconds": FIRST_CONTEXT_END_SECONDS,
        },
        "label_rule": {
            "positive": "10-s window center lies inside a seizure annotation",
            "negative": "window center is at least 30 s from the nearest seizure boundary",
            "ambiguous": "within 30 s of a seizure boundary but not inside seizure; excluded from training",
        },
        "counts": {
            "eligible_patient_detector_units": 72,
            "unique_patients": int(eligible.patient_id.nunique()),
            "eligible_by_detector": eligible.groupby("model").size().astype(int).to_dict(),
            "patient_windows": int(len(windows)),
            "patient_detector_windows": int(len(units)),
            "cross_phase_edfs": cross_phase,
            "fit_units_with_both_classes": int(fit_summary.has_both_train_classes.sum()),
            "dev_units_with_positive": int((unit_summary[unit_summary.phase == "dev"].positive > 0).sum()),
            "holdout_units_with_positive": int((unit_summary[unit_summary.phase == "holdout"].positive > 0).sum()),
        },
        "training_sampler": {
            "effective_batch_size": 64,
            "positive": 32,
            "negative": 32,
            "positive_sampling": "seizure event uniformly, then window uniformly",
            "negative_sampling": "EDF uniformly, then non-seizure 30-s block and window",
            "replacement": "fit only when needed",
            "loss": "BCEWithLogitsLoss(pos_weight=1)",
        },
        "source_manifests": source_manifests,
        "source_eligibility": str(ELIGIBILITY.relative_to(ROOT)),
        "source_eligibility_sha256": sha256(ELIGIBILITY),
        "files": {
            str(path.relative_to(OUT)): sha256(path)
            for path in (windows_path, units_path, unit_summary_path, detector_summary_path, phases_path)
        },
        "holdout_probabilities_read": False,
        "model_training": False,
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (OUT / "manifest.sha256").write_text(sha256(manifest_path) + "  manifest.json\n")
    print(detector_summary.to_string(index=False))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
