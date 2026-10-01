"""Freeze the standard GroupKFold TUSZ external-validation cohort.

The cohort is the complete TUSZ v2.0.6 ``eval`` partition.  It is frozen from
the dataset inventory before any standard-model external output is read.  No
model checkpoint, threshold, or performance value participates in selection.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import mne


ROOT = Path("/root/b_false_alarm_atlas")
RUN_ID = "v3-groupkfold-confirmatory-v1"
REPORT_DIR = ROOT / "outputs" / "reports" / RUN_ID
DATA_ROOT = Path("/mnt/d/TUH_EEG/TUSZ_v2.0.6")
EVAL_ROOT = DATA_ROOT / "edf" / "eval"
COHORT_ID = "tusz-groupkfold-external-v1"
OUTPUT = REPORT_DIR / "tusz_groupkfold_external_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _annotations(path: Path) -> list[dict[str, str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines()]
    lines = [line for line in lines if line and not line.startswith("#")]
    return list(csv.DictReader(lines))


def main() -> None:
    if not EVAL_ROOT.exists():
        raise FileNotFoundError(EVAL_ROOT)
    patient_dirs = sorted(path for path in EVAL_ROOT.iterdir() if path.is_dir())
    if not patient_dirs:
        raise RuntimeError(f"no TUSZ eval patient directories under {EVAL_ROOT}")

    patient_rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    montage_counts: Counter[str] = Counter()
    sampling_counts: Counter[str] = Counter()
    total_duration_s = 0.0
    seizure_events = 0
    for patient_dir in patient_dirs:
        patient_records = sorted(patient_dir.rglob("*.edf"))
        patient_labels: list[str] = []
        for edf in patient_records:
            bi = Path(str(edf.with_suffix(".csv")) + "_bi")
            if bi.exists():
                patient_labels.extend(str(row.get("label", "")).lower() for row in _annotations(bi))
        patient_rows.append(
            {
                "patient_id": patient_dir.name,
                "has_seizure_annotation": "seiz" in patient_labels,
                "record_count": len(patient_records),
            }
        )
        for edf in patient_records:
            bi = Path(str(edf.with_suffix(".csv")) + "_bi")
            if not bi.exists():
                raise FileNotFoundError(f"missing bi annotation for {edf}")
            annotations = _annotations(bi)
            seizure_rows = [
                {
                    "start_s": float(row["start_time"]),
                    "end_s": float(row["stop_time"]),
                    "label": str(row["label"]),
                }
                for row in annotations
                if str(row.get("label", "")).lower() == "seiz"
            ]
            raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
            sampling_hz = float(raw.info["sfreq"])
            duration_s = float(raw.n_times / sampling_hz)
            relative = edf.relative_to(DATA_ROOT).as_posix()
            total_duration_s += duration_s
            seizure_events += len(seizure_rows)
            montage_counts[edf.parent.name] += 1
            sampling_counts[str(sampling_hz)] += 1
            records.append(
                {
                    "patient_id": patient_dir.name,
                    "relative_edf": relative,
                    "edf_bytes": edf.stat().st_size,
                    "edf_sha256": _sha256(edf),
                    "csv_bi_sha256": _sha256(bi),
                    "duration_s": duration_s,
                    "sampling_hz": sampling_hz,
                    "montage": edf.parent.name,
                    "seizure_events": seizure_rows,
                }
            )
        print(
            f"TUSZ_GROUPKFOLD_FREEZE_PROGRESS patients={len(patient_rows)}/{len(patient_dirs)} "
            f"records={len(records)}",
            flush=True,
        )

    manifest = {
        "release_id": RUN_ID,
        "external_cohort_id": COHORT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "TUSZ_v2.0.6 eval",
        "dataset_root": str(DATA_ROOT),
        "selection_partition": "eval",
        "selection_rule": "retain every patient and EDF in the frozen TUSZ eval partition",
        "selection_depended_on_model_output": False,
        "selection_depended_on_model_performance": False,
        "patient_count": len(patient_rows),
        "seizure_present_patient_count": sum(bool(row["has_seizure_annotation"]) for row in patient_rows),
        "background_only_patient_count": sum(not bool(row["has_seizure_annotation"]) for row in patient_rows),
        "record_count": len(records),
        "total_duration_s": total_duration_s,
        "total_duration_h": total_duration_s / 3600.0,
        "seizure_event_count": seizure_events,
        "montage_counts": dict(sorted(montage_counts.items())),
        "sampling_hz_counts": dict(sorted(sampling_counts.items())),
        "patients": patient_rows,
        "records": records,
        "models": ["singlem", "cbramod", "tcn_gat"],
        "folds": [0, 1, 2, 3, 4],
        "seeds": [17, 42, 3407],
        "checkpoint_policy": "standard GroupKFold split0-4, seed17/42/3407, final step_05000 checkpoint",
        "threshold_policy": "each standard run's final validation-only threshold from completed.json",
        "model_training": False,
        "threshold_retuning": False,
        "patient_removal_after_results": False,
        "interpretation": "frozen external cohort inventory; event-level scoring follows only after format and inference audits",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    digest = _sha256(OUTPUT)
    OUTPUT.with_suffix(".sha256").write_text(digest + "\n")
    print(
        json.dumps(
            {
                "manifest": str(OUTPUT),
                "sha256": digest,
                "patients": len(patient_rows),
                "records": len(records),
                "hours": total_duration_s / 3600.0,
                "seizure_events": seizure_events,
                "montages": dict(sorted(montage_counts.items())),
                "sampling_hz": dict(sorted(sampling_counts.items())),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
