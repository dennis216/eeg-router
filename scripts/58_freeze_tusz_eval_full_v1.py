"""Freeze all 43 TUSZ eval patients and all eval EDF records for confirmation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import mne


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports" / "v3-compact-extension-v1"
DATA_ROOT = Path("/mnt/d/TUH_EEG/TUSZ_v2.0.6")
EVAL_ROOT = DATA_ROOT / "edf" / "eval"
COHORT_ID = "tusz-eval-full-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def annotations(path: Path) -> list[dict[str, str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    lines = [line for line in lines if line and not line.startswith("#")]
    return list(csv.DictReader(lines))


def main() -> None:
    patient_rows = []
    records = []
    montage_counts = Counter()
    sampling_counts = Counter()
    total_duration_s = 0.0
    seizure_events = 0
    for patient_dir in sorted(path for path in EVAL_ROOT.iterdir() if path.is_dir()):
        patient_records = sorted(patient_dir.rglob("*.edf"))
        patient_labels = []
        for edf in patient_records:
            bi = Path(str(edf.with_suffix(".csv")) + "_bi")
            patient_labels.extend(row["label"].lower() for row in annotations(bi))
        patient_rows.append({
            "patient_id": patient_dir.name,
            "has_seizure_annotation": "seiz" in patient_labels,
            "records": len(patient_records),
            "selection_sha256_rank": hashlib.sha256(f"v3-compact-extension-v1|{COHORT_ID}|{patient_dir.name}".encode()).hexdigest(),
        })
        for edf in patient_records:
            bi = Path(str(edf.with_suffix(".csv")) + "_bi")
            ann = annotations(bi)
            seizure_rows = [{"start_s": float(row["start_time"]), "end_s": float(row["stop_time"]), "label": row["label"]} for row in ann if row["label"].lower() == "seiz"]
            raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
            sampling_hz = float(raw.info["sfreq"])
            duration_s = float(raw.n_times / sampling_hz)
            relative = edf.relative_to(DATA_ROOT).as_posix()
            total_duration_s += duration_s
            seizure_events += len(seizure_rows)
            montage_counts[edf.parent.name] += 1
            sampling_counts[str(sampling_hz)] += 1
            records.append({
                "patient_id": patient_dir.name,
                "relative_edf": relative,
                "edf_bytes": edf.stat().st_size,
                "edf_sha256": sha256(edf),
                "csv_bi_sha256": sha256(bi),
                "duration_s": duration_s,
                "sampling_hz": sampling_hz,
                "montage": edf.parent.name,
                "seizure_events": seizure_rows,
            })
        print(f"FREEZE_PROGRESS patients={len(patient_rows)}/43 records={len(records)}", flush=True)
    manifest = {
        "release_id": "v3-compact-extension-v1",
        "cohort_id": COHORT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "TUSZ_v2.0.6 eval",
        "dataset_root": str(DATA_ROOT),
        "selection_partition": "eval",
        "selection_rule": "retain all 43 eval patients and every EDF record; patient/record list frozen before model output review",
        "selection_depended_on_model_output": False,
        "selection_depended_on_model_performance": False,
        "patient_count": len(patient_rows),
        "seizure_present_patient_count": sum(row["has_seizure_annotation"] for row in patient_rows),
        "background_only_patient_count": sum(not row["has_seizure_annotation"] for row in patient_rows),
        "record_count": len(records),
        "total_duration_s": total_duration_s,
        "total_duration_h": total_duration_s / 3600.0,
        "seizure_event_count": seizure_events,
        "montage_counts": dict(sorted(montage_counts.items())),
        "sampling_hz_counts": dict(sorted(sampling_counts.items())),
        "patients": patient_rows,
        "records": records,
        "models": ["singlem", "cbramod", "tcn_gat"],
        "checkpoint_policy": "split17 seed17 final 5000-update checkpoint for each model",
        "threshold_policy": "each model's frozen CHB validation-only split17 seed17 final threshold",
        "model_training": False,
        "threshold_retuning": False,
        "interpretation": "full TUSZ eval confirmation cohort; no tuning and no patient exclusion after scoring",
    }
    output = REPORT_DIR / "tusz_eval_full_v1_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = sha256(output)
    (REPORT_DIR / "tusz_eval_full_v1_manifest.sha256").write_text(digest + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), "sha256": digest, "patients": len(patient_rows), "records": len(records), "hours": total_duration_s / 3600.0, "seizure_events": seizure_events, "montages": dict(montage_counts), "sampling_hz": dict(sampling_counts)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
