"""Freeze the first patient-hashed TUSZ eval external-validation subset."""

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
COHORT_ID = "tusz-eval-subset-v1"
PATIENTS_PER_STRATUM = 5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def annotations(path: Path) -> list[dict[str, str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    lines = [line for line in lines if line and not line.startswith("#")]
    return list(csv.DictReader(lines))


def main() -> None:
    patients = []
    for directory in sorted(path for path in EVAL_ROOT.iterdir() if path.is_dir()):
        edfs = sorted(directory.rglob("*.edf"))
        labels = []
        for edf in edfs:
            labels.extend(row["label"].lower() for row in annotations(Path(str(edf.with_suffix(".csv")) + "_bi")))
        rank = hashlib.sha256(f"v3-compact-extension-v1|{COHORT_ID}|{directory.name}".encode()).hexdigest()
        patients.append({
            "patient_id": directory.name,
            "has_seizure_annotation": "seiz" in labels,
            "records": len(edfs),
            "selection_sha256_rank": rank,
        })
    selected_patients = []
    for stratum in (True, False):
        pool = sorted((p for p in patients if p["has_seizure_annotation"] is stratum), key=lambda p: p["selection_sha256_rank"])
        selected_patients.extend(pool[:PATIENTS_PER_STRATUM])
    selected_ids = {p["patient_id"] for p in selected_patients}

    records = []
    total_duration = 0.0
    seizure_events = 0
    montage_counts = Counter()
    sampling_counts = Counter()
    for patient_id in sorted(selected_ids):
        for edf in sorted((EVAL_ROOT / patient_id).rglob("*.edf")):
            rel = edf.relative_to(DATA_ROOT).as_posix()
            bi = Path(str(edf.with_suffix(".csv")) + "_bi")
            ann = annotations(bi)
            seizures = [
                {"start_s": float(a["start_time"]), "end_s": float(a["stop_time"]), "label": a["label"]}
                for a in ann if a["label"].lower() == "seiz"
            ]
            raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
            duration = float(raw.n_times / raw.info["sfreq"])
            sampling = float(raw.info["sfreq"])
            montage = edf.parent.name
            total_duration += duration
            seizure_events += len(seizures)
            montage_counts[montage] += 1
            sampling_counts[str(sampling)] += 1
            records.append({
                "patient_id": patient_id,
                "relative_edf": rel,
                "edf_bytes": edf.stat().st_size,
                "edf_sha256": sha256(edf),
                "csv_bi_sha256": sha256(bi),
                "duration_s": duration,
                "sampling_hz": sampling,
                "montage": montage,
                "seizure_events": seizures,
            })

    manifest = {
        "release_id": "v3-compact-extension-v1",
        "cohort_id": COHORT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "TUSZ_v2.0.6 eval",
        "dataset_root": str(DATA_ROOT),
        "selection_partition": "eval",
        "selection_rule": "select five lowest SHA256-ranked patient IDs independently within seizure-present and background-only strata; retain every EDF for selected patients",
        "selection_key": "SHA256(v3-compact-extension-v1|tusz-eval-subset-v1|patient_id)",
        "selection_depended_on_model_output": False,
        "selection_depended_on_model_performance": False,
        "patients_per_stratum": PATIENTS_PER_STRATUM,
        "patients": sorted(selected_patients, key=lambda p: p["patient_id"]),
        "patient_count": len(selected_ids),
        "seizure_present_patient_count": sum(p["has_seizure_annotation"] for p in selected_patients),
        "background_only_patient_count": sum(not p["has_seizure_annotation"] for p in selected_patients),
        "record_count": len(records),
        "total_duration_s": total_duration,
        "total_duration_h": total_duration / 3600.0,
        "seizure_event_count": seizure_events,
        "montage_counts": dict(sorted(montage_counts.items())),
        "sampling_hz_counts": dict(sorted(sampling_counts.items())),
        "records": records,
        "models": ["singlem", "cbramod", "tcn_gat"],
        "checkpoint_policy": "split17 seed17 final 5000-update checkpoint for each model",
        "threshold_policy": "each model's frozen CHB validation-only split17 seed17 final threshold",
        "model_training": False,
        "threshold_retuning": False,
        "test_patient_removal_after_results": False,
        "interpretation": "preliminary frozen external subset validation; not full TUSZ eval",
    }
    out = REPORT_DIR / "tusz_eval_subset_v1_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (REPORT_DIR / "tusz_eval_subset_v1_manifest.sha256").write_text(sha256(out) + "\n")
    print(json.dumps({
        "manifest": str(out), "sha256": sha256(out), "patients": manifest["patient_count"],
        "records": manifest["record_count"], "hours": manifest["total_duration_h"],
        "seizure_events": manifest["seizure_event_count"], "montages": manifest["montage_counts"],
        "sampling_hz": manifest["sampling_hz_counts"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
