"""Freeze a deterministic outer-train discovery subset for H2/H3.

This subset is not a new train/validation/test split.  It is a compute-bounded
subset of the already frozen outer-train patients, selected before reading any
model outputs.  Whole recordings are retained so eventization remains causal
and contiguous; patient round-robin selection prevents one patient dominating
the discovery fit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path("/root/b_false_alarm_atlas")
MANIFEST_DIR = ROOT / "manifests" / "groupkfold_cv_v1"
MAX_WINDOWS = 50_000
FOLDS = tuple(range(5))


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _select(fold: int, windows: pd.DataFrame, train_patients: list[str]) -> pd.DataFrame:
    eligible = windows[windows.patient.isin(train_patients) & windows.train_eligible].copy()
    recording_rows = (
        eligible.groupby(["patient", "recording"], sort=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    by_patient: dict[str, list[dict[str, object]]] = {}
    for patient, group in recording_rows.groupby("patient", sort=True):
        records = group.to_dict(orient="records")
        records.sort(key=lambda row: _digest(f"{fold}:{row['recording']}"))
        by_patient[str(patient)] = records
    selected: list[str] = []
    used = 0
    while True:
        progress = False
        for patient in sorted(by_patient):
            while by_patient[patient]:
                row = by_patient[patient].pop(0)
                rows = int(row["rows"])
                # Keep whole recordings.  Oversized individual recordings are
                # skipped rather than window-cropped, which preserves event
                # continuity and keeps the selection deterministic.
                if rows > MAX_WINDOWS or used + rows > MAX_WINDOWS:
                    continue
                selected.append(str(row["recording"]))
                used += rows
                progress = True
                break
        if not progress:
            break
    subset = eligible[eligible.recording.astype(str).isin(selected)].reset_index(drop=True)
    if subset.empty:
        raise RuntimeError(f"empty discovery subset for fold {fold}")
    return subset


def main() -> None:
    windows = pd.read_parquet(ROOT / "manifests" / "windows.parquet")
    rows: list[dict[str, object]] = []
    for fold in FOLDS:
        split_path = MANIFEST_DIR / f"fold_{fold}.json"
        split = json.loads(split_path.read_text())
        subset = _select(fold, windows, [str(value) for value in split["train"]])
        output_path = MANIFEST_DIR / f"discovery_subset_fold_{fold}.parquet"
        subset.to_parquet(output_path, index=False)
        metadata = {
            "release_id": "v3-groupkfold-confirmatory-v1",
            "subset_id": "outer-train-discovery-subset-v1",
            "fold": fold,
            "selection_inputs": ["frozen outer-train patient IDs", "recording IDs", "window row counts"],
            "model_outputs_read": False,
            "test_read": False,
            "selection_algorithm": "sha256(fold:recording) order; patient round-robin; whole recordings; max 50000 windows",
            "max_windows": MAX_WINDOWS,
            "rows": int(len(subset)),
            "patients": sorted(subset.patient.astype(str).unique().tolist()),
            "recordings": sorted(subset.recording.astype(str).unique().tolist()),
            "source_fold_sha256": _sha256(split_path),
            "subset_sha256": _sha256(output_path),
            "output": str(output_path.relative_to(ROOT)),
        }
        (MANIFEST_DIR / f"discovery_subset_fold_{fold}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        rows.append(
            {
                "fold": fold,
                "rows": len(subset),
                "patients": subset.patient.nunique(),
                "recordings": subset.recording.nunique(),
            }
        )
    manifest = {
        "release_id": "v3-groupkfold-confirmatory-v1",
        "subset_id": "outer-train-discovery-subset-v1",
        "max_windows": MAX_WINDOWS,
        "folds": rows,
        "model_outputs_read": False,
        "test_read": False,
    }
    (MANIFEST_DIR / "discovery_subset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
