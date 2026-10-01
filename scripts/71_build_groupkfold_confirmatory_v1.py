"""Build the paper-standard patient-grouped confirmatory manifest.

The split assignment intentionally uses one row per canonical patient.  The
only grouping variable passed to scikit-learn is the patient identifier; EEG
duration, seizure count, signal quality, predictions, and model outcomes are
never used to optimize fold membership.  Recording/seizure summaries are
written only after assignment for audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


OUTER_FOLDS = 5
INNER_VALIDATION_FRACTION = 0.20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def patient_frame(recordings: pd.DataFrame) -> pd.DataFrame:
    if "patient_id" not in recordings.columns:
        raise ValueError("recordings manifest must contain patient_id")
    patients = sorted(recordings["patient_id"].astype(str).unique())
    if len(patients) != 22:
        raise ValueError(f"expected 22 canonical patients, found {len(patients)}")
    # One row per group is deliberate.  No burden or duration column is passed
    # to either GroupKFold or GroupShuffleSplit.
    return pd.DataFrame({"patient_id": patients})


def audit_stats(
    recordings: pd.DataFrame, seizures: pd.DataFrame, patients: list[str]
) -> dict[str, float]:
    subset = recordings[recordings.patient_id.astype(str).isin(patients)]
    seizure_subset = seizures[seizures.patient_id.astype(str).isin(patients)]
    return {
        "patients": float(len(patients)),
        "hours": float(subset.duration_s.sum()) / 3600.0,
        "seizures": float(len(seizure_subset)),
    }


def validate(folds: list[dict[str, object]], patients: set[str]) -> None:
    seen_test: list[str] = []
    for fold in folds:
        train = set(fold["train"])
        validation = set(fold["validation"])
        test = set(fold["test"])
        if train & validation or train & test or validation & test:
            raise RuntimeError(f"patient overlap in fold {fold['fold']}")
        if train | validation | test != patients:
            raise RuntimeError(f"patient coverage failure in fold {fold['fold']}")
        seen_test.extend(sorted(test))
    if sorted(seen_test) != sorted(patients):
        raise RuntimeError("each patient must occur in exactly one outer test fold")


def build_manifest(root: Path, out: Path, inner_seed: int) -> dict[str, object]:
    recordings_path = root / "manifests/recordings.parquet"
    seizures_path = root / "manifests/seizures.parquet"
    recordings = pd.read_parquet(recordings_path)
    seizures = pd.read_parquet(seizures_path)
    frame = patient_frame(recordings)
    groups = frame["patient_id"].to_numpy()
    outer = GroupKFold(n_splits=OUTER_FOLDS)
    folds: list[dict[str, object]] = []

    for fold_id, (development_idx, test_idx) in enumerate(
        outer.split(frame, groups=groups)
    ):
        development = frame.iloc[development_idx].reset_index(drop=True)
        test = sorted(frame.iloc[test_idx].patient_id.astype(str).tolist())
        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=INNER_VALIDATION_FRACTION,
            random_state=inner_seed + fold_id,
        )
        train_idx, validation_idx = next(
            inner.split(development, groups=development["patient_id"].to_numpy())
        )
        train = sorted(development.iloc[train_idx].patient_id.astype(str).tolist())
        validation = sorted(
            development.iloc[validation_idx].patient_id.astype(str).tolist()
        )
        folds.append(
            {
                "fold": fold_id,
                "algorithm": {
                    "outer": "sklearn.model_selection.GroupKFold",
                    "outer_n_splits": OUTER_FOLDS,
                    "inner": "sklearn.model_selection.GroupShuffleSplit",
                    "inner_test_size": INNER_VALIDATION_FRACTION,
                    "inner_random_state": inner_seed + fold_id,
                    "group_column": "canonical patient_id",
                    "assignment_inputs": ["canonical patient_id only"],
                },
                "train": train,
                "validation": validation,
                "test": test,
                "stats": {
                    "train": audit_stats(recordings, seizures, train),
                    "validation": audit_stats(recordings, seizures, validation),
                    "test": audit_stats(recordings, seizures, test),
                },
            }
        )

    patients = set(frame.patient_id.astype(str))
    validate(folds, patients)
    out.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        (out / f"fold_{fold['fold']}.json").write_text(
            json.dumps(fold, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    manifest: dict[str, object] = {
        "protocol": "v3-groupkfold-confirmatory-v1",
        "algorithm": "sklearn GroupKFold outer + GroupShuffleSplit inner",
        "assignment_inputs": ["canonical patient_id only"],
        "group_unit": "canonical patient",
        "patient_count": len(patients),
        "outer_fold_count": OUTER_FOLDS,
        "inner_validation_fraction": INNER_VALIDATION_FRACTION,
        "inner_seed_base": inner_seed,
        "source_recordings_sha256": sha256(recordings_path),
        "source_seizures_sha256": sha256(seizures_path),
        "folds": folds,
        "validation_policy": "patient-level GroupShuffleSplit within each outer development set",
        "test_policy": "outer patient-level test is evaluated once after validation threshold freeze",
        "forbidden_inputs": [
            "seizure count",
            "recording duration",
            "signal quality",
            "model predictions",
            "test outcomes",
        ],
    }
    (out / "cv_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("manifests/groupkfold_cv_v1"))
    parser.add_argument("--inner-seed", type=int, default=20260819)
    args = parser.parse_args()
    manifest = build_manifest(args.root.resolve(), args.out.resolve(), args.inner_seed)
    print(
        json.dumps(
            {
                "manifest": str(args.out.resolve() / "cv_manifest.json"),
                "folds": len(manifest["folds"]),
                "patients": manifest["patient_count"],
                "algorithm": manifest["algorithm"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
