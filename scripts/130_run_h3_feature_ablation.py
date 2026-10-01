"""Assess whether H3 source-model identification relies on EEG or detector-output features.

Four predeclared analyses use the frozen patient-level GroupKFold feature table:
full / EEG-only / detector-behavior-only, all patient-centered, plus the
uncentered full representation as a sensitivity analysis. The classifier and
outer partitions exactly follow the original H3 design. Macro-AUROC chance is
0.5; three-class accuracy chance is reported separately as 1/3.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


SOURCE_ROOT = Path(r"\\wsl.localhost\Ubuntu-24.04\root\b_false_alarm_atlas")
SOURCE_RUN = "v3-groupkfold-confirmatory-v1"
FEATURE_PATH = SOURCE_ROOT / "outputs" / "reports" / SOURCE_RUN / "h2h3_fp_features.parquet"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "reports" / "research-a-robustness-corrections-v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
FOLDS = tuple(range(5))
BEHAVIOR_FEATURES = ("event_duration_s", "peak_probability")
EEG_FEATURES = (
    "rms_median_uv", "rms_iqr_uv", "spatial_rms_dispersion", "spatial_rms_max_min_log",
    "line_noise_ratio", "clipping_fraction", "flat_fraction", "relative_delta_power",
    "relative_theta_power", "relative_alpha_power", "relative_beta_power", "relative_gamma_power", "high_low_ratio",
)
ANALYSES = {
    "full_patient_centered": {"features": BEHAVIOR_FEATURES + EEG_FEATURES, "patient_centered": True},
    "eeg_only_patient_centered": {"features": EEG_FEATURES, "patient_centered": True},
    "detector_behavior_only_patient_centered": {"features": BEHAVIOR_FEATURES, "patient_centered": True},
    "full_uncentered_sensitivity": {"features": BEHAVIOR_FEATURES + EEG_FEATURES, "patient_centered": False},
}
BOOTSTRAPS = 2000
RNG_SEED = 20260826


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(train: pd.DataFrame, target: pd.DataFrame, features: tuple[str, ...], centered: bool) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    tr, te = train.copy(), target.copy()
    if centered:
        for frame in (tr, te):
            frame.loc[:, features] = frame.loc[:, features] - frame.groupby("patient")[list(features)].transform("mean")
    train_values = tr.loc[:, features].replace([np.inf, -np.inf], np.nan)
    test_values = te.loc[:, features].replace([np.inf, -np.inf], np.nan)
    medians = train_values.median().fillna(0.0)
    train_values, test_values = train_values.fillna(medians), test_values.fillna(medians)
    lower, upper = train_values.quantile(.01), train_values.quantile(.99)
    train_values = train_values.clip(lower=lower, upper=upper, axis="columns")
    test_values = test_values.clip(lower=lower, upper=upper, axis="columns")
    scaler = StandardScaler().fit(train_values)
    return scaler.transform(train_values), scaler.transform(test_values), scaler


def grouped_bootstrap(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Patient-resampling CI without copying DataFrames at each replicate."""
    patient_values = frame.patient.astype(str).to_numpy()
    patients = np.asarray(sorted(np.unique(patient_values)))
    labels = frame.label.to_numpy(dtype=int)
    probabilities = np.vstack(frame.probabilities)
    row_indices = [np.flatnonzero(patient_values == patient) for patient in patients]
    values: list[float] = []
    for _ in range(BOOTSTRAPS):
        selected = rng.integers(0, len(patients), size=len(patients))
        rows = np.concatenate([row_indices[index] for index in selected])
        try:
            score = roc_auc_score(
                labels[rows],
                probabilities[rows],
                multi_class="ovr",
                average="macro",
                labels=list(range(len(MODELS))),
            )
            values.append(float(score))
        except ValueError:
            continue
    return np.asarray(values, dtype=float)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "h3_feature_ablation_manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite formal result: {manifest_path}")
    data = pd.read_parquet(FEATURE_PATH)
    labels = {name: index for index, name in enumerate(MODELS)}
    fold_rows: list[dict[str, object]] = []
    oof_frames: list[pd.DataFrame] = []
    for analysis_id, config in ANALYSES.items():
        features = tuple(config["features"])
        for fold in FOLDS:
            frame = data[data.split == fold].copy()
            train, test = frame[frame.partition == "train"].copy(), frame[frame.partition == "test"].copy()
            if train.empty or test.empty:
                raise RuntimeError(f"empty train/test on fold {fold}")
            x_train, x_test, _ = prepare(train, test, features, bool(config["patient_centered"]))
            y_train = train.model.map(labels).to_numpy()
            y_test = test.model.map(labels).to_numpy()
            clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=20260821 + fold)
            clf.fit(x_train, y_train)
            probability = clf.predict_proba(x_test)
            prediction = clf.predict(x_test)
            auc = float(
                roc_auc_score(
                    y_test,
                    probability,
                    multi_class="ovr",
                    average="macro",
                    labels=list(range(len(MODELS))),
                )
            )
            fold_rows.append({"analysis_id": analysis_id, "fold": fold, "features": json.dumps(features), "feature_count": len(features), "patient_centered": bool(config["patient_centered"]), "train_events": len(train), "test_events": len(test), "test_macro_auroc": auc, "test_accuracy": float(accuracy_score(y_test, prediction)), "test_balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)), "macro_auroc_chance": .5, "accuracy_chance": 1/3, "test_used_for_selection": False})
            oof = test[["patient", "recording", "model", "seed", "start_s", "end_s"]].copy()
            oof["analysis_id"], oof["fold"], oof["label"] = analysis_id, fold, y_test
            oof["probabilities"] = [list(row) for row in probability]
            oof_frames.append(oof)
            print(json.dumps({"analysis": analysis_id, "fold": fold, "macro_auroc": auc}), flush=True)
    oof = pd.concat(oof_frames, ignore_index=True)
    fold_table = pd.DataFrame(fold_rows)
    summary_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(RNG_SEED)
    for analysis_id in ANALYSES:
        target = oof[oof.analysis_id == analysis_id].copy()
        observed = float(
            roc_auc_score(
                target.label,
                np.vstack(target.probabilities),
                multi_class="ovr",
                average="macro",
                labels=list(range(len(MODELS))),
            )
        )
        boot = grouped_bootstrap(target, rng)
        summary_rows.append({"analysis_id": analysis_id, "oof_patients": target.patient.nunique(), "oof_events": len(target), "oof_macro_auroc": observed, "patient_bootstrap_q025": float(np.quantile(boot, .025)), "patient_bootstrap_q975": float(np.quantile(boot, .975)), "bootstrap_replicates": len(boot), "macro_auroc_chance": .5, "accuracy_chance": 1/3})
    fold_table.to_csv(OUT / "h3_feature_ablation_by_fold.csv", index=False)
    oof.to_parquet(OUT / "h3_feature_ablation_oof_predictions.parquet", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / "h3_feature_ablation_summary.csv", index=False)
    manifest = {"analysis_id": "h3-feature-ablation-v1", "status": "complete", "source_run": SOURCE_RUN, "feature_source_sha256": sha256(FEATURE_PATH), "feature_sets": {key: {"features": list(value["features"]), "patient_centered": value["patient_centered"]} for key, value in ANALYSES.items()}, "classifier": "multinomial_logistic_C1_class_weight_balanced", "outer_grouping": "canonical patient", "fit_partition": "outer train only", "test_used_for_selection": False, "macro_auroc_chance": .5, "accuracy_chance": 1/3, "patient_bootstrap_replicates": BOOTSTRAPS, "model_training": False, "created_utc": datetime.now(timezone.utc).isoformat()}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
