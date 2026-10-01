"""One additional H3 sensitivity analysis: EEG-only features without patient centring.

This is a post-hoc descriptive robustness analysis of frozen H3 feature rows.
It does not fit a seizure detector or use labels from the outer test for model
selection. It addresses whether the modest EEG-only source identification in
the centred analysis depends on transductive within-patient centring.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


SOURCE_ROOT = Path(r"\\wsl.localhost\Ubuntu-24.04\root\b_false_alarm_atlas")
OUT = Path(__file__).resolve().parent.parent / "outputs" / "reports" / "research-a-robustness-corrections-v1"
FEATURE_PATH = SOURCE_ROOT / "outputs" / "reports" / "v3-groupkfold-confirmatory-v1" / "h2h3_fp_features.parquet"
MODELS = ("singlem", "cbramod", "tcn_gat")
FOLDS = tuple(range(5))
FEATURES = (
    "rms_median_uv", "rms_iqr_uv", "spatial_rms_dispersion", "spatial_rms_max_min_log",
    "line_noise_ratio", "clipping_fraction", "flat_fraction", "relative_delta_power",
    "relative_theta_power", "relative_alpha_power", "relative_beta_power",
    "relative_gamma_power", "high_low_ratio",
)
BOOTSTRAPS = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transform(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_values = train.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan)
    test_values = test.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan)
    medians = train_values.median().fillna(0.0)
    train_values, test_values = train_values.fillna(medians), test_values.fillna(medians)
    lower, upper = train_values.quantile(.01), train_values.quantile(.99)
    train_values = train_values.clip(lower=lower, upper=upper, axis="columns")
    test_values = test_values.clip(lower=lower, upper=upper, axis="columns")
    scaler = StandardScaler().fit(train_values)
    return scaler.transform(train_values), scaler.transform(test_values)


def bootstrap(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    patients = np.asarray(sorted(frame.patient.astype(str).unique()))
    patient_values = frame.patient.astype(str).to_numpy()
    labels = frame.label.to_numpy(int)
    probabilities = np.vstack(frame.probabilities)
    index = [np.flatnonzero(patient_values == patient) for patient in patients]
    values = []
    for _ in range(BOOTSTRAPS):
        rows = np.concatenate([index[i] for i in rng.integers(0, len(patients), size=len(patients))])
        try:
            values.append(roc_auc_score(labels[rows], probabilities[rows], multi_class="ovr", average="macro", labels=[0, 1, 2]))
        except ValueError:
            pass
    return np.asarray(values, dtype=float)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "h3_eegonly_uncentered_manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite formal result: {manifest_path}")
    data = pd.read_parquet(FEATURE_PATH)
    label_map = {name: index for index, name in enumerate(MODELS)}
    frames, folds = [], []
    for fold in FOLDS:
        frame = data[data.split == fold].copy()
        train, test = frame[frame.partition == "train"], frame[frame.partition == "test"]
        x_train, x_test = transform(train, test)
        y_train, y_test = train.model.map(label_map).to_numpy(), test.model.map(label_map).to_numpy()
        clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=20260826 + fold)
        clf.fit(x_train, y_train)
        prob = clf.predict_proba(x_test)
        auc = float(roc_auc_score(y_test, prob, multi_class="ovr", average="macro", labels=[0, 1, 2]))
        folds.append({"fold": fold, "test_macro_auroc": auc, "macro_auroc_chance": .5, "accuracy_chance": 1 / 3})
        out = test[["patient", "recording", "model", "seed", "start_s", "end_s"]].copy()
        out["fold"], out["label"] = fold, y_test
        out["probabilities"] = [list(x) for x in prob]
        frames.append(out)
        print(json.dumps({"fold": fold, "macro_auroc": auc}), flush=True)
    oof = pd.concat(frames, ignore_index=True)
    observed = float(roc_auc_score(oof.label, np.vstack(oof.probabilities), multi_class="ovr", average="macro", labels=[0, 1, 2]))
    values = bootstrap(oof, np.random.default_rng(20260826))
    summary = {"analysis_id": "eeg_only_uncentered", "oof_patients": int(oof.patient.nunique()), "oof_events": int(len(oof)), "oof_macro_auroc": observed, "patient_bootstrap_q025": float(np.quantile(values, .025)), "patient_bootstrap_q975": float(np.quantile(values, .975)), "bootstrap_replicates": int(len(values)), "macro_auroc_chance": .5, "accuracy_chance": 1 / 3}
    pd.DataFrame(folds).to_csv(OUT / "h3_eegonly_uncentered_by_fold.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUT / "h3_eegonly_uncentered_summary.csv", index=False)
    manifest_path.write_text(json.dumps({"analysis_id": "h3-eegonly-uncentered-v1", "status": "complete", "source_run": "v3-groupkfold-confirmatory-v1", "feature_source_sha256": sha256(FEATURE_PATH), "features": list(FEATURES), "patient_centered": False, "classifier": "multinomial_logistic_C1_class_weight_balanced", "fit_partition": "outer train only", "test_used_for_selection": False, "model_training": False, "macro_auroc_chance": .5, "accuracy_chance": 1 / 3, "patient_bootstrap_replicates": BOOTSTRAPS, "created_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
