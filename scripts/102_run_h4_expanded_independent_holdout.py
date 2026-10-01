"""H4-expanded patient-level independent holdout router.

This analysis freezes a new patient-level calibration/validation/holdout split
using ordinary GroupShuffleSplit, fits only on calibration patients, selects
the router work point on validation patients, and evaluates the holdout once.
The parent release has historically touched these CHB patients, so the manifest
explicitly distinguishes independence within this new analysis from a pristine
historical/external holdout.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
from itertools import product
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

H4_SCRIPT = Path("/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/work/b_false_alarm_atlas/scripts/84_run_groupkfold_h4_router.py")
spec = importlib.util.spec_from_file_location("h4", H4_SCRIPT)
h4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h4)

ROOT = h4.ROOT
SOURCE = h4.REPORT_DIR
FEATURE_PATH = h4.FEATURE_PATH
OUT = ROOT / "outputs/reports/h4-expanded-independent-holdout-v1"
SPLIT_MANIFEST = OUT / "split_manifest.json"
SEEDS = (17, 42, 3407)
BACKBONE_SEED = 17
SPLIT_SEED_HOLDOUT = 20260824
SPLIT_SEED_VALIDATION = 20260825
HOLDOUT_PATIENTS = 5
VALIDATION_PATIENTS = 4
BOOTSTRAP = 2000
BOUNDARY = "independent_within_new_analysis_not_pristine_historical_holdout"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def make_split(patients):
    patients = np.asarray(sorted(map(str, patients)))
    outer = GroupShuffleSplit(n_splits=1, test_size=HOLDOUT_PATIENTS, random_state=SPLIT_SEED_HOLDOUT)
    dev_idx, hold_idx = next(outer.split(patients, groups=patients))
    development = patients[dev_idx]
    holdout = patients[hold_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_PATIENTS, random_state=SPLIT_SEED_VALIDATION)
    cal_idx, val_idx = next(inner.split(development, groups=development))
    calibration = development[cal_idx]
    validation = development[val_idx]
    if set(calibration) & set(validation) or set(calibration) & set(holdout) or set(validation) & set(holdout):
        raise RuntimeError("patient overlap in new holdout split")
    return calibration.tolist(), validation.tolist(), holdout.tolist()

def router_matrix(frame: pd.DataFrame, aware: bool):
    # Reuse the frozen H4 feature schema. All feature selection is frozen before
    # holdout scoring; no holdout feature is used to choose columns.
    return h4._router_matrix(frame, aware)

def fit_router(train, validation, target_sensitivity, aware, seed, existing_model=None):
    train_x, columns = router_matrix(train, aware)
    val_x, _ = router_matrix(validation, aware)
    y = (train.event_type.astype(str) == "tp").astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise RuntimeError("new holdout calibration fit lacks both TP and FP")
    model = existing_model or Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            alpha=1e-4, batch_size=64, learning_rate_init=3e-4,
            max_iter=150, random_state=seed, early_stopping=False,
        )),
    ])
    if existing_model is None:
        model.fit(train_x, y)
    scores = model.predict_proba(val_x)[:, 1]
    thresholds = sorted(set(np.round(np.linspace(0.05, 0.99, 48), 3)) |
                       set(np.round(np.quantile(scores, np.linspace(0.50, 0.995, 20)), 3)))
    scored = []
    starts, patients = h4._evaluation_meta(validation)
    seizures = pd.read_parquet(ROOT / "manifests/seizures.parquet")
    recordings = pd.read_parquet(ROOT / "manifests/recordings.parquet")
    durations = recordings.set_index("recording_id")["duration_s"].astype(float).to_dict()
    candidates = h4._candidate_groups(validation)
    candidate_scores = model.predict_proba(router_matrix(candidates, aware)[0])[:, 1]
    for threshold in thresholds:
        selected = h4._router_select(candidates, candidate_scores, float(threshold))
        scored.append((float(threshold), h4._score_events(selected, starts, patients, seizures, durations)))
    eligible = [row for row in scored if row[1]["sensitivity"] >= target_sensitivity - 0.03]
    pool = eligible if eligible else scored
    threshold, score = sorted(pool, key=lambda row: (
        row[1]["fa_per_24h"], -row[1]["sensitivity"],
        row[1]["mean_delay_s"] if np.isfinite(row[1]["mean_delay_s"]) else np.inf,
    ))[0]
    return model, float(threshold), score, columns

def patient_bootstrap(events, method, seed):
    meta_events = events.rename(columns={"event_start": "start_s", "event_end": "end_s"})
    starts, patients = h4._evaluation_meta(meta_events)
    seizures = pd.read_parquet(ROOT / "manifests/seizures.parquet")
    recordings = pd.read_parquet(ROOT / "manifests/recordings.parquet")
    durations = recordings.set_index("recording_id")["duration_s"].astype(float).to_dict()
    frame = h4._patient_stats(events, starts, patients, seizures, durations)
    rng = np.random.default_rng(20260826 + seed)
    n = len(frame)
    sens, fa, delay = [], [], []
    for _ in range(BOOTSTRAP):
        sample = frame.iloc[rng.integers(0, n, size=n)]
        sens.append(sample.sensitivity.mean())
        fa.append(sample.fa_per_24h.mean())
        delay.append(sample.mean_delay_s.fillna(0.0).mean())
    return {
        "method": method, "router_seed": seed, "patients": n, "bootstrap_replicates": BOOTSTRAP,
        "sensitivity_mean": float(frame.sensitivity.mean()),
        "sensitivity_ci_low": float(np.quantile(sens, 0.025)),
        "sensitivity_ci_high": float(np.quantile(sens, 0.975)),
        "fa_per_24h_mean": float(frame.fa_per_24h.mean()),
        "fa_per_24h_ci_low": float(np.quantile(fa, 0.025)),
        "fa_per_24h_ci_high": float(np.quantile(fa, 0.975)),
        "mean_delay_ci_low": float(np.quantile(delay, 0.025)),
        "mean_delay_ci_high": float(np.quantile(delay, 0.975)),
    }

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(FEATURE_PATH)
    features = features[features.seed == BACKBONE_SEED].copy()
    patients = sorted(features.patient.astype(str).unique())
    calibration, validation, holdout = make_split(patients)
    split_manifest = {
        "analysis_id": "h4-expanded-independent-holdout-v1",
        "status": "frozen_before_holdout_scoring",
        "split_unit": "canonical_patient",
        "selection_method": "GroupShuffleSplit",
        "holdout_patients": holdout,
        "calibration_patients": calibration,
        "validation_patients": validation,
        "holdout_patient_count": len(holdout),
        "calibration_patient_count": len(calibration),
        "validation_patient_count": len(validation),
        "backbone_seed": BACKBONE_SEED,
        "holdout_random_state": SPLIT_SEED_HOLDOUT,
        "validation_random_state": SPLIT_SEED_VALIDATION,
        "historical_claim_boundary": BOUNDARY,
        "source_feature_sha256": sha256(FEATURE_PATH),
    }
    SPLIT_MANIFEST.write_text(json.dumps(split_manifest, indent=2, sort_keys=True) + "\n")
    split_hash = sha256(SPLIT_MANIFEST)
    assign = {p: "calibration" for p in calibration}
    assign.update({p: "validation" for p in validation})
    assign.update({p: "holdout" for p in holdout})
    features["new_partition"] = features.patient.astype(str).map(assign)
    calibration_df = features[features.new_partition == "calibration"].copy()
    validation_df = features[features.new_partition == "validation"].copy()
    holdout_df = features[features.new_partition == "holdout"].copy()
    seizures = pd.read_parquet(ROOT / "manifests/seizures.parquet")
    recordings = pd.read_parquet(ROOT / "manifests/recordings.parquet")
    durations = recordings.set_index("recording_id")["duration_s"].astype(float).to_dict()
    val_starts, val_patients = h4._evaluation_meta(validation_df)
    hold_starts, hold_patients = h4._evaluation_meta(holdout_df)
    rows_val, rows_hold, rows_boot, artifacts = [], [], [], []
    calibration_candidates = h4._candidate_groups(calibration_df)
    validation_candidates = h4._candidate_groups(validation_df)
    holdout_candidates = h4._candidate_groups(holdout_df)
    target = 0.0
    model_scores = {}
    for model in h4.MODELS:
        events = h4._event_rows_from_model(validation_df, model)
        score = h4._score_events(events, val_starts, val_patients, seizures, durations)
        model_scores[model] = score
        target = max(target, score["sensitivity"])
    eligible = [(m, s) for m, s in model_scores.items() if s["sensitivity"] >= target - 0.03]
    best_model, best_score = sorted(eligible or list(model_scores.items()), key=lambda x: (x[1]["fa_per_24h"], -x[1]["sensitivity"]))[0]
    rows_val.append({"method": "best_single", "router_seed": None, "selected_model": best_model, **best_score})
    fitted = {}
    for router_seed in SEEDS:
        for variant, aware in (("router_model_aware", True), ("router_no_model_id", False)):
            path = OUT / f"h4_{variant}_seed{router_seed}.joblib"
            existing = joblib.load(path) if path.exists() else None
            model, threshold, score, columns = fit_router(calibration_candidates, validation_candidates, target, aware, router_seed, existing_model=existing)
            fitted[(router_seed, variant)] = (model, threshold, columns, aware)
            rows_val.append({
                "method": variant, "router_seed": router_seed, "threshold": threshold,
                "threshold_source": "new_holdout_validation_only", "model_aware": aware,
                **score,
            })
            joblib.dump(model, path)
            artifacts.append({"method": variant, "router_seed": router_seed, "path": path.name, "sha256": sha256(path), "feature_columns": columns})
    hold_methods = {"best_single": h4._event_rows_from_model(holdout_df, best_model)}
    for (router_seed, variant), (model, threshold, columns, aware) in fitted.items():
        scores = model.predict_proba(router_matrix(holdout_candidates, aware)[0])[:, 1]
        hold_methods[f"{variant}_seed{router_seed}"] = h4._router_select(holdout_candidates, scores, threshold)
    for method, events in hold_methods.items():
        score = h4._score_events(events, hold_starts, hold_patients, seizures, durations)
        rows_hold.append({"method": method, "test_evaluation_count": 1, "threshold_source": "new_holdout_validation_only" if method != "best_single" else "frozen_backbone_event_threshold", **score})
        rows_boot.append(patient_bootstrap(events, method, 0 if method == "best_single" else int(method.rsplit("seed", 1)[1])))
    pd.DataFrame(rows_val).to_csv(OUT / "validation_metrics.csv", index=False)
    pd.DataFrame(rows_hold).to_csv(OUT / "holdout_metrics.csv", index=False)
    pd.DataFrame(rows_boot).to_csv(OUT / "holdout_patient_bootstrap.csv", index=False)
    manifest = {
        "analysis_id": "h4-expanded-independent-holdout-v1",
        "status": "complete",
        "claim_boundary": BOUNDARY,
        "source_release": "v3-groupkfold-confirmatory-v1",
        "backbone_seed": BACKBONE_SEED,
        "router_seeds": list(SEEDS),
        "split_manifest_sha256": split_hash,
        "feature_table_sha256": sha256(FEATURE_PATH),
        "calibration_fit": True,
        "validation_workpoint_selection": True,
        "holdout_used_for_selection": False,
        "holdout_evaluation_count_per_method": 1,
        "holdout_patient_count": len(holdout),
        "patient_bootstrap_replicates": BOOTSTRAP,
        "model_training": "router_only",
        "router_variants": ["router_model_aware", "router_no_model_id"],
        "outputs": ["validation_metrics.csv", "holdout_metrics.csv", "holdout_patient_bootstrap.csv"],
        "artifacts": artifacts,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": manifest["status"], "holdout_patients": holdout, "calibration_patients": calibration, "validation_patients": validation, "holdout_rows": len(rows_hold)}, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
