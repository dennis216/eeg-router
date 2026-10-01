"""Run the standard patient-level GroupKFold H4 router.

The router is intentionally independent of the older compact-v3 scripts.  For
each outer fold/seed it fits only on that fold's ``train`` discovery subset,
locks thresholds on validation, and evaluates the frozen test partition once.
No test score is used for model, threshold, or feature selection.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bfa.evaluation.eventize import Event, eventize
from bfa.evaluation.match import match_events


ROOT = Path("/root/b_false_alarm_atlas")
RUN_ID = "v3-groupkfold-confirmatory-v1"
REPORT_DIR = ROOT / "outputs" / "reports" / RUN_ID
FEATURE_PATH = REPORT_DIR / "h4_groupkfold_event_features.parquet"
GATE_PATH = REPORT_DIR / "gates.json"
MODELS = ("singlem", "cbramod", "tcn_gat")
FOLDS = tuple(range(5))
SEEDS = (17, 42, 3407)
MAX_ONSET_DIFF_S = 30.0
N_BOOTSTRAP = 2000
RNG_SEED = 20260821
SIGNAL_FEATURES = (
    "event_duration_s",
    "peak_probability",
    "rms_median_uv",
    "rms_iqr_uv",
    "spatial_rms_dispersion",
    "spatial_rms_max_min_log",
    "line_noise_ratio",
    "clipping_fraction",
    "flat_fraction",
    "relative_delta_power",
    "relative_theta_power",
    "relative_alpha_power",
    "relative_beta_power",
    "relative_gamma_power",
    "high_low_ratio",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _truths(seizures: pd.DataFrame, recording: str, evaluation_start_s: float) -> list[tuple[float, float]]:
    rows = seizures[seizures.recording_id.astype(str) == str(recording)]
    return [
        (max(float(row.start_s), evaluation_start_s), float(row.end_s))
        for row in rows.itertuples(index=False)
        if float(row.end_s) > evaluation_start_s
    ]


def _evaluation_meta(frame: pd.DataFrame) -> tuple[dict[str, float], dict[str, str]]:
    starts: dict[str, float] = {}
    patients: dict[str, str] = {}
    for recording, group in frame.groupby("recording", sort=False):
        ordered = group.sort_values("end_s", kind="stable")
        if ordered.empty:
            continue
        key = str(recording)
        starts[key] = float(ordered.end_s.iloc[0])
        patients[key] = str(ordered.patient.iloc[0])
    return starts, patients


def _score_events(
    events: pd.DataFrame,
    evaluation_starts: dict[str, float],
    evaluation_patients: dict[str, str],
    seizures: pd.DataFrame,
    durations: dict[str, float],
) -> dict[str, float]:
    true_positive = 0
    false_alarm = 0
    truth_count = 0
    nonseizure_hours = 0.0
    delays: list[float] = []
    for recording, patient in evaluation_patients.items():
        group = events[events.recording.astype(str) == recording] if not events.empty else events
        predictions = []
        if not group.empty:
            for row in group.sort_values("event_start", kind="stable").itertuples(index=False):
                predictions.append((float(row.event_start), float(row.event_end), float(row.peak_probability)))
        truth = _truths(seizures, recording, evaluation_starts[recording])
        pred_events = [Event(start, end, peak) for start, end, peak in predictions]
        matched = match_events(pred_events, truth)
        true_positive += len(matched.pairs)
        false_alarm += len(matched.unmatched_predictions)
        truth_count += len(truth)
        for pair in matched.pairs:
            delays.append(max(0.0, pred_events[pair.prediction_index].start_s - truth[pair.truth_index][0]))
        duration = float(durations[recording])
        seizure_seconds = sum(max(0.0, end - start) for start, end in truth)
        nonseizure_hours += max(0.0, duration - evaluation_starts[recording] - seizure_seconds) / 3600.0
    return {
        "true_positive_events": int(true_positive),
        "false_alarm_events": int(false_alarm),
        "truth_events": int(truth_count),
        "sensitivity": float(true_positive / truth_count) if truth_count else 0.0,
        "fa_per_24h": float(false_alarm * 24.0 / nonseizure_hours) if nonseizure_hours else math.inf,
        "mean_delay_s": float(np.mean(delays)) if delays else math.nan,
        "nonseizure_hours": nonseizure_hours,
    }


def _event_rows_from_model(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    subset = frame[frame.model == model]
    return subset.rename(columns={"start_s": "event_start", "end_s": "event_end"})[
        ["patient", "recording", "event_start", "event_end", "peak_probability"]
    ].copy()


def _load_probability_table(fold: int, seed: int, model: str, partition: str) -> pd.DataFrame:
    run = ROOT / "runs" / RUN_ID / model / f"split{fold}_seed{seed}_main"
    path = (
        run / "validation" / "probabilities_step_05000.parquet"
        if partition == "validation"
        else run / "test" / "probabilities.parquet"
    )
    table = pd.read_parquet(path)
    return table[["patient", "recording", "end", "probability"]].copy()


def _combined_events(
    tables: dict[str, pd.DataFrame], weights: tuple[float, float, float], threshold: float
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for model in MODELS:
        frame = tables[model].rename(columns={"probability": f"probability_{model}"})
        keys = ["patient", "recording", "end"]
        merged = frame if merged is None else merged.merge(frame, on=keys, how="inner", validate="one_to_one")
    if merged is None or merged.empty:
        return pd.DataFrame(columns=["patient", "recording", "event_start", "event_end", "peak_probability"])
    merged["score"] = sum(
        weight * merged[f"probability_{model}"] for weight, model in zip(weights, MODELS, strict=True)
    )
    rows: list[dict[str, object]] = []
    for recording, group in merged.groupby("recording", sort=False):
        ordered = group.sort_values("end", kind="stable")
        predictions = eventize(
            ordered.end.to_numpy(dtype=float),
            ordered.score.to_numpy(dtype=float),
            threshold=float(threshold),
        )
        patient = str(ordered.patient.iloc[0])
        for event in predictions:
            rows.append(
                {
                    "patient": patient,
                    "recording": str(recording),
                    "event_start": float(event.start_s),
                    "event_end": float(event.end_s),
                    "peak_probability": float(event.peak_probability),
                }
            )
    return pd.DataFrame(rows)


def _candidate_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Group near-synchronous model events and attach group-level features."""
    output: list[dict[str, object]] = []
    for recording, rec in frame.groupby("recording", sort=True):
        ordered = rec.sort_values(["start_s", "end_s", "model"], kind="stable")
        groups: list[dict[str, object]] = []
        for row in ordered.to_dict(orient="records"):
            start = float(row["start_s"])
            if not groups or start - float(groups[-1]["anchor_start"]) > MAX_ONSET_DIFF_S:
                groups.append({"anchor_start": start, "decision_end": float(row["end_s"]), "members": []})
            groups[-1]["members"].append(row)
            groups[-1]["decision_end"] = max(float(groups[-1]["decision_end"]), float(row["end_s"]))
        patient = str(ordered.patient.iloc[0])
        for group_index, group in enumerate(groups):
            members: dict[str, dict[str, object]] = {}
            for row in group["members"]:
                model = str(row["model"])
                if model not in members or float(row["peak_probability"]) > float(members[model]["peak_probability"]):
                    members[model] = row
            peaks = {
                model: float(members[model]["peak_probability"]) if model in members else 0.0
                for model in MODELS
            }
            group_id = f"{patient}|{recording}|{group_index}"
            for model, row in members.items():
                record = dict(row)
                record.update(
                    {
                        "group_id": group_id,
                        "group_decision_end": float(group["decision_end"]),
                        "group_peak_singlem": peaks["singlem"],
                        "group_peak_cbramod": peaks["cbramod"],
                        "group_peak_tcn_gat": peaks["tcn_gat"],
                        "group_uncertainty_singlem": peaks["singlem"] * (1.0 - peaks["singlem"]),
                        "group_uncertainty_cbramod": peaks["cbramod"] * (1.0 - peaks["cbramod"]),
                        "group_uncertainty_tcn_gat": peaks["tcn_gat"] * (1.0 - peaks["tcn_gat"]),
                        "source_model": model,
                    }
                )
                output.append(record)
    return pd.DataFrame(output)


def _router_columns() -> list[str]:
    columns = [f"group_peak_{model}" for model in MODELS]
    columns += [f"group_uncertainty_{model}" for model in MODELS]
    columns += [f"source_{model}" for model in MODELS]
    columns += [f"member_{feature}" for feature in SIGNAL_FEATURES]
    return columns


def _router_matrix(frame: pd.DataFrame, model_aware: bool) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    for model in MODELS:
        result[f"source_{model}"] = (result.source_model == model).astype(float) if model_aware else 0.0
    for feature in SIGNAL_FEATURES:
        result[f"member_{feature}"] = pd.to_numeric(result[feature], errors="coerce").fillna(0.0)
    columns = _router_columns()
    return result[columns].replace([np.inf, -np.inf], np.nan), columns


def _router_select(candidates: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=["patient", "recording", "event_start", "event_end", "peak_probability", "router_score", "selected_model", "group_id"])
    frame = candidates.copy()
    frame["router_score"] = np.asarray(scores, dtype=float)
    rows: list[dict[str, object]] = []
    for _, group in frame.groupby("group_id", sort=True):
        row = group.sort_values(["router_score", "peak_probability"], ascending=[False, False]).iloc[0]
        if float(row.router_score) < threshold:
            continue
        rows.append(
            {
                "patient": str(row.patient),
                "recording": str(row.recording),
                "event_start": float(row.start_s),
                "event_end": float(row.group_decision_end),
                "peak_probability": float(row.peak_probability),
                "router_score": float(row.router_score),
                "selected_model": str(row.source_model),
                "group_id": str(row.group_id),
            }
        )
    return pd.DataFrame(rows)


def _fit_router(train: pd.DataFrame, validation: pd.DataFrame, target_sensitivity: float, aware: bool, *, split: int, seed: int):
    train_x, columns = _router_matrix(train, aware)
    val_x, _ = _router_matrix(validation, aware)
    y = (train.event_type.astype(str) == "tp").astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise RuntimeError(f"router train fit lacks TP/FP split={split} seed={seed}")
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=64,
                    learning_rate_init=3e-4,
                    max_iter=150,
                    random_state=RNG_SEED + split * 100 + seed,
                    early_stopping=False,
                ),
            ),
        ]
    )
    model.fit(train_x, y)
    scores = model.predict_proba(val_x)[:, 1]
    thresholds = sorted(set(np.round(np.linspace(0.05, 0.99, 48), 3)) | set(np.round(np.quantile(scores, np.linspace(0.50, 0.995, 20)), 3)))
    return model, scores, thresholds, columns


def _patient_stats(events: pd.DataFrame, starts: dict[str, float], patients: dict[str, str], seizures: pd.DataFrame, durations: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for patient in sorted(set(patients.values())):
        patient_recordings = [recording for recording, value in patients.items() if value == patient]
        tp = fp = truth_count = 0
        hours = 0.0
        delays: list[float] = []
        for recording in patient_recordings:
            truth = _truths(seizures, recording, starts[recording])
            truth_count += len(truth)
            hours += max(0.0, float(durations[recording]) - starts[recording] - sum(max(0.0, b - a) for a, b in truth)) / 3600.0
            group = events[events.recording.astype(str) == recording] if not events.empty else events
            predictions = [
                Event(float(row.event_start), float(row.event_end), float(row.peak_probability))
                for row in group.itertuples(index=False)
            ]
            matched = match_events(predictions, truth)
            tp += len(matched.pairs)
            fp += len(matched.unmatched_predictions)
            delays.extend(max(0.0, predictions[p.prediction_index].start_s - truth[p.truth_index][0]) for p in matched.pairs)
        rows.append({"patient": patient, "sensitivity": float(tp / truth_count) if truth_count else 0.0, "fa_per_24h": float(fp * 24.0 / hours) if hours else 0.0, "mean_delay_s": float(np.mean(delays)) if delays else math.nan, "tp": tp, "fp": fp, "truth": truth_count})
    return pd.DataFrame(rows)


def _bootstrap(frame: pd.DataFrame, method: str, split: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(RNG_SEED + split * 1000 + seed)
    n = len(frame)
    if n == 0:
        return {"method": method, "split": split, "seed": seed, "bootstrap_replicates": N_BOOTSTRAP}
    sensitivity = np.empty(N_BOOTSTRAP)
    false_alarm = np.empty(N_BOOTSTRAP)
    delay = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        sample = frame.iloc[rng.integers(0, n, size=n)]
        sensitivity[index] = sample.sensitivity.mean()
        false_alarm[index] = sample.fa_per_24h.mean()
        delay[index] = sample.mean_delay_s.fillna(0.0).mean()
    return {
        "method": method,
        "split": split,
        "seed": seed,
        "bootstrap_replicates": N_BOOTSTRAP,
        "sensitivity_mean": float(frame.sensitivity.mean()),
        "sensitivity_q025": float(np.quantile(sensitivity, 0.025)),
        "sensitivity_q975": float(np.quantile(sensitivity, 0.975)),
        "fa_per_24h_mean": float(frame.fa_per_24h.mean()),
        "fa_per_24h_q025": float(np.quantile(false_alarm, 0.025)),
        "fa_per_24h_q975": float(np.quantile(false_alarm, 0.975)),
        "mean_delay_s_q025": float(np.quantile(delay, 0.025)),
        "mean_delay_s_q975": float(np.quantile(delay, 0.975)),
    }


def _threshold_for_combined(tables: dict[str, pd.DataFrame], weights: tuple[float, float, float], target: float, seizures: pd.DataFrame, durations: dict[str, float], starts: dict[str, float], patients: dict[str, str]) -> tuple[float, dict[str, float]]:
    combined = _combined_events(tables, weights, 0.0)
    values = pd.concat([tables[model].probability for model in MODELS], ignore_index=True).to_numpy(float)
    thresholds = sorted(set(np.round(np.linspace(0.01, 0.99, 49), 3)) | set(np.round(np.quantile(values, np.linspace(0.50, 0.995, 20)), 3)))
    scored: list[tuple[float, dict[str, float]]] = []
    for threshold in thresholds:
        events = _combined_events(tables, weights, float(threshold))
        scored.append((float(threshold), _score_events(events, starts, patients, seizures, durations)))
    eligible = [row for row in scored if row[1]["sensitivity"] >= target - 0.03]
    pool = eligible if eligible else scored
    return sorted(pool, key=lambda row: (row[1]["fa_per_24h"], -row[1]["sensitivity"], row[1]["mean_delay_s"] if math.isfinite(row[1]["mean_delay_s"]) else math.inf))[0]


def main() -> None:
    gates = json.loads(GATE_PATH.read_text())
    if gates.get("branch") != "router_candidate":
        raise RuntimeError(f"H4 gate branch is {gates.get('branch')!r}; router not permitted")
    features = pd.read_parquet(FEATURE_PATH)
    seizures = pd.read_parquet(ROOT / "manifests" / "seizures.parquet")
    recordings = pd.read_parquet(ROOT / "manifests" / "recordings.parquet")
    durations = recordings.set_index("recording_id")["duration_s"].astype(float).to_dict()
    oracle = pd.read_csv(REPORT_DIR / "h4_groupkfold_validation_oracle.csv")
    validation_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    waterfall_rows: list[dict[str, object]] = []
    artifacts: list[dict[str, object]] = []
    for fold in FOLDS:
        for seed in SEEDS:
            fold_features = features[(features.split == fold) & (features.seed == seed)].copy()
            train = fold_features[fold_features.partition == "train"].copy()
            validation = fold_features[fold_features.partition == "validation"].copy()
            test = fold_features[fold_features.partition == "test"].copy()
            train_candidates = _candidate_groups(train)
            validation_candidates = _candidate_groups(validation)
            test_candidates = _candidate_groups(test)
            validation_starts, validation_patients = _evaluation_meta(validation)
            test_starts, test_patients = _evaluation_meta(test)
            best_single_metrics: dict[str, dict[str, float]] = {}
            model_thresholds: dict[str, float] = {}
            val_model_events: dict[str, pd.DataFrame] = {}
            target_sensitivity = 0.0
            for model in MODELS:
                run = ROOT / "runs" / RUN_ID / model / f"split{fold}_seed{seed}_main"
                metrics = json.loads((run / "validation" / "metrics_step_05000.json").read_text())
                threshold = float(metrics["selected_event_operating_point"]["threshold"])
                model_thresholds[model] = threshold
                events = _event_rows_from_model(validation, model)
                val_model_events[model] = events
                score = _score_events(events, validation_starts, validation_patients, seizures, durations)
                best_single_metrics[model] = score
                target_sensitivity = max(target_sensitivity, score["sensitivity"])
            eligible = [(model, score) for model, score in best_single_metrics.items() if score["sensitivity"] >= target_sensitivity - 0.03]
            if not eligible:
                eligible = list(best_single_metrics.items())
            best_model, best_score = sorted(eligible, key=lambda item: (item[1]["fa_per_24h"], -item[1]["sensitivity"]))[0]
            validation_rows.append({"fold": fold, "seed": seed, "method": "best_single", "selected_model": best_model, "threshold_source": "frozen_validation_model_threshold", "threshold": model_thresholds[best_model], **best_score})

            validation_tables = {model: _load_probability_table(fold, seed, model, "validation") for model in MODELS}
            equal_weights = (1 / 3, 1 / 3, 1 / 3)
            oracle_row = oracle[(oracle.fold == fold) & (oracle.seed == seed)].iloc[0]
            fixed_weights = tuple(float(oracle_row[f"oracle_weight_{model}"]) for model in MODELS)
            for method, weights in (("equal_weight_average", equal_weights), ("validation_fixed_weights", fixed_weights)):
                threshold, score = _threshold_for_combined(validation_tables, weights, target_sensitivity, seizures, durations, validation_starts, validation_patients)
                validation_rows.append({"fold": fold, "seed": seed, "method": method, "threshold_source": "validation_only", "weight_singlem": weights[0], "weight_cbramod": weights[1], "weight_tcn_gat": weights[2], "threshold": threshold, **score})

            fitted: dict[str, tuple[Pipeline, float]] = {}
            for variant, aware in (("router_model_aware", True), ("router_no_model_id", False)):
                model, scores, thresholds, columns = _fit_router(train_candidates, validation_candidates, target_sensitivity, aware, split=fold, seed=seed)
                scored: list[tuple[float, dict[str, float]]] = []
                for threshold in thresholds:
                    selected = _router_select(validation_candidates, scores, float(threshold))
                    scored.append((float(threshold), _score_events(selected, validation_starts, validation_patients, seizures, durations)))
                eligible_scores = [row for row in scored if row[1]["sensitivity"] >= target_sensitivity - 0.03]
                pool = eligible_scores if eligible_scores else scored
                threshold, score = sorted(pool, key=lambda row: (row[1]["fa_per_24h"], -row[1]["sensitivity"], row[1]["mean_delay_s"] if math.isfinite(row[1]["mean_delay_s"]) else math.inf))[0]
                fitted[variant] = (model, float(threshold))
                validation_rows.append({"fold": fold, "seed": seed, "method": variant, "threshold_source": "validation_only", "threshold": threshold, "router_feature_count": len(columns), "model_aware": aware, **score})
                artifact_path = REPORT_DIR / f"h4_{variant}_fold{fold}_seed{seed}.joblib"
                joblib.dump(model, artifact_path)
                artifacts.append({"fold": fold, "seed": seed, "method": variant, "path": artifact_path.name, "sha256": _sha256(artifact_path), "feature_columns": columns, "model_aware": aware})

            # Test files are opened only after all validation workpoints above
            # have been frozen.
            test_tables = {model: _load_probability_table(fold, seed, model, "test") for model in MODELS}
            methods: dict[str, pd.DataFrame] = {
                "best_single": _event_rows_from_model(test, best_model),
                "equal_weight_average": _combined_events(test_tables, equal_weights, float(next(row["threshold"] for row in validation_rows[::-1] if row["fold"] == fold and row["seed"] == seed and row["method"] == "equal_weight_average"))),
                "validation_fixed_weights": _combined_events(test_tables, fixed_weights, float(next(row["threshold"] for row in validation_rows[::-1] if row["fold"] == fold and row["seed"] == seed and row["method"] == "validation_fixed_weights"))),
            }
            for variant, (model, threshold) in fitted.items():
                aware = variant == "router_model_aware"
                matrix, _ = _router_matrix(test_candidates, aware)
                scores = model.predict_proba(matrix)[:, 1]
                methods[variant] = _router_select(test_candidates, scores, threshold)
            best_patient = None
            for method, events in methods.items():
                score = _score_events(events, test_starts, test_patients, seizures, durations)
                test_rows.append({"fold": fold, "seed": seed, "method": method, "test_evaluation_count": 1, "threshold_source": "validation_only", **score})
                patient_frame = _patient_stats(events, test_starts, test_patients, seizures, durations)
                bootstrap_rows.append(_bootstrap(patient_frame, method, fold, seed))
                if method == "best_single":
                    best_patient = patient_frame.set_index("patient")
                elif best_patient is not None:
                    for row in patient_frame.itertuples(index=False):
                        baseline = best_patient.loc[row.patient]
                        waterfall_rows.append({"fold": fold, "seed": seed, "patient": row.patient, "method": method, "delta_fa_per_24h": float(row.fa_per_24h - baseline.fa_per_24h), "delta_sensitivity_pp": float((row.sensitivity - baseline.sensitivity) * 100.0), "delta_delay_s": float((row.mean_delay_s if math.isfinite(row.mean_delay_s) else 0.0) - (baseline.mean_delay_s if math.isfinite(baseline.mean_delay_s) else 0.0))})
            print(f"H4_GROUPKFOLD_ROUTER_DONE fold={fold} seed={seed} train_candidates={len(train_candidates)} validation_candidates={len(validation_candidates)} test_candidates={len(test_candidates)}", flush=True)

    pd.DataFrame(validation_rows).to_csv(REPORT_DIR / "h4_groupkfold_router_validation.csv", index=False)
    pd.DataFrame(test_rows).to_csv(REPORT_DIR / "h4_groupkfold_router_test.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(REPORT_DIR / "h4_groupkfold_router_patient_bootstrap.csv", index=False)
    pd.DataFrame(waterfall_rows).to_csv(REPORT_DIR / "h4_groupkfold_router_waterfall.csv", index=False)
    manifest = {
        "release_id": RUN_ID,
        "branch": "router_candidate",
        "train_fit_partition": "outer_train_discovery",
        "validation_workpoint_selection": True,
        "test_used_for_selection": False,
        "test_evaluation_count_per_method_fold_seed": 1,
        "router_variants": ["router_model_aware", "router_no_model_id"],
        "baselines": ["best_single", "equal_weight_average", "validation_fixed_weights"],
        "bootstrap_replicates": N_BOOTSTRAP,
        "inputs": {
            "gates_sha256": _sha256(GATE_PATH),
            "feature_manifest_sha256": _sha256(REPORT_DIR / "h4_groupkfold_event_features_manifest.json"),
            "feature_table_sha256": _sha256(FEATURE_PATH),
            "oracle_table_sha256": _sha256(REPORT_DIR / "h4_groupkfold_validation_oracle.csv"),
        },
        "outputs": {
            "validation": "h4_groupkfold_router_validation.csv",
            "test": "h4_groupkfold_router_test.csv",
            "patient_bootstrap": "h4_groupkfold_router_patient_bootstrap.csv",
            "waterfall": "h4_groupkfold_router_waterfall.csv",
            "artifacts": artifacts,
        },
    }
    manifest_path = REPORT_DIR / "h4_groupkfold_router_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"test_rows": len(test_rows), "validation_rows": len(validation_rows), "bootstrap_rows": len(bootstrap_rows), "manifest": str(manifest_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
