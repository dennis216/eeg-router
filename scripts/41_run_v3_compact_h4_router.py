"""Run the frozen H4 router branch with train/validation/test separation.

The router is an event-level model-selection policy.  It scores the available
members of a temporally grouped alarm, chooses one model, and suppresses the
group when its score is below the validation-locked work point.  The phenotype
version receives a frozen H2 prototype distance; the matched control receives a
constant-zero distance column so its MLP has the same parameter count.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bfa.evaluation.eventize import Event, eventize
from bfa.evaluation.match import match_events


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "outputs" / "reports" / "v3-compact-extension-v1"
EVENT_DIR = ROOT / "outputs" / "events"
FEATURE_PATH = EVENT_DIR / "v3-compact-extension-v1_h4_event_features.parquet"
MODELS = ("singlem", "cbramod", "tcn_gat")
RUNS = (
    ("BSN-pilot-v1", 17, (17, 42, 3407)),
    ("BSN-pilot-v1-multisplit", 42, (17, 42)),
    ("v3-lite-2split", 42, (3407,)),
)
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
MAX_ONSET_DIFF_S = 30.0
TARGET_SENSITIVITY = 0.80
N_BOOTSTRAP = 2000
RNG_SEED = 20260811


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truths(seizures: pd.DataFrame, recording: str, evaluation_start: float) -> list[tuple[float, float]]:
    rows = seizures[seizures.recording_id == recording]
    return [
        (max(float(row.start_s), evaluation_start), float(row.end_s))
        for row in rows.itertuples(index=False)
        if float(row.end_s) > evaluation_start
    ]


def _load_probability_table(namespace: str, split: int, seed: int, model: str, partition: str) -> pd.DataFrame:
    run = ROOT / "runs" / namespace / model / f"split{split}_seed{seed}_main"
    path = (
        run / "validation" / "probabilities_step_05000.parquet"
        if partition == "validation"
        else run / "test" / "probabilities.parquet"
    )
    table = pd.read_parquet(path)
    return table[["patient", "recording", "start", "end", "probability"]].copy()


def _evaluation_meta(table: pd.DataFrame) -> tuple[dict[str, float], dict[str, str]]:
    starts: dict[str, float] = {}
    patients: dict[str, str] = {}
    for recording, group in table.groupby("recording", sort=False):
        ordered = group.sort_values("end", kind="stable")
        starts[str(recording)] = float(ordered.end.iloc[0])
        patients[str(recording)] = str(ordered.patient.iloc[0])
    return starts, patients


def _score_events(
    events: pd.DataFrame,
    evaluation_starts: dict[str, float],
    evaluation_patients: dict[str, str],
    seizures: pd.DataFrame,
    recordings: pd.DataFrame,
) -> dict[str, float]:
    durations = recordings.set_index("recording_id")["duration_s"].astype(float).to_dict()
    tp = fp = truth_count = 0
    delays: list[float] = []
    for recording, patient in evaluation_patients.items():
        group = events[events.recording.astype(str) == recording] if not events.empty else events
        predictions = [] if group.empty else [
            Event(float(row.event_start), float(row.event_end), float(row.peak_probability))
            for row in group.sort_values("event_start", kind="stable").itertuples(index=False)
        ]
        truth = _truths(seizures, recording, evaluation_starts[recording])
        matched = match_events(predictions, truth)
        tp += len(matched.pairs)
        fp += len(matched.unmatched_predictions)
        truth_count += len(truth)
        for pair in matched.pairs:
            delays.append(max(0.0, predictions[pair.prediction_index].start_s - truth[pair.truth_index][0]))
    nonseizure_hours = 0.0
    for recording in evaluation_patients:
        start = evaluation_starts[recording]
        duration = float(durations[recording])
        truth = _truths(seizures, recording, start)
        seizure_seconds = sum(max(0.0, end - begin) for begin, end in truth)
        nonseizure_hours += max(0.0, duration - start - seizure_seconds) / 3600.0
    sensitivity = float(tp / truth_count) if truth_count else 0.0
    fa = float(fp * 24.0 / nonseizure_hours) if nonseizure_hours else math.inf
    return {
        "true_positive_events": int(tp),
        "false_alarm_events": int(fp),
        "truth_events": int(truth_count),
        "sensitivity": sensitivity,
        "fa_per_24h": fa,
        "mean_delay_s": float(np.mean(delays)) if delays else math.nan,
        "nonseizure_hours": nonseizure_hours,
    }


def _event_rows_from_probabilities(table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for recording, group in table.groupby("recording", sort=False):
        ordered = group.sort_values("end", kind="stable")
        predictions = eventize(
            ordered.end.to_numpy(float), ordered.probability.to_numpy(float), threshold=float(threshold)
        )
        patient = str(ordered.patient.iloc[0])
        for event in predictions:
            rows.append(
                {
                    "patient": patient,
                    "recording": str(recording),
                    "event_start": event.start_s,
                    "event_end": event.end_s,
                    "peak_probability": event.peak_probability,
                }
            )
    return pd.DataFrame(rows, columns=["patient", "recording", "event_start", "event_end", "peak_probability"])


def _score_weighted(
    tables: dict[str, pd.DataFrame],
    weights: tuple[float, float, float],
    threshold: float,
    seizures: pd.DataFrame,
    recordings: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float], dict[str, str]]:
    merged: pd.DataFrame | None = None
    for model in MODELS:
        frame = tables[model].rename(columns={"probability": f"probability_{model}"})
        keys = ["patient", "recording", "end"]
        merged = frame if merged is None else merged.merge(frame, on=keys, how="inner", validate="one_to_one")
    merged["probability"] = sum(
        weight * merged[f"probability_{model}"] for weight, model in zip(weights, MODELS, strict=True)
    )
    table = merged[["patient", "recording", "end", "probability"]]
    events = _event_rows_from_probabilities(table, threshold)
    starts, patients = _evaluation_meta(tables[MODELS[0]])
    return events, _score_events(events, starts, patients, seizures, recordings), starts, patients


def _choose_threshold(
    tables: dict[str, pd.DataFrame],
    weights: tuple[float, float, float],
    seizures: pd.DataFrame,
    recordings: pd.DataFrame,
) -> tuple[float, dict[str, float]]:
    values = pd.concat([tables[model].probability for model in MODELS], ignore_index=True).to_numpy(float)
    thresholds = sorted(set(np.round(np.linspace(0.01, 0.99, 49), 3)) | set(np.round(np.quantile(values, np.linspace(0.50, 0.995, 20)), 3)))
    rows: list[tuple[float, dict[str, float]]] = []
    for threshold in thresholds:
        _, score, _, _ = _score_weighted(tables, weights, float(threshold), seizures, recordings)
        rows.append((float(threshold), score))
    eligible = [row for row in rows if row[1]["sensitivity"] >= TARGET_SENSITIVITY]
    pool = eligible if eligible else rows
    selected = sorted(
        pool,
        key=lambda row: (
            row[1]["fa_per_24h"],
            -row[1]["sensitivity"],
            row[1]["mean_delay_s"] if math.isfinite(row[1]["mean_delay_s"]) else math.inf,
        ),
    )[0]
    return selected


def _add_frozen_phenotype_distance(features: pd.DataFrame) -> pd.DataFrame:
    output = features.copy()
    output["phenotype_distance"] = 0.0
    selection = pd.read_csv(REPORT_DIR / "h2_cluster_selection.csv").set_index("split")
    for split in sorted(output.split.unique()):
        validation = output[(output.split == split) & (output.partition == "validation") & (output.event_type == "fp")].copy()
        if validation.empty:
            continue
        lower = validation.loc[:, SIGNAL_FEATURES].quantile(0.01)
        upper = validation.loc[:, SIGNAL_FEATURES].quantile(0.99)
        clipped = validation.loc[:, SIGNAL_FEATURES].clip(lower=lower, upper=upper, axis="columns")
        scaler = StandardScaler().fit(clipped)
        points = scaler.transform(clipped)
        pca = PCA(n_components=0.90, svd_solver="full", random_state=RNG_SEED).fit(points)
        k = int(selection.loc[split, "selected_k"])
        clusterer = KMeans(n_clusters=k, n_init=50, random_state=RNG_SEED).fit(pca.transform(points))
        mask = output.split == split
        target = output.loc[mask, SIGNAL_FEATURES].clip(lower=lower, upper=upper, axis="columns")
        target_points = pca.transform(scaler.transform(target))
        distances = np.sqrt(((target_points[:, None, :] - clusterer.cluster_centers_[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        output.loc[mask, "phenotype_distance"] = distances
    return output


def _candidate_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Group near-synchronous model events; decision time is max member end."""
    rows: list[dict[str, object]] = []
    for recording, recording_frame in frame.groupby("recording", sort=True):
        ordered = recording_frame.sort_values(["start_s", "end_s", "model"], kind="stable")
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
            peaks = {model: float(members[model]["peak_probability"]) if model in members else 0.0 for model in MODELS}
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
                rows.append(record)
    return pd.DataFrame(rows)


def _router_feature_columns() -> list[str]:
    columns = []
    columns.extend([f"group_peak_{model}" for model in MODELS])
    columns.extend([f"group_uncertainty_{model}" for model in MODELS])
    columns.extend([f"source_{model}" for model in MODELS])
    columns.extend([f"member_{feature}" for feature in SIGNAL_FEATURES])
    columns.append("phenotype_distance")
    return columns


def _router_matrix(candidates: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = candidates.copy()
    for model in MODELS:
        frame[f"source_{model}"] = (frame.source_model == model).astype(float)
    for feature in SIGNAL_FEATURES:
        frame[f"member_{feature}"] = pd.to_numeric(frame[feature], errors="coerce").fillna(0.0)
    columns = _router_feature_columns()
    return frame[columns].replace([np.inf, -np.inf], np.nan), columns


def _router_selected(candidates: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    frame = candidates.copy()
    frame["router_score"] = np.asarray(scores, dtype=float)
    selected_rows: list[dict[str, object]] = []
    for _, group in frame.groupby("group_id", sort=True):
        row = group.sort_values(["router_score", "peak_probability"], ascending=[False, False]).iloc[0]
        if float(row.router_score) < threshold:
            continue
        selected_rows.append(
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
    return pd.DataFrame(
        selected_rows,
        columns=["patient", "recording", "event_start", "event_end", "peak_probability", "router_score", "selected_model", "group_id"],
    )


def _fit_router(
    train_candidates: pd.DataFrame,
    validation_candidates: pd.DataFrame,
    eval_starts: dict[str, float],
    eval_patients: dict[str, str],
    seizures: pd.DataFrame,
    recordings: pd.DataFrame,
) -> tuple[Pipeline, dict[str, object], np.ndarray, np.ndarray]:
    train_x, columns = _router_matrix(train_candidates)
    validation_x, _ = _router_matrix(validation_candidates)
    y = (train_candidates.event_type.astype(str) == "tp").astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise RuntimeError("H4 router train fit requires both TP and FP events")
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=32,
                    learning_rate_init=3e-4,
                    max_iter=150,
                    random_state=RNG_SEED,
                    early_stopping=False,
                ),
            ),
        ]
    )
    model.fit(train_x, y)
    validation_scores = model.predict_proba(validation_x)[:, 1]
    values = sorted(set(np.round(np.linspace(0.05, 0.99, 48), 3)) | set(np.round(np.quantile(validation_scores, np.linspace(0.50, 0.995, 20)), 3)))
    scored: list[tuple[float, dict[str, float]]] = []
    for threshold in values:
        selected = _router_selected(validation_candidates, validation_scores, float(threshold))
        score = _score_events(selected, eval_starts, eval_patients, seizures, recordings)
        scored.append((float(threshold), score))
    eligible = [row for row in scored if row[1]["sensitivity"] >= TARGET_SENSITIVITY]
    pool = eligible if eligible else scored
    threshold, score = sorted(pool, key=lambda row: (row[1]["fa_per_24h"], -row[1]["sensitivity"], row[1]["mean_delay_s"] if math.isfinite(row[1]["mean_delay_s"]) else math.inf))[0]
    return model, {"threshold": threshold, **score, "feature_columns": columns, "hidden_layers": [128, 64]}, validation_scores, np.asarray([threshold], dtype=float)


def _patient_stats(
    events: pd.DataFrame,
    eval_starts: dict[str, float],
    eval_patients: dict[str, str],
    seizures: pd.DataFrame,
    recordings: pd.DataFrame,
) -> pd.DataFrame:
    durations = recordings.set_index("recording_id")["duration_s"].astype(float).to_dict()
    rows: list[dict[str, object]] = []
    for patient in sorted(set(eval_patients.values())):
        tp = fp = truth_count = 0
        delays: list[float] = []
        hours = 0.0
        for recording, rec_patient in eval_patients.items():
            if rec_patient != patient:
                continue
            start = eval_starts[recording]
            truth = _truths(seizures, recording, start)
            truth_count += len(truth)
            hours += max(0.0, float(durations[recording]) - start - sum(max(0.0, b - a) for a, b in truth)) / 3600.0
            group = events[events.recording.astype(str) == recording] if not events.empty else events
            predictions = [Event(float(row.event_start), float(row.event_end), float(row.peak_probability)) for row in group.itertuples(index=False)]
            matched = match_events(predictions, truth)
            tp += len(matched.pairs)
            fp += len(matched.unmatched_predictions)
            delays.extend(max(0.0, predictions[p.prediction_index].start_s - truth[p.truth_index][0]) for p in matched.pairs)
        rows.append(
            {
                "patient": patient,
                "sensitivity": float(tp / truth_count) if truth_count else 0.0,
                "fa_per_24h": float(fp * 24.0 / hours) if hours else 0.0,
                "mean_delay_s": float(np.mean(delays)) if delays else math.nan,
                "tp": tp,
                "fp": fp,
                "truth": truth_count,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(patient_frame: pd.DataFrame, method: str, split: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(RNG_SEED + split * 100 + seed)
    n = len(patient_frame)
    sensitivities = np.empty(N_BOOTSTRAP, dtype=float)
    false_alarms = np.empty(N_BOOTSTRAP, dtype=float)
    delays = np.empty(N_BOOTSTRAP, dtype=float)
    for index in range(N_BOOTSTRAP):
        sampled = rng.integers(0, n, size=n)
        subset = patient_frame.iloc[sampled]
        sensitivities[index] = float(subset.sensitivity.mean())
        false_alarms[index] = float(subset.fa_per_24h.mean())
        delays[index] = float(subset.mean_delay_s.fillna(0.0).mean())
    return {
        "method": method,
        "split": split,
        "seed": seed,
        "bootstrap_replicates": N_BOOTSTRAP,
        "sensitivity_q025": float(np.quantile(sensitivities, 0.025)),
        "sensitivity_q975": float(np.quantile(sensitivities, 0.975)),
        "fa_per_24h_q025": float(np.quantile(false_alarms, 0.025)),
        "fa_per_24h_q975": float(np.quantile(false_alarms, 0.975)),
        "mean_delay_s_q025": float(np.quantile(delays, 0.025)),
        "mean_delay_s_q975": float(np.quantile(delays, 0.975)),
    }


def _gate_weights(split: int, seed: int) -> tuple[float, float, float]:
    gates = json.loads((REPORT_DIR / "gates.json").read_text())
    for group in gates["oracle_gate"]["groups"]:
        if int(group["split"]) == split and int(group["seed"]) == seed:
            oracle = group["oracle"]
            return tuple(float(oracle[f"weight_{model}"]) for model in MODELS)
    return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


def main() -> None:
    features = _add_frozen_phenotype_distance(pd.read_parquet(FEATURE_PATH))
    seizures = pd.read_parquet(ROOT / "manifests" / "seizures.parquet")
    recordings = pd.read_parquet(ROOT / "manifests" / "recordings.parquet")
    test_results: list[dict[str, object]] = []
    validation_workpoints: list[dict[str, object]] = []
    patient_bootstrap: list[dict[str, object]] = []
    waterfall: list[dict[str, object]] = []
    model_artifacts: list[dict[str, object]] = []
    for namespace, split, seeds in RUNS:
        for seed in seeds:
            # The model-event features are frozen before this loop.  Test is
            # evaluated only after all train/validation choices below are done.
            train_frame = features[(features.namespace == namespace) & (features.split == split) & (features.seed == seed) & (features.partition == "train_fit_subset")].copy()
            validation_frame = features[(features.namespace == namespace) & (features.split == split) & (features.seed == seed) & (features.partition == "validation")].copy()
            train_candidates = _candidate_groups(train_frame)
            validation_candidates = _candidate_groups(validation_frame)
            validation_tables = {model: _load_probability_table(namespace, split, seed, model, "validation") for model in MODELS}
            validation_starts, validation_patients = _evaluation_meta(validation_tables[MODELS[0]])
            router_models: dict[str, Pipeline] = {}
            router_workpoints: dict[str, dict[str, object]] = {}
            for variant, zero_pheno in (("router_phenotype", False), ("router_no_phenotype", True)):
                train_variant = train_candidates.copy()
                val_variant = validation_candidates.copy()
                if zero_pheno:
                    train_variant["phenotype_distance"] = 0.0
                    val_variant["phenotype_distance"] = 0.0
                model, workpoint, val_scores, _ = _fit_router(
                    train_variant,
                    val_variant,
                    validation_starts,
                    validation_patients,
                    seizures,
                    recordings,
                )
                router_models[variant] = model
                router_workpoints[variant] = workpoint
                validation_workpoints.append({"namespace": namespace, "split": split, "seed": seed, "method": variant, **{key: value for key, value in workpoint.items() if key not in {"feature_columns"}}})
                model_path = REPORT_DIR / f"h4_{variant}_split{split}_seed{seed}.joblib"
                joblib.dump(model, model_path)
                model_artifacts.append({"method": variant, "split": split, "seed": seed, "path": str(model_path), "sha256": _sha256(model_path), "train_rows": int(len(train_variant)), "feature_columns": workpoint["feature_columns"], "zeroed_phenotype_distance": zero_pheno})

            # Validation-lock the two probability-only baselines.
            equal_weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
            equal_threshold, equal_val_score = _choose_threshold(validation_tables, equal_weights, seizures, recordings)
            fixed_weights = _gate_weights(split, seed)
            fixed_threshold, fixed_val_score = _choose_threshold(validation_tables, fixed_weights, seizures, recordings)
            model_validation_scores: dict[str, dict[str, float]] = {}
            model_thresholds: dict[str, float] = {}
            for model in MODELS:
                run = ROOT / "runs" / namespace / model / f"split{split}_seed{seed}_main"
                metrics = json.loads((run / "validation" / "metrics_step_05000.json").read_text())
                threshold = float(metrics["selected_event_operating_point"]["threshold"])
                model_thresholds[model] = threshold
                event_rows = _event_rows_from_probabilities(validation_tables[model], threshold)
                model_validation_scores[model] = _score_events(event_rows, validation_starts, validation_patients, seizures, recordings)
            eligible = [(model, score) for model, score in model_validation_scores.items() if score["sensitivity"] >= TARGET_SENSITIVITY - 0.03]
            if not eligible:
                eligible = list(model_validation_scores.items())
            best_model, best_val_score = sorted(eligible, key=lambda item: (item[1]["fa_per_24h"], -item[1]["sensitivity"]))[0]
            validation_workpoints.extend(
                [
                    {"namespace": namespace, "split": split, "seed": seed, "method": "best_single", "selected_model": best_model, "threshold": model_thresholds[best_model], **best_val_score},
                    {"namespace": namespace, "split": split, "seed": seed, "method": "equal_weight_average", "weight_singlem": 1 / 3, "weight_cbramod": 1 / 3, "weight_tcn_gat": 1 / 3, "threshold": equal_threshold, **equal_val_score},
                    {"namespace": namespace, "split": split, "seed": seed, "method": "validation_fixed_weights", "weight_singlem": fixed_weights[0], "weight_cbramod": fixed_weights[1], "weight_tcn_gat": fixed_weights[2], "threshold": fixed_threshold, **fixed_val_score},
                ]
            )

            # Test is read only after model fitting and validation work points
            # are frozen. No test threshold or score is used for selection.
            test_frame = features[(features.namespace == namespace) & (features.split == split) & (features.seed == seed) & (features.partition == "test")].copy()
            test_candidates = _candidate_groups(test_frame)
            test_tables = {model: _load_probability_table(namespace, split, seed, model, "test") for model in MODELS}
            test_starts, test_patients = _evaluation_meta(test_tables[MODELS[0]])
            method_events: dict[str, pd.DataFrame] = {}
            method_events["best_single"] = _event_rows_from_probabilities(test_tables[best_model], model_thresholds[best_model])
            method_events["equal_weight_average"], _, _, _ = _score_weighted(test_tables, equal_weights, equal_threshold, seizures, recordings)
            method_events["validation_fixed_weights"], _, _, _ = _score_weighted(test_tables, fixed_weights, fixed_threshold, seizures, recordings)
            for variant, model in router_models.items():
                candidate = test_candidates.copy()
                if variant == "router_no_phenotype":
                    candidate["phenotype_distance"] = 0.0
                x, _ = _router_matrix(candidate)
                scores = model.predict_proba(x)[:, 1]
                method_events[variant] = _router_selected(candidate, scores, float(router_workpoints[variant]["threshold"]))
            for method, event_rows in method_events.items():
                score = _score_events(event_rows, test_starts, test_patients, seizures, recordings)
                test_results.append({"namespace": namespace, "split": split, "seed": seed, "partition": "test", "method": method, "test_evaluation_count": 1, "selected_model": best_model if method == "best_single" else None, "threshold_source": "validation_only", **score})
                patient_frame = _patient_stats(event_rows, test_starts, test_patients, seizures, recordings)
                patient_bootstrap.append(_bootstrap(patient_frame, method, split, seed))
                if method != "best_single":
                    best_patient = _patient_stats(method_events["best_single"], test_starts, test_patients, seizures, recordings).set_index("patient")
                    for row in patient_frame.itertuples(index=False):
                        baseline = best_patient.loc[row.patient]
                        waterfall.append({"namespace": namespace, "split": split, "seed": seed, "patient": row.patient, "method": method, "delta_fa_per_24h": float(row.fa_per_24h - baseline.fa_per_24h), "delta_sensitivity_pp": float((row.sensitivity - baseline.sensitivity) * 100.0), "delta_delay_s": float((row.mean_delay_s if math.isfinite(row.mean_delay_s) else 0.0) - (baseline.mean_delay_s if math.isfinite(baseline.mean_delay_s) else 0.0))})
            print(f"H4_ROUTER_TEST_DONE split={split} seed={seed} best_single={best_model} router_pheno_threshold={router_workpoints['router_phenotype']['threshold']:.3f}", flush=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(test_results).to_csv(REPORT_DIR / "h4_router_test_results.csv", index=False)
    pd.DataFrame(validation_workpoints).to_csv(REPORT_DIR / "h4_router_validation_workpoints.csv", index=False)
    pd.DataFrame(patient_bootstrap).to_csv(REPORT_DIR / "h4_router_patient_bootstrap.csv", index=False)
    pd.DataFrame(waterfall).to_csv(REPORT_DIR / "h4_router_waterfall.csv", index=False)
    manifest = {
        "release_id": "v3-compact-extension-v1",
        "branch": "router",
        "train_fit_partition": "train_fit_subset",
        "train_fit_subset_max_windows": 5000,
        "validation_workpoint_target_sensitivity": TARGET_SENSITIVITY,
        "test_used_for_selection": False,
        "test_evaluation_count_per_method_run": 1,
        "router_hidden_layers": [128, 64],
        "router_variants": ["router_phenotype", "router_no_phenotype"],
        "probability_baselines": ["best_single", "equal_weight_average", "validation_fixed_weights"],
        "inputs": {
            "gates_sha256": _sha256(REPORT_DIR / "gates.json"),
            "h4_event_features_sha256": _sha256(FEATURE_PATH),
            "train_event_table_sha256": json.loads((REPORT_DIR / "train_event_table_manifest.json").read_text())["event_table_sha256"],
            "split17_train_fit_manifest_sha256": _sha256(ROOT / "manifests" / "h4_train_fit_index_split17.parquet"),
            "split42_train_fit_manifest_sha256": _sha256(ROOT / "manifests" / "h4_train_fit_index_split42.parquet"),
        },
        "outputs": {
            "test_results": "h4_router_test_results.csv",
            "validation_workpoints": "h4_router_validation_workpoints.csv",
            "patient_bootstrap": "h4_router_patient_bootstrap.csv",
            "waterfall": "h4_router_waterfall.csv",
            "model_artifacts": model_artifacts,
        },
        "interpretation": "preliminary compute-budgeted H4 router evaluation; no H5 or TUH was run",
    }
    (REPORT_DIR / "h4_router_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"test_rows": len(test_results), "bootstrap_rows": len(patient_bootstrap), "waterfall_rows": len(waterfall), "manifest": str(REPORT_DIR / 'h4_router_manifest.json')}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
