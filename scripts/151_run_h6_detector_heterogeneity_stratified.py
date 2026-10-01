"""Patient--detector heterogeneity and stratified router analysis.

This is an exploratory extension of the frozen H6 chronological
personalisation experiment.  It does not retrain any seizure-detector
backbone and it never changes a frozen detector threshold.  The only fitted
models in this script are small, explicitly marked ``router_only`` logistic
routers.  Every router is fitted on an early calibration portion, a later
calibration portion selects its operating threshold, and the chronological
holdout is evaluated once.

The analysis has three purposes:

1. quantify the patient x detector relationship (does the preferred detector
   vary by patient?);
2. rank patients by calibration-only FP event disagreement and train a
   patient-specific router for every patient (representative extremes are
   reported, but no patient is silently dropped from the statistics);
3. pool the highest- and lowest-disagreement patients into two group routers,
   evaluate each on its own chronological holdouts, and cross-apply the
   routers to the opposite stratum.

The input event table is the already frozen standard GroupKFold OOF event
table.  This namespace is deliberately separate from h4 and h6 outputs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
RUN_ID = "h6-detector-heterogeneity-stratified-v1"
REPORT_DIR = ROOT / "outputs" / "reports" / RUN_ID
SOURCE_RUN = "v3-groupkfold-confirmatory-v1"
H4_FEATURE_PATH = ROOT / "outputs" / "reports" / SOURCE_RUN / "h4_groupkfold_event_features.parquet"
H4_FEATURE_MANIFEST_PATH = ROOT / "outputs" / "reports" / SOURCE_RUN / "h4_groupkfold_event_features_manifest.json"
H6_SPLIT_PATH = ROOT / "outputs" / "reports" / "h6-supervised-personalized-router-v1" / "chronological_split_manifest.json"
SEIZURE_PATH = ROOT / "manifests" / "seizures.parquet"
RECORDING_PATH = ROOT / "manifests" / "recordings.parquet"
MODELS = ("singlem", "cbramod", "tcn_gat")
SEEDS = (17, 42, 3407)
N_EXTREME = 5
N_BOOTSTRAP = 2000
RNG_SEED = 20260828
MAX_ONSET_DIFF_S = 30.0
DEFAULT_EVALUATION_START_S = 60.0

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_progress(phase: str, **payload: object) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "analysis_id": RUN_ID,
        "phase": phase,
        "updated_unix": time.time(),
        **payload,
    }
    (REPORT_DIR / "progress.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


def load_h4_helpers():
    """Load the frozen H4 event grouping and scoring primitives."""

    path = ROOT / "scripts" / "84_run_groupkfold_h4_router.py"
    if not path.exists():
        # The workspace mirror contains the source even when the WSL project
        # copy predates it.  This fallback keeps execution reproducible.
        mirror = Path("/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/work/b_false_alarm_atlas/scripts/84_run_groupkfold_h4_router.py")
        path = mirror if mirror.exists() else path
    spec = importlib.util.spec_from_file_location("frozen_h4_router", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen H4 helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def model_matrix(frame: pd.DataFrame, model_aware: bool) -> tuple[pd.DataFrame, list[str]]:
    """Return the H4-compatible candidate matrix for a router."""

    result = frame.copy()
    for model in MODELS:
        result[f"source_{model}"] = (
            (result.source_model.astype(str) == model).astype(float) if model_aware else 0.0
        )
    for feature in SIGNAL_FEATURES:
        result[f"member_{feature}"] = pd.to_numeric(result[feature], errors="coerce").fillna(0.0)
    columns = [f"group_peak_{model}" for model in MODELS]
    columns += [f"group_uncertainty_{model}" for model in MODELS]
    columns += [f"source_{model}" for model in MODELS]
    columns += [f"member_{feature}" for feature in SIGNAL_FEATURES]
    return result[columns].replace([np.inf, -np.inf], np.nan), columns


def build_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Group near-synchronous detector events using the frozen H4 rule."""

    if frame.empty:
        return pd.DataFrame()
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
                output.append(record)
    return pd.DataFrame(output)


def truth_for_recording(seizures: pd.DataFrame, recording: str) -> list[tuple[float, float]]:
    rows = seizures[seizures.recording_id.astype(str) == str(recording)]
    return [
        (max(DEFAULT_EVALUATION_START_S, float(row.start_s)), float(row.end_s))
        for row in rows.itertuples(index=False)
        if float(row.end_s) > DEFAULT_EVALUATION_START_S
    ]


def score_events(
    events: pd.DataFrame,
    recordings: Iterable[str],
    seizures: pd.DataFrame,
    durations: dict[str, float],
) -> dict[str, float]:
    """Score an event table at the frozen event-level operating definition."""

    from bfa.evaluation.eventize import Event
    from bfa.evaluation.match import match_events

    true_positive = 0
    false_alarm = 0
    truth_count = 0
    nonseizure_hours = 0.0
    delays: list[float] = []
    if events is None or events.empty or "recording" not in events.columns or "event_start" not in events.columns:
        events = pd.DataFrame(columns=["patient", "recording", "event_start", "event_end", "peak_probability"])
    for recording in sorted(set(str(r) for r in recordings)):
        truth = truth_for_recording(seizures, recording)
        truth_count += len(truth)
        seizure_seconds = sum(max(0.0, b - a) for a, b in truth)
        nonseizure_hours += max(0.0, float(durations.get(recording, 0.0)) - DEFAULT_EVALUATION_START_S - seizure_seconds) / 3600.0
        group = events[events.recording.astype(str) == recording] if not events.empty else events
        predictions = [
            Event(float(row.event_start), float(row.event_end), float(row.peak_probability))
            for row in group.sort_values("event_start", kind="stable").itertuples(index=False)
        ]
        matched = match_events(predictions, truth)
        true_positive += len(matched.pairs)
        false_alarm += len(matched.unmatched_predictions)
        for pair in matched.pairs:
            delays.append(max(0.0, predictions[pair.prediction_index].start_s - truth[pair.truth_index][0]))
    return {
        "true_positive_events": int(true_positive),
        "false_alarm_events": int(false_alarm),
        "truth_events": int(truth_count),
        "sensitivity": float(true_positive / truth_count) if truth_count else 0.0,
        "fa_per_24h": float(false_alarm * 24.0 / nonseizure_hours) if nonseizure_hours else math.inf,
        "mean_delay_s": float(np.mean(delays)) if delays else math.nan,
        "nonseizure_hours": float(nonseizure_hours),
    }


def event_rows_to_predictions(rows: pd.DataFrame) -> pd.DataFrame:
    if rows is None or rows.empty:
        return pd.DataFrame(columns=["patient", "recording", "event_start", "event_end", "peak_probability"])
    return rows.rename(columns={"start_s": "event_start", "end_s": "event_end"})[
        ["patient", "recording", "event_start", "event_end", "peak_probability"]
    ].copy()


def select_router_events(candidates: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    output_columns = [
        "patient",
        "recording",
        "event_start",
        "event_end",
        "peak_probability",
        "selected_model",
        "group_id",
        "router_score",
    ]
    if candidates is None or candidates.empty:
        return pd.DataFrame(columns=output_columns)
    frame = candidates.copy()
    frame["router_score"] = np.asarray(scores, dtype=float)
    rows: list[dict[str, object]] = []
    for _, group in frame.groupby("group_id", sort=True):
        row = group.sort_values(["router_score", "peak_probability"], ascending=[False, False]).iloc[0]
        if float(row.router_score) < float(threshold):
            continue
        rows.append(
            {
                "patient": str(row.patient),
                "recording": str(row.recording),
                "event_start": float(row.start_s),
                "event_end": float(row.group_decision_end),
                "peak_probability": float(row.peak_probability),
                "selected_model": str(row.source_model),
                "group_id": str(row.group_id),
                "router_score": float(row.router_score),
            }
        )
    # Keep the schema even when the threshold suppresses every candidate.
    return pd.DataFrame(rows, columns=output_columns)


def threshold_grid(scores: np.ndarray) -> list[float]:
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    values = set(np.round(np.linspace(0.05, 0.95, 19), 3).tolist())
    if finite.size:
        values.update(np.round(np.quantile(finite, np.linspace(0.10, 0.99, 18)), 3).tolist())
    return sorted(float(x) for x in values)


def fit_router(
    fit_candidates: pd.DataFrame,
    selection_candidates: pd.DataFrame,
    target_sensitivity: float,
    model_aware: bool,
    selection_recordings: list[str],
    seizures: pd.DataFrame,
    durations: dict[str, float],
    random_state: int,
) -> tuple[Pipeline | None, float, list[str], dict[str, object]]:
    """Fit a router and choose its threshold using only late calibration data."""

    fit_candidates = fit_candidates if fit_candidates is not None else pd.DataFrame()
    selection_candidates = selection_candidates if selection_candidates is not None else pd.DataFrame()
    if fit_candidates.empty:
        return None, 0.5, [], {"status": "no_fit_candidates"}
    x_fit, columns = model_matrix(fit_candidates, model_aware)
    y_fit = (fit_candidates.event_type.astype(str) == "tp").astype(int).to_numpy()
    if len(np.unique(y_fit)) < 2:
        return None, 0.5, columns, {"status": "single_class_fit", "positive_rate": float(np.mean(y_fit))}
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=500,
                    solver="liblinear",
                    random_state=int(random_state),
                ),
            ),
        ]
    )
    model.fit(x_fit, y_fit)
    if selection_candidates.empty:
        return model, 0.5, columns, {"status": "fit_no_selection_candidates", "fit_rows": int(len(fit_candidates))}
    x_selection, _ = model_matrix(selection_candidates, model_aware)
    selection_scores = model.predict_proba(x_selection)[:, 1]
    scored: list[tuple[float, dict[str, float]]] = []
    for threshold in threshold_grid(selection_scores):
        selected = select_router_events(selection_candidates, selection_scores, threshold)
        scored.append((threshold, score_events(selected, selection_recordings, seizures, durations)))
    eligible = [row for row in scored if row[1]["sensitivity"] >= float(target_sensitivity) - 0.03]
    pool = eligible if eligible else scored
    best = sorted(
        pool,
        key=lambda row: (
            row[1]["fa_per_24h"],
            -row[1]["sensitivity"],
            row[1]["mean_delay_s"] if math.isfinite(row[1]["mean_delay_s"]) else math.inf,
            row[0],
        ),
    )[0]
    return model, float(best[0]), columns, {
        "status": "fit_and_selection_complete",
        "fit_rows": int(len(fit_candidates)),
        "selection_rows": int(len(selection_candidates)),
        "selection_threshold_candidates": int(len(scored)),
        "selection_target_sensitivity": float(target_sensitivity),
        "selection_sensitivity": float(best[1]["sensitivity"]),
        "selection_fa_per_24h": float(best[1]["fa_per_24h"]),
    }


def patient_record_sets(split_manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(row["patient"]): row for row in split_manifest["patients"]}


def pairwise_fp_jaccard(candidates: pd.DataFrame) -> dict[str, float]:
    """Jaccard of FP event groups for the three detector pairs."""

    pairs = [("singlem", "cbramod"), ("singlem", "tcn_gat"), ("cbramod", "tcn_gat")]
    values: dict[str, float] = {}
    for left, right in pairs:
        if candidates.empty:
            values[f"jaccard_{left}_{right}"] = math.nan
            continue
        intersection = 0
        union = 0
        for _, group in candidates.groupby("group_id", sort=False):
            sources = set(group.source_model.astype(str))
            has_left, has_right = left in sources, right in sources
            intersection += int(has_left and has_right)
            union += int(has_left or has_right)
        values[f"jaccard_{left}_{right}"] = float(intersection / union) if union else math.nan
    finite = [v for v in values.values() if math.isfinite(v)]
    values["fp_event_jaccard_mean"] = float(np.mean(finite)) if finite else math.nan
    values["fp_disagreement_index"] = float(1.0 - np.mean(finite)) if finite else math.nan
    return values


def safe_friedman(table: pd.DataFrame, value: str) -> dict[str, object]:
    wide = table.pivot(index="patient", columns="model", values=value).dropna()
    if len(wide) < 3:
        return {"statistic": math.nan, "p_value": math.nan, "n_patients": int(len(wide)), "status": "insufficient"}
    result = friedmanchisquare(*(wide[model].to_numpy(float) for model in MODELS))
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "n_patients": int(len(wide)), "status": "complete"}


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: (math.inf if not math.isfinite(item[1]) else item[1]))
    adjusted = [math.nan] * len(p_values)
    m = len(p_values)
    running = 0.0
    for rank, (idx, p_value) in enumerate(indexed):
        if not math.isfinite(p_value):
            continue
        current = min(1.0, float(p_value) * (m - rank))
        running = max(running, current)
        adjusted[idx] = running
    return adjusted


def pairwise_wilcoxon(table: pd.DataFrame, value: str) -> list[dict[str, object]]:
    wide = table.pivot(index="patient", columns="model", values=value).dropna()
    rows: list[dict[str, object]] = []
    raw: list[float] = []
    for i, left in enumerate(MODELS):
        for right in MODELS[i + 1 :]:
            if len(wide) < 3:
                statistic, p_value = math.nan, math.nan
            else:
                result = wilcoxon(wide[left], wide[right], zero_method="wilcox", alternative="two-sided", method="auto")
                statistic, p_value = float(result.statistic), float(result.pvalue)
            raw.append(p_value)
            rows.append({"metric": value, "left_model": left, "right_model": right, "n_patients": int(len(wide)), "statistic": statistic, "p_value": p_value})
    adjusted = holm_adjust(raw)
    for row, value_adj in zip(rows, adjusted, strict=True):
        row["p_holm"] = value_adj
    return rows


def bootstrap_summary(patient_metrics: pd.DataFrame, method: str, stratum: str | None = None) -> dict[str, object]:
    subset = patient_metrics[(patient_metrics.method == method) & (patient_metrics.stratum == stratum)] if stratum is not None else patient_metrics[patient_metrics.method == method]
    if subset.empty:
        return {"method": method, "stratum": stratum, "bootstrap_replicates": N_BOOTSTRAP, "status": "empty"}
    per_patient = subset.groupby("patient", as_index=False)[["sensitivity", "fa_per_24h", "mean_delay_s"]].mean()
    rng = np.random.default_rng(RNG_SEED + sum(ord(c) for c in method) + (0 if stratum is None else sum(ord(c) for c in stratum)))
    n = len(per_patient)
    sens = np.empty(N_BOOTSTRAP)
    fa = np.empty(N_BOOTSTRAP)
    delay = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        sample = per_patient.iloc[rng.integers(0, n, size=n)]
        sens[i] = sample.sensitivity.mean()
        fa[i] = sample.fa_per_24h.mean()
        delay[i] = sample.mean_delay_s.fillna(0.0).mean()
    return {
        "method": method,
        "stratum": stratum,
        "patients": int(n),
        "bootstrap_replicates": N_BOOTSTRAP,
        "sensitivity_mean": float(per_patient.sensitivity.mean()),
        "sensitivity_q025": float(np.quantile(sens, 0.025)),
        "sensitivity_q975": float(np.quantile(sens, 0.975)),
        "fa_per_24h_mean": float(per_patient.fa_per_24h.mean()),
        "fa_per_24h_q025": float(np.quantile(fa, 0.025)),
        "fa_per_24h_q975": float(np.quantile(fa, 0.975)),
        "mean_delay_s_mean": float(per_patient.mean_delay_s.fillna(0.0).mean()),
        "mean_delay_s_q025": float(np.quantile(delay, 0.025)),
        "mean_delay_s_q975": float(np.quantile(delay, 0.975)),
        "status": "complete",
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_progress("loading", status="started")
    for path in (H4_FEATURE_PATH, H4_FEATURE_MANIFEST_PATH, H6_SPLIT_PATH, SEIZURE_PATH, RECORDING_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    features = pd.read_parquet(H4_FEATURE_PATH)
    split_manifest = json.loads(H6_SPLIT_PATH.read_text())
    split_by_patient = patient_record_sets(split_manifest)
    seizures = pd.read_parquet(SEIZURE_PATH)
    recordings = pd.read_parquet(RECORDING_PATH)
    durations = recordings.set_index("recording_id")["duration_s"].astype(float).to_dict()
    # Only the frozen standard outer-test event rows are used.  No new OOF
    # extraction, threshold selection, or detector training occurs here.
    test_features = features[features.partition.astype(str) == "test"].copy()
    test_features["patient"] = test_features.patient.astype(str)
    test_features["recording"] = test_features.recording.astype(str)
    test_features["model"] = test_features.model.astype(str)
    test_features["seed"] = test_features.seed.astype(int)
    test_features["fold"] = test_features.split.astype(int)
    if set(split_by_patient) != set(test_features.patient.unique()):
        raise RuntimeError("chronological manifest and standard outer-test patients do not match")
    write_progress("loaded", patients=len(split_by_patient), feature_rows=len(test_features))

    # Per-patient/per-seed calibration and holdout detector metrics and FP
    # disagreement are the main source table for every subsequent analysis.
    detector_rows: list[dict[str, object]] = []
    disagreement_rows: list[dict[str, object]] = []
    candidate_cache: dict[tuple[str, int, str], pd.DataFrame] = {}
    for p_index, patient in enumerate(sorted(split_by_patient)):
        manifest_row = split_by_patient[patient]
        calibration_recordings = [str(x) for x in manifest_row["calibration_recordings"]]
        holdout_recordings = [str(x) for x in manifest_row["holdout_recordings"]]
        patient_rows = test_features[test_features.patient == patient]
        for seed in SEEDS:
            seed_rows = patient_rows[patient_rows.seed == seed]
            cal_rows = seed_rows[seed_rows.recording.isin(calibration_recordings)]
            hold_rows = seed_rows[seed_rows.recording.isin(holdout_recordings)]
            # Build a single candidate table for FP disagreement.  Each row is
            # a detector event, and each group is a ±30-s near-synchronous set.
            fp = cal_rows[cal_rows.event_type.astype(str) == "fp"]
            fp_candidates = build_candidates(fp)
            disagreement = pairwise_fp_jaccard(fp_candidates)
            disagreement_rows.append({"patient": patient, "fold": int(manifest_row["fold"]), "seed": seed, "calibration_fp_event_rows": int(len(fp)), "calibration_fp_candidate_rows": int(len(fp_candidates)), **disagreement})
            for partition_name, rows, record_list in (("calibration", cal_rows, calibration_recordings), ("holdout", hold_rows, holdout_recordings)):
                for model in MODELS:
                    model_rows = rows[rows.model == model]
                    metrics = score_events(event_rows_to_predictions(model_rows), record_list, seizures, durations)
                    detector_rows.append({"patient": patient, "fold": int(manifest_row["fold"]), "seed": seed, "partition": partition_name, "model": model, "recordings": len(record_list), "event_rows": int(len(model_rows)), **metrics})
            # Cache all-event candidate groups for router fits/evaluation.
            candidate_cache[(patient, seed, "calibration")] = build_candidates(cal_rows)
            candidate_cache[(patient, seed, "holdout")] = build_candidates(hold_rows)
        if p_index == 0 or (p_index + 1) % 5 == 0:
            write_progress("patient_metrics", patients_done=p_index + 1, patients_total=len(split_by_patient))

    detector_df = pd.DataFrame(detector_rows)
    disagreement_df = pd.DataFrame(disagreement_rows)
    detector_df.to_csv(REPORT_DIR / "patient_detector_metrics.csv", index=False)
    disagreement_df.to_csv(REPORT_DIR / "patient_fp_disagreement_by_seed.csv", index=False)

    # Aggregate calibration-only heterogeneity across model seeds.  Strata are
    # frozen once here and are never recomputed from holdout results.
    cal = detector_df[detector_df.partition == "calibration"].copy()
    cal["log1p_fa_per_24h"] = np.log1p(cal.fa_per_24h.replace([np.inf, -np.inf], np.nan).fillna(0.0))
    wide_sens = cal.pivot_table(index=["patient", "seed"], columns="model", values="sensitivity", aggfunc="first")
    wide_fa = cal.pivot_table(index=["patient", "seed"], columns="model", values="log1p_fa_per_24h", aggfunc="first")
    perf_rows: list[dict[str, object]] = []
    for (patient, seed), row in wide_sens.iterrows():
        sens_values = [float(row.get(model, math.nan)) for model in MODELS]
        fa_values = [float(wide_fa.loc[(patient, seed)].get(model, math.nan)) for model in MODELS]
        valid_s = [x for x in sens_values if math.isfinite(x)]
        valid_f = [x for x in fa_values if math.isfinite(x)]
        perf_rows.append({"patient": patient, "seed": int(seed), "sensitivity_range": float(max(valid_s) - min(valid_s)) if valid_s else math.nan, "log_fa_range": float(max(valid_f) - min(valid_f)) if valid_f else math.nan, "best_sensitivity_model": MODELS[int(np.nanargmax(sens_values))] if valid_s else "", "best_low_fa_model": MODELS[int(np.nanargmin(fa_values))] if valid_f else ""})
    perf_df = pd.DataFrame(perf_rows)
    heterogeneity = disagreement_df.groupby("patient", as_index=False).agg(
        fp_disagreement_index=("fp_disagreement_index", "mean"),
        fp_event_jaccard_mean=("fp_event_jaccard_mean", "mean"),
        jaccard_singlem_cbramod=("jaccard_singlem_cbramod", "mean"),
        jaccard_singlem_tcn_gat=("jaccard_singlem_tcn_gat", "mean"),
        jaccard_cbramod_tcn_gat=("jaccard_cbramod_tcn_gat", "mean"),
    ).merge(perf_df.groupby("patient", as_index=False).agg(sensitivity_range_mean=("sensitivity_range", "mean"), log_fa_range_mean=("log_fa_range", "mean")), on="patient", how="left")
    heterogeneity = heterogeneity.sort_values(["fp_disagreement_index", "sensitivity_range_mean", "patient"], ascending=[False, False, True], kind="stable").reset_index(drop=True)
    heterogeneity["rank_high_to_low"] = np.arange(1, len(heterogeneity) + 1)
    heterogeneity["stratum"] = "middle"
    heterogeneity.loc[: N_EXTREME - 1, "stratum"] = "high_disagreement"
    heterogeneity.loc[len(heterogeneity) - N_EXTREME :, "stratum"] = "low_disagreement"
    heterogeneity.to_csv(REPORT_DIR / "patient_heterogeneity_ranking.csv", index=False)
    high_patients = heterogeneity.loc[heterogeneity.stratum == "high_disagreement", "patient"].tolist()
    low_patients = heterogeneity.loc[heterogeneity.stratum == "low_disagreement", "patient"].tolist()
    stratum_by_patient = heterogeneity.set_index("patient").stratum.to_dict()
    write_progress("strata_frozen", high_patients=high_patients, low_patients=low_patients)

    # Patient x detector relationship summary and paired non-parametric tests.
    patient_detector = cal.groupby(["patient", "model"], as_index=False).agg(
        sensitivity_mean=("sensitivity", "mean"), sensitivity_sd=("sensitivity", "std"),
        fa_per_24h_mean=("fa_per_24h", "mean"), fa_per_24h_sd=("fa_per_24h", "std"),
        mean_delay_s_mean=("mean_delay_s", "mean"),
    )
    patient_detector["log_fa_per_24h_mean"] = np.log1p(patient_detector.fa_per_24h_mean.replace([np.inf, -np.inf], np.nan).fillna(0.0))
    patient_detector["stratum"] = patient_detector.patient.map(stratum_by_patient)
    patient_detector.to_csv(REPORT_DIR / "patient_detector_relationship.csv", index=False)
    calibration_patient_mean = cal.groupby(["patient", "model"], as_index=False).agg(sensitivity=("sensitivity", "mean"), log_fa_per_24h=("log1p_fa_per_24h", "mean"))
    relationship_stats: dict[str, object] = {
        "calibration_friedman_sensitivity": safe_friedman(calibration_patient_mean.rename(columns={"sensitivity": "sensitivity"}), "sensitivity"),
        "calibration_friedman_log1p_fa": safe_friedman(calibration_patient_mean.rename(columns={"log_fa_per_24h": "log1p_fa_per_24h"}), "log1p_fa_per_24h"),
        "pairwise_wilcoxon_sensitivity": pairwise_wilcoxon(calibration_patient_mean, "sensitivity"),
        "pairwise_wilcoxon_log1p_fa": pairwise_wilcoxon(calibration_patient_mean, "log_fa_per_24h"),
        "preferred_detector_definition": "among detectors within 0.03 sensitivity of the patient's calibration maximum, choose minimum calibration FA/24h; ties resolved by sensitivity then model name",
    }
    preferred_rows: list[dict[str, object]] = []
    for patient in sorted(split_by_patient):
        for seed in SEEDS:
            row = cal[(cal.patient == patient) & (cal.seed == seed)].set_index("model")
            max_sens = float(row.sensitivity.max()) if not row.empty else 0.0
            eligible = row[row.sensitivity >= max_sens - 0.03]
            if eligible.empty:
                eligible = row
            preferred = sorted(eligible.reset_index().to_dict(orient="records"), key=lambda x: (float(x["fa_per_24h"]), -float(x["sensitivity"]), str(x["model"])))[0] if not eligible.empty else {"model": "none", "sensitivity": math.nan, "fa_per_24h": math.nan}
            preferred_rows.append({"patient": patient, "seed": seed, "preferred_model": preferred["model"], "preferred_sensitivity": float(preferred["sensitivity"]), "preferred_fa_per_24h": float(preferred["fa_per_24h"]), "max_sensitivity": max_sens, "stratum": stratum_by_patient[patient]})
    preferred_df = pd.DataFrame(preferred_rows)
    preferred_df.to_csv(REPORT_DIR / "patient_detector_preference.csv", index=False)
    relationship_stats["preferred_model_counts"] = preferred_df.preferred_model.value_counts().to_dict()
    relationship_stats["preferred_model_patient_consistency"] = preferred_df.groupby("patient").preferred_model.nunique().describe().to_dict()
    (REPORT_DIR / "patient_detector_relationship_stats.json").write_text(json.dumps(relationship_stats, indent=2, sort_keys=True, default=float) + "\n")

    # Router fits.  The within-patient routers and pooled stratum routers use
    # exactly the same candidate construction and chronological rule.  Each
    # row below is a holdout evaluation; no holdout label is used to choose a
    # router, threshold, stratum, or representative patient.
    write_progress("router_fits", status="started")
    router_artifacts: list[dict[str, object]] = []
    router_models: dict[tuple[str, int, str], tuple[Pipeline | None, float, bool, list[str], dict[str, object]]] = {}
    router_validation_rows: list[dict[str, object]] = []
    holdout_rows: list[dict[str, object]] = []

    def fit_and_store(key: tuple[str, int, str], fit_df: pd.DataFrame, select_df: pd.DataFrame, target: float, model_aware: bool, selection_recordings: list[str]) -> None:
        label, seed, variant = key
        model, threshold, columns, fit_info = fit_router(
            fit_df,
            select_df,
            target,
            model_aware,
            selection_recordings,
            seizures,
            durations,
            RNG_SEED + seed + len(label),
        )
        router_models[key] = (model, threshold, model_aware, columns, fit_info)
        router_validation_rows.append({"router_scope": label, "seed": seed, "variant": variant, "model_aware": model_aware, "threshold_source": "late_calibration_only", "threshold": threshold, **fit_info})
        if model is not None:
            artifact_path = REPORT_DIR / f"router_{label}_{variant}_seed{seed}.joblib"
            joblib.dump(model, artifact_path)
            router_artifacts.append({"router_scope": label, "seed": seed, "variant": variant, "model_aware": model_aware, "path": artifact_path.name, "sha256": sha256_file(artifact_path), "feature_columns": columns})

    # Fit patient routers and score holdouts.
    for index, patient in enumerate(sorted(split_by_patient)):
        manifest_row = split_by_patient[patient]
        cal_records = [str(x) for x in manifest_row["calibration_recordings"]]
        hold_records = [str(x) for x in manifest_row["holdout_recordings"]]
        n_fit = max(1, int(math.floor(0.70 * len(cal_records))))
        if len(cal_records) >= 3:
            n_fit = min(n_fit, len(cal_records) - 1)
        fit_records = cal_records[:n_fit]
        select_records = cal_records[n_fit:]
        for seed in SEEDS:
            cal_candidates = candidate_cache[(patient, seed, "calibration")]
            hold_candidates = candidate_cache[(patient, seed, "holdout")]
            fit_df = cal_candidates[cal_candidates.recording.isin(fit_records)] if not cal_candidates.empty else cal_candidates
            select_df = cal_candidates[cal_candidates.recording.isin(select_records)] if not cal_candidates.empty else cal_candidates
            cal_metric = cal[(cal.patient == patient) & (cal.seed == seed)]
            target = float(cal_metric.sensitivity.max()) if not cal_metric.empty else 0.0
            for variant, aware in (("model_aware", True), ("no_model_id", False)):
                key = (patient, seed, variant)
                fit_and_store(key, fit_df, select_df, target, aware, select_records)
                model, threshold, _, columns, fit_info = router_models[key]
                if model is None or hold_candidates.empty:
                    selected = event_rows_to_predictions(pd.DataFrame())
                else:
                    x_hold, _ = model_matrix(hold_candidates, aware)
                    scores = model.predict_proba(x_hold)[:, 1]
                    selected = select_router_events(hold_candidates, scores, threshold)
                metrics = score_events(selected, hold_records, seizures, durations)
                holdout_rows.append({"router_scope": patient, "patient": patient, "stratum": stratum_by_patient[patient], "fold": int(manifest_row["fold"]), "seed": seed, "method": f"single_patient_router_{variant}", "threshold_source": "late_calibration_only", "holdout_labels_used_for_selection": False, "test_evaluation_count": 1, "fit_status": fit_info.get("status"), "threshold": threshold, **metrics})
            # Frozen calibration-selected best single baseline, evaluated once.
            best_row = preferred_df[(preferred_df.patient == patient) & (preferred_df.seed == seed)].iloc[0]
            best_model = str(best_row.preferred_model)
            best_events = event_rows_to_predictions(test_features[(test_features.patient == patient) & (test_features.seed == seed) & (test_features.recording.isin(hold_records)) & (test_features.model == best_model)])
            metrics = score_events(best_events, hold_records, seizures, durations)
            holdout_rows.append({"router_scope": patient, "patient": patient, "stratum": stratum_by_patient[patient], "fold": int(manifest_row["fold"]), "seed": seed, "method": "patient_calibration_best_single", "threshold_source": "frozen_detector_validation_threshold", "holdout_labels_used_for_selection": False, "test_evaluation_count": 1, "fit_status": "frozen_baseline", "threshold": math.nan, **metrics})
        if index == 0 or (index + 1) % 5 == 0:
            write_progress("patient_routers", patients_done=index + 1, patients_total=len(split_by_patient))

    # Pooled high- and low-disagreement routers.  A router is fit separately
    # for each seed and stratum; it is evaluated both on its own stratum and
    # cross-applied to the opposite stratum without retraining.
    for stratum, patients in (("high_disagreement", high_patients), ("low_disagreement", low_patients)):
        for seed in SEEDS:
            fit_frames: list[pd.DataFrame] = []
            select_frames: list[pd.DataFrame] = []
            select_records: list[str] = []
            all_cal_metrics = cal[(cal.patient.isin(patients)) & (cal.seed == seed)]
            target = float(all_cal_metrics.groupby("model").sensitivity.mean().max()) if not all_cal_metrics.empty else 0.0
            for patient in patients:
                manifest_row = split_by_patient[patient]
                records = [str(x) for x in manifest_row["calibration_recordings"]]
                n_fit = max(1, int(math.floor(0.70 * len(records))))
                if len(records) >= 3:
                    n_fit = min(n_fit, len(records) - 1)
                fit_records = records[:n_fit]
                late_records = records[n_fit:]
                candidates = candidate_cache[(patient, seed, "calibration")]
                fit_frames.append(candidates[candidates.recording.isin(fit_records)] if not candidates.empty else candidates)
                select_frames.append(candidates[candidates.recording.isin(late_records)] if not candidates.empty else candidates)
                select_records.extend(late_records)
            pooled_fit = pd.concat(fit_frames, ignore_index=True) if fit_frames else pd.DataFrame()
            pooled_select = pd.concat(select_frames, ignore_index=True) if select_frames else pd.DataFrame()
            for variant, aware in (("model_aware", True), ("no_model_id", False)):
                key = (stratum, seed, variant)
                fit_and_store(key, pooled_fit, pooled_select, target, aware, select_records)
                model, threshold, _, columns, fit_info = router_models[key]
                for eval_stratum, eval_patients in (("high_disagreement", high_patients), ("low_disagreement", low_patients)):
                    for patient in eval_patients:
                        manifest_row = split_by_patient[patient]
                        hold_records = [str(x) for x in manifest_row["holdout_recordings"]]
                        hold_candidates = candidate_cache[(patient, seed, "holdout")]
                        if model is None or hold_candidates.empty:
                            selected = event_rows_to_predictions(pd.DataFrame())
                        else:
                            x_hold, _ = model_matrix(hold_candidates, aware)
                            selected = select_router_events(hold_candidates, model.predict_proba(x_hold)[:, 1], threshold)
                        metrics = score_events(selected, hold_records, seizures, durations)
                        holdout_rows.append({"router_scope": stratum, "patient": patient, "stratum": eval_stratum, "trained_stratum": stratum, "fold": int(manifest_row["fold"]), "seed": seed, "method": f"group_router_{stratum}_{variant}", "threshold_source": "late_calibration_only", "holdout_labels_used_for_selection": False, "test_evaluation_count": 1, "fit_status": fit_info.get("status"), "threshold": threshold, **metrics})

    validation_df = pd.DataFrame(router_validation_rows)
    holdout_df = pd.DataFrame(holdout_rows)
    validation_df.to_csv(REPORT_DIR / "router_validation.csv", index=False)
    holdout_df.to_csv(REPORT_DIR / "router_holdout_patient_metrics.csv", index=False)
    pd.DataFrame(router_artifacts).to_json(REPORT_DIR / "router_artifacts.json", orient="records", indent=2)

    # Representative extremes are a presentation aid only; all 22 patients
    # remain in the quantitative tables above.
    representative = heterogeneity.head(3).assign(selection="top_3_high_disagreement")
    representative = pd.concat([representative, heterogeneity.tail(3).assign(selection="bottom_3_low_disagreement")], ignore_index=True)
    representative.to_csv(REPORT_DIR / "representative_patients.csv", index=False)

    bootstrap_rows: list[dict[str, object]] = []
    methods = sorted(holdout_df.method.dropna().unique().tolist())
    for method in methods:
        bootstrap_rows.append(bootstrap_summary(holdout_df, method, None))
        if method.startswith("group_router_"):
            bootstrap_rows.append(bootstrap_summary(holdout_df, method, "high_disagreement"))
            bootstrap_rows.append(bootstrap_summary(holdout_df, method, "low_disagreement"))
    pd.DataFrame(bootstrap_rows).to_csv(REPORT_DIR / "patient_bootstrap_summary.csv", index=False)

    # Aggregate method summaries; this is descriptive and does not select a
    # method or threshold from holdout outcomes.
    method_summary = holdout_df.groupby(["method", "stratum"], dropna=False, as_index=False).agg(
        patients=("patient", "nunique"), rows=("patient", "size"), sensitivity_mean=("sensitivity", "mean"),
        fa_per_24h_mean=("fa_per_24h", "mean"), mean_delay_s=("mean_delay_s", "mean"),
    )
    method_summary.to_csv(REPORT_DIR / "method_summary.csv", index=False)

    # Reproducibility manifest.  Explicitly document that the outer-test
    # records are inherited from the already inspected H6 exploratory source;
    # this add-on is not a pristine patient-independent confirmation.
    manifest = {
        "analysis_id": RUN_ID,
        "status": "complete_exploratory_chronological_holdout",
        "claim_boundary": "exploratory within-patient chronological holdout; inherited parent H6 source whose outer-test results were previously inspected; not patient-independent generalization",
        "question": "whether detector preference and router behavior vary with patient-specific detector disagreement",
        "models": list(MODELS),
        "patients": int(len(split_by_patient)),
        "seeds": list(SEEDS),
        "input_partition": "standard v3-groupkfold-confirmatory-v1 outer-test OOF events only",
        "stratification": {"primary_score": "mean calibration-only FP event disagreement index = 1 - mean pairwise ±30-s FP-event Jaccard", "secondary_scores": ["calibration sensitivity range", "calibration log1p FA/24h range"], "n_extreme_each": N_EXTREME, "high_patients": high_patients, "low_patients": low_patients, "holdout_used_for_selection": False},
        "chronological_protocol": {"calibration": "first 70% of each patient's frozen calibration recording list used for router fit", "router_selection": "remaining 30% of calibration recordings used for threshold selection", "holdout": "frozen later recording list evaluated once", "evaluation_start_s": DEFAULT_EVALUATION_START_S, "boundary_buffer_inherited_s": float(split_manifest.get("boundary_buffer_s", 30.0))},
        "router_training": "router_only logistic regression; no detector-backbone retraining; features are frozen H4 candidate group peaks, uncertainty, signal features, and optional source-model indicators",
        "threshold_source": "late_calibration_only",
        "test_evaluation_count": 1,
        "patient_bootstrap_replicates": N_BOOTSTRAP,
        "source_hashes": {"h4_features": sha256_file(H4_FEATURE_PATH), "h4_features_manifest": sha256_file(H4_FEATURE_MANIFEST_PATH), "h6_split_manifest": sha256_file(H6_SPLIT_PATH), "seizures": sha256_file(SEIZURE_PATH), "recordings": sha256_file(RECORDING_PATH), "preflight": sha256_file(REPORT_DIR / "no_duplicate_overlap_preflight.json") if (REPORT_DIR / "no_duplicate_overlap_preflight.json").exists() else None},
        "outputs": {"patient_detector_metrics": "patient_detector_metrics.csv", "patient_detector_relationship": "patient_detector_relationship.csv", "relationship_stats": "patient_detector_relationship_stats.json", "patient_fp_disagreement_by_seed": "patient_fp_disagreement_by_seed.csv", "patient_heterogeneity_ranking": "patient_heterogeneity_ranking.csv", "router_validation": "router_validation.csv", "router_holdout_patient_metrics": "router_holdout_patient_metrics.csv", "router_artifacts": "router_artifacts.json", "patient_bootstrap_summary": "patient_bootstrap_summary.csv", "method_summary": "method_summary.csv", "representative_patients": "representative_patients.csv"},
    }
    (REPORT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=float) + "\n")
    write_progress("complete", status="complete", artifacts=len(router_artifacts), holdout_rows=len(holdout_df))


if __name__ == "__main__":
    main()
