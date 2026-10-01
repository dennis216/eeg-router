"""Read-only adequacy audit for the 45 frozen standard GroupKFold experts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


ROOT = Path("/root/b_false_alarm_atlas")
RUN_ID = "v3-groupkfold-confirmatory-v1"
SOURCE = ROOT / "outputs/reports" / RUN_ID
OUT = ROOT / "outputs/reports/h4-final-linkage-expert-audit-v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
FOLDS = range(5)
SEEDS = (17, 42, 3407)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    total = len(y)
    value = 0.0
    for index in range(bins):
        chosen = ids == index
        if chosen.any():
            value += chosen.mean() * abs(float(y[chosen].mean()) - float(p[chosen].mean()))
    return float(value)


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray, seed: int) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    if len(y) > 200_000:
        idx = rng.choice(len(y), 200_000, replace=False)
        y, p = y[idx], p[idx]
    x = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return math.nan, math.nan, len(y)
    fit = LogisticRegression(C=1e6, solver="lbfgs", max_iter=300).fit(x, y)
    return float(fit.coef_[0, 0]), float(fit.intercept_[0]), len(y)


def probability_metrics(path: Path, seed: int) -> dict[str, float]:
    table = pd.read_parquet(path, columns=["label", "probability"])
    table = table.dropna(subset=["label", "probability"])
    y = table.label.astype(int).to_numpy()
    p = np.clip(table.probability.astype("float64").to_numpy(), 1e-8, 1 - 1e-8)
    slope, intercept, n_cal = calibration_slope_intercept(y, p, seed)
    return {
        "rows": len(y), "positive_fraction": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece15": ece(y, p), "calibration_slope": slope, "calibration_intercept": intercept,
        "calibration_sample_n": n_cal,
    }


def dominance(sweep: pd.DataFrame, sensitivity: float, fa: float) -> tuple[bool, float, float, float]:
    better = sweep[(sweep.event_sensitivity >= sensitivity - 1e-12) & (sweep.fa_per_24h <= fa + 1e-12)].copy()
    strict = better[(better.event_sensitivity > sensitivity + 1e-12) | (better.fa_per_24h < fa - 1e-12)]
    if strict.empty:
        return False, math.nan, math.nan, math.nan
    row = strict.sort_values(["fa_per_24h", "event_sensitivity"], ascending=[True, False]).iloc[0]
    return True, float(row.threshold), float(row.event_sensitivity), float(row.fa_per_24h)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    curves = pd.read_parquet(SOURCE / "sensitivity_fa_curves.parquet")

    for model_index, model in enumerate(MODELS):
        for fold in FOLDS:
            for seed in SEEDS:
                run = ROOT / "runs" / RUN_ID / model / f"split{fold}_seed{seed}_main"
                completed = json.loads((run / "completed.json").read_text())
                manifest = json.loads((run / "run_manifest.json").read_text())
                history = pd.read_json(run / "training_history.jsonl", lines=True)
                finite = bool(np.isfinite(history[["loss", "gradient_norm", "learning_rate"]].to_numpy()).all())
                first = float(history.loss.head(100).mean())
                last = float(history.loss.tail(100).mean())
                loss_slope = float(np.polyfit(history["update"].to_numpy(float), history.loss.to_numpy(float), 1)[0])
                validations: list[dict[str, object]] = []
                metric_paths = sorted((run / "validation").glob("metrics_step_*.json"))
                for metric_path in metric_paths:
                    step = int(metric_path.stem.rsplit("_", 1)[-1])
                    metric = json.loads(metric_path.read_text())
                    op = metric["selected_event_operating_point"]
                    row = {
                        "model": model, "fold": fold, "seed": seed, "step": step,
                        "window_auroc": float(metric["window_auroc"]),
                        "window_average_precision": float(metric["window_average_precision"]),
                        "event_sensitivity": float(op["event_sensitivity"]), "fa_per_24h": float(op["fa_per_24h"]),
                        "threshold": float(op["threshold"]),
                    }
                    validations.append(row)
                    checkpoint_rows.append(row)
                val = pd.DataFrame(validations)
                final = val[val.step == 5000].iloc[0]
                best_ap = val.sort_values(["window_average_precision", "window_auroc"], ascending=False).iloc[0]
                best_auc = val.sort_values(["window_auroc", "window_average_precision"], ascending=False).iloc[0]
                # Within-step validation Pareto audit of the frozen operating point.
                val_sweep = pd.read_parquet(run / "validation/threshold_sweep_step_05000.parquet")
                dominated_val, dom_thr, dom_sens, dom_fa = dominance(val_sweep, float(final.event_sensitivity), float(final.fa_per_24h))
                rho, _ = spearmanr(val_sweep.threshold, val_sweep.fa_per_24h)
                fa_diff = np.diff(val_sweep.sort_values("threshold").fa_per_24h.to_numpy(float))
                sensitivity_diff = np.diff(val_sweep.sort_values("threshold").event_sensitivity.to_numpy(float))

                test_score = completed["test"]["score"]
                test_curve = curves[(curves.model == model) & (curves.fold == fold) & (curves.seed == seed)]
                dominated_test, test_dom_thr, test_dom_sens, test_dom_fa = dominance(
                    test_curve, float(test_score["event_sensitivity"]), float(test_score["fa_per_24h"])
                )
                run_rows.append({
                    "model": model, "fold": fold, "seed": seed, "completed": True,
                    "optimizer_updates": int(completed["optimizer_updates"]),
                    "test_evaluation_count": int(completed["test"]["test_evaluation_count"]),
                    "threshold_source": str(completed["test"]["threshold_source"]),
                    "frozen_backbone_feature_cache": bool(manifest.get("frozen_backbone_feature_cache", False)),
                    "training_finite": finite, "loss_first100": first, "loss_last100": last,
                    "validation_checkpoint_metrics_count": int(len(val)),
                    "loss_last_to_first_ratio": last / first if first else math.nan, "loss_linear_slope": loss_slope,
                    "final_step": 5000, "best_val_ap_step": int(best_ap.step),
                    "best_val_auc_step": int(best_auc.step), "final_is_best_ap": int(best_ap.step) == 5000,
                    "final_is_best_auc": int(best_auc.step) == 5000,
                    "final_val_auroc": float(final.window_auroc), "final_val_ap": float(final.window_average_precision),
                    "final_val_sensitivity": float(final.event_sensitivity), "final_val_fa_per_24h": float(final.fa_per_24h),
                    "validation_threshold": float(final.threshold), "validation_point_dominated": dominated_val,
                    "validation_dominating_threshold": dom_thr, "validation_dominating_sensitivity": dom_sens,
                    "validation_dominating_fa_per_24h": dom_fa,
                    "validation_threshold_fa_spearman": float(rho),
                    "validation_fa_monotonicity_violations": int((fa_diff > 1e-12).sum()),
                    "validation_sensitivity_monotonicity_violations": int((sensitivity_diff > 1e-12).sum()),
                    "test_sensitivity": float(test_score["event_sensitivity"]),
                    "test_fa_per_24h": float(test_score["fa_per_24h"]),
                    "test_point_dominated_descriptively": dominated_test,
                    "test_dominating_threshold_descriptive": test_dom_thr,
                    "test_dominating_sensitivity_descriptive": test_dom_sens,
                    "test_dominating_fa_per_24h_descriptive": test_dom_fa,
                })

                for partition, path in (
                    ("validation", run / "validation/probabilities_step_05000.parquet"),
                    ("test", run / "test/probabilities.parquet"),
                ):
                    calibration_rows.append({
                        "model": model, "fold": fold, "seed": seed, "partition": partition,
                        **probability_metrics(path, 20260904 + model_index * 10000 + fold * 100 + seed),
                    })

                for target in (.60, .70, .80):
                    eligible = test_curve[test_curve.event_sensitivity >= target]
                    if not eligible.empty:
                        point = eligible.sort_values(["fa_per_24h", "event_sensitivity"], ascending=[True, False]).iloc[0]
                        curve_rows.append({"model": model, "fold": fold, "seed": seed, "match_type": "sensitivity", "target": target,
                                           "achieved_sensitivity": float(point.event_sensitivity), "achieved_fa_per_24h": float(point.fa_per_24h), "threshold": float(point.threshold)})
                for budget in (50.0, 100.0, 200.0):
                    eligible = test_curve[test_curve.fa_per_24h <= budget]
                    if not eligible.empty:
                        point = eligible.sort_values(["event_sensitivity", "fa_per_24h"], ascending=[False, True]).iloc[0]
                        curve_rows.append({"model": model, "fold": fold, "seed": seed, "match_type": "fa_budget", "target": budget,
                                           "achieved_sensitivity": float(point.event_sensitivity), "achieved_fa_per_24h": float(point.fa_per_24h), "threshold": float(point.threshold)})

    runs = pd.DataFrame(run_rows)
    checkpoints = pd.DataFrame(checkpoint_rows)
    calibration = pd.DataFrame(calibration_rows)
    matched = pd.DataFrame(curve_rows)
    runs.to_csv(OUT / "expert_run_audit.csv", index=False)
    checkpoints.to_csv(OUT / "expert_validation_checkpoint_curves.csv", index=False)
    calibration.to_csv(OUT / "expert_calibration.csv", index=False)
    matched.to_csv(OUT / "expert_matched_operating_points_descriptive.csv", index=False)

    summary = (
        runs.groupby("model", as_index=False)
        .agg(
            runs=("seed", "size"), all_finite=("training_finite", "all"),
            mean_loss_ratio=("loss_last_to_first_ratio", "mean"), final_best_ap_fraction=("final_is_best_ap", "mean"),
            final_best_auc_fraction=("final_is_best_auc", "mean"), dominated_validation_fraction=("validation_point_dominated", "mean"),
            dominated_test_fraction=("test_point_dominated_descriptively", "mean"),
            mean_val_auroc=("final_val_auroc", "mean"), mean_val_ap=("final_val_ap", "mean"),
            mean_test_sensitivity=("test_sensitivity", "mean"), mean_test_fa_per_24h=("test_fa_per_24h", "mean"),
            median_threshold=("validation_threshold", "median"), mean_threshold_fa_spearman=("validation_threshold_fa_spearman", "mean"),
        )
    )
    cal_summary = calibration.groupby(["model", "partition"], as_index=False)[["auroc", "average_precision", "brier", "log_loss", "ece15", "calibration_slope", "calibration_intercept"]].mean()
    matched_summary = matched.groupby(["model", "match_type", "target"], as_index=False)[["achieved_sensitivity", "achieved_fa_per_24h"]].mean()
    summary.to_csv(OUT / "expert_adequacy_summary.csv", index=False)
    cal_summary.to_csv(OUT / "expert_calibration_summary.csv", index=False)
    matched_summary.to_csv(OUT / "expert_matched_operating_summary_descriptive.csv", index=False)
    manifest = {
        "stage": "expert_adequacy_complete", "runs": len(runs),
        "all_runs_complete": bool(runs.completed.all()), "all_training_finite": bool(runs.training_finite.all()),
        "all_test_evaluation_count_one": bool((runs.test_evaluation_count == 1).all()),
        "all_thresholds_validation_only": bool(runs.threshold_source.str.contains("validation_only").all()),
        "test_curve_use": "descriptive audit only; never used to select a model or threshold",
        "calibration_slope_sample_cap": 200000,
    }
    (OUT / "stage2_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\nEXPERT SUMMARY\n", summary.to_string(index=False))
    print("\nCALIBRATION\n", cal_summary.to_string(index=False))
    print("\nMATCHED CURVES\n", matched_summary.to_string(index=False))
    print("\nMANIFEST\n", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
