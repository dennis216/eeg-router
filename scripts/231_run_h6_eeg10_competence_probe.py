"""Run the Train-only patient-OOF EEG10 competence mechanism probe.

This is a diagnostic gate, not a clinical router evaluation.  Three independent
one-vs-all logistic heads estimate each frozen expert's same-window correctness.
The main EEG10 representation is compared with output-only and combined models.
Patient-disjoint OOF predictions, a patient-preserving cyclic null, and a
2,000-replicate patient bootstrap are used without reading Dev or Eval.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
NAMESPACE = "h6-eeg-conditionable-competence-probe-v1"
SOURCE = ROOT / "outputs/reports" / NAMESPACE / "same_window_conditions_v2"
OUT = ROOT / "outputs/reports" / NAMESPACE / "eeg10_probe_train_oof_v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
CONDITIONS = ("frozen", "matched_sensitivity", "matched_fa")
EEG10 = (
    "rms_median_uv", "rms_iqr_uv", "spatial_rms_dispersion", "spatial_rms_max_min_log",
    "relative_delta_power", "relative_theta_power", "relative_alpha_power",
    "relative_beta_power", "relative_gamma_power", "high_low_ratio",
)
OUTPUTS = tuple(f"ema_probability_{model}" for model in MODELS)
REPRESENTATIONS = {"eeg10": EEG10, "output_only": OUTPUTS, "eeg10_plus_outputs": EEG10 + OUTPUTS}
N_SPLITS = 5
N_NULL = 1000
N_BOOTSTRAP = 2000
SEED = 20260831


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    result: dict[str, float | int] = {"n": int(len(y)), "positives": int(y.sum()), "prevalence": float(y.mean())}
    if len(np.unique(y)) < 2:
        result.update({"auroc": float("nan"), "auprc": float("nan")})
    else:
        result.update({"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p))})
    result["brier"] = float(brier_score_loss(y, p))
    return result


def fit_oof(frame: pd.DataFrame, features: tuple[str, ...], targets: tuple[str, ...]) -> np.ndarray:
    x = frame.loc[:, features].to_numpy(dtype=np.float64)
    groups = frame.patient_id.astype(str).to_numpy()
    outputs = np.full((len(frame), len(targets)), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=N_SPLITS)
    for fold, (train_index, test_index) in enumerate(splitter.split(x, groups=groups), start=1):
        for target_index, target in enumerate(targets):
            y_train = frame[target].to_numpy(dtype=int)[train_index]
            if len(np.unique(y_train)) < 2:
                outputs[test_index, target_index] = float(y_train[0])
                continue
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced", solver="liblinear", max_iter=1000,
                    random_state=SEED + fold * 100 + target_index,
                ),
            )
            estimator.fit(x[train_index], y_train)
            outputs[test_index, target_index] = estimator.predict_proba(x[test_index])[:, 1]
        print(f"OOF {features[0]}... fold {fold}/{N_SPLITS}", flush=True)
    if not np.isfinite(outputs).all():
        raise RuntimeError("non-finite OOF predictions")
    return outputs


def patient_cyclic_null(y: np.ndarray, patient_indices: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    permuted = np.empty_like(y)
    for indices in patient_indices:
        if len(indices) < 2:
            permuted[indices] = y[indices]
        else:
            shift = int(rng.integers(1, len(indices)))
            permuted[indices] = np.roll(y[indices], shift)
    return permuted


def auc_from_fixed_ranks(y: np.ndarray, ranks: np.ndarray) -> float:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    return float((ranks[y.astype(bool)].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def cyclic_null_test(frame: pd.DataFrame, target: str, predictions: np.ndarray, seed_offset: int) -> dict[str, object]:
    order = frame.assign(_row=np.arange(len(frame))).sort_values(["patient_id", "recording", "decision_time_s"])
    patient_indices = [group._row.to_numpy(dtype=int) for _, group in order.groupby("patient_id", sort=False)]
    y = frame[target].to_numpy(dtype=np.int8)
    ranks = rankdata(predictions, method="average")
    observed = auc_from_fixed_ranks(y, ranks)
    rng = np.random.default_rng(SEED + seed_offset)
    null = np.empty(N_NULL, dtype=float)
    for replicate in range(N_NULL):
        null[replicate] = auc_from_fixed_ranks(patient_cyclic_null(y, patient_indices, rng), ranks)
    p_value = float((1 + np.count_nonzero(null >= observed)) / (N_NULL + 1))
    return {
        "observed_auroc": observed,
        "null_mean": float(np.nanmean(null)),
        "null_sd": float(np.nanstd(null, ddof=1)),
        "null_q025": float(np.nanquantile(null, 0.025)),
        "null_q975": float(np.nanquantile(null, 0.975)),
        "p_value_one_sided": p_value,
        "replicates": N_NULL,
        "null_contract": "independent cyclic shift within each patient after ordering by recording and time; patient prevalence preserved",
    }


def patient_bootstrap_auc(frame: pd.DataFrame, target: str, predictions: np.ndarray, seed_offset: int) -> dict[str, float | int]:
    patients = frame.patient_id.astype(str).to_numpy()
    unique_patients, inverse = np.unique(patients, return_inverse=True)
    patient_rows = [np.flatnonzero(inverse == index) for index in range(len(unique_patients))]
    y = frame[target].to_numpy(dtype=int)
    rng = np.random.default_rng(SEED + 10000 + seed_offset)
    values = np.empty(N_BOOTSTRAP, dtype=float)
    for replicate in range(N_BOOTSTRAP):
        sampled = rng.integers(0, len(patient_rows), size=len(patient_rows))
        indices = np.concatenate([patient_rows[index] for index in sampled])
        if len(np.unique(y[indices])) < 2:
            values[replicate] = np.nan
        else:
            values[replicate] = roc_auc_score(y[indices], predictions[indices])
    return {
        "replicates": N_BOOTSTRAP,
        "valid_replicates": int(np.isfinite(values).sum()),
        "auroc_q025": float(np.nanquantile(values, 0.025)),
        "auroc_median": float(np.nanmedian(values)),
        "auroc_q975": float(np.nanquantile(values, 0.975)),
    }


def bh_fdr(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result.tolist()


def selection_summary(frame: pd.DataFrame, predictions: np.ndarray) -> dict[str, object]:
    truth = frame[[f"correct_{model}" for model in MODELS]].to_numpy(dtype=int)
    chosen = np.argmax(predictions, axis=1)
    chosen_correct = truth[np.arange(len(frame)), chosen]
    actionable = frame.actionable.to_numpy(dtype=bool)
    shares = {model: float(np.mean(chosen == index)) for index, model in enumerate(MODELS)}
    baseline_expert = int(np.argmax(truth.mean(axis=0)))
    random_expected = float(np.mean(frame.loc[actionable, "correct_count"].to_numpy() / 3.0)) if actionable.any() else float("nan")
    return {
        "selection_share": shares,
        "max_selection_share": float(max(shares.values())),
        "all_rows_chosen_correct": float(chosen_correct.mean()),
        "actionable_rows": int(actionable.sum()),
        "actionable_top_competence_accuracy": float(chosen_correct[actionable].mean()) if actionable.any() else float("nan"),
        "actionable_global_best_fixed_accuracy": float(truth[actionable, baseline_expert].mean()) if actionable.any() else float("nan"),
        "actionable_random_expected_accuracy": random_expected,
        "global_best_fixed_expert": MODELS[baseline_expert],
        "collapse_over_95pct": bool(max(shares.values()) > 0.95),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite frozen probe: {manifest_path}")
    all_metrics: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    output_hashes: dict[str, str] = {}

    for condition_index, condition in enumerate(CONDITIONS):
        source_path = SOURCE / f"train_{condition}.parquet"
        frame = pd.read_parquet(source_path).sort_values(["patient_id", "recording", "decision_time_s"]).reset_index(drop=True)
        targets = tuple(f"correct_{model}" for model in MODELS)
        condition_predictions: dict[str, np.ndarray] = {}
        for representation, features in REPRESENTATIONS.items():
            print(f"{condition}: fitting {representation}", flush=True)
            predictions = fit_oof(frame, features, targets)
            condition_predictions[representation] = predictions
            for model_index, model in enumerate(MODELS):
                y = frame[f"correct_{model}"].to_numpy(dtype=int)
                p = predictions[:, model_index]
                for subgroup, mask in (
                    ("all", np.ones(len(frame), dtype=bool)),
                    ("seizure", frame.current_has_seizure.to_numpy(dtype=bool)),
                    ("background", ~frame.current_has_seizure.to_numpy(dtype=bool)),
                ):
                    all_metrics.append({
                        "condition": condition, "representation": representation,
                        "expert": model, "subgroup": subgroup,
                        **safe_metrics(y[mask], p[mask]),
                    })
            selection_rows.append({
                "condition": condition, "representation": representation,
                **selection_summary(frame, predictions),
            })

        primary = condition_predictions["eeg10"]
        for model_index, model in enumerate(MODELS):
            target = f"correct_{model}"
            test = cyclic_null_test(frame, target, primary[:, model_index], condition_index * 10 + model_index)
            null_rows.append({"condition": condition, "expert": model, **test})
            ci = patient_bootstrap_auc(frame, target, primary[:, model_index], condition_index * 10 + model_index)
            bootstrap_rows.append({"condition": condition, "expert": model, **ci})

        # Same-capacity diagnostic probes: seizure/background recognition and
        # background all-FP suppression risk.  These identify whether EEG10 is
        # merely solving an easier surrogate task.
        diagnostic_targets = ("current_has_seizure", "all_fp_background")
        frame["all_fp_background"] = (
            frame.current_has_seizure.eq(0)
            & frame[[f"prediction_{model}" for model in MODELS]].eq(1).all(axis=1)
        ).astype(np.int8)
        diagnostic_predictions = fit_oof(frame, EEG10, diagnostic_targets)
        for target_index, target in enumerate(diagnostic_targets):
            diagnostic_rows.append({
                "condition": condition, "target": target,
                **safe_metrics(frame[target].to_numpy(dtype=int), diagnostic_predictions[:, target_index]),
            })

        oof = frame[["sample_id", "patient_id", "recording", "decision_time_s", "current_has_seizure", "correct_mask", "actionable"]].copy()
        for representation, predictions in condition_predictions.items():
            for model_index, model in enumerate(MODELS):
                oof[f"p_{representation}_{model}"] = predictions[:, model_index]
        oof_path = OUT / f"oof_predictions_{condition}.parquet"
        oof.to_parquet(oof_path, index=False)
        output_hashes[str(oof_path.relative_to(ROOT))] = sha256(oof_path)

    q_values = bh_fdr([float(row["p_value_one_sided"]) for row in null_rows])
    for row, q_value in zip(null_rows, q_values):
        row["q_value_bh_across_9_heads"] = q_value
        row["significant_q05"] = bool(q_value < 0.05)

    tables = {
        "metrics.csv": pd.DataFrame(all_metrics),
        "patient_cyclic_null.csv": pd.DataFrame(null_rows),
        "patient_bootstrap_auroc.csv": pd.DataFrame(bootstrap_rows),
        "selection_summary.json": selection_rows,
        "diagnostic_probes.csv": pd.DataFrame(diagnostic_rows),
    }
    for name, value in tables.items():
        path = OUT / name
        if isinstance(value, pd.DataFrame):
            value.to_csv(path, index=False)
        else:
            path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output_hashes[str(path.relative_to(ROOT))] = sha256(path)

    null_frame = pd.DataFrame(null_rows)
    selection_frame = pd.DataFrame(selection_rows)
    gates: dict[str, object] = {}
    for condition in CONDITIONS:
        condition_null = null_frame[null_frame.condition.eq(condition)]
        condition_selection = selection_frame[
            selection_frame.condition.eq(condition) & selection_frame.representation.eq("eeg10")
        ].iloc[0]
        gates[condition] = {
            "significant_heads_q05": int(condition_null.significant_q05.sum()),
            "gate1_at_least_two_heads": bool(condition_null.significant_q05.sum() >= 2),
            "max_selection_share": float(condition_selection.max_selection_share),
            "collapse_over_95pct": bool(condition_selection.collapse_over_95pct),
        }

    manifest = {
        "release_id": NAMESPACE,
        "stage": "train_patient_oof_eeg10_competence_probe_v1",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dev_read": False,
        "eval_read": False,
        "expert_training": False,
        "model_training": "class-balanced one-vs-all logistic diagnostic probes only",
        "patient_split": f"GroupKFold(n_splits={N_SPLITS}) by canonical patient_id",
        "representations": {key: list(value) for key, value in REPRESENTATIONS.items()},
        "null_replicates": N_NULL,
        "patient_bootstrap_replicates": N_BOOTSTRAP,
        "source_manifest_sha256": sha256(SOURCE / "manifest.json"),
        "gates": gates,
        "output_hashes": output_hashes,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (OUT / "manifest.sha256").write_text(f"{sha256(manifest_path)}  manifest.json\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
