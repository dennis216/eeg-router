"""Condition-specific EEG10 competence probes for the H6 mechanism gate.

Separate patient-OOF models are trained within seizure and background rows.
This checks whether the full-data probe failed on seizure competence merely
because background rows dominated training.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
NAMESPACE = "h6-eeg-conditionable-competence-probe-v1"
SOURCE = ROOT / "outputs/reports" / NAMESPACE / "same_window_conditions_v2"
OUT = ROOT / "outputs/reports" / NAMESPACE / "conditional_eeg10_probe_train_oof_v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
CONDITIONS = ("matched_sensitivity", "matched_fa")
EEG10 = (
    "rms_median_uv", "rms_iqr_uv", "spatial_rms_dispersion", "spatial_rms_max_min_log",
    "relative_delta_power", "relative_theta_power", "relative_alpha_power",
    "relative_beta_power", "relative_gamma_power", "high_low_ratio",
)
SEED = 20260831
N_NULL = 1000
N_BOOTSTRAP = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_oof(frame: pd.DataFrame, target: str, seed_offset: int) -> np.ndarray:
    x = frame.loc[:, EEG10].to_numpy(dtype=float)
    y = frame[target].to_numpy(dtype=int)
    groups = frame.patient_id.astype(str).to_numpy()
    output = np.full(len(frame), np.nan)
    splitter = GroupKFold(n_splits=5)
    for fold, (train_index, test_index) in enumerate(splitter.split(x, groups=groups), start=1):
        if len(np.unique(y[train_index])) < 2:
            output[test_index] = float(y[train_index][0])
            continue
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", solver="liblinear", max_iter=1000,
                random_state=SEED + seed_offset + fold,
            ),
        )
        estimator.fit(x[train_index], y[train_index])
        output[test_index] = estimator.predict_proba(x[test_index])[:, 1]
    if not np.isfinite(output).all():
        raise RuntimeError("non-finite conditional OOF predictions")
    return output


def auc_rank(y: np.ndarray, ranks: np.ndarray) -> float:
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def null_test(frame: pd.DataFrame, target: str, p: np.ndarray, seed_offset: int) -> dict[str, float | int]:
    y = frame[target].to_numpy(dtype=np.int8)
    ranks = rankdata(p, method="average")
    observed = auc_rank(y, ranks)
    order = frame.assign(_row=np.arange(len(frame))).sort_values(["patient_id", "recording", "decision_time_s"])
    patient_rows = [group._row.to_numpy(dtype=int) for _, group in order.groupby("patient_id", sort=False)]
    rng = np.random.default_rng(SEED + seed_offset)
    null = np.empty(N_NULL)
    for replicate in range(N_NULL):
        shuffled = np.empty_like(y)
        for rows in patient_rows:
            if len(rows) < 2:
                shuffled[rows] = y[rows]
            else:
                shuffled[rows] = np.roll(y[rows], int(rng.integers(1, len(rows))))
        null[replicate] = auc_rank(shuffled, ranks)
    return {
        "observed_auroc": observed,
        "null_mean": float(np.nanmean(null)),
        "null_q025": float(np.nanquantile(null, 0.025)),
        "null_q975": float(np.nanquantile(null, 0.975)),
        "p_value_one_sided": float((1 + np.count_nonzero(null >= observed)) / (N_NULL + 1)),
    }


def bootstrap(frame: pd.DataFrame, target: str, p: np.ndarray, seed_offset: int) -> dict[str, float | int]:
    y = frame[target].to_numpy(dtype=int)
    patient = frame.patient_id.astype(str).to_numpy()
    unique, inverse = np.unique(patient, return_inverse=True)
    rows = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    rng = np.random.default_rng(SEED + 10000 + seed_offset)
    values = np.empty(N_BOOTSTRAP)
    for replicate in range(N_BOOTSTRAP):
        sampled = rng.integers(0, len(rows), size=len(rows))
        index = np.concatenate([rows[item] for item in sampled])
        values[replicate] = roc_auc_score(y[index], p[index]) if len(np.unique(y[index])) == 2 else np.nan
    return {
        "bootstrap_valid": int(np.isfinite(values).sum()),
        "auroc_q025": float(np.nanquantile(values, 0.025)),
        "auroc_median": float(np.nanmedian(values)),
        "auroc_q975": float(np.nanquantile(values, 0.975)),
    }


def bh(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result.tolist()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite frozen probe: {manifest_path}")
    rows: list[dict[str, object]] = []
    predictions_out: list[pd.DataFrame] = []
    test_index = 0
    for condition in CONDITIONS:
        source = pd.read_parquet(SOURCE / f"train_{condition}.parquet")
        for subgroup, selector in (
            ("seizure", source.current_has_seizure.eq(1)),
            ("background", source.current_has_seizure.eq(0)),
        ):
            frame = source.loc[selector].sort_values(["patient_id", "recording", "decision_time_s"]).reset_index(drop=True)
            output = frame[["sample_id", "patient_id", "recording", "decision_time_s"]].copy()
            for model in MODELS:
                target = f"correct_{model}"
                y = frame[target].to_numpy(dtype=int)
                p = fit_oof(frame, target, test_index * 10) if len(np.unique(y)) == 2 else np.full(len(frame), float(y[0]))
                output[f"truth_{model}"] = y
                output[f"p_{model}"] = p
                record: dict[str, object] = {
                    "condition": condition, "subgroup": subgroup, "expert": model,
                    "n": int(len(y)), "patients": int(frame.patient_id.nunique()),
                    "positives": int(y.sum()), "prevalence": float(y.mean()),
                    "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
                    "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
                }
                if len(np.unique(y)) == 2:
                    record.update(null_test(frame, target, p, test_index * 100))
                    record.update(bootstrap(frame, target, p, test_index * 100))
                rows.append(record)
                test_index += 1
            output["condition"] = condition
            output["subgroup"] = subgroup
            predictions_out.append(output)
            print(f"completed {condition} / {subgroup}", flush=True)

    table = pd.DataFrame(rows)
    valid = table.p_value_one_sided.notna()
    table.loc[valid, "q_value_bh"] = bh(table.loc[valid, "p_value_one_sided"].tolist())
    table["significant_q05"] = table.q_value_bh.lt(0.05).fillna(False)
    metrics_path = OUT / "conditional_metrics.csv"
    prediction_path = OUT / "oof_predictions.parquet"
    table.to_csv(metrics_path, index=False)
    pd.concat(predictions_out, ignore_index=True).to_parquet(prediction_path, index=False)

    seizure = table[table.subgroup.eq("seizure")]
    gates = {}
    for condition in CONDITIONS:
        subset = seizure[seizure.condition.eq(condition)]
        gates[condition] = {
            "seizure_heads_auroc_above_0_5": int((subset.auroc > 0.5).sum()),
            "seizure_heads_significant_q05": int(subset.significant_q05.sum()),
            "gate2_requires_two_seizure_heads": bool(((subset.auroc > 0.5) & subset.significant_q05).sum() >= 2),
        }
    manifest = {
        "release_id": NAMESPACE,
        "stage": "train_patient_oof_condition_specific_eeg10_competence_probe_v1",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dev_read": False,
        "eval_read": False,
        "expert_training": False,
        "model_training": "separate class-balanced OVA logistic probes within seizure and background",
        "patient_split": "GroupKFold(n_splits=5) by canonical patient_id within each subgroup",
        "null_replicates": N_NULL,
        "patient_bootstrap_replicates": N_BOOTSTRAP,
        "gates": gates,
        "source_manifest_sha256": sha256(SOURCE / "manifest.json"),
        "outputs": {
            str(metrics_path.relative_to(ROOT)): sha256(metrics_path),
            str(prediction_path.relative_to(ROOT)): sha256(prediction_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (OUT / "manifest.sha256").write_text(f"{sha256(manifest_path)}  manifest.json\n")
    print(table.to_string(index=False))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
