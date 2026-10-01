"""Locked Train-to-Dev evaluation of condition-specific EEG10 competence.

Hyperparameters and feature definitions are inherited unchanged from the
Train-only conditional probe.  Dev is evaluated once for mechanism validation;
no threshold or architecture selection is performed and Eval remains unread.
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
NAMESPACE = "h6-eeg-conditionable-competence-probe-v1"
SOURCE = ROOT / "outputs/reports" / NAMESPACE / "same_window_conditions_v2"
TRAIN_PROBE = ROOT / "outputs/reports" / NAMESPACE / "conditional_eeg10_probe_train_oof_v1/manifest.json"
OUT = ROOT / "outputs/reports" / NAMESPACE / "conditional_eeg10_probe_dev_once_v1"
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


def auc_rank(y: np.ndarray, ranks: np.ndarray) -> float:
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cyclic_null(frame: pd.DataFrame, y: np.ndarray, p: np.ndarray, seed_offset: int) -> dict[str, float]:
    ranks = rankdata(p, method="average")
    observed = auc_rank(y, ranks)
    order = frame.assign(_row=np.arange(len(frame))).sort_values(["patient_id", "recording", "decision_time_s"])
    patient_rows = [group._row.to_numpy(dtype=int) for _, group in order.groupby("patient_id", sort=False)]
    rng = np.random.default_rng(SEED + seed_offset)
    values = np.empty(N_NULL)
    for replicate in range(N_NULL):
        shifted = np.empty_like(y)
        for rows in patient_rows:
            if len(rows) < 2:
                shifted[rows] = y[rows]
            else:
                shifted[rows] = np.roll(y[rows], int(rng.integers(1, len(rows))))
        values[replicate] = auc_rank(shifted, ranks)
    return {
        "null_mean": float(np.nanmean(values)),
        "null_q025": float(np.nanquantile(values, 0.025)),
        "null_q975": float(np.nanquantile(values, 0.975)),
        "p_value_one_sided": float((1 + np.count_nonzero(values >= observed)) / (N_NULL + 1)),
    }


def bootstrap(frame: pd.DataFrame, y: np.ndarray, p: np.ndarray, seed_offset: int) -> dict[str, float | int]:
    patient = frame.patient_id.astype(str).to_numpy()
    unique, inverse = np.unique(patient, return_inverse=True)
    rows = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    rng = np.random.default_rng(SEED + 10000 + seed_offset)
    values = np.empty(N_BOOTSTRAP)
    for replicate in range(N_BOOTSTRAP):
        sampled = rng.integers(0, len(rows), len(rows))
        index = np.concatenate([rows[item] for item in sampled])
        values[replicate] = roc_auc_score(y[index], p[index]) if len(np.unique(y[index])) == 2 else np.nan
    return {
        "bootstrap_valid": int(np.isfinite(values).sum()),
        "auroc_q025": float(np.nanquantile(values, 0.025)),
        "auroc_median": float(np.nanmedian(values)),
        "auroc_q975": float(np.nanquantile(values, 0.975)),
    }


def bh(values: list[float]) -> list[float]:
    p = np.asarray(values)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result.tolist()


def main() -> None:
    if not TRAIN_PROBE.exists():
        raise FileNotFoundError("Train-only gate must be frozen before Dev evaluation")
    train_gate = json.loads(TRAIN_PROBE.read_text())
    if not all(train_gate["gates"][condition]["gate2_requires_two_seizure_heads"] for condition in CONDITIONS):
        raise RuntimeError("Train-only conditional gate did not authorize Dev evaluation")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing a second Dev evaluation: {manifest_path}")
    records: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    test_index = 0
    for condition in CONDITIONS:
        train_all = pd.read_parquet(SOURCE / f"train_{condition}.parquet")
        dev_all = pd.read_parquet(SOURCE / f"dev_{condition}.parquet")
        for subgroup, value in (("seizure", 1), ("background", 0)):
            train = train_all[train_all.current_has_seizure.eq(value)].copy()
            dev = dev_all[dev_all.current_has_seizure.eq(value)].copy().reset_index(drop=True)
            x_train = train.loc[:, EEG10].to_numpy(dtype=float)
            x_dev = dev.loc[:, EEG10].to_numpy(dtype=float)
            output = dev[["sample_id", "patient_id", "recording", "decision_time_s"]].copy()
            for model in MODELS:
                target = f"correct_{model}"
                y_train = train[target].to_numpy(dtype=int)
                y_dev = dev[target].to_numpy(dtype=int)
                if len(np.unique(y_train)) < 2:
                    p_dev = np.full(len(dev), float(y_train[0]))
                else:
                    estimator = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            class_weight="balanced", solver="liblinear", max_iter=1000,
                            random_state=SEED + test_index,
                        ),
                    )
                    estimator.fit(x_train, y_train)
                    p_dev = estimator.predict_proba(x_dev)[:, 1]
                output[f"truth_{model}"] = y_dev
                output[f"p_{model}"] = p_dev
                record: dict[str, object] = {
                    "condition": condition, "subgroup": subgroup, "expert": model,
                    "train_n": int(len(train)), "dev_n": int(len(dev)),
                    "train_patients": int(train.patient_id.nunique()), "dev_patients": int(dev.patient_id.nunique()),
                    "dev_positives": int(y_dev.sum()), "dev_prevalence": float(y_dev.mean()),
                    "auroc": float(roc_auc_score(y_dev, p_dev)) if len(np.unique(y_dev)) == 2 else float("nan"),
                    "auprc": float(average_precision_score(y_dev, p_dev)) if len(np.unique(y_dev)) == 2 else float("nan"),
                    "brier": float(brier_score_loss(y_dev, p_dev)),
                }
                if len(np.unique(y_dev)) == 2:
                    record.update(cyclic_null(dev, y_dev, p_dev, test_index * 100))
                    record.update(bootstrap(dev, y_dev, p_dev, test_index * 100))
                records.append(record)
                test_index += 1
            output["condition"] = condition
            output["subgroup"] = subgroup
            predictions.append(output)
            print(f"evaluated {condition} / {subgroup}", flush=True)
    metrics = pd.DataFrame(records)
    valid = metrics.p_value_one_sided.notna()
    metrics.loc[valid, "q_value_bh"] = bh(metrics.loc[valid, "p_value_one_sided"].tolist())
    metrics["significant_q05"] = metrics.q_value_bh.lt(0.05).fillna(False)
    metrics_path = OUT / "dev_metrics.csv"
    predictions_path = OUT / "dev_predictions.parquet"
    metrics.to_csv(metrics_path, index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(predictions_path, index=False)
    gates = {}
    for condition in CONDITIONS:
        seizure = metrics[(metrics.condition.eq(condition)) & metrics.subgroup.eq("seizure")]
        gates[condition] = {
            "seizure_heads_auroc_above_0_5": int((seizure.auroc > 0.5).sum()),
            "seizure_heads_significant_q05": int(seizure.significant_q05.sum()),
            "dev_mechanism_replication": bool(((seizure.auroc > 0.5) & seizure.significant_q05).sum() >= 2),
        }
    manifest = {
        "release_id": NAMESPACE,
        "stage": "locked_train_to_dev_condition_specific_eeg10_probe_v1",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dev_evaluation_count": 1,
        "dev_used_for_model_or_threshold_selection": False,
        "eval_read": False,
        "expert_training": False,
        "gates": gates,
        "train_probe_manifest_sha256": sha256(TRAIN_PROBE),
        "source_manifest_sha256": sha256(SOURCE / "manifest.json"),
        "outputs": {
            str(metrics_path.relative_to(ROOT)): sha256(metrics_path),
            str(predictions_path.relative_to(ROOT)): sha256(predictions_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (OUT / "manifest.sha256").write_text(f"{sha256(manifest_path)}  manifest.json\n")
    print(metrics.to_string(index=False))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
