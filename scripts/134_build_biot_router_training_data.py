from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parent.parent
NAMESPACE = "chb-mit-biot-router-minimal-v1"
REPORT = ROOT / "outputs/reports" / NAMESPACE
RUNS = ROOT / "runs" / NAMESPACE / "formal"
MODELS = ["singlem", "cbramod", "tcn_gat"]
KEYS = ["patient", "recording", "start", "end", "relative_path"]


def threshold_relative_signed_confidence(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Return scale-free confidence relative to each expert's frozen threshold.

    A positive value means the expert's thresholded decision is correct and a
    negative value means it is wrong.  Correct and incorrect sides of the
    threshold use their respective maximum distances, so the result is
    bounded in [-1, 1] and comparable across experts with different thresholds.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float).reshape(-1)
    t = np.asarray(thresholds, dtype=float).reshape(-1)
    if p.ndim != 2 or p.shape[0] != y.shape[0] or p.shape[1] != t.shape[0]:
        raise ValueError("probability, label, and threshold shapes are incompatible")
    positive = y[:, None] == 1.0
    raw_signed = np.where(positive, p - t[None, :], t[None, :] - p)
    correct_side_denominator = np.where(positive, 1.0 - t[None, :], t[None, :])
    wrong_side_denominator = np.where(positive, t[None, :], 1.0 - t[None, :])
    denominator = np.where(raw_signed >= 0.0, correct_side_denominator, wrong_side_denominator)
    denominator = np.maximum(denominator, np.finfo(float).eps)
    confidence = raw_signed / denominator
    confidence[~np.isfinite(y), :] = np.nan
    return confidence


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True, choices=range(5))
    args = ap.parse_args()
    out_dir = RUNS / f"round_{args.round}/router"
    manifest_path = out_dir / "router_training_manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite frozen router data: {manifest_path}")
    split = json.loads((REPORT / f"round_{args.round}_split.json").read_text())
    merged: pd.DataFrame | None = None
    thresholds: dict[str, float] = {}
    source_hashes: dict[str, str] = {}
    for model in MODELS:
        model_dir = RUNS / f"round_{args.round}/{model}"
        table_path = model_dir / "router_dev_probabilities.parquet"
        lock_path = model_dir / "threshold_lock.json"
        if not table_path.is_file() or not lock_path.is_file():
            raise FileNotFoundError(f"missing formal expert output for {model}")
        table = pd.read_parquet(table_path)
        threshold = float(json.loads(lock_path.read_text())["threshold"])
        thresholds[model] = threshold
        source_hashes[f"{model}_dev"] = sha256_file(table_path)
        source_hashes[f"{model}_threshold"] = sha256_file(lock_path)
        keep = KEYS + ["label", "probability"]
        table = table[keep].rename(columns={"probability": f"prob_{model}"})
        if merged is None:
            merged = table
        else:
            merged = merged.merge(table, on=KEYS, how="inner", validate="one_to_one", suffixes=("", f"_{model}"))
    if merged is None or merged.empty:
        raise RuntimeError("no Router Dev rows after expert merge")
    labels = merged["label"].to_numpy(dtype=float)
    # Boundary windows can legitimately have unknown (NaN) labels. Compare
    # only finite labels across the merged expert tables; NaN must remain
    # unknown rather than causing a valid merge to fail.
    known_labels = labels[np.isfinite(labels)]
    if known_labels.size == 0 or not np.all(known_labels == known_labels[0]):
        raise RuntimeError("label merge invariant violated for known labels")
    p = merged[[f"prob_{m}" for m in MODELS]].to_numpy(dtype=float)
    t = np.array([thresholds[m] for m in MODELS], dtype=float)
    predicted_positive = p >= t[None, :]
    seizure_mask = labels == 1
    disagreement_mask = (labels == 0) & (predicted_positive.sum(axis=1) > 0) & (predicted_positive.sum(axis=1) < 3)
    hard_negative_mask = (labels == 0) & (predicted_positive.any(axis=1))
    background_mask = (labels == 0) & ~hard_negative_mask
    rng = np.random.default_rng(3407)
    chosen: set[int] = set(np.flatnonzero(seizure_mask).tolist())
    def add_capped(mask: np.ndarray, cap_per_patient: int) -> None:
        for patient, group in merged.loc[mask].groupby("patient", sort=True):
            idx = group.index.to_numpy()
            if len(idx) > cap_per_patient:
                idx = np.sort(rng.choice(idx, size=cap_per_patient, replace=False))
            chosen.update(int(x) for x in idx)
    add_capped(disagreement_mask, 4000)
    add_capped(hard_negative_mask, 4000)
    add_capped(background_mask, 300)
    selected = merged.loc[sorted(chosen)].reset_index(drop=True)
    selected_p = selected[[f"prob_{m}" for m in MODELS]].to_numpy(dtype=float)
    selected_y = selected["label"].to_numpy(dtype=float)
    selected_raw_signed = np.where(
        selected_y[:, None] == 1,
        selected_p - t[None, :],
        t[None, :] - selected_p,
    )
    selected_confidence = threshold_relative_signed_confidence(selected_p, selected_y, t)
    selected_predicted_positive = selected_p >= t[None, :]
    selected_known = np.isfinite(selected_y)
    selected_correct = np.zeros_like(selected_predicted_positive, dtype=bool)
    selected_correct[selected_known] = selected_predicted_positive[selected_known] == (selected_y[selected_known, None] == 1.0)
    target_logits = np.clip(selected_confidence / 0.10, -8.0, 8.0)
    any_correct = selected_correct.any(axis=1)
    target_logits = np.where(any_correct[:, None], np.where(selected_correct, target_logits, -np.inf), target_logits)
    target = softmax(target_logits, axis=1)
    # When multiple experts are clearly correct, retain a soft preference but
    # distribute the blend only among the correct experts.
    multi_correct = selected_correct.sum(axis=1) >= 2
    correct_count = np.maximum(selected_correct.sum(axis=1), 1)[:, None]
    uniform_correct = selected_correct / correct_count
    target[multi_correct] = 0.65 * target[multi_correct] + 0.35 * uniform_correct[multi_correct]
    for j, model in enumerate(MODELS):
        selected[f"competence_target_{model}"] = target[:, j]
        selected[f"raw_signed_margin_{model}"] = selected_raw_signed[:, j]
        selected[f"threshold_relative_confidence_{model}"] = selected_confidence[:, j]
        selected[f"expert_positive_{model}"] = selected_p[:, j] >= thresholds[model]
    groups = selected["patient"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=3407)
    train_idx, val_idx = next(splitter.split(selected, groups=groups))
    train_patients = sorted(selected.iloc[train_idx].patient.unique().tolist())
    val_patients = sorted(selected.iloc[val_idx].patient.unique().tolist())
    if not train_patients or not val_patients or set(train_patients) & set(val_patients):
        raise RuntimeError("invalid Router inner patient split")
    selected["router_inner_partition"] = np.where(selected.patient.isin(train_patients), "inner_train", "inner_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(out_dir / "router_training_targets.parquet", index=False)
    merged.to_parquet(out_dir / "router_dev_merged_expert_outputs.parquet", index=False)
    (out_dir / "router_inner_split.json").write_text(
        json.dumps(
            {
                "group_unit": "canonical_patient_id",
                "algorithm": "sklearn_GroupShuffleSplit",
                "random_state": 3407,
                "train_patients": train_patients,
                "validation_patients": val_patients,
                "test_used": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    payload = {
        "protocol_id": NAMESPACE,
        "round": args.round,
        "status": "frozen",
        "router_dev_patients": split["validation"],
        "inner_train_patients": train_patients,
        "inner_validation_patients": val_patients,
        "counts": {
            "merged_router_dev_rows": int(len(merged)),
            "selected_router_rows": int(len(selected)),
            "seizure_rows": int(seizure_mask.sum()),
            "disagreement_rows": int(disagreement_mask.sum()),
            "hard_negative_rows": int(hard_negative_mask.sum()),
            "background_rows": int(background_mask.sum()),
            "selected_inner_train_rows": int(len(train_idx)),
            "selected_inner_validation_rows": int(len(val_idx)),
        },
        "thresholds": thresholds,
        "competence_target": {
            "definition": "bounded_threshold_relative_signed_confidence; correct side uses seizure (1-threshold) or nonseizure threshold, wrong side uses the opposite side distance",
            "correctness_first": True,
            "unknown_label_policy": "exclude_nan_labels_from_targets",
            "raw_signed_margin_preserved_as": "raw_signed_margin_<model>",
            "temperature": 0.10,
            "clip": [-8.0, 8.0],
            "multi_correct_blend": "0.65*masked-softmax + 0.35*uniform-over-correct-experts when at least two experts are correct",
            "all_wrong_policy": "softmax over all threshold-relative confidences; highest confidence is least-wrong expert",
        },
        "source_hashes": source_hashes,
        "test_used_for_selection": False,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest_path.chmod(0o444)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
