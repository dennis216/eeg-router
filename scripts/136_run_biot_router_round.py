from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
NAMESPACE = "chb-mit-biot-router-minimal-v1"
REPORT = ROOT / "outputs/reports" / NAMESPACE
RUNS = ROOT / "runs" / NAMESPACE / "formal"
MODELS = ["singlem", "cbramod", "tcn_gat"]
KEYS = ["patient", "recording", "start", "end", "relative_path"]
sys.path.insert(0, str(ROOT / "third_party/BIOT"))
from bfa.evaluation.eventize import eventize  # noqa: E402
from bfa.evaluation.match import match_events  # noqa: E402


def load_v3_helpers():
    import importlib.util

    path = ROOT / "scripts/28_run_v3_unified.py"
    spec = importlib.util.spec_from_file_location("v3_unified_helpers_for_router", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v3 = load_v3_helpers()


class RouterHead(torch.nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(256, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def threshold_relative_signed_confidence(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Compute comparable confidence relative to each frozen threshold.

    Correct and incorrect sides use their respective maximum distances from
    the threshold.  The sign therefore encodes correctness and the magnitude
    is bounded in [-1, 1], making experts with different thresholds comparable.
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


def select_correctness_first_oracle(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
    *,
    unknown_fallback_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select a correct expert first, then maximize threshold-relative confidence.

    If no expert is correct for a window, the least-wrong expert (largest
    signed confidence) is selected so the selector remains total and can be
    eventized on the same 10-second grid.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float).reshape(-1)
    t = np.asarray(thresholds, dtype=float).reshape(-1)
    confidence = threshold_relative_signed_confidence(p, y, t)
    known = np.isfinite(y)
    predicted_positive = p >= t[None, :]
    correct = np.zeros_like(predicted_positive, dtype=bool)
    correct[known] = predicted_positive[known] == (y[known, None] == 1.0)
    masked = np.where(correct, confidence, -np.inf)
    selected = np.argmax(masked, axis=1)
    unknown = ~known
    selected[unknown] = int(unknown_fallback_index)
    no_correct = known & ~correct.any(axis=1)
    if np.any(no_correct):
        selected[no_correct] = np.argmax(confidence[no_correct], axis=1)
    return selected, confidence, correct


def summarize_oracle_confidence(
    confidence: np.ndarray,
    correct: np.ndarray,
    selected: np.ndarray,
    labels: np.ndarray,
    best_fixed_index: int,
) -> dict[str, float]:
    """Compare Oracle confidence and correctness with the frozen best expert."""
    row_index = np.arange(len(selected))
    selected_confidence = confidence[row_index, selected]
    fixed_confidence = confidence[:, best_fixed_index]
    selected_correct = correct[row_index, selected]
    fixed_correct = correct[:, best_fixed_index]
    label_array = np.asarray(labels, dtype=float)
    known = np.isfinite(label_array)
    if not np.any(known):
        raise ValueError("cannot summarize Oracle confidence without known labels")
    confidence_gain = selected_confidence - fixed_confidence
    def subset(mask: np.ndarray) -> dict[str, float | int]:
        if not np.any(mask):
            return {"n_windows": 0}
        return {
            "n_windows": int(mask.sum()),
            "oracle_mean_threshold_relative_confidence": float(np.nanmean(selected_confidence[mask])),
            "best_fixed_mean_threshold_relative_confidence": float(np.nanmean(fixed_confidence[mask])),
            "oracle_mean_confidence_gain_vs_best_fixed": float(np.nanmean(confidence_gain[mask])),
            "oracle_median_confidence_gain_vs_best_fixed": float(np.nanmedian(confidence_gain[mask])),
            "oracle_correct_window_fraction": float(np.mean(selected_correct[mask])),
            "best_fixed_correct_window_fraction": float(np.mean(fixed_correct[mask])),
            "oracle_correct_window_gain_pp_vs_best_fixed": float(100.0 * (np.mean(selected_correct[mask]) - np.mean(fixed_correct[mask]))),
        }

    overall = {
        "known_window_fraction": float(np.mean(known)),
        "unknown_window_count": int((~known).sum()),
        **subset(known),
    }
    overall["by_label"] = {
        "seizure_detection_windows": subset(known & (label_array == 1.0)),
        "nonseizure_false_alarm_windows": subset(known & (label_array == 0.0)),
    }
    return overall


def truth_intervals(seizures: pd.DataFrame, recording: str, start: float) -> list[tuple[float, float]]:
    rows = seizures[seizures.recording_id == recording]
    return [(max(float(r.start_s), start), float(r.end_s)) for r in rows.itertuples(index=False) if float(r.end_s) > start]


def score_selected(
    table: pd.DataFrame,
    selections: np.ndarray,
    thresholds: np.ndarray,
    seizures: pd.DataFrame,
    recordings: pd.DataFrame,
) -> dict[str, float | int]:
    if len(table) != len(selections):
        raise ValueError("selection/table length mismatch")
    probs = table[[f"prob_{m}" for m in MODELS]].to_numpy(dtype=float)
    selected = probs[np.arange(len(table)), selections]
    margins = selected - thresholds[selections]
    working = table[KEYS + ["label"]].copy()
    working["margin"] = margins
    tp = fa = truth_count = 0
    nonseizure_seconds = 0.0
    for recording_id, group in working.groupby("recording", sort=False):
        group = group.sort_values("end", kind="stable")
        times = group.end.to_numpy(dtype=float)
        if len(times) < 2:
            continue
        predictions = eventize(times, group.margin.to_numpy(dtype=float), threshold=0.0)
        evaluation_start = float(times[0])
        duration = float(recordings.loc[recordings.recording_id == recording_id, "duration_s"].iloc[0])
        truths = truth_intervals(seizures, str(recording_id), evaluation_start)
        matched = match_events(predictions, truths)
        tp += len(matched.pairs)
        fa += len(matched.unmatched_predictions)
        truth_count += len(truths)
        nonseizure_seconds += max(0.0, duration - evaluation_start - sum(max(0.0, e - s) for s, e in truths))
    hours = nonseizure_seconds / 3600.0
    return {
        "true_positive_events": tp,
        "false_alarm_events": fa,
        "truth_events": truth_count,
        "event_sensitivity": tp / truth_count if truth_count else 1.0,
        "nonseizure_hours": hours,
        "fa_per_24h": fa * 24.0 / hours if hours > 0 else float("nan"),
    }


def choose_best_fixed(dev: pd.DataFrame, thresholds: dict[str, float], seizures: pd.DataFrame, recordings: pd.DataFrame) -> tuple[str, dict[str, dict[str, float | int]]]:
    scores: dict[str, dict[str, float | int]] = {}
    for model in MODELS:
        scores[model] = v3.score_probabilities(dev[KEYS + ["label", f"prob_{model}"]].rename(columns={f"prob_{model}": "probability"}), seizures, recordings, thresholds[model])
    feasible = [m for m in MODELS if float(scores[m]["event_sensitivity"]) >= 0.80]
    if feasible:
        best = min(feasible, key=lambda m: (float(scores[m]["fa_per_24h"]), -float(scores[m]["event_sensitivity"])))
    else:
        best = max(MODELS, key=lambda m: (float(scores[m]["event_sensitivity"]), -float(scores[m]["fa_per_24h"])))
    return best, scores


def load_embedding_table(table: pd.DataFrame, embedding_index: pd.DataFrame, memmap: np.memmap) -> np.ndarray:
    joined = table[KEYS].merge(embedding_index, on=KEYS, how="left", validate="one_to_one", sort=False)
    if joined.window_row_id.isna().any():
        raise RuntimeError("missing BIOT embedding for one or more windows")
    ids = joined.window_row_id.to_numpy(dtype=np.int64)
    return np.asarray(memmap[ids], dtype=np.float32)


def train_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden: int,
    dropout: float,
    lr: float,
    seed: int,
    updates: int = 300,
) -> RouterHead:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RouterHead(hidden, dropout).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=min(256, len(dataset)), shuffle=True, num_workers=0, drop_last=False)
    iterator = iter(loader)
    model.train()
    for update in range(1, updates + 1):
        try:
            xb, yb = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            xb, yb = next(iterator)
        xb = xb.cuda(non_blocking=True)
        yb = yb.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = -(yb * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True, choices=range(5))
    ap.add_argument("--router-updates", type=int, default=300)
    args = ap.parse_args()
    round_dir = RUNS / f"round_{args.round}"
    router_dir = round_dir / "router"
    result_path = router_dir / "router_test_metrics.json"
    if result_path.exists():
        print(result_path.read_text())
        return
    target_path = router_dir / "router_training_targets.parquet"
    if not target_path.is_file():
        raise FileNotFoundError(target_path)
    dev_tables = []
    for model in MODELS:
        dev_tables.append(pd.read_parquet(round_dir / model / "router_dev_probabilities.parquet").rename(columns={"probability": f"prob_{model}"}))
    dev = dev_tables[0][KEYS + ["label", "prob_singlem"]]
    for table, model in zip(dev_tables[1:], MODELS[1:], strict=True):
        dev = dev.merge(table[KEYS + [f"prob_{model}"]], on=KEYS, how="inner", validate="one_to_one", sort=False)
    dev_known_labels = dev["label"].to_numpy(dtype=float)
    dev_known_labels = dev_known_labels[np.isfinite(dev_known_labels)]
    if dev_known_labels.size == 0 or not np.all(dev_known_labels == dev_known_labels[0]):
        raise RuntimeError("Router Dev label merge invariant violated for known labels")
    test_tables = []
    for model in MODELS:
        test_tables.append(pd.read_parquet(round_dir / model / "final_test_probabilities.parquet").rename(columns={"probability": f"prob_{model}"}))
    test = test_tables[0][KEYS + ["label", "prob_singlem"]]
    for table, model in zip(test_tables[1:], MODELS[1:], strict=True):
        test = test.merge(table[KEYS + [f"prob_{model}"]], on=KEYS, how="inner", validate="one_to_one", sort=False)
    test_known_labels = test["label"].to_numpy(dtype=float)
    test_known_labels = test_known_labels[np.isfinite(test_known_labels)]
    if test_known_labels.size == 0 or not np.all(test_known_labels == test_known_labels[0]):
        raise RuntimeError("Final Test label merge invariant violated for known labels")
    thresholds = {m: float(json.loads((round_dir / m / "threshold_lock.json").read_text())["threshold"]) for m in MODELS}
    threshold_array = np.array([thresholds[m] for m in MODELS], dtype=float)
    targets = pd.read_parquet(target_path)
    embedding_index = pd.read_parquet("/mnt/d/EEGData/bfa_biot_prest_embeddings_v1/embedding_index.parquet")
    rows = len(embedding_index)
    memmap = np.memmap("/mnt/d/EEGData/bfa_biot_prest_embeddings_v1/embeddings.float16.memmap", mode="r", dtype=np.float16, shape=(rows, 256))
    x_all_dev = load_embedding_table(dev, embedding_index, memmap)
    x_all_test = load_embedding_table(test, embedding_index, memmap)
    x_target = load_embedding_table(targets, embedding_index, memmap)
    target_cols = [f"competence_target_{m}" for m in MODELS]
    y_target = targets[target_cols].to_numpy(dtype=np.float32)
    inner_train_mask = targets.router_inner_partition.to_numpy() == "inner_train"
    inner_val_mask = ~inner_train_mask
    seizures = pd.read_parquet(ROOT / "manifests/seizures.parquet")
    recordings = pd.read_parquet(ROOT / "manifests/recordings.parquet")
    inner_table = dev[dev.patient.isin(targets.loc[inner_val_mask, "patient"].unique())].reset_index(drop=True)
    inner_dev_x = load_embedding_table(inner_table, embedding_index, memmap)
    best_fixed, fixed_dev_scores = choose_best_fixed(dev, thresholds, seizures, recordings)
    pilot_rows: list[dict[str, object]] = []
    best_model: RouterHead | None = None
    best_cfg: dict[str, object] | None = None
    best_key: tuple[float, float] | None = None
    for lr in (1e-3, 3e-4):
        for hidden in (64, 128):
            for dropout in (0.2, 0.4):
                model = train_head(
                    x_target[inner_train_mask], y_target[inner_train_mask],
                    x_target[inner_val_mask], y_target[inner_val_mask],
                    hidden=hidden, dropout=dropout, lr=lr, seed=3407, updates=args.router_updates,
                )
                with torch.inference_mode():
                    selected_inner = model(torch.from_numpy(inner_dev_x).cuda()).argmax(dim=1).cpu().numpy()
                score = score_selected(inner_table, selected_inner, threshold_array, seizures, recordings)
                row = {"learning_rate": lr, "hidden": hidden, "dropout": dropout, **score}
                pilot_rows.append(row)
                key = (float(score["fa_per_24h"]) if float(score["event_sensitivity"]) >= 0.80 else 1e9 + (0.80 - float(score["event_sensitivity"])) * 1e6, -float(score["event_sensitivity"]))
                if best_key is None or key < best_key:
                    best_key, best_model, best_cfg = key, model, {"learning_rate": lr, "hidden": hidden, "dropout": dropout}
    if best_model is None or best_cfg is None:
        raise RuntimeError("router pilot produced no model")
    pd.DataFrame(pilot_rows).to_csv(router_dir / "router_pilot_validation.csv", index=False)
    torch.save({"state_dict": best_model.state_dict(), "config": best_cfg, "protocol_id": NAMESPACE, "round": args.round}, router_dir / "router_head.pt")
    with torch.inference_mode():
        selected_dev = best_model(torch.from_numpy(x_all_dev).cuda()).argmax(dim=1).cpu().numpy()
        selected_test = best_model(torch.from_numpy(x_all_test).cuda()).argmax(dim=1).cpu().numpy()
    dev_router_score = score_selected(dev, selected_dev, threshold_array, seizures, recordings)
    test_router_score = score_selected(test, selected_test, threshold_array, seizures, recordings)
    dev_oracle_sel, dev_oracle_confidence, dev_oracle_correct = select_correctness_first_oracle(
        dev[[f"prob_{m}" for m in MODELS]].to_numpy(dtype=float),
        dev.label.to_numpy(dtype=float),
        threshold_array,
        unknown_fallback_index=MODELS.index(best_fixed),
    )
    test_oracle_sel, test_oracle_confidence, test_oracle_correct = select_correctness_first_oracle(
        test[[f"prob_{m}" for m in MODELS]].to_numpy(dtype=float),
        test.label.to_numpy(dtype=float),
        threshold_array,
        unknown_fallback_index=MODELS.index(best_fixed),
    )
    dev_oracle_score = score_selected(dev, dev_oracle_sel, threshold_array, seizures, recordings)
    test_oracle_score = score_selected(test, test_oracle_sel, threshold_array, seizures, recordings)
    fixed_test_scores: dict[str, dict[str, float | int]] = {}
    for model in MODELS:
        fixed_test_scores[model] = v3.score_probabilities(test[KEYS + ["label", f"prob_{model}"]].rename(columns={f"prob_{model}": "probability"}), seizures, recordings, thresholds[model])
    best_fixed_index = MODELS.index(best_fixed)
    dev_oracle_confidence_summary = summarize_oracle_confidence(
        dev_oracle_confidence, dev_oracle_correct, dev_oracle_sel, dev.label.to_numpy(dtype=float), best_fixed_index
    )
    test_oracle_confidence_summary = summarize_oracle_confidence(
        test_oracle_confidence, test_oracle_correct, test_oracle_sel, test.label.to_numpy(dtype=float), best_fixed_index
    )
    selection_table = test[KEYS + ["label"]].copy()
    selection_table["router_selected_expert"] = [MODELS[i] for i in selected_test]
    selection_table["oracle_selected_expert"] = [MODELS[i] for i in test_oracle_sel]
    selection_table["router_oracle_agree"] = selected_test == test_oracle_sel
    selection_table["oracle_selected_confidence"] = test_oracle_confidence[np.arange(len(test)), test_oracle_sel]
    selection_table["oracle_had_correct_expert"] = test_oracle_correct.any(axis=1)
    selection_table.to_parquet(router_dir / "router_test_selections.parquet", index=False)
    summary = {
        "protocol_id": NAMESPACE,
        "round": args.round,
        "best_fixed_expert": best_fixed,
        "validation": {"fixed": fixed_dev_scores, "best_fixed": fixed_dev_scores[best_fixed], "oracle": dev_oracle_score, "oracle_confidence": dev_oracle_confidence_summary, "router": dev_router_score},
        "final_test": {"fixed": fixed_test_scores, "best_fixed": fixed_test_scores[best_fixed], "oracle": test_oracle_score, "oracle_confidence": test_oracle_confidence_summary, "router": test_router_score},
        "router_oracle_top1_agreement": float(np.mean(selected_test == test_oracle_sel)),
        "router_selection_proportions": {m: float(np.mean(selected_test == i)) for i, m in enumerate(MODELS)},
        "oracle_selection_proportions": {m: float(np.mean(test_oracle_sel == i)) for i, m in enumerate(MODELS)},
        "oracle_selection_rule": "correctness_first_then_threshold_relative_confidence",
        "confidence_definition": "bounded_threshold_relative_signed_confidence: correct side normalized by its available endpoint distance; wrong side normalized by the opposite endpoint distance",
        "oracle_unknown_label_policy": "exclude_nan_from_correctness_and_confidence; fallback_to_validation_best_fixed; retain_window_for_eventization",
        "oracle_no_correct_expert_fraction": {
            "validation": float(np.mean(~dev_oracle_correct.any(axis=1))),
            "final_test": float(np.mean(~test_oracle_correct.any(axis=1))),
        },
        "router_config": best_cfg,
        "thresholds": thresholds,
        "test_evaluation_count": 1,
        "threshold_source": "expert_train_only",
        "test_used_for_router_selection": False,
        "completed_at_unix": time.time(),
    }
    result_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (router_dir / "router_manifest.json").write_text(json.dumps({"protocol_id": NAMESPACE, "round": args.round, "status": "frozen", "router_test_metrics_sha256": sha256_file(result_path), "pilot_validation_sha256": sha256_file(router_dir / "router_pilot_validation.csv"), "head_sha256": sha256_file(router_dir / "router_head.pt"), "test_evaluation_count": 1, "threshold_source": "expert_train_only", "test_used_for_selection": False}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
