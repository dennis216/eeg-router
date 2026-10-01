"""Compute a label-aware event-selection oracle for the frozen standard H4 outputs.

This is an audit-only upper bound.  For each true seizure in the frozen
outer-test partitions, the oracle may choose the best detector event if at
least one detector caught that seizure, and it abstains on every other event.
It uses outer-test labels, so it is not deployable and must never be used for
threshold/model tuning.  The result is the maximum headroom available to any
future selector at the existing detector thresholds.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/b_false_alarm_atlas")
RUN_ID = "v3-groupkfold-confirmatory-v1"
REPORT_DIR = ROOT / "outputs" / "reports" / RUN_ID
OUT_DIR = ROOT / "outputs" / "reports" / "h4-event-oracle-ceiling-v1"
FEATURE_PATH = REPORT_DIR / "h4_groupkfold_event_features.parquet"
BASELINE_PATH = REPORT_DIR / "h4_groupkfold_router_test.csv"
MODELS = ("singlem", "cbramod", "tcn_gat")
FOLDS = tuple(range(5))
SEEDS = (17, 42, 3407)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_h4_helpers():
    path = ROOT / "scripts" / "84_run_groupkfold_h4_router.py"
    spec = importlib.util.spec_from_file_location("h4_router_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _oracle_select(test: pd.DataFrame, seizures: pd.DataFrame, starts: dict[str, float], h4) -> pd.DataFrame:
    """Select one detector event per truth event, if any detector caught it.

    Matching is performed separately for each detector with the frozen
    event-matching rule.  We then keep the candidate with the smallest onset
    delay for each recording/truth index.  This avoids counting duplicate
    detector alarms as false alarms while retaining one detection per seizure.
    """
    columns = [
        "patient",
        "recording",
        "event_start",
        "event_end",
        "peak_probability",
        "group_id",
        "selected_model",
        "oracle_label",
        "truth_index",
        "onset_delay_s",
    ]
    rows: list[dict[str, object]] = []
    if test.empty:
        return pd.DataFrame(columns=columns)
    candidates: dict[tuple[str, int], list[dict[str, object]]] = {}
    for recording, rec in test.groupby("recording", sort=False):
        recording = str(recording)
        truth = h4._truths(seizures, recording, starts[recording])
        if not truth:
            continue
        for model in MODELS:
            model_rows = rec[rec.model == model]
            predictions = [h4.Event(float(r.start_s), float(r.end_s), float(r.peak_probability)) for r in model_rows.itertuples(index=False)]
            matched = h4.match_events(predictions, truth)
            for pair in matched.pairs:
                row = model_rows.iloc[pair.prediction_index]
                # The frozen event table should agree with the independent
                # match.  Keep the explicit label check for auditability.
                if str(row.event_type) != "tp":
                    continue
                truth_index = int(pair.truth_index)
                delay = max(0.0, float(row.start_s) - float(truth[truth_index][0]))
                candidates.setdefault((recording, truth_index), []).append(
                    {
                        "patient": str(row.patient),
                        "recording": recording,
                        "event_start": float(row.start_s),
                        "event_end": float(row.end_s),
                        "peak_probability": float(row.peak_probability),
                        "group_id": "",
                        "selected_model": str(row.model),
                        "oracle_label": "tp",
                        "truth_index": truth_index,
                        "onset_delay_s": delay,
                    }
                )
    for key, values in sorted(candidates.items()):
        # Earliest valid detection is the most conservative representative;
        # peak probability only breaks exact-delay ties.
        rows.append(sorted(values, key=lambda r: (float(r["onset_delay_s"]), -float(r["peak_probability"]), str(r["selected_model"])))[0])
    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    h4 = _load_h4_helpers()
    features = pd.read_parquet(FEATURE_PATH)
    baseline = pd.read_csv(BASELINE_PATH)
    seizures = pd.read_parquet(ROOT / "manifests" / "seizures.parquet")
    recordings = pd.read_parquet(ROOT / "manifests" / "recordings.parquet")
    durations = recordings.set_index("recording_id")["duration_s"].astype(float).to_dict()
    rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for fold in FOLDS:
        for seed in SEEDS:
            test = features[(features.partition == "test") & (features.split == fold) & (features.seed == seed)].copy()
            starts, patients = h4._evaluation_meta(test)
            candidates = h4._candidate_groups(test)
            selected = _oracle_select(test, seizures, starts, h4)
            score = h4._score_events(selected, starts, patients, seizures, durations)
            base = baseline[(baseline.fold == fold) & (baseline.seed == seed) & (baseline.method == "best_single")]
            if len(base) != 1:
                raise RuntimeError(f"expected one best_single baseline for fold={fold} seed={seed}, got {len(base)}")
            b = base.iloc[0]
            rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "method": "label_aware_event_oracle",
                    "baseline_method": "best_single",
                    "baseline_sensitivity": float(b.sensitivity),
                    "baseline_fa_per_24h": float(b.fa_per_24h),
                    "oracle_sensitivity": score["sensitivity"],
                    "oracle_fa_per_24h": score["fa_per_24h"],
                    "sensitivity_delta_pp": 100.0 * (score["sensitivity"] - float(b.sensitivity)),
                    "fa_absolute_reduction_per_24h": float(b.fa_per_24h) - score["fa_per_24h"],
                    "fa_relative_reduction": (float(b.fa_per_24h) - score["fa_per_24h"]) / float(b.fa_per_24h) if float(b.fa_per_24h) > 0 else math.nan,
                    "oracle_true_positive_events": score["true_positive_events"],
                    "oracle_false_alarm_events": score["false_alarm_events"],
                    "oracle_truth_events": score["truth_events"],
                    "candidate_groups": int(candidates.group_id.nunique()) if not candidates.empty else 0,
                    "selected_truth_events": int(selected.shape[0]),
                    "test_labels_used_for_selection": True,
                }
            )
            for record in selected.to_dict(orient="records"):
                selected_rows.append({"fold": fold, "seed": seed, **record})
            print(f"H4_EVENT_ORACLE_DONE fold={fold} seed={seed} groups={rows[-1]['candidate_groups']} selected={rows[-1]['selected_truth_events']}", flush=True)

    per_run = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(OUT_DIR / "per_run.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(OUT_DIR / "selected_events.csv", index=False)
    method_rows: list[dict[str, object]] = []
    for method, group in baseline.groupby("method", sort=True):
        method_rows.append(
            {
                "method": str(method),
                "n_runs": int(len(group)),
                "macro_mean_sensitivity": float(group.sensitivity.mean()),
                "macro_median_sensitivity": float(group.sensitivity.median()),
                "pooled_sensitivity": float(group.true_positive_events.sum() / group.truth_events.sum()),
                "macro_mean_fa_per_24h": float(group.fa_per_24h.mean()),
                "macro_median_fa_per_24h": float(group.fa_per_24h.median()),
                "pooled_fa_per_24h": float(group.false_alarm_events.sum() * 24.0 / group.nonseizure_hours.sum()),
            }
        )
    method_rows.append(
        {
            "method": "label_aware_event_oracle",
            "n_runs": int(len(per_run)),
            "macro_mean_sensitivity": float(per_run.oracle_sensitivity.mean()),
            "macro_median_sensitivity": float(per_run.oracle_sensitivity.median()),
            "pooled_sensitivity": float(per_run.oracle_true_positive_events.sum() / per_run.oracle_truth_events.sum()),
            "macro_mean_fa_per_24h": float(per_run.oracle_fa_per_24h.mean()),
            "macro_median_fa_per_24h": float(per_run.oracle_fa_per_24h.median()),
            "pooled_fa_per_24h": float(per_run.oracle_false_alarm_events.sum() * 24.0 / baseline[baseline.method == "best_single"].nonseizure_hours.sum()),
        }
    )
    pd.DataFrame(method_rows).to_csv(OUT_DIR / "method_summary.csv", index=False)
    summary = {
        "status": "audit_only_non_deployable",
        "oracle_definition": "for each frozen outer-test truth event, select the earliest valid detector event among SingLEM/CBraMod/TCN-GAT if any detector caught it at its frozen threshold; abstain on all other alarms",
        "comparison_baseline": "best_single selected from validation-only frozen detector thresholds",
        "test_labels_used_for_oracle": True,
        "model_training": False,
        "threshold_selection": False,
        "test_evaluation_count_per_fold_seed": 1,
        "n_runs": int(len(per_run)),
        "mean": {col: float(per_run[col].mean()) for col in ["baseline_sensitivity", "oracle_sensitivity", "sensitivity_delta_pp", "baseline_fa_per_24h", "oracle_fa_per_24h", "fa_absolute_reduction_per_24h", "fa_relative_reduction"]},
        "median": {col: float(per_run[col].median()) for col in ["baseline_sensitivity", "oracle_sensitivity", "sensitivity_delta_pp", "baseline_fa_per_24h", "oracle_fa_per_24h", "fa_absolute_reduction_per_24h", "fa_relative_reduction"]},
        "pooled": {
            "baseline_sensitivity": float(per_run.baseline_sensitivity.mul(per_run.oracle_truth_events).sum() / per_run.oracle_truth_events.sum()),
            "oracle_sensitivity": float(per_run.oracle_true_positive_events.sum() / per_run.oracle_truth_events.sum()),
            "baseline_fa_per_24h": float((baseline[baseline.method == "best_single"].false_alarm_events.sum() * 24.0) / baseline[baseline.method == "best_single"].nonseizure_hours.sum()),
            "oracle_fa_per_24h": float((per_run.oracle_false_alarm_events.sum() * 24.0) / baseline[baseline.method == "best_single"].nonseizure_hours.sum()),
        },
        "runs_with_fa_reduction": int((per_run.fa_absolute_reduction_per_24h > 0).sum()),
        "runs_with_no_sensitivity_loss": int((per_run.sensitivity_delta_pp >= 0).sum()),
        "inputs": {"features_sha256": _sha256(FEATURE_PATH), "baseline_sha256": _sha256(BASELINE_PATH)},
        "outputs": {"per_run": "per_run.csv", "selected_events": "selected_events.csv", "method_summary": "method_summary.csv"},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
