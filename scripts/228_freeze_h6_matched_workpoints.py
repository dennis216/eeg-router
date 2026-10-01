"""Select Train-only matched-sensitivity and matched-FA expert workpoints.

The common targets are anchored to the frozen CBraMod operating point on
TUSZ Train.  No Dev or Eval outcomes are read for threshold selection.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bfa.evaluation.eventize import eventize
from bfa.evaluation.match import match_events


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
NAMESPACE = "h6-eeg-conditionable-competence-probe-v1"
OUT = ROOT / "outputs/reports" / NAMESPACE / "matched_workpoints"
MANIFEST = ROOT / "outputs/reports/h6-tusz-raw-pre-router-v1/manifests/train_manifest.json"
EXPERT_ROOT = ROOT / "outputs/tuh_external/h6_tusz_raw_pre_router_v1/train"
MODELS = ("singlem", "cbramod", "tcn_gat")
REFERENCE_MODEL = "cbramod"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def truths(record: dict[str, object]) -> list[tuple[float, float]]:
    return [
        (max(70.0, float(event["start_s"])), float(event["end_s"]))
        for event in record.get("seizure_events", [])
        if float(event["end_s"]) > 70.0
    ]


def load_model(model: str, records: list[dict[str, object]]):
    loaded = []
    probabilities = []
    frozen_threshold = None
    for index, record in enumerate(records, start=1):
        recording = str(record["relative_edf"])
        base = EXPERT_ROOT / model / "fold0_seed17"
        probability_path = base / "probabilities" / Path(recording).with_suffix(".parquet")
        metadata_path = base / "metadata" / Path(recording).with_suffix(".json")
        table = pd.read_parquet(probability_path, columns=["window_end_s", "probability"]).sort_values("window_end_s")
        metadata = json.loads(metadata_path.read_text())
        threshold = float(metadata["threshold"])
        if frozen_threshold is None:
            frozen_threshold = threshold
        elif not np.isclose(frozen_threshold, threshold):
            raise RuntimeError(f"non-constant frozen threshold for {model}")
        times = table.window_end_s.to_numpy(dtype=float)
        values = table.probability.to_numpy(dtype=float)
        loaded.append((times, values, truths(record), float(record["duration_s"])))
        probabilities.append(values)
        if index % 1000 == 0:
            print(f"{model}: loaded {index}/{len(records)}", flush=True)
    return loaded, np.concatenate(probabilities), float(frozen_threshold)


def score(loaded, threshold: float) -> dict[str, float | int]:
    tp = fp = truth_count = 0
    nonseizure_s = 0.0
    delay_sum = 0.0
    delay_count = 0
    for times, probabilities, truth, duration in loaded:
        predictions = eventize(times, probabilities, threshold=threshold) if len(times) >= 2 else []
        matched = match_events(predictions, truth)
        truth_count += len(truth)
        tp += len(matched.pairs)
        fp += len(matched.unmatched_predictions)
        seizure_s = sum(max(0.0, end - start) for start, end in truth)
        nonseizure_s += max(0.0, duration - 70.0 - seizure_s)
        for pair in matched.pairs:
            delay_sum += max(0.0, predictions[pair.prediction_index].start_s - truth[pair.truth_index][0])
            delay_count += 1
    return {
        "threshold": float(threshold),
        "truth_events": int(truth_count),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(truth_count - tp),
        "sensitivity": float(tp / truth_count) if truth_count else float("nan"),
        "fa_per_24h": float(fp / (nonseizure_s / 86400.0)) if nonseizure_s else float("nan"),
        "mean_delay_s": float(delay_sum / delay_count) if delay_count else float("nan"),
    }


def threshold_grid(values: np.ndarray, frozen: float) -> np.ndarray:
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, 61))
    extra = np.asarray([0.0, 0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.99, frozen, 1.000001])
    return np.unique(np.clip(np.concatenate([quantiles, extra]), 0.0, 1.000001))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    output_manifest = OUT / "manifest.json"
    if output_manifest.exists():
        raise RuntimeError(f"refusing to overwrite frozen workpoints: {output_manifest}")
    cohort = json.loads(MANIFEST.read_text())
    records = list(cohort["records"])
    curves = []
    frozen_rows = {}
    for model in MODELS:
        loaded, values, frozen = load_model(model, records)
        grid = threshold_grid(values, frozen)
        model_rows = []
        for index, threshold in enumerate(grid, start=1):
            row = {"model": model, **score(loaded, float(threshold)), "is_frozen_threshold": bool(np.isclose(threshold, frozen))}
            model_rows.append(row)
            if index % 20 == 0:
                print(f"{model}: scored {index}/{len(grid)} thresholds", flush=True)
        curve = pd.DataFrame(model_rows)
        if not curve.is_frozen_threshold.any():
            raise RuntimeError(f"frozen threshold missing for {model}")
        frozen_rows[model] = curve[curve.is_frozen_threshold].iloc[0].to_dict()
        curves.append(curve)

    curve_frame = pd.concat(curves, ignore_index=True)
    reference = frozen_rows[REFERENCE_MODEL]
    target_sensitivity = float(reference["sensitivity"])
    target_fa = float(reference["fa_per_24h"])
    selections: dict[str, dict[str, dict[str, float]]] = {
        "frozen": {}, "matched_sensitivity": {}, "matched_fa": {}
    }
    for model in MODELS:
        model_curve = curve_frame[curve_frame.model.eq(model)].copy()
        frozen_row = model_curve[model_curve.is_frozen_threshold].iloc[0]
        sensitivity_row = model_curve.assign(distance=(model_curve.sensitivity - target_sensitivity).abs()).sort_values(
            ["distance", "fa_per_24h", "threshold"], ascending=[True, True, False]
        ).iloc[0]
        fa_row = model_curve.assign(distance=(model_curve.fa_per_24h - target_fa).abs()).sort_values(
            ["distance", "sensitivity", "threshold"], ascending=[True, False, False]
        ).iloc[0]
        for condition, row in (("frozen", frozen_row), ("matched_sensitivity", sensitivity_row), ("matched_fa", fa_row)):
            selections[condition][model] = {
                key: float(row[key]) for key in ("threshold", "sensitivity", "fa_per_24h", "mean_delay_s")
            }

    curve_path = OUT / "train_threshold_curves.csv"
    workpoint_path = OUT / "workpoints.json"
    curve_frame.to_csv(curve_path, index=False)
    workpoints = {
        "release_id": NAMESPACE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_partition": "TUSZ Train only",
        "eval_read": False,
        "reference_model": REFERENCE_MODEL,
        "targets": {"sensitivity": target_sensitivity, "fa_per_24h": target_fa},
        "selection_rules": {
            "matched_sensitivity": "minimum absolute sensitivity distance; tie lower FA/24h then higher threshold",
            "matched_fa": "minimum absolute FA/24h distance; tie higher sensitivity then higher threshold",
        },
        "conditions": selections,
    }
    workpoint_path.write_text(json.dumps(workpoints, indent=2, sort_keys=True) + "\n")
    manifest = {
        "release_id": NAMESPACE,
        "stage": "train_only_matched_workpoint_selection",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "eval_read": False,
        "dev_used_for_selection": False,
        "source_manifest_sha256": sha256(MANIFEST),
        "threshold_curve_sha256": sha256(curve_path),
        "workpoints_sha256": sha256(workpoint_path),
        "workpoints": workpoints,
    }
    output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (OUT / "manifest.sha256").write_text(f"{sha256(output_manifest)}  manifest.json\n")
    print(json.dumps(workpoints, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
