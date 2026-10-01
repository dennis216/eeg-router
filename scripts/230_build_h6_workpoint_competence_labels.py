"""Build same-window competence labels at frozen non-degenerate workpoints.

The source EEG10 table is reused, but expert decisions are recomputed from the
same causal EMA probabilities used by event scoring.  This avoids defining
window correctness from an un-smoothed endpoint while selecting thresholds on
smoothed event streams.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bfa.evaluation.eventize import causal_ema


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
NAMESPACE = "h6-eeg-conditionable-competence-probe-v1"
SOURCE = ROOT / "outputs/reports" / NAMESPACE / "same_window"
WORKPOINT = ROOT / "outputs/reports" / NAMESPACE / "matched_workpoints_nondegenerate_v2/workpoints.json"
EXPERT_ROOT = ROOT / "outputs/tuh_external/h6_tusz_raw_pre_router_v1"
OUT = ROOT / "outputs/reports" / NAMESPACE / "same_window_conditions_v2"
MODELS = ("singlem", "cbramod", "tcn_gat")
CONDITIONS = ("frozen", "matched_sensitivity", "matched_fa")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def smoothed_probability_at_times(path: Path, requested: np.ndarray) -> np.ndarray:
    table = pd.read_parquet(path, columns=["window_end_s", "probability"]).sort_values("window_end_s")
    times = table.window_end_s.to_numpy(dtype=float)
    smooth = causal_ema(table.probability.to_numpy(dtype=float))
    indices = np.searchsorted(times, requested)
    indices = np.clip(indices, 0, len(times) - 1)
    exact = np.isclose(times[indices], requested, atol=1e-6)
    if not exact.all():
        bad = requested[~exact][:5].tolist()
        raise RuntimeError(f"decision times missing from probability stream {path}: {bad}")
    return smooth[indices]


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(frame)),
        "patients": int(frame.patient_id.nunique()),
        "recordings": int(frame.recording.nunique()),
        "seizure_rows": int(frame.current_has_seizure.sum()),
        "mask_counts": {str(k): int(v) for k, v in frame.correct_mask.value_counts().sort_index().items()},
        "correct_prevalence": {model: float(frame[f"correct_{model}"].mean()) for model in MODELS},
        "prediction_prevalence": {model: float(frame[f"prediction_{model}"].mean()) for model in MODELS},
        "actionable_rows": int(frame.actionable.sum()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite frozen labels: {manifest_path}")
    workpoints = json.loads(WORKPOINT.read_text())
    outputs: dict[str, str] = {}
    summaries: dict[str, dict[str, object]] = {}
    for partition in ("train", "dev"):
        source_path = SOURCE / f"{partition}_same_window_competence_eeg10.parquet"
        base = pd.read_parquet(source_path)
        smooth_columns = {model: np.empty(len(base), dtype=np.float32) for model in MODELS}
        for recording_index, (recording, rows) in enumerate(base.groupby("recording", sort=False), start=1):
            requested = rows.decision_time_s.to_numpy(dtype=float)
            for model in MODELS:
                path = (
                    EXPERT_ROOT / partition / model / "fold0_seed17" / "probabilities"
                    / Path(str(recording)).with_suffix(".parquet")
                )
                smooth_columns[model][rows.index.to_numpy()] = smoothed_probability_at_times(path, requested)
            if recording_index % 500 == 0:
                print(f"{partition}: EMA aligned {recording_index}/{base.recording.nunique()} recordings", flush=True)
        for model in MODELS:
            base[f"ema_probability_{model}"] = smooth_columns[model]

        truth = base.current_has_seizure.to_numpy(dtype=np.int8)
        for condition in CONDITIONS:
            frame = base.copy()
            for model in MODELS:
                threshold = float(workpoints["conditions"][condition][model]["threshold"])
                prediction = (frame[f"ema_probability_{model}"].to_numpy() >= threshold).astype(np.int8)
                frame[f"threshold_{model}"] = threshold
                frame[f"prediction_{model}"] = prediction
                frame[f"correct_{model}"] = (prediction == truth).astype(np.int8)
            correctness = frame[[f"correct_{model}" for model in MODELS]].astype(int)
            frame["correct_mask"] = correctness.astype(str).agg("".join, axis=1)
            frame["correct_count"] = correctness.sum(axis=1).astype(np.int8)
            frame["actionable"] = frame.correct_count.between(1, 2)
            frame["all_correct"] = frame.correct_count.eq(3)
            frame["all_wrong"] = frame.correct_count.eq(0)
            frame["workpoint_condition"] = condition
            frame["label_contract"] = "same_completed_10s_window_causal_ema"
            output = OUT / f"{partition}_{condition}.parquet"
            frame.to_parquet(output, index=False)
            key = str(output.relative_to(ROOT))
            outputs[key] = sha256(output)
            summaries[f"{partition}:{condition}"] = summarize(frame)

    manifest = {
        "release_id": NAMESPACE,
        "stage": "same_window_causal_ema_competence_labels_v2",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_training": False,
        "expert_training": False,
        "threshold_selection_partition": "TUSZ Train only",
        "dev_used_for_threshold_selection": False,
        "eval_read": False,
        "source_same_window_manifest_sha256": sha256(SOURCE / "manifest.json"),
        "workpoint_sha256": sha256(WORKPOINT),
        "conditions": list(CONDITIONS),
        "summaries": summaries,
        "output_hashes": outputs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (OUT / "manifest.sha256").write_text(f"{sha256(manifest_path)}  manifest.json\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
