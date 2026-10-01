"""Development-only test inference for validation-selected saved checkpoints.

The first pass is restricted to seed 3407 across five patient-held-out folds.
Checkpoint and threshold selection use validation only.  No training occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from bfa.evaluation.eventize import eventize


ROOT = Path("/root/b_false_alarm_atlas")
SOURCE_ID = "v3-groupkfold-confirmatory-v1"
ANALYSIS = ROOT / "outputs" / "reports" / "h4-router-revision-v1"
OUT = ANALYSIS / "selected_checkpoint_seed3407"
MODELS = ("singlem", "cbramod", "tcn_gat")
FOLDS = range(5)
SEED = 3407


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def events_from_probabilities(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for recording, group in frame.groupby("recording", sort=False):
        ordered = group.sort_values("end", kind="stable")
        for event in eventize(
            ordered.end.to_numpy(float),
            ordered.probability.to_numpy(float),
            threshold=float(threshold),
        ):
            rows.append(
                {
                    "patient": str(ordered.patient.iloc[0]),
                    "recording": str(recording),
                    "event_start": float(event.start_s),
                    "event_end": float(event.end_s),
                    "peak_probability": float(event.peak_probability),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["patient", "recording", "event_start", "event_end", "peak_probability"],
    )


def select_threshold(
    probability_path: Path,
    sweep_path: Path,
    fold: int,
    revised,
    recordings: pd.DataFrame,
    seizures: pd.DataFrame,
    durations: dict[str, float],
) -> tuple[float, dict[str, float]]:
    probabilities = pd.read_parquet(probability_path)
    starts, patients = revised.partition_metadata(fold, "validation", recordings, probabilities)
    frozen = pd.read_parquet(sweep_path)
    max_sensitivity = float(frozen.event_sensitivity.max())
    thresholds = frozen.loc[
        frozen.event_sensitivity >= max_sensitivity - 0.03, "threshold"
    ].to_numpy(float)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        events = events_from_probabilities(probabilities, float(threshold))
        score = revised.score_events(events, starts, patients, seizures, durations)
        rows.append({"threshold": float(threshold), **score})
    candidates = pd.DataFrame(rows)
    corrected_max = float(candidates.sensitivity.max())
    eligible = candidates[candidates.sensitivity >= corrected_max - 0.03].copy()
    selected = eligible.sort_values(
        ["false_alarm_minutes_per_24h", "fa_per_24h", "mean_delay_s", "threshold"],
        ascending=[True, True, True, False],
        na_position="last",
        kind="stable",
    ).iloc[0]
    return float(selected.threshold), selected.drop(labels=["threshold"]).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("BFA_ROOT", str(ROOT))
    os.environ.setdefault(
        "BFA_CONFIG", str(ROOT / "configs" / "standard" / "v3_groupkfold_confirmatory_v1.yaml")
    )
    runner = load_module(
        Path("/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/work/b_false_alarm_atlas/scripts/11_run_bsn_pilot.py"),
        "h4_checkpoint_runner",
    )
    revised = load_module(ROOT / "scripts" / "251_run_h4_router_revision_diagnostics.py", "h4_revised")
    config = yaml.safe_load(runner.CONFIG_PATH.read_text())
    windows = pd.read_parquet(ROOT / config["cohort"]["window_manifest"])
    recordings = pd.read_parquet(ROOT / "manifests" / "recordings.parquet")
    seizures = pd.read_parquet(ROOT / "manifests" / "seizures.parquet")
    durations = recordings.set_index("recording_id").duration_s.astype(float).to_dict()
    selected = pd.read_csv(ANALYSIS / "validation_checkpoint_selection.csv")
    locks: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []

    for fold in FOLDS:
        split = json.loads((ROOT / "manifests" / "groupkfold_cv_v1" / f"fold_{fold}.json").read_text())
        test_index = windows[
            windows.patient.isin(split["test"]) & ~windows.warmup
        ].reset_index(drop=True)
        for model in MODELS:
            choice = selected[
                (selected.model == model) & (selected.fold == fold) & (selected.seed == SEED)
            ].iloc[0]
            step = int(choice.selected_step_by_validation_ap)
            run = ROOT / "runs" / SOURCE_ID / model / f"split{fold}_seed{SEED}_main"
            checkpoint = run / "checkpoints" / f"step_{step:05d}.pt"
            validation_probability = run / "validation" / f"probabilities_step_{step:05d}.parquet"
            validation_sweep = run / "validation" / f"threshold_sweep_step_{step:05d}.parquet"
            threshold, validation_score = select_threshold(
                validation_probability,
                validation_sweep,
                fold,
                revised,
                recordings,
                seizures,
                durations,
            )
            locks.append(
                {"fold": fold, "seed": SEED, "model": model, "step": step, "threshold": threshold, **validation_score}
            )
            output = OUT / f"fold{fold}_{model}_seed{SEED}_step{step:05d}.parquet"
            meta_path = output.with_suffix(".json")
            checkpoint_hash = sha256(checkpoint)
            if output.exists() and meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if meta.get("checkpoint_sha256") != checkpoint_hash:
                    raise RuntimeError(f"existing output checkpoint mismatch: {output}")
                table = pd.read_parquet(output)
                print(f"SELECTED_CHECKPOINT_TEST_EXISTS fold={fold} model={model} step={step}", flush=True)
            else:
                print(f"SELECTED_CHECKPOINT_TEST_START fold={fold} model={model} step={step}", flush=True)
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                encoder = runner._model(model).cuda()
                head = runner.SharedContextHead().cuda()
                encoder.load_state_dict(state["encoder"], strict=False)
                head.load_state_dict(state["head"], strict=True)
                encoder.eval()
                head.eval()
                probabilities = runner._predict(
                    encoder,
                    head,
                    test_index,
                    model,
                    batch_size=64,
                    workers=max(0, int(args.workers)),
                )
                table = runner._probability_table(test_index, probabilities)
                table.to_parquet(output, index=False)
                meta_path.write_text(
                    json.dumps(
                        {
                            "fold": fold,
                            "seed": SEED,
                            "model": model,
                            "step": step,
                            "checkpoint_sha256": checkpoint_hash,
                            "probability_sha256": sha256(output),
                            "rows": len(table),
                            "threshold_source": "validation_only_alarm_time_lock",
                            "test_evaluation_count": 1,
                            "development_reanalysis": True,
                        },
                        indent=2,
                        sort_keys=True,
                    ) + "\n"
                )
                del encoder, head, state, probabilities
                torch.cuda.empty_cache()
                print(f"SELECTED_CHECKPOINT_TEST_DONE fold={fold} model={model} step={step}", flush=True)
            starts, patients = revised.partition_metadata(fold, "test", recordings, table)
            events = events_from_probabilities(table, threshold)
            score = revised.score_events(events, starts, patients, seizures, durations)
            test_rows.append(
                {"fold": fold, "seed": SEED, "method": model, "selected_model": model, "step": step, "threshold": threshold, **score}
            )

    locks_frame = pd.DataFrame(locks)
    tests = pd.DataFrame(test_rows)
    locks_frame.to_csv(OUT / "validation_locks.csv", index=False)
    tests.to_csv(OUT / "test_metrics.csv", index=False)
    best_rows: list[dict[str, object]] = []
    for fold in FOLDS:
        validation_fold = locks_frame[locks_frame.fold == fold]
        target = float(validation_fold.sensitivity.max())
        eligible = validation_fold[validation_fold.sensitivity >= target - 0.03]
        best = eligible.sort_values(
            ["false_alarm_minutes_per_24h", "fa_per_24h", "mean_delay_s"],
            kind="stable",
        ).iloc[0]
        result = tests[(tests.fold == fold) & (tests.method == best.model)].iloc[0].to_dict()
        result["method"] = "best_single_selected_checkpoint"
        result["selected_model"] = str(best.model)
        best_rows.append(result)
    combined = pd.concat([tests, pd.DataFrame(best_rows)], ignore_index=True)
    combined.to_csv(OUT / "test_metrics_with_best_single.csv", index=False)
    summary = combined.groupby("method", as_index=False).agg(
        runs=("fold", "size"),
        mean_sensitivity=("sensitivity", "mean"),
        mean_fa_per_24h=("fa_per_24h", "mean"),
        mean_false_alarm_minutes_per_24h=("false_alarm_minutes_per_24h", "mean"),
        mean_alarm_minutes_per_24h=("alarm_minutes_per_24h", "mean"),
        mean_delay_s=("mean_delay_s", "mean"),
    )
    summary.to_csv(OUT / "summary.csv", index=False)
    manifest = {
        "analysis_id": "h4-selected-checkpoint-seed3407-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_development_only" if len(tests) == 15 else "incomplete",
        "training_performed": False,
        "folds": 5,
        "seed": SEED,
        "test_used_for_selection": False,
        "checkpoint_source": "validation AUPRC",
        "threshold_source": "validation sensitivity gate then false-alarm minutes/24h",
        "test_evaluation_count_per_model_fold": 1,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
