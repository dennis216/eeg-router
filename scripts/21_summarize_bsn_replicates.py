"""Create the nine-run BSN pilot replication summary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MODELS = ("singlem", "cbramod", "tcn_gat")
SEEDS = (17, 42, 3407)


def _hash_ok(path: Path) -> bool:
    return hashlib.sha256(path.read_bytes()).hexdigest() == path.with_suffix(path.suffix + ".sha256").read_text().strip()


def main() -> None:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for seed in SEEDS:
            run = ROOT / "runs" / "BSN-pilot-v1" / model / f"split17_seed{seed}_main"
            completed = json.loads((run / "completed.json").read_text())
            score = completed["test"]["score"]
            validation = completed["final_validation"]["selected_event_operating_point"]
            checkpoint = run / "checkpoints/step_05000.pt"
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "optimizer_updates": completed["optimizer_updates"],
                    "validation_threshold": validation["threshold"],
                    "test_event_sensitivity": score["event_sensitivity"],
                    "test_fa_per_24h": score["fa_per_24h"],
                    "test_true_positive_events": score["true_positive_events"],
                    "test_truth_events": score["truth_events"],
                    "test_false_alarm_events": score["false_alarm_events"],
                    "test_evaluation_count": completed["test"]["test_evaluation_count"],
                    "run_wall_hours": completed["training_and_validation_wall_seconds"] / 3600,
                    "final_checkpoint_sha256_ok": _hash_ok(checkpoint),
                }
            )
    summary = pd.DataFrame(rows)
    aggregate = (
        summary.groupby("model", as_index=False)
        .agg(
            seeds=("seed", "count"),
            mean_test_event_sensitivity=("test_event_sensitivity", "mean"),
            sd_test_event_sensitivity=("test_event_sensitivity", "std"),
            mean_test_fa_per_24h=("test_fa_per_24h", "mean"),
            sd_test_fa_per_24h=("test_fa_per_24h", "std"),
            mean_run_wall_hours=("run_wall_hours", "mean"),
        )
    )
    output = ROOT / "outputs/reports/BSN-pilot-v1"
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "model_summary_3seeds.csv", index=False)
    aggregate.to_csv(output / "model_summary_3seeds_aggregate.csv", index=False)
    integrity = {
        "completed_runs": int(len(summary)),
        "expected_runs": 9,
        "all_updates_5000": bool((summary.optimizer_updates == 5000).all()),
        "all_test_evaluation_count_one": bool((summary.test_evaluation_count == 1).all()),
        "all_checkpoint_sha256_ok": bool(summary.final_checkpoint_sha256_ok.all()),
        "all_models_have_three_seeds": bool((summary.groupby("model").size() == 3).all()),
    }
    (output / "replication_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(aggregate.to_string(index=False))
    print(json.dumps(integrity, sort_keys=True))


if __name__ == "__main__":
    main()
