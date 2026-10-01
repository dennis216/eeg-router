from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def sha256_ok(checkpoint: Path) -> bool:
    import hashlib

    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return digest == checkpoint.with_suffix(checkpoint.suffix + ".sha256").read_text().strip()


def main() -> None:
    run_root = ROOT / "runs/BSN-pilot-v1"
    rows = []
    curves = []
    for model in ("singlem", "cbramod", "tcn_gat"):
        model_root = run_root / model / "split17_seed17_main"
        completed = json.loads((model_root / "completed.json").read_text())
        validation = completed["final_validation"]
        test = completed["test"]["score"]
        checkpoint = model_root / "checkpoints/step_05000.pt"
        rows.append(
            {
                "model": model,
                "optimizer_updates": completed["optimizer_updates"],
                "validation_window_auroc": validation["window_auroc"],
                "validation_average_precision": validation["window_average_precision"],
                "validation_threshold": validation["selected_event_operating_point"]["threshold"],
                "validation_event_sensitivity": validation["selected_event_operating_point"]["event_sensitivity"],
                "validation_fa_per_24h": validation["selected_event_operating_point"]["fa_per_24h"],
                "test_event_sensitivity": test["event_sensitivity"],
                "test_fa_per_24h": test["fa_per_24h"],
                "test_true_positive_events": test["true_positive_events"],
                "test_truth_events": test["truth_events"],
                "test_false_alarm_events": test["false_alarm_events"],
                "test_evaluation_count": completed["test"]["test_evaluation_count"],
                "run_wall_hours": completed["training_and_validation_wall_seconds"] / 3600,
                "final_checkpoint_sha256_ok": sha256_ok(checkpoint),
            }
        )
        for path in sorted((model_root / "validation").glob("metrics_step_*.json")):
            metrics = json.loads(path.read_text())
            curves.append(
                {
                    "model": model,
                    "update": metrics["update"],
                    "window_auroc": metrics["window_auroc"],
                    "average_precision": metrics["window_average_precision"],
                    "selected_fa_per_24h": metrics["selected_event_operating_point"]["fa_per_24h"],
                }
            )
    summary = pd.DataFrame(rows)
    curve_table = pd.DataFrame(curves)
    output = ROOT / "outputs/reports/BSN-pilot-v1"
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "model_summary.csv", index=False)
    curve_table.to_csv(output / "validation_curves.csv", index=False)
    integrity = {
        "models_completed": int(len(summary)),
        "all_optimizer_updates_5000": bool((summary.optimizer_updates == 5000).all()),
        "all_test_evaluation_count_one": bool((summary.test_evaluation_count == 1).all()),
        "all_final_checkpoint_hashes_ok": bool(summary.final_checkpoint_sha256_ok.all()),
    }
    (output / "integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(curve_table.to_string(index=False))
    print(json.dumps(integrity, sort_keys=True))


if __name__ == "__main__":
    main()
