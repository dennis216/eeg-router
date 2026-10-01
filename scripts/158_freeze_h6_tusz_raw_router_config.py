"""Freeze the H6 raw-router architecture and label/inference contract."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/b_false_alarm_atlas")
OUT = ROOT / "outputs/reports/h6-tusz-raw-pre-router-v1"
CONFIG = OUT / "raw_router_config.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if CONFIG.exists():
        raise RuntimeError(f"refusing to overwrite frozen config: {CONFIG}")
    payload = {
        "release_id": "h6-tusz-raw-pre-router-v1",
        "status": "frozen_before_formal_training",
        "dataset": "TUSZ_v2.0.6",
        "partitions": {"train": "official TUSZ Train", "dev": "official TUSZ Dev", "eval": "official TUSZ Eval"},
        "expert_selection": {"models": ["singlem", "cbramod", "tcn_gat"], "checkpoint_run": "v3-groupkfold-confirmatory-v1/split0_seed17_main", "checkpoint_update": 5000, "expert_retraining": False, "threshold_source": "frozen CHB validation-only threshold"},
        "input": {"description": "past 60 seconds canonical bipolar raw EEG", "channels": 16, "sampling_hz": 128, "shape": [16, 7680], "preprocessing": "canonical bipolar microvolts; polyphase resample only; no expert output, probability, threshold, model ID, filter, clipping, or scaling"},
        "target": {"future_horizon_seconds": 60, "stride_seconds": 60, "multi_label_expert_competence": True, "simultaneous_correct_priority": "0.3 uniform + 0.7 softmax((calibrated_confidence - 0.01*delay)/tau), tau=0.25", "unroutable": "seizure horizon with no correct expert", "reject": "background horizon in which all three experts emit at least one FP event"},
        "model": {"backbone": "Conv1d(16,32,k15,s4)+BN+ReLU -> Conv1d(32,64,k9,s4)+BN+ReLU -> Conv1d(64,128,k7,s4)+BN+ReLU -> GlobalAvgPool", "heads": {"expert_competence": "Linear(128,3) independent sigmoid outputs", "reject": "Linear(128,1) independent sigmoid", "future_seizure_risk": "Linear(128,1) independent sigmoid"}},
        "loss": {"competence": "balanced per-expert BCE", "priority_cross_entropy_weight": 0.05, "reject_bce_weight": 0.2, "future_seizure_risk_bce_weight": 0.1, "reject_oversampling_max": 5, "expert_output_in_router_input": False},
        "inference": {"default": "select max competence", "reject_allowed_if": ["reject_probability >= tau_reject", "all expert competence < tau_expert", "future seizure risk < tau_seizure"], "fallback": "max competence expert", "threshold_selection": "Dev only; sensitivity loss <=3 percentage points versus best fixed expert, then minimize FA/24h", "eval_threshold_retuning": False},
        "evaluation": {"metrics": ["sensitivity", "FA/24h", "mean detection delay"], "comparators": ["each frozen expert", "best fixed expert", "equal event ensemble", "router without reject", "router with reject", "interval oracle"], "bootstrap": {"unit": "patient", "replicates": 2000, "seed": 20260829}, "eval_evaluation_count_per_method": 1},
        "selection_used_model_output": False,
        "selection_used_eval_performance": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    CONFIG.with_suffix(".sha256").write_text(sha256(CONFIG) + "\n")
    print(json.dumps({"config": str(CONFIG), "sha256": sha256(CONFIG), "status": payload["status"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
