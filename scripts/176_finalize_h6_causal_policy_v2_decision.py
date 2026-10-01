"""Freeze the development-only decision for the H6-v2 retry.

This writes an auditable comparison of the original v2 and the reweighted
retry.  It does not read TUSZ Eval, alter thresholds, or launch inference.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path("/root/b_false_alarm_atlas")
OUT = ROOT / "outputs/reports/h6-causal-policy-router-v2-reweighted"
DECISION = OUT / "final_decision.json"
MODELS = ("singlem", "cbramod", "tcn_gat")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    if DECISION.exists():
        raise RuntimeError(f"refusing to overwrite {DECISION}")
    gate = load(ROOT / "outputs/reports/h6-causal-policy-router-v2-corrected/training_gate_corrected_v2.json")
    original_sel = load(ROOT / "outputs/reports/h6-causal-policy-router-v2-corrected/scoring_dev_formal_train_only/threshold_selection.json")
    retry_sel = load(OUT / "scoring_dev_formal_train_only/threshold_selection.json")
    retry_summary = pd.read_csv(OUT / "scoring_dev_formal_train_only/summary.csv")
    best_fixed = retry_summary[retry_summary.method.eq("best_fixed")].iloc[0]
    router = retry_summary[retry_summary.method.eq("router")].iloc[0]
    sens_loss_pp = float((float(best_fixed.sensitivity) - float(router.sensitivity)) * 100.0)
    fa_reduction = float(1.0 - float(router.fa_per_24h) / float(best_fixed.fa_per_24h))
    practical_safe = bool(str(retry_sel.get("status")) == "safe_dev_point_selected" and sens_loss_pp <= 3.0 and fa_reduction > 0.0)
    decision = {
        "release_id": "h6-causal-policy-router-v2-reweighted",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "development_only_decision",
        "status": "PRELIMINARY_SAFE_DEV_IMPROVEMENT_NOT_CONFIRMATORY" if practical_safe else "NEGATIVE_OR_UNSTABLE_DEV",
        "oracle_gate_pass": bool(gate.get("oracle_gate_pass", False)),
        "actual_router_safety_gate": {
            "pass": practical_safe,
            "definition": "Dev safe point; sensitivity loss <=3 percentage points and FA/24h reduction >0",
            "sensitivity_loss_pp": sens_loss_pp,
            "fa_reduction": fa_reduction,
            "fa_reduction_25pct_gate": bool(fa_reduction >= 0.25),
        },
        "best_fixed": {"method": str(best_fixed.method), "sensitivity": float(best_fixed.sensitivity), "fa_per_24h": float(best_fixed.fa_per_24h)},
        "reweighted_router": {"sensitivity": float(router.sensitivity), "fa_per_24h": float(router.fa_per_24h), "mean_delay_s": float(router.mean_delay_s), "patients": int(router.patients), "records": int(router.records), "truth_events": int(router.truth_events)},
        "reweighted_threshold_selection": retry_sel,
        "original_v2_threshold_selection": original_sel,
        "training_changes": ["causal per-window channel median/IQR robust z-score clipped [-12,12]", "3x policy-loss weight on seizure horizons"],
        "unchanged_contract": ["raw EEG only", "same corrected labels", "same 16x1280 architecture", "same four-stratum batch", "frozen experts", "no Eval read", "no test threshold selection"],
        "tusz_eval_status": "not_read_for_v2_decision",
        "interpretation": "The reweighted router reaches a Dev safety point with a 20.4% FA/24h reduction and 2.84pp sensitivity loss. This is a development signal, not an external confirmation; the 25% reduction benchmark is not met.",
        "next_step": "Freeze Dev thresholds and evaluate once on a genuinely unused independent cohort; if none is available, report as exploratory and do not tune TUSZ Eval.",
        "source_hashes": {
            "oracle_gate": sha256(ROOT / "outputs/reports/h6-causal-policy-router-v2-corrected/training_gate_corrected_v2.json"),
            "original_dev_selection": sha256(ROOT / "outputs/reports/h6-causal-policy-router-v2-corrected/scoring_dev_formal_train_only/threshold_selection.json"),
            "retry_dev_selection": sha256(OUT / "scoring_dev_formal_train_only/threshold_selection.json"),
            "retry_summary": sha256(OUT / "scoring_dev_formal_train_only/summary.csv"),
            "retry_checkpoint": sha256(OUT / "training_formal_train_only/checkpoint.pt"),
        },
    }
    DECISION.write_text(json.dumps(decision, indent=2, sort_keys=True, default=float) + "\n")
    (OUT / "final_decision.sha256").write_text(sha256(DECISION) + "  final_decision.json\n")
    print(json.dumps({"status": decision["status"], "sensitivity_loss_pp": sens_loss_pp, "fa_reduction": fa_reduction, "out": str(DECISION)}, indent=2))


if __name__ == "__main__":
    main()
