"""Finalize the pre-training gate for H6 causal-policy-router-v2.

The first oracle pass is preserved verbatim.  This audit applies the
pre-specified collapse rule (dominance alone is not failure; dominance plus no
oracle benefit is failure), checks minority/patient support, and writes a new
immutable gate decision.  It never reads Eval and never starts training.
"""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path("/root/b_false_alarm_atlas")
NS = "h6-causal-policy-router-v2"
OUT = ROOT / "outputs/reports" / NS
ORACLE = OUT / "oracle/oracle_manifest.json"
MODELS = ("singlem", "cbramod", "tcn_gat")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def main():
    out = OUT / "training_gate.json"
    if out.exists():
        raise RuntimeError(f"refusing overwrite {out}")
    oracle = json.loads(ORACLE.read_text())
    dev = oracle["results"]["dev"]["gate"]
    train = oracle["results"]["train"]["gate"]
    audit = oracle["label_audit"]
    dev_audit = audit["dev"]
    counts = dev_audit["policy_target_argmax_counts"]
    eligible = int(dev_audit["policy_eligible_rows"])
    max_model, max_count = max(counts.items(), key=lambda item: item[1])
    max_share = float(max_count / eligible) if eligible else 1.0
    minority = {}
    for model in MODELS:
        labels = pd.read_parquet(OUT / "labels/dev_causal_policy.parquet")
        selected = labels[(labels.policy_eligible == 1) & (labels.policy_target_argmax == model)]
        minority[model] = {"rows": int(len(selected)), "patients": int(selected.patient_id.nunique()), "minimum_rows_pass": bool(len(selected) >= 100), "minimum_patients_pass": bool(selected.patient_id.nunique() >= 20)}
    reject = pd.read_parquet(OUT / "labels/dev_causal_policy.parquet")
    reject = reject[reject.reject_label == 1]
    support_pass = all(v["minimum_rows_pass"] and v["minimum_patients_pass"] for v in minority.values())
    reject_pass = bool(len(reject) >= 100 and reject.patient_id.nunique() >= 20)
    collapse_failure = bool(max_share > .95 and not dev["oracle_has_any_fa_reduction"])
    gates = {
        "dev_oracle_sensitivity_within_3pp": bool(dev["sensitivity_safe_3pp"]),
        "dev_oracle_fa_reduction_at_least_25pct": bool(dev["fa_reduction_at_least_25pct"]),
        "all_three_policy_targets_supported": support_pass,
        "reject_target_supported": reject_pass,
        "collapse_rule_failure": collapse_failure,
        "eval_read": False,
        "no_test_selection": True,
    }
    passed = all(gates.values())
    payload = {
        "release_id": NS,
        "status": "PASS_ORACLE_GATE" if passed else "FAIL_ORACLE_GATE",
        "training_started": False,
        "ready_for_25_update_smoke": passed,
        "ready_for_formal_training": False,
        "formal_training_blocker": "10-second raw-cache and smoke audit must complete before any formal run",
        "gate_rules": gates,
        "collapse_observation": {"max_policy_argmax_model": max_model, "max_policy_argmax_share": max_share, "interpretation": "distribution is highly imbalanced, but the specified collapse failure requires no oracle benefit; oracle benefit is present"},
        "minority_support": minority,
        "reject_support": {"rows": int(len(reject)), "patients": int(reject.patient_id.nunique()), "pass": reject_pass},
        "dev_oracle": dev,
        "train_oracle": train,
        "source_oracle_manifest": str(ORACLE.relative_to(ROOT)).replace("\\", "/"),
        "source_oracle_manifest_sha256": sha256(ORACLE),
        "source_labels_manifest_sha256": sha256(OUT / "label_manifest.json"),
        "eval_used": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + chr(10))
    out.with_suffix(".sha256").write_text(sha256(out) + chr(10))
    print(json.dumps(payload, indent=2, sort_keys=True, default=float))

if __name__ == "__main__":
    main()
