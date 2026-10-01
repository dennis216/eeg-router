"""Finalize the corrected H6 causal-policy gate.

This is an immutable replacement for the first bookkeeping audit. It applies
the stated collapse rule to the corrected labels and keeps Eval excluded.
"""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path("/root/b_false_alarm_atlas")
NS = "h6-causal-policy-router-v2-corrected"
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
    target = OUT / "training_gate_corrected_v2.json"
    if target.exists():
        raise RuntimeError(f"refusing overwrite {target}")
    oracle = json.loads(ORACLE.read_text())
    dev = oracle["results"]["dev"]["gate"]
    train = oracle["results"]["train"]["gate"]
    labels = pd.read_parquet(OUT / "labels/dev_causal_policy.parquet")
    eligible = labels[labels.policy_eligible == 1]
    counts = eligible.policy_target_argmax.value_counts().to_dict()
    max_model = str(max(counts, key=counts.get)) if counts else "none"
    max_share = float(max(counts.values()) / len(eligible)) if len(eligible) else 1.0
    minority = {}
    for m in MODELS:
        selected = eligible[eligible.policy_target_argmax == m]
        minority[m] = {"rows": int(len(selected)), "patients": int(selected.patient_id.nunique()), "rows_pass": bool(len(selected) >= 100), "patients_pass": bool(selected.patient_id.nunique() >= 20)}
    reject = labels[labels.reject_label == 1]
    conditions = {
        "dev_oracle_sensitivity_within_3pp": bool(dev["sensitivity_safe_3pp"]),
        "dev_oracle_fa_reduction_at_least_25pct": bool(dev["fa_reduction_at_least_25pct"]),
        "all_three_policy_targets_supported": all(x["rows_pass"] and x["patients_pass"] for x in minority.values()),
        "reject_target_supported": bool(len(reject) >= 100 and reject.patient_id.nunique() >= 20),
        "collapse_rule_not_triggered": not bool(max_share > .95 and not dev["oracle_has_any_fa_reduction"]),
        "eval_not_read": True,
        "no_test_selection": True,
    }
    passed = all(conditions.values())
    payload = {"release_id": NS,"status":"PASS_ORACLE_GATE" if passed else "FAIL_ORACLE_GATE","oracle_gate_pass":passed,"ready_for_25_update_smoke":passed,"ready_for_formal_training":False,"next_required_stage":"25_update_smoke_with_fixed_four_strata_batch","conditions":conditions,"collapse_observation":{"max_policy_argmax_model":max_model,"max_policy_argmax_share":max_share,"rule":"failure only when >95% and no oracle benefit"},"minority_support":minority,"reject_support":{"rows":int(len(reject)),"patients":int(reject.patient_id.nunique())},"dev_oracle":dev,"train_oracle":train,"source_oracle_manifest":"outputs/reports/h6-causal-policy-router-v2-corrected/oracle/oracle_manifest.json","source_oracle_manifest_sha256":sha256(ORACLE),"labels_manifest_sha256":sha256(OUT/"label_manifest.json"),"eval_read":False,"test_threshold_selection":False,"created_utc":datetime.now(timezone.utc).isoformat()}
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + chr(10))
    target.with_suffix(".sha256").write_text(sha256(target) + chr(10))
    print(json.dumps(payload, indent=2, sort_keys=True, default=float))

if __name__ == "__main__":
    main()
