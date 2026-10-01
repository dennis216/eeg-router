"""Write the corrected immutable H6 v2 gate.

The first gate file is retained as an audit trail. Its eval_read=false field
is a provenance flag, not a boolean requirement; this script fixes that
bookkeeping error without changing labels, oracle metrics, or thresholds.
"""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/b_false_alarm_atlas")
OUT = ROOT / "outputs/reports/h6-causal-policy-router-v2"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def main():
    source = OUT / "training_gate.json"
    target = OUT / "training_gate_corrected.json"
    if target.exists():
        raise RuntimeError(f"refusing overwrite {target}")
    old = json.loads(source.read_text())
    gates = old["gate_rules"]
    # eval_read is intentionally false as a status flag. The corresponding
    # pass condition is its negation and is recorded separately here.
    conditions = {
        "dev_oracle_sensitivity_within_3pp": bool(gates["dev_oracle_sensitivity_within_3pp"]),
        "dev_oracle_fa_reduction_at_least_25pct": bool(gates["dev_oracle_fa_reduction_at_least_25pct"]),
        "all_three_policy_targets_supported": bool(gates["all_three_policy_targets_supported"]),
        "reject_target_supported": bool(gates["reject_target_supported"]),
        "collapse_rule_not_triggered": not bool(gates["collapse_rule_failure"]),
        "eval_not_read": not bool(old.get("eval_used", True)),
        "no_test_selection": bool(gates["no_test_selection"]),
    }
    passed = all(conditions.values())
    payload = {
        "release_id": "h6-causal-policy-router-v2",
        "status": "PASS_ORACLE_GATE" if passed else "FAIL_ORACLE_GATE",
        "oracle_gate_pass": passed,
        "ready_for_25_update_smoke": passed,
        "ready_for_formal_training": False,
        "next_required_stage": "build_10_second_raw_cache_then_25_update_smoke",
        "formal_training_blocker": "raw-cache contract and smoke audit are not yet complete",
        "conditions": conditions,
        "source_gate_audit": "outputs/reports/h6-causal-policy-router-v2/training_gate.json",
        "source_gate_audit_sha256": sha256(source),
        "labels_manifest_sha256": sha256(OUT / "label_manifest.json"),
        "oracle_manifest_sha256": sha256(OUT / "oracle/oracle_manifest.json"),
        "eval_read": False,
        "test_threshold_selection": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + chr(10))
    target.with_suffix(".sha256").write_text(sha256(target) + chr(10))
    print(json.dumps(payload, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
