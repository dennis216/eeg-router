from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent.parent
NAMESPACE = "chb-mit-biot-router-minimal-v1"
REPORT = ROOT / "outputs" / "reports" / NAMESPACE


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-existing", action="store_true")
    args = ap.parse_args()
    cfg_path = ROOT / "configs" / "chb_mit_biot_router_minimal_v1.yaml"
    plan_path = Path(
        "/mnt/c/Users/User/AppData/Roaming/Tencent/WeMeet/Global/Data/IM/file/2026-8/"
        "13645446218310344276/CHB_MIT_BIOT_Router_Minimal_Experiment_Plan.md"
    )
    base_path = ROOT / "manifests" / "groupkfold_cv_v1" / "cv_manifest.json"
    biot_path = ROOT / "third_party/BIOT/pretrained-models/EEG-PREST-16-channels.ckpt"
    old_runs = ROOT / "runs" / "v3-groupkfold-confirmatory-v1"
    if REPORT.exists() and not args.allow_existing:
        raise SystemExit(f"refusing to overwrite existing namespace: {REPORT}")
    if not cfg_path.is_file() or not plan_path.is_file() or not base_path.is_file():
        raise SystemExit("required plan/config/base manifest is missing")
    if not biot_path.is_file():
        raise SystemExit("official BIOT-PREST checkpoint is missing")
    cfg = yaml.safe_load(cfg_path.read_text())
    base = json.loads(base_path.read_text())
    alg = str(base.get("algorithm", ""))
    if "GroupKFold" not in alg or "greedy" in alg.lower():
        raise SystemExit(f"base manifest is not the required standard GroupKFold: {alg}")
    folds = base.get("folds", [])
    if len(folds) != 5:
        raise SystemExit("base manifest must contain five outer folds")
    test_sets = [set(f["test"]) for f in folds]
    if set.intersection(*test_sets):
        raise SystemExit("outer test folds overlap")
    all_patients = set.union(*test_sets)
    if len(all_patients) != 22:
        raise SystemExit(f"expected 22 canonical patients, got {len(all_patients)}")
    proc = subprocess.run(["ps", "-eo", "pid,etime,cmd"], text=True, capture_output=True, check=True)
    current_pid = str(os.getpid())
    active = [
        line
        for line in proc.stdout.splitlines()
        if (NAMESPACE in line or "129_" in line) and not line.lstrip().startswith(current_pid + " ")
    ]
    if active:
        raise SystemExit("active plan process detected: " + " | ".join(active))
    payload = {
        "protocol_id": NAMESPACE,
        "created_at_unix": time.time(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "config_path": str(cfg_path.relative_to(ROOT)),
        "config_sha256": sha256_file(cfg_path),
        "base_cv_manifest": str(base_path.relative_to(ROOT)),
        "base_cv_sha256": sha256_file(base_path),
        "base_algorithm": alg,
        "base_patient_count": len(all_patients),
        "base_folds": [{"fold": f["fold"], "test": f["test"]} for f in folds],
        "biot_checkpoint": str(biot_path.relative_to(ROOT)),
        "biot_checkpoint_sha256": sha256_file(biot_path),
        "biot_expected_sha256": cfg["biot"]["checkpoint_sha256"],
        "biot_hash_ok": sha256_file(biot_path) == cfg["biot"]["checkpoint_sha256"],
        "old_namespace_preserved": old_runs.is_dir(),
        "old_run_count": len(list(old_runs.rglob("completed.json"))) if old_runs.is_dir() else 0,
        "formal_checkpoint_reuse": False,
        "status": "PASS",
    }
    if not payload["biot_hash_ok"]:
        raise SystemExit("BIOT checkpoint hash mismatch")
    REPORT.mkdir(parents=True, exist_ok=False)
    (REPORT / "preflight.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
