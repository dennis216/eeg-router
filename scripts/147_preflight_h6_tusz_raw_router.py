"""Preflight and freeze the TUSZ raw-EEG pre-router namespace.

This script is deliberately read-only with respect to existing experiments.  It
creates only the new namespace manifest after verifying that no old run is
active and that the official TUSZ train/dev/eval partitions are present.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/b_false_alarm_atlas")
DATA_ROOT = Path("/mnt/d/TUH_EEG/TUSZ_v2.0.6")
OUT = ROOT / "outputs/reports/h6-tusz-raw-pre-router-v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def count_partition(name: str) -> dict[str, int]:
    root = DATA_ROOT / "edf" / name
    if not root.is_dir():
        raise FileNotFoundError(root)
    # The TUSZ tree is on a mounted Windows volume; a Python recursive stat of
    # every file can take minutes.  Use the filesystem's find implementation
    # for the inventory and inspect one annotation per partition here.  The
    # full per-record annotation audit is performed by the manifest builder.
    def nlines(args: list[str]) -> int:
        out = subprocess.check_output(args, text=True)
        return sum(1 for line in out.splitlines() if line.strip())

    patients = nlines(["find", str(root), "-mindepth", "1", "-maxdepth", "1", "-type", "d"])
    records = nlines(["find", str(root), "-type", "f", "-iname", "*.edf"])
    sample = subprocess.check_output(
        ["find", str(root), "-type", "f", "-iname", "*.csv_bi", "-print", "-quit"],
        text=True,
    ).strip()
    if not sample:
        raise FileNotFoundError(f"no _bi annotation found in {name}")
    return {"patients": patients, "records": records, "missing_bi": "not_scanned"}


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing namespace: {OUT}")
    partitions = {name: count_partition(name) for name in ("train", "dev", "eval")}
    expected = {"train": (579, 5057), "dev": (53, 2203), "eval": (43, 880)}
    for name, (patients, records) in expected.items():
        got = partitions[name]
        if (got["patients"], got["records"]) != (patients, records):
            raise RuntimeError(f"unexpected TUSZ {name} inventory: {got}, expected {(patients, records)}")
    ps = subprocess.run(["ps", "-eo", "pid,etime,cmd"], text=True, capture_output=True, check=True)
    current = str(os.getpid())
    forbidden = ("h6-tusz-raw-pre-router-v1", "147_preflight_h6_tusz_raw_router", "148_build_h6_tusz")
    active = [line for line in ps.stdout.splitlines() if any(x in line for x in forbidden) and not line.lstrip().startswith(current + " ")]
    if active:
        raise RuntimeError("active new-router process detected: " + " | ".join(active))
    status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    payload = {
        "protocol_id": "h6-tusz-raw-pre-router-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "TUSZ_v2.0.6 official train/dev/eval",
        "dataset_root": str(DATA_ROOT),
        "partitions": partitions,
        "expected_inventory": expected,
        "group_unit": "patient_id",
        "raw_view": {"channels": 16, "sampling_hz": 128, "context_seconds": 60, "samples": 7680},
        "future_horizon_seconds": 60,
        "router_input_excludes_expert_outputs": True,
        "expert_checkpoint_policy": "frozen v3-groupkfold-confirmatory-v1 checkpoints; no expert retraining",
        "old_results_preserved": True,
        "git_status_at_preflight": status,
        "status": "PASS",
    }
    OUT.mkdir(parents=True, exist_ok=False)
    path = OUT / "preflight_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (OUT / "preflight_manifest.sha256").write_text(sha256(path) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
