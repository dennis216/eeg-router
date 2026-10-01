"""Freeze the standard GroupKFold preflight inputs before confirmation runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUN_ID = "v3-groupkfold-confirmatory-v1"
OUT = ROOT / "outputs" / "reports" / RUN_ID / "preflight_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    tracked = {
        "protocol": ROOT / "protocols" / "v3-groupkfold-confirmatory-v1.md",
        "config": ROOT / "configs" / "standard" / "v3_groupkfold_confirmatory_v1.yaml",
        "cv_manifest": ROOT / "manifests" / "groupkfold_cv_v1" / "cv_manifest.json",
        "runner": ROOT / "scripts" / "11_run_bsn_pilot.py",
        "queue": ROOT / "scripts" / "run_groupkfold_confirmatory_v1.sh",
        "split_builder": ROOT / "scripts" / "71_build_groupkfold_confirmatory_v1.py",
        "run_audit": ROOT / "scripts" / "72_audit_groupkfold_confirmatory_v1.py",
        "oof_builder": ROOT / "scripts" / "73_build_groupkfold_confirmatory_oof.py",
    }
    missing = [str(path) for path in tracked.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing preflight files: " + ", ".join(missing))
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        commit = "unavailable"
    manifest = {
        "namespace": RUN_ID,
        "frozen_at_unix": time.time(),
        "git_commit_at_preflight": commit,
        "split_algorithm": "sklearn GroupKFold(n_splits=5) outer + GroupShuffleSplit(test_size=0.20) inner",
        "group_unit": "canonical patient ID only",
        "formal_runs": 45,
        "smoke_updates": 25,
        "test_threshold_policy": "validation-only; test evaluated once per formal run",
        "files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in tracked.values()
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
