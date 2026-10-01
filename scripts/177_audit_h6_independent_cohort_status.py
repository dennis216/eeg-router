"""Record whether a genuinely unused H6 external evaluation cohort exists."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/b_false_alarm_atlas")
OUT = ROOT / "outputs/reports/h6-causal-policy-router-v2-reweighted"
STATUS = OUT / "independent_cohort_status.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if STATUS.exists():
        raise RuntimeError(f"refusing to overwrite {STATUS}")
    candidates = [
        {
            "name": "Siena",
            "paths_checked": ["/mnt/d/EEGData", "/mnt/d/TUH_EEG"],
            "found": False,
            "usable_for_h6": False,
            "reason": "No Siena directory or seizure-labeled Siena manifest was found locally.",
        },
        {
            "name": "TUSZ_v2.0.6",
            "paths_checked": ["/mnt/d/TUH_EEG/TUSZ_v2.0.6"],
            "found": True,
            "usable_for_h6": False,
            "reason": "Train, Dev and Eval recordings were already used by h6-tusz-raw-pre-router-v1; a new split of the same recordings is not an independent cohort.",
        },
        {
            "name": "TUAR",
            "paths_checked": ["/mnt/d/TUH_EEG/TUAR"],
            "found": True,
            "usable_for_h6": False,
            "reason": "Artifact annotations are available, but this corpus is not a seizure-detector event-level ground-truth cohort for H6 sensitivity/FA scoring.",
        },
        {
            "name": "TUEV",
            "paths_checked": ["/mnt/d/TUH_EEG/TUEV"],
            "found": True,
            "usable_for_h6": False,
            "reason": "Artifact labels are available, but there is no compatible seizure event ground truth for H6 routing performance.",
        },
    ]
    source_files = []
    for p in [
        ROOT / "outputs/reports/h6-causal-policy-router-v2-reweighted/final_decision.json",
        ROOT / "outputs/reports/h6-causal-policy-router-v2-reweighted/scoring_dev_formal_train_only/manifest.json",
        ROOT / "outputs/reports/h6-tusz-raw-pre-router-v1/cohort_freeze_manifest.json",
    ]:
        if p.exists():
            source_files.append({"path": str(p.relative_to(ROOT)), "sha256": sha256(p)})
    result = {
        "release_id": "h6-causal-policy-router-v2-reweighted",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "NO_ELIGIBLE_UNUSED_EXTERNAL_COHORT",
        "independent_cohort_required": True,
        "eval_read_for_this_audit": False,
        "candidates": candidates,
        "selection_rule": "must have raw EEG, seizure event ground truth, compatible channel/time contract, and no prior H6-v1/v2 exposure",
        "next_action": "Obtain or explicitly lock a genuinely unused seizure-labeled cohort (for example Siena) and run one frozen external evaluation; until then report H6-v2 as Dev-only exploratory evidence.",
        "source_files": source_files,
    }
    STATUS.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (OUT / "independent_cohort_status.sha256").write_text(sha256(STATUS) + "  independent_cohort_status.json\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
