from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NAMESPACE = "chb-mit-biot-router-minimal-v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def readonly_write(path: Path, payload: dict) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=ROOT / "manifests/groupkfold_cv_v1/cv_manifest.json")
    args = ap.parse_args()
    base = json.loads(args.base.read_text())
    folds = base["folds"]
    by_fold = {int(f["fold"]): list(f["test"]) for f in folds}
    if sorted(by_fold) != [0, 1, 2, 3, 4]:
        raise ValueError("base fold ids must be 0..4")
    patients = set()
    for p in by_fold.values():
        if patients.intersection(p):
            raise ValueError("base folds overlap")
        patients.update(p)
    if len(patients) != 22:
        raise ValueError(f"expected 22 patients, got {len(patients)}")
    rotations = [
        {"round": 0, "expert_train_folds": [2, 3, 4], "router_dev_fold": 1, "final_test_fold": 0},
        {"round": 1, "expert_train_folds": [3, 4, 0], "router_dev_fold": 2, "final_test_fold": 1},
        {"round": 2, "expert_train_folds": [4, 0, 1], "router_dev_fold": 3, "final_test_fold": 2},
        {"round": 3, "expert_train_folds": [0, 1, 2], "router_dev_fold": 4, "final_test_fold": 3},
        {"round": 4, "expert_train_folds": [1, 2, 3], "router_dev_fold": 0, "final_test_fold": 4},
    ]
    for r in rotations:
        used = set(r["expert_train_folds"]) | {r["router_dev_fold"], r["final_test_fold"]}
        if used != set(range(5)) or r["router_dev_fold"] == r["final_test_fold"]:
            raise ValueError(f"invalid rotation {r}")
        r["expert_train_patients"] = sum((by_fold[f] for f in r["expert_train_folds"]), [])
        r["router_dev_patients"] = by_fold[r["router_dev_fold"]]
        r["final_test_patients"] = by_fold[r["final_test_fold"]]
    payload = {
        "protocol_id": NAMESPACE,
        "status": "frozen",
        "created_at_unix": time.time(),
        "base_manifest": str(args.base.relative_to(ROOT)),
        "base_manifest_sha256": sha256_file(args.base),
        "algorithm": "standard_sklearn_GroupKFold_patient_only_derived_3_1_1_rotation",
        "group_unit": "canonical_patient_id",
        "patient_count": len(patients),
        "fold_count": 5,
        "rotations": rotations,
        "overlap_check": "PASS",
        "window_split_rule": "all EDFs events and windows follow canonical patient",
        "test_accessed_for_selection": False,
    }
    out = ROOT / "outputs/reports" / NAMESPACE / "split_manifest.json"
    readonly_write(out, payload)
    # Per-round split files are useful to the runners and are immutable as well.
    for r in rotations:
        readonly_write(
            ROOT / "outputs/reports" / NAMESPACE / f"round_{r['round']}_split.json",
            {
                "protocol_id": NAMESPACE,
                "round": r["round"],
                "train": r["expert_train_patients"],
                "validation": r["router_dev_patients"],
                "test": r["final_test_patients"],
                "expert_train_folds": r["expert_train_folds"],
                "router_dev_fold": r["router_dev_fold"],
                "final_test_fold": r["final_test_fold"],
            },
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
