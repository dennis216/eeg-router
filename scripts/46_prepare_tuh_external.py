"""Status-only TUH preparation, channel-mapping smoke, and frozen-artifact audit."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from bfa.preprocessing.channels import CHANNELS, MissingCanonicalChannel, canonical_order, normalize_channel


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "outputs" / "reports" / "v3-compact-extension-v1"
TUH_CANDIDATES = (
    Path("/mnt/d/EEGData/TUH"),
    Path("/mnt/d/TUH"),
    Path("/mnt/d/EEGData/tuh"),
    Path("/mnt/c/Users/User/Downloads/TUH"),
    Path("/mnt/c/Users/User/Downloads/TUSZ"),
    Path("/mnt/c/Users/User/Documents/TUH"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _channel_smoke() -> dict[str, object]:
    synthetic = [f"EEG {name}-REF" for name in reversed(CHANNELS)]
    indices = canonical_order(synthetic)
    normalized = [normalize_channel(name) for name in synthetic]
    missing_error = None
    try:
        canonical_order(synthetic[:-1])
    except MissingCanonicalChannel as exc:
        missing_error = str(exc)
    return {
        "canonical_channel_count": len(CHANNELS),
        "normalization_smoke_pass": normalized[0] == CHANNELS[-1],
        "reverse_order_mapping_pass": sorted(indices) == list(range(len(CHANNELS))),
        "missing_channel_guard_pass": missing_error is not None,
        "missing_channel_error": missing_error,
        "canonical_channels": list(CHANNELS),
    }


def main() -> None:
    existing = []
    for candidate in TUH_CANDIDATES:
        if candidate.exists():
            edf_count = sum(1 for path in candidate.rglob("*") if path.is_file() and path.name.lower().endswith((".edf", ".edf.gz")))
            existing.append({"path": str(candidate), "readable": os.access(candidate, os.R_OK), "edf_files": edf_count})
    tuh_available = bool(existing and any(row["edf_files"] > 0 for row in existing))
    frozen_files = [
        REPORT_DIR / "h4_negative_freeze.json",
        REPORT_DIR / "h4_frozen_inputs.json",
        REPORT_DIR / "h5_descriptive_transfer_manifest.json",
        REPORT_DIR / "feature_dictionary.csv",
    ]
    frozen_audit = {
        str(path.name): {"exists": path.exists(), "sha256": _sha256(path) if path.exists() else None}
        for path in frozen_files
    }
    status = {
        "release_id": "v3-compact-extension-v1",
        "dataset": "TUH/TUSZ family",
        "status": "ready_for_mapping_when_data_is_available" if tuh_available else "missing_dataset_in_current_environment",
        "tuh_available": tuh_available,
        "candidate_roots": [str(path) for path in TUH_CANDIDATES],
        "existing_candidate_roots": existing,
        "permissions_check": "pass" if not existing or all(row["readable"] for row in existing) else "permission_issue",
        "disk_free_gib": float(shutil.disk_usage("/").free / 2**30),
        "channel_mapping_smoke": _channel_smoke(),
        "frozen_artifact_audit": frozen_audit,
        "construct_validation": "pending_tuh_files" if not tuh_available else "not_started_until_channel_mapping",
        "model_retraining": False,
        "threshold_retuning": False,
        "notes": "No external performance number is reported because no TUH/TUSZ EDF or annotation files were found in the checked roots.",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "tuh_external_status.json"
    path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    (REPORT_DIR / "tuh_channel_mapping_smoke.json").write_text(json.dumps(status["channel_mapping_smoke"], indent=2, sort_keys=True) + "\n")
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
