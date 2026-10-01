"""Update release metadata after freezing H4 as preliminary_negative."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "outputs" / "reports" / "v3-compact-extension-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    path = REPORT_DIR / "release_manifest.json"
    release = json.loads(path.read_text())
    freeze = json.loads((REPORT_DIR / "h4_negative_freeze.json").read_text())
    h4 = release["analyses"]["h4"]
    h4["status"] = "preliminary_negative"
    h4["functional_success"] = False
    h4["negative_freeze"] = "outputs/reports/v3-compact-extension-v1/h4_negative_freeze.json"
    h4["negative_freeze_sha256"] = sha256(REPORT_DIR / "h4_negative_freeze.json")
    h4["gate_sha256"] = sha256(REPORT_DIR / "gates.json")
    h4["router_manifest_sha256"] = sha256(REPORT_DIR / "h4_router_manifest.json")
    h4["test_results_locked"] = True
    h4["test_retuning_forbidden"] = True
    h4["interpretation"] = "preliminary negative under the 5000-window train-only compute budget; all split-seed and phenotype/no-phenotype comparisons preserved; h4-expanded-v1 requires new holdout or TUH validation"
    release["interpretation"] = [
        "preliminary",
        "compute-matched",
        "h4-preliminary-negative",
        "h4-test-results-locked",
        "h5-descriptive-transfer-only-pending",
        "tuh-external-validation-pending",
        "no_full_v3_completion_claim",
        "no_architecture_superiority_claim_from_pilot_alone",
    ]
    path.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": release["status"], "h4_status": h4["status"], "freeze_sha256": freeze["artifacts"]["gates_sha256"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
