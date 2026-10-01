"""Run a deterministic TUSZ format and channel-mapping smoke for standard H5.

This smoke reads two frozen eval records (one seizure-present and one
background-only), verifies annotations, derives the canonical bipolar montage,
and constructs each model's frozen preprocessing view.  It does not load a
checkpoint, tune a threshold, or report external performance.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bfa.preprocessing.tuh import (
    MODEL_VIEW_CONFIG,
    load_tuh_canonical_raw_uv,
    tuh_model_view,
    tuh_quality_features,
)


ROOT = Path("/root/b_false_alarm_atlas")
RUN_ID = "v3-groupkfold-confirmatory-v1"
REPORT_DIR = ROOT / "outputs" / "reports" / RUN_ID
MANIFEST = REPORT_DIR / "tusz_groupkfold_external_manifest.json"
OUTPUT = REPORT_DIR / "tusz_groupkfold_format_smoke.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _annotation_rows(path: Path) -> list[dict[str, str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines()]
    return list(csv.DictReader([line for line in lines if line and not line.startswith("#")]))


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    records = manifest["records"]
    positives = [record for record in records if record["seizure_events"]]
    negatives = [record for record in records if not record["seizure_events"]]
    if not positives or not negatives:
        raise RuntimeError("frozen eval cohort lacks seizure-present or background-only record")
    rank = lambda record: hashlib.sha256(
        f"{RUN_ID}|tusz-groupkfold-format-smoke|{record['relative_edf']}".encode()
    ).hexdigest()
    selected = [min(positives, key=rank), min(negatives, key=rank)]
    results: list[dict[str, object]] = []
    for record in selected:
        path = Path(manifest["dataset_root"]) / record["relative_edf"]
        bi = Path(str(path.with_suffix(".csv")) + "_bi")
        raw_uv, source_hz, source_channels = load_tuh_canonical_raw_uv(path)
        quality = tuh_quality_features(raw_uv, source_hz)
        record_result: dict[str, object] = {
            "patient_id": record["patient_id"],
            "relative_edf": record["relative_edf"],
            "role": "seizure_present" if record["seizure_events"] else "background_only",
            "edf_sha256_manifest_match": _sha256(path) == record["edf_sha256"],
            "csv_bi_sha256_manifest_match": _sha256(bi) == record["csv_bi_sha256"],
            "annotation_rows": len(_annotation_rows(bi)),
            "source_sampling_hz": source_hz,
            "source_channel_count": len(source_channels),
            "source_shape_uv": list(raw_uv.shape),
            "source_finite": bool(np.isfinite(raw_uv).all()),
            "quality_shape": list(quality.shape),
            "quality_finite": bool(np.isfinite(quality).all()),
            "model_views": {},
        }
        for model in ("singlem", "cbramod", "tcn_gat"):
            view = tuh_model_view(raw_uv, source_hz, model)
            expected_hz = int(MODEL_VIEW_CONFIG[model]["sampling_hz"])
            record_result["model_views"][model] = {
                "shape": list(view.shape),
                "expected_sampling_hz": expected_hz,
                "finite": bool(np.isfinite(view).all()),
                "abs_p99": float(np.quantile(np.abs(view), 0.99)),
                "outside_minus1_5_plus1_5_fraction": float(np.mean(np.abs(view) > 1.5)) if model == "singlem" else None,
            }
        results.append(record_result)

    passed = all(
        result["edf_sha256_manifest_match"]
        and result["csv_bi_sha256_manifest_match"]
        and result["source_finite"]
        and result["quality_finite"]
        and all(view["finite"] for view in result["model_views"].values())
        for result in results
    )
    output = {
        "release_id": RUN_ID,
        "external_cohort_id": manifest["external_cohort_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "standard_tusz_format_and_channel_mapping_smoke",
        "cohort_manifest_sha256": _sha256(MANIFEST),
        "selected_records": results,
        "result_count": len(results),
        "all_pass": bool(passed and len(results) == 2),
        "model_training": False,
        "threshold_retuning": False,
        "external_performance_reported": False,
        "next_gate": "run one frozen standard checkpoint smoke" if passed else "repair TUH format/channel adapter",
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all_pass": output["all_pass"], "result_count": len(results), "output": str(OUTPUT), "next_gate": output["next_gate"]}, indent=2))


if __name__ == "__main__":
    main()
