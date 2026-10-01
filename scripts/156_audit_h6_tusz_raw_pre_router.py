"""Audit the frozen-data, no-leakage, and single-Eval contract for H6."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch


ROOT = Path("/root/b_false_alarm_atlas")
NAMESPACE = "h6-tusz-raw-pre-router-v1"
# The report namespace uses hyphens, while the frozen expert inference queue
# was deliberately stored under its existing underscore-qualified directory.
# Keep this explicit so the audit checks the real provenance files rather than
# silently treating all nine expert runs as missing.
EXPERT_NAMESPACE = "h6_tusz_raw_pre_router_v1"
OUT = ROOT / "outputs/reports" / NAMESPACE
MODELS = ("singlem", "cbramod", "tcn_gat")
PARTITIONS = ("train", "dev", "eval")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(problems: list[str], message: str) -> None:
    problems.append(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-checkpoint", type=Path)
    parser.add_argument("--eval-scoring-dir", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    problems: list[str] = []
    cohort = OUT / "cohort_freeze_manifest.json"
    if not cohort.exists():
        fail(problems, f"missing cohort freeze: {cohort}")
    else:
        cohort_payload = json.loads(cohort.read_text())
        if cohort_payload.get("patient_disjoint") is not True or cohort_payload.get("record_disjoint") is not True:
            fail(problems, "cohort is not patient/record disjoint")
    config = OUT / "raw_router_config.json"
    if not config.exists():
        fail(problems, f"missing frozen router config: {config}")
    else:
        config_payload = json.loads(config.read_text())
        input_contract = config_payload.get("input", {})
        if input_contract.get("shape") != [16, 7680] or input_contract.get("expert_output_in_input") is True:
            fail(problems, "raw-input shape/expert-output contract invalid")
        if "expert output" not in str(input_contract.get("preprocessing", "")).lower():
            fail(problems, "raw-input preprocessing contract does not exclude expert outputs")
    amendment = OUT / "raw_router_config_amendment_equal_probability_v1.json"
    if not amendment.exists():
        fail(problems, f"missing scoring protocol amendment: {amendment}")
    progress_rows = []
    for partition in PARTITIONS:
        for model in MODELS:
            run = ROOT / "outputs/tuh_external" / EXPERT_NAMESPACE / partition / model / "fold0_seed17"
            progress_path = run / "progress.json"
            if not progress_path.exists():
                if args.require_complete:
                    fail(problems, f"missing expert progress: {run}")
                continue
            progress = json.loads(progress_path.read_text())
            progress_rows.append(progress)
            # The historical Dev queue sometimes retained the string
            # ``running_or_incomplete`` after writing every record.  Treat an
            # exact completed/total count with zero failures as complete; a
            # partial count is still a hard failure.
            records_completed = int(progress.get("records_completed", -1))
            records_total = int(progress.get("records_total", -1))
            records_failed = int(progress.get("records_failed", 0))
            if progress.get("status") != "complete" and not (
                records_total > 0 and records_completed == records_total and records_failed == 0
            ):
                fail(problems, f"expert incomplete: {partition}/{model}")
            if records_failed:
                fail(problems, f"expert failures: {partition}/{model}: {progress.get('records_failed')}")
            if progress.get("model_training") is not False or progress.get("threshold_retuning") is not False:
                fail(problems, f"expert provenance flags invalid: {partition}/{model}")
            if int(progress.get("checkpoint_update", 5000)) != 5000:
                fail(problems, f"checkpoint update not 5000: {partition}/{model}")
    for label in sorted((OUT / "labels").glob("*.parquet")) if (OUT / "labels").exists() else []:
        frame = pd.read_parquet(label)
        for required in ["correct_mask", "reject_label", "unroutable", "cache_path"]:
            if required not in frame:
                fail(problems, f"missing label column {required}: {label.name}")
        if len(frame) and frame["correct_mask"].astype(str).str.len().ne(3).any():
            fail(problems, f"invalid three-expert masks: {label.name}")
    if args.require_complete:
        for partition in PARTITIONS:
            progress_path = OUT / "raw_cache" / f"{partition}_progress.json"
            if not progress_path.exists():
                fail(problems, f"missing raw-cache progress: {partition}")
            else:
                progress = json.loads(progress_path.read_text())
                if progress.get("status") != "complete" or int(progress.get("records_failed", 0)):
                    fail(problems, f"raw-cache incomplete/failed: {partition}")
            samples = list((OUT / "raw_cache" / partition).rglob("*.npz")) if (OUT / "raw_cache" / partition).exists() else []
            if not samples:
                fail(problems, f"missing raw-cache arrays: {partition}")
            else:
                import numpy as np
                try:
                    array = np.load(samples[0], allow_pickle=False)["x"]
                    if array.ndim != 3 or tuple(array.shape[1:]) != (16, 7680):
                        fail(problems, f"raw-cache shape invalid: {partition}: {array.shape}")
                except Exception as exc:
                    fail(problems, f"raw-cache unreadable: {partition}: {exc}")
    if args.training_checkpoint:
        checkpoint = args.training_checkpoint.resolve()
        if not checkpoint.exists():
            fail(problems, f"missing router checkpoint: {checkpoint}")
        else:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            keys = set(state.get("model", {}))
            for required in ["competence_head.weight", "reject_head.weight", "seizure_risk_head.weight"]:
                if required not in keys:
                    fail(problems, f"router checkpoint missing {required}")
    scoring_dir = args.eval_scoring_dir.resolve() if args.eval_scoring_dir else OUT / "scoring/eval"
    if (scoring_dir / "manifest.json").exists():
        scoring = json.loads((scoring_dir / "manifest.json").read_text())
        if scoring.get("threshold_source") != "TUSZ Dev only":
            fail(problems, "Eval threshold source is not Dev-only")
        if scoring.get("eval_evaluation_count_per_method") != 1:
            fail(problems, "Eval evaluation count is not one")
        if scoring.get("bootstrap_replicates") != 2000 or scoring.get("bootstrap_unit") != "patient":
            fail(problems, "Eval bootstrap contract invalid")
        if scoring.get("holdout_used_for_selection") is not False:
            fail(problems, "Eval was used for selection")
        if scoring.get("protocol_amendment_sha256") and not amendment.exists():
            fail(problems, "Eval references missing protocol amendment")
        if "equal_ensemble_threshold" not in scoring.get("thresholds", {}):
            fail(problems, "Eval is missing frozen equal-ensemble threshold")
    elif args.require_complete:
        fail(problems, f"missing Eval scoring manifest: {scoring_dir / 'manifest.json'}")
    payload = {"release_id": NAMESPACE, "status": "PASS" if not problems else "FAIL", "problems": problems, "expert_progress_rows": len(progress_rows), "cohort_sha256": sha256(cohort) if cohort.exists() else None, "created_utc": datetime.now(timezone.utc).isoformat()}
    audit = OUT / "h6_raw_pre_router_audit.json"
    audit.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    audit.with_suffix(".sha256").write_text(sha256(audit) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if problems:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
