"""Audit the immutable GroupKFold confirmatory matrix before OOF aggregation.

The audit is deliberately strict: it only accepts the new patient-level
GroupKFold namespace, requires all 45 runs to be complete, and verifies that
each formal test set was evaluated exactly once.  It never edits old queues.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUN_ID = "v3-groupkfold-confirmatory-v1"
RUN_ROOT = ROOT / "runs" / RUN_ID
MANIFEST_ROOT = ROOT / "manifests" / "groupkfold_cv_v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
FOLDS = tuple(range(5))
SEEDS = (17, 42, 3407)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    cv_manifest_path = MANIFEST_ROOT / "cv_manifest.json"
    if not cv_manifest_path.exists():
        raise FileNotFoundError(cv_manifest_path)
    cv_manifest = json.loads(cv_manifest_path.read_text())
    algorithm = str(cv_manifest.get("algorithm", ""))
    if "GroupKFold" not in algorithm or "GroupShuffleSplit" not in algorithm:
        raise RuntimeError(f"unexpected split algorithm: {algorithm!r}")
    if cv_manifest.get("assignment_inputs") != ["canonical patient_id only"]:
        raise RuntimeError("split manifest was not assigned from patient ID only")

    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for fold in FOLDS:
        split_path = MANIFEST_ROOT / f"fold_{fold}.json"
        split = json.loads(split_path.read_text())
        if split.get("fold") != fold:
            failures.append(f"fold {fold}: fold manifest index mismatch")
        all_sets = [set(split[part]) for part in ("train", "validation", "test")]
        if len(set.union(*all_sets)) != sum(len(s) for s in all_sets):
            failures.append(f"fold {fold}: patient appears in multiple partitions")
        for model in MODELS:
            for seed in SEEDS:
                run_dir = RUN_ROOT / model / f"split{fold}_seed{seed}_main"
                completed_path = run_dir / "completed.json"
                manifest_path = run_dir / "run_manifest.json"
                probability_path = run_dir / "test" / "probabilities.parquet"
                metrics_path = run_dir / "test" / "metrics.json"
                required = (completed_path, manifest_path, probability_path, metrics_path)
                missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
                if missing:
                    failures.append(
                        f"{model} fold {fold} seed {seed}: missing " + ", ".join(missing)
                    )
                    continue
                run_manifest = json.loads(manifest_path.read_text())
                completed = json.loads(completed_path.read_text())
                test_metrics = json.loads(metrics_path.read_text())
                expected_split = str(split_path.relative_to(ROOT)).replace("\\", "/")
                if run_manifest.get("pilot_id") != RUN_ID:
                    failures.append(f"{model} fold {fold} seed {seed}: wrong namespace")
                if run_manifest.get("split_file") != expected_split:
                    failures.append(f"{model} fold {fold} seed {seed}: wrong split file")
                if run_manifest.get("model") != model or int(run_manifest.get("seed", -1)) != seed:
                    failures.append(f"{model} fold {fold} seed {seed}: run identity mismatch")
                if not bool(run_manifest.get("test_accessed")):
                    failures.append(f"{model} fold {fold} seed {seed}: test was not finalized")
                if int(test_metrics.get("test_evaluation_count", -1)) != 1:
                    failures.append(f"{model} fold {fold} seed {seed}: test count is not one")
                if completed.get("test", {}).get("test_evaluation_count") != 1:
                    failures.append(f"{model} fold {fold} seed {seed}: completed test count is not one")
                expected_hash = str(test_metrics.get("probability_sha256", ""))
                if expected_hash != _sha256(probability_path):
                    failures.append(f"{model} fold {fold} seed {seed}: probability hash mismatch")
                rows.append(
                    {
                        "model": model,
                        "fold": fold,
                        "seed": seed,
                        "run_dir": str(run_dir.relative_to(ROOT)),
                        "test_probability_sha256": expected_hash,
                        "threshold": float(test_metrics["score"]["threshold"]),
                        "test_evaluation_count": int(test_metrics["test_evaluation_count"]),
                        "validation_threshold": float(run_manifest.get("validation_threshold", float("nan"))),
                    }
                )

    expected_count = len(MODELS) * len(FOLDS) * len(SEEDS)
    if len(rows) != expected_count:
        failures.append(f"completed run count {len(rows)} != expected {expected_count}")
    if failures:
        report = {"namespace": RUN_ID, "matrix_complete": False, "failures": failures, "runs_found": rows}
        out = RUN_ROOT / "groupkfold_confirmatory_audit.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(json.dumps(report, indent=2, sort_keys=True))

    report = {
        "namespace": RUN_ID,
        "matrix_complete": True,
        "expected_runs": expected_count,
        "completed_runs": len(rows),
        "models": list(MODELS),
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "split_algorithm": algorithm,
        "group_column": "canonical patient_id",
        "test_evaluation_count_required": 1,
        "run_records": rows,
        "cv_manifest_sha256": _sha256(cv_manifest_path),
    }
    out = RUN_ROOT / "groupkfold_confirmatory_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
