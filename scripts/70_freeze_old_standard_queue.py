"""Freeze the superseded greedy standard-confirmatory queue before replacement.

This script does not modify the old run namespace.  It records the dirty git
state, archives project-owned source/configuration files, and hashes the old
manifest, completed run metadata, and the partial TCN-GAT checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def archive(root: Path, destination: Path, paths: list[Path]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as stream:
        for path in paths:
            if not path.exists():
                continue
            stream.add(path, arcname=str(path.relative_to(root)))


def main() -> None:
    root = Path.cwd().resolve()
    snapshot = root / "outputs/archives/v3-standard-greedy-queue-freeze-v1"
    snapshot.mkdir(parents=True, exist_ok=True)
    status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True)
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=root)
    base_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    (snapshot / "git_status.txt").write_text(status, encoding="utf-8")
    (snapshot / "working_tree.diff").write_bytes(diff)
    (snapshot / "base_commit.txt").write_text(base_commit + "\n", encoding="utf-8")

    code_paths = [root / name for name in ("src", "scripts", "configs", "protocols", "tests")]
    for name in ("pyproject.toml", "README.md"):
        path = root / name
        if path.exists():
            code_paths.append(path)
    archive(root, snapshot / "code_snapshot.tar.gz", code_paths)

    selected = [
        root / "manifests/standard_cv_v1",
        root / "configs/standard/v3_standard_confirmatory_v1.yaml",
    ]
    old_root = root / "runs/v3-standard-confirmatory-v1"
    for pattern in (
        "**/completed.json",
        "**/run_manifest.json",
        "**/test/metrics.json",
        "**/validation/metrics_step_05000.json",
        "**/checkpoints/step_03000.pt",
        "**/checkpoints/step_03000.pt.sha256",
    ):
        selected.extend(old_root.glob(pattern))
    archive(root, snapshot / "frozen_old_queue_metadata.tar.gz", selected)

    paths: list[Path] = [
        root / "configs/standard/v3_standard_confirmatory_v1.yaml",
        root / "manifests/standard_cv_v1/cv_manifest.json",
        root
        / "runs/v3-standard-confirmatory-v1/tcn_gat/split2_seed17_main/checkpoints/step_03000.pt",
    ]
    paths.extend(sorted(root.glob("manifests/standard_cv_v1/fold_*.json")))
    for pattern in (
        "**/completed.json",
        "**/run_manifest.json",
        "**/test/metrics.json",
        "**/validation/metrics_step_05000.json",
        "**/checkpoints/step_03000.pt.sha256",
    ):
        paths.extend(sorted(old_root.glob(pattern)))

    records: list[dict[str, object]] = []
    seen: set[Path] = set()
    for path in paths:
        path = path.resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        records.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    manifest = {
        "snapshot_id": "v3-standard-greedy-queue-freeze-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_commit": base_commit,
        "working_tree_dirty": bool(status.strip()),
        "old_run_namespace": "v3-standard-confirmatory-v1",
        "old_run_completed_count": len(list(old_root.glob("**/completed.json"))),
        "partial_checkpoint": "runs/v3-standard-confirmatory-v1/tcn_gat/split2_seed17_main/checkpoints/step_03000.pt",
        "new_runs_must_not_write_old_namespace": True,
        "artifacts": {
            "git_status": "git_status.txt",
            "working_tree_diff": "working_tree.diff",
            "code_snapshot": "code_snapshot.tar.gz",
            "frozen_old_queue_metadata": "frozen_old_queue_metadata.tar.gz",
        },
        "file_hashes": records,
    }
    (snapshot / "freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "snapshot": str(snapshot),
                "completed": manifest["old_run_completed_count"],
                "hashed_files": len(records),
                "code_archive_bytes": (snapshot / "code_snapshot.tar.gz").stat().st_size,
                "metadata_archive_bytes": (snapshot / "frozen_old_queue_metadata.tar.gz").stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
