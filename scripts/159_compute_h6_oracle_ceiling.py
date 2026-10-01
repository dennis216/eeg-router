"""Compute the interval-oracle ceiling before fitting the raw router.

The oracle uses only the offline future competence labels and frozen expert
event streams.  It is not a deployable method and is never used to choose an
Eval threshold; it quantifies the maximum benefit available to any causal
selector under the same 60-s action grid.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path("/root/b_false_alarm_atlas")
OUT = ROOT / "outputs/reports/h6-tusz-raw-pre-router-v1"


def load_scorer():
    path = ROOT / "scripts/155_score_h6_tusz_raw_pre_router.py"
    spec = importlib.util.spec_from_file_location("h6_scorer", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=["dev", "eval"], required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite oracle ceiling: {output}")
    scorer = load_scorer()
    manifest = json.loads(args.manifest.resolve().read_text())
    labels = pd.read_parquet(args.labels.resolve())
    events, thresholds, _ensemble_cache = scorer._read_event_cache(manifest, args.expert_root.resolve())
    rows = scorer._score_actions(labels, manifest, events, "oracle_interval", None)
    summary = scorer._summary(rows)
    payload = {
        "release_id": "h6-tusz-raw-pre-router-v1",
        "partition": args.partition,
        "status": "oracle_ceiling_only",
        "summary": summary,
        "expert_thresholds": thresholds,
        "manifest": str(args.manifest.resolve().relative_to(ROOT)),
        "labels": str(args.labels.resolve().relative_to(ROOT)),
        "expert_root": str(args.expert_root.resolve().relative_to(ROOT)),
        "selection_used": False,
        "deployable": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
