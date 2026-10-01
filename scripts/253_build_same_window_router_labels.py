from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bfa.same_window_moe import ACTIONS, EXPERTS, build_policy_targets, distribution_audit


ROOT = Path("/root/b_false_alarm_atlas")
DEFAULT_PROBABILITY_ROOT = ROOT / "outputs/reports/same-window-router-first-moe-v1/expert_probabilities/tusz"
DEFAULT_THRESHOLDS = ROOT / "outputs/reports/same-window-router-first-moe-v1/expert_audit/thresholds.json"
DEFAULT_OUTPUT = ROOT / "outputs/reports/same-window-router-first-moe-v1/router_labels"
KEYS = ["patient_id", "recording", "window_start_s", "window_end_s", "label"]


def load_aligned(root: Path, partition: str) -> pd.DataFrame:
    merged = None
    for model in EXPERTS:
        path = root / partition / f"{model}.parquet"
        if path.is_file():
            frame = pd.read_parquet(path)
        else:
            files = sorted((root / partition / model).rglob("*.parquet"))
            if not files:
                raise FileNotFoundError(path)
            frame = pd.concat((pd.read_parquet(file) for file in files), ignore_index=True)
        forbidden = [column for column in frame if "future" in column.lower() or "horizon" in column.lower()]
        if forbidden:
            raise RuntimeError(f"future-policy columns are forbidden: {forbidden}")
        missing = set(KEYS + ["probability"]) - set(frame.columns)
        if missing:
            raise RuntimeError(f"{path} lacks {sorted(missing)}")
        metadata = [
            column for column in (
                "source_cache_path", "source_cache_path_v1", "cache_path", "cache_row",
                "relative_edf", "source_sampling_hz",
            ) if column in frame.columns
        ] if merged is None else []
        frame = frame[KEYS + metadata + ["probability"]].rename(columns={"probability": f"probability_{model}"})
        merged = frame if merged is None else merged.merge(frame, on=KEYS, how="inner", validate="one_to_one")
    if merged is None or merged.empty:
        raise RuntimeError(f"no aligned expert rows for {partition}")
    length = merged.window_end_s.to_numpy(float) - merged.window_start_s.to_numpy(float)
    if not np.allclose(length, 10.0):
        raise RuntimeError("router and expert windows must be exactly 10 seconds")
    return merged.sort_values(["recording", "window_start_s"], kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probability-root", type=Path, default=DEFAULT_PROBABILITY_ROOT)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--partitions", nargs="+", choices=("train", "val", "test"), default=("train", "val"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    thresholds = json.loads(args.thresholds.read_text())
    all_distribution = []
    audit_payload = {}
    for partition in args.partitions:
        frame = load_aligned(args.probability_root, partition)
        p = frame[[f"probability_{model}" for model in EXPERTS]].to_numpy(float)
        t = np.array([float(thresholds[model]) for model in EXPERTS])
        result = build_policy_targets(frame.label.to_numpy(float), p, t)
        frame["policy_action_eligible"] = result["eligible"]
        frame["policy_action_id"] = result["hard_action"]
        frame["policy_action"] = pd.Series(result["hard_action"]).map(dict(enumerate(ACTIONS))).fillna("unknown")
        frame["outcome_group"] = result["outcome_group"]
        for index, action in enumerate(ACTIONS):
            frame[f"policy_target_{action}"] = result["target"][:, index].astype(np.float32)
        for index, model in enumerate(EXPERTS):
            frame[f"threshold_{model}"] = t[index]
            frame[f"confidence_{model}"] = result["confidence"][:, index].astype(np.float32)
            frame[f"positive_{model}"] = result["expert_positive"][:, index]
            frame[f"correct_{model}"] = result["expert_correct"][:, index]
        output_path = args.output / f"{partition}.parquet"
        frame.to_parquet(output_path, index=False)
        distribution, warnings = distribution_audit(frame, partition)
        all_distribution.append(distribution)
        audit_payload[partition] = {
            "rows": int(len(frame)),
            "eligible_rows": int(frame.policy_action_eligible.sum()),
            "seizure_rows": int((frame.label == 1).sum()),
            "background_rows": int((frame.label == 0).sum()),
            "outcome_groups": frame.outcome_group.value_counts(dropna=False).to_dict(),
            "warnings": warnings,
        }
        patient = frame[frame.policy_action_eligible].groupby(["patient_id", "policy_action"], as_index=False).size()
        patient.to_csv(args.output / f"{partition}_patient_action_distribution.csv", index=False)
    pd.concat(all_distribution, ignore_index=True).to_csv(args.output / "natural_action_distribution.csv", index=False)
    (args.output / "audit.json").write_text(json.dumps(audit_payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
