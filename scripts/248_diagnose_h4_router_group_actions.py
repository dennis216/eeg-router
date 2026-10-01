"""Read-only diagnostic of frozen H4 model-aware router actions by group kind."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path("/root/b_false_alarm_atlas")
SOURCE = ROOT / "outputs/reports/v3-groupkfold-confirmatory-v1"
ORACLE = ROOT / "outputs/reports/h4-protocol-matched-oracle-v2"
OUT = Path("/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/reports/h4-router-background-actions-v1")
MODELS = ("singlem", "cbramod", "tcn_gat")
MINIMUM_OVERLAP_S = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_h4():
    path = ROOT / "scripts/84_run_groupkfold_h4_router.py"
    spec = importlib.util.spec_from_file_location("h4_group_action_diagnostic", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def validation_row(validation: pd.DataFrame, fold: int, seed: int) -> pd.Series:
    row = validation[
        (validation.fold == fold)
        & (validation.seed == seed)
        & (validation.method == "router_model_aware")
    ]
    if len(row) != 1:
        raise RuntimeError(f"expected one model-aware validation row: {fold=} {seed=}")
    return row.iloc[0]


def classify_groups(candidates: pd.DataFrame, starts: dict[str, float], seizures: pd.DataFrame, h4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_id, group in candidates.groupby("group_id", sort=True):
        recording = str(group.recording.iloc[0])
        truths = h4._truths(seizures, recording, starts[recording])
        associated = any(
            overlap(
                float(action.start_s),
                float(action.group_decision_end),
                float(truth_start),
                float(truth_end),
            )
            >= MINIMUM_OVERLAP_S
            for action in group.itertuples(index=False)
            for truth_start, truth_end in truths
        )
        rows.append(
            {
                "group_id": str(group_id),
                "recording": recording,
                "patient": str(group.patient.iloc[0]),
                "group_kind": "seizure_associated" if associated else "background",
            }
        )
    return pd.DataFrame(rows)


def summarize(labels: pd.DataFrame, selected_ids: set[str], fold: int, seed: int) -> dict[str, object]:
    row: dict[str, object] = {"fold": fold, "seed": seed}
    for kind, prefix in (("background", "background"), ("seizure_associated", "seizure_associated")):
        ids = set(labels.loc[labels.group_kind == kind, "group_id"].astype(str))
        emitted = len(ids & selected_ids)
        abstained = len(ids - selected_ids)
        total = len(ids)
        row[f"{prefix}_groups"] = total
        row[f"{prefix}_emitted"] = emitted
        row[f"{prefix}_emitted_proportion"] = emitted / total if total else float("nan")
        row[f"{prefix}_abstained"] = abstained
        row[f"{prefix}_abstained_proportion"] = abstained / total if total else float("nan")
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    h4 = load_h4()
    manifest = json.loads((ORACLE / "manifest.json").read_text())
    event_path = SOURCE / "h4_groupkfold_event_features.parquet"
    validation_path = SOURCE / "h4_groupkfold_router_validation.csv"
    if sha256(event_path) != manifest["source_hashes"]["event_features"]:
        raise RuntimeError("event feature hash differs from oracle manifest")
    if sha256(validation_path) != manifest["source_hashes"]["h4_validation"]:
        raise RuntimeError("validation table hash differs from oracle manifest")

    features = pd.read_parquet(event_path)
    validation = pd.read_csv(validation_path)
    seizures = pd.read_parquet(ROOT / "manifests/seizures.parquet")
    rows: list[dict[str, object]] = []
    parity: list[dict[str, object]] = []
    policy = pd.read_csv(ORACLE / "policy_per_run.csv")

    test = features[features.partition.astype(str) == "test"].copy()
    for (fold_value, seed_value), frame in test.groupby(["split", "seed"], sort=True):
        fold, seed = int(fold_value), int(seed_value)
        candidates = h4._candidate_groups(frame.copy())
        starts, _ = h4._evaluation_meta(frame)
        labels = classify_groups(candidates, starts, seizures, h4)
        lock = validation_row(validation, fold, seed)
        model_path = SOURCE / f"h4_router_model_aware_fold{fold}_seed{seed}.joblib"
        model = joblib.load(model_path)
        matrix, _ = h4._router_matrix(candidates, True)
        scores = model.predict_proba(matrix)[:, 1]
        selected = h4._router_select(candidates, scores, float(lock.threshold))
        selected_ids = set(selected.group_id.astype(str)) if not selected.empty else set()
        if not selected_ids.issubset(set(labels.group_id.astype(str))):
            raise RuntimeError("selected group absent from candidate labels")
        rows.append(summarize(labels, selected_ids, fold, seed))

        expected = policy[
            (policy.fold == fold)
            & (policy.seed == seed)
            & (policy.policy == "router_model_aware")
        ].iloc[0]
        expected_selected = int(expected.true_positive_events + expected.false_alarm_events)
        parity.append(
            {
                "fold": fold,
                "seed": seed,
                "selected_groups": len(selected_ids),
                "scored_tp_plus_fp": expected_selected,
                "difference": len(selected_ids) - expected_selected,
            }
        )

    per_run = pd.DataFrame(rows).sort_values(["fold", "seed"]).reset_index(drop=True)
    if len(per_run) != 15:
        raise RuntimeError(f"expected 15 held-out evaluations, found {len(per_run)}")
    parity_frame = pd.DataFrame(parity)
    if (parity_frame.difference != 0).any():
        raise RuntimeError("router selection parity failed")

    summary_rows: list[dict[str, object]] = []
    for prefix in ("background", "seizure_associated"):
        total = int(per_run[f"{prefix}_groups"].sum())
        emitted = int(per_run[f"{prefix}_emitted"].sum())
        abstained = int(per_run[f"{prefix}_abstained"].sum())
        summary_rows.append(
            {
                "group_kind": prefix,
                "evaluations": len(per_run),
                "mean_groups_per_evaluation": per_run[f"{prefix}_groups"].mean(),
                "mean_emitted_per_evaluation": per_run[f"{prefix}_emitted"].mean(),
                "mean_abstained_per_evaluation": per_run[f"{prefix}_abstained"].mean(),
                "mean_fold_seed_emitted_proportion": per_run[f"{prefix}_emitted_proportion"].mean(),
                "mean_fold_seed_abstained_proportion": per_run[f"{prefix}_abstained_proportion"].mean(),
                "aggregate_groups": total,
                "aggregate_emitted": emitted,
                "pooled_emitted_proportion": emitted / total,
                "aggregate_abstained": abstained,
                "pooled_abstained_proportion": abstained / total,
            }
        )
    summary = pd.DataFrame(summary_rows)
    per_run.to_csv(OUT / "model_aware_router_actions_per_fold_seed.csv", index=False)
    summary.to_csv(OUT / "model_aware_router_actions_summary.csv", index=False)
    parity_frame.to_csv(OUT / "selection_count_parity.csv", index=False)
    audit = {
        "analysis_id": "h4-router-background-actions-v1",
        "status": "complete_read_only_diagnostic",
        "source_protocol": manifest["source_protocol"],
        "candidate_group_rule": "unchanged H4 onset grouping within 30 seconds",
        "seizure_associated_definition": "at least one existing member action, extended to the unchanged group decision end, overlaps an eligible seizure by at least 1 second",
        "background_definition": "no existing member action in the candidate group overlaps an eligible seizure by at least 1 second",
        "router_action": "emit the highest-scoring member when its frozen model-aware score meets the validation-locked threshold; otherwise abstain",
        "aggregation": "unweighted mean of the 15 fold-seed proportions; aggregate raw counts and pooled proportions reported separately",
        "evaluations": len(per_run),
        "source_hashes_verified_against_oracle_manifest": True,
        "router_selection_count_parity": True,
        "test_used_for_training_or_threshold_selection": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(per_run.to_string(index=False))
    print("\nSUMMARY\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
