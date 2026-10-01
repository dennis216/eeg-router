"""Final H1 x H4 linkage, oracle-gap, and coverage audit.

This is a read-only downstream analysis of the frozen
``v3-groupkfold-confirmatory-v1`` outputs.  It reconstructs the exact H4
event-candidate groups and learned-router decisions, adds a forced-selection
hindsight oracle, links local detector sharedness to router retention/regret,
and resolves the truth/recording coverage questions.  No model, threshold,
split, or test prediction is changed.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path("/root/b_false_alarm_atlas")
RUN_ID = "v3-groupkfold-confirmatory-v1"
SOURCE = ROOT / "outputs/reports" / RUN_ID
ORACLE_DIR = ROOT / "outputs/reports/h4-protocol-matched-oracle-v2"
OUT = ROOT / "outputs/reports/h4-final-linkage-expert-audit-v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
SEEDS = (17, 42, 3407)
N_BOOTSTRAP = 2000
RNG_SEED = 20260904
MIN_OVERLAP = 1.0


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ov(a: float, b: float, c: float, d: float) -> float:
    return max(0.0, min(b, d) - max(a, c))


def mask_name(models: set[str]) -> str:
    return "+".join(model for model in MODELS if model in models) if models else "none"


def validation_row(validation: pd.DataFrame, fold: int, seed: int, method: str) -> pd.Series:
    row = validation[(validation.fold == fold) & (validation.seed == seed) & (validation.method == method)]
    if len(row) != 1:
        raise RuntimeError(f"missing validation row {fold=} {seed=} {method=}")
    return row.iloc[0]


def raw_group_members(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Reproduce H4 anchor grouping while retaining every raw event member."""
    result: dict[str, pd.DataFrame] = {}
    for recording, rec in frame.groupby("recording", sort=True):
        ordered = rec.sort_values(["start_s", "end_s", "model"], kind="stable")
        groups: list[dict[str, object]] = []
        for row in ordered.to_dict(orient="records"):
            start = float(row["start_s"])
            if not groups or start - float(groups[-1]["anchor_start"]) > 30.0:
                groups.append({"anchor_start": start, "members": []})
            groups[-1]["members"].append(row)
        patient = str(ordered.patient.iloc[0])
        for index, group in enumerate(groups):
            result[f"{patient}|{recording}|{index}"] = pd.DataFrame(group["members"])
    return result


def router_decisions(candidates: pd.DataFrame, fold: int, seed: int, method: str, aware: bool, h4, validation: pd.DataFrame):
    model = joblib.load(SOURCE / f"h4_{method}_fold{fold}_seed{seed}.joblib")
    matrix, _ = h4._router_matrix(candidates, aware)
    scores = model.predict_proba(matrix)[:, 1]
    scored = candidates.copy()
    scored["candidate_router_score"] = scores
    threshold = float(validation_row(validation, fold, seed, method).threshold)
    chosen = h4._router_select(candidates, scores, threshold)
    return scored, chosen, threshold


def actual_matches(events: pd.DataFrame, starts: dict[str, float], seizures: pd.DataFrame, h4):
    matched_groups: set[str] = set()
    matched_truths: set[tuple[str, int]] = set()
    if events.empty or "recording" not in events.columns:
        return matched_groups, matched_truths
    for recording, start in starts.items():
        rec = events[events.recording.astype(str) == str(recording)].sort_values("event_start", kind="stable")
        truths = h4._truths(seizures, str(recording), float(start))
        predictions = [h4.Event(float(x.event_start), float(x.event_end), float(x.peak_probability)) for x in rec.itertuples(index=False)]
        matched = h4.match_events(predictions, truths)
        ids = rec.group_id.astype(str).tolist() if "group_id" in rec else [""] * len(rec)
        for pair in matched.pairs:
            matched_groups.add(ids[pair.prediction_index])
            matched_truths.add((str(recording), int(pair.truth_index)))
    return matched_groups, matched_truths


def forced_selection(actions: pd.DataFrame, oracle_selected: pd.DataFrame) -> pd.DataFrame:
    """Emit every H4 group, preserving hindsight assignments where available."""
    selected_by_group = {str(row.group_id): row for row in oracle_selected.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    for group_id, group in actions.groupby("group_id", sort=True):
        group_id = str(group_id)
        if group_id in selected_by_group:
            row = selected_by_group[group_id]
            rows.append({
                "patient": str(row.patient), "recording": str(row.recording),
                "event_start": float(row.event_start), "event_end": float(row.event_end),
                "peak_probability": float(row.peak_probability),
                "selected_model": str(row.selected_model), "group_id": group_id,
            })
        else:
            row = group.sort_values(["peak_probability"], ascending=False, kind="stable").iloc[0]
            rows.append({
                "patient": str(row.patient), "recording": str(row.recording),
                "event_start": float(row.event_start), "event_end": float(row.event_end),
                "peak_probability": float(row.peak_probability),
                "selected_model": str(row.selected_model), "group_id": group_id,
            })
    return pd.DataFrame(rows)


def proximity(raw: pd.DataFrame, all_rec: pd.DataFrame) -> dict[str, object]:
    anchor = raw.sort_values(["start_s", "end_s", "model"], kind="stable").iloc[0]
    others = all_rec[all_rec.model.astype(str) != str(anchor.model)].copy()
    delta = np.abs(others.start_s.to_numpy(float) - float(anchor.start_s)) if not others.empty else np.array([])
    out: dict[str, object] = {
        "anchor_model": str(anchor.model),
        "anchor_start_s": float(anchor.start_s),
        "nearest_other_detector_alarm_s": float(delta.min()) if len(delta) else math.nan,
    }
    for tolerance in (10, 30, 60):
        inside = others[np.abs(others.start_s.astype(float) - float(anchor.start_s)) <= tolerance]
        out[f"other_alarm_count_within_{tolerance}s"] = int(len(inside))
        out[f"other_detector_count_within_{tolerance}s"] = int(inside.model.astype(str).nunique())
    return out


def patient_category_bootstrap(groups: pd.DataFrame, router: str) -> pd.DataFrame:
    subset = groups[(groups.router == router) & (groups.group_kind == "background")].copy()
    patient_rates = (
        subset.groupby(["patient", "detector_count"], as_index=False)
        .agg(retained_rate=("retained", "mean"), regret_rate=("router_regret", "mean"), groups=("group_id", "size"))
    )
    patient_rates.to_csv(OUT / f"patient_sharedness_rates_{router}.csv", index=False)
    rng = np.random.default_rng(RNG_SEED + (0 if router == "router_model_aware" else 1))
    rows: list[dict[str, object]] = []
    for outcome in ("retained_rate", "regret_rate"):
        for target in (2, 3):
            wide = patient_rates.pivot(index="patient", columns="detector_count", values=outcome)
            if 1 not in wide or target not in wide:
                continue
            pair = wide[[1, target]].dropna()
            delta = pair[target] - pair[1]
            boot = np.empty(N_BOOTSTRAP)
            values = delta.to_numpy(float)
            for i in range(N_BOOTSTRAP):
                boot[i] = rng.choice(values, size=len(values), replace=True).mean()
            rows.append({
                "router": router, "outcome": outcome, "comparison": f"{target}-detector minus single",
                "patients": len(values), "delta": float(values.mean()),
                "ci025": float(np.quantile(boot, .025)), "ci975": float(np.quantile(boot, .975)),
            })
    return pd.DataFrame(rows)


def gee_association(groups: pd.DataFrame, router: str) -> pd.DataFrame:
    frame = groups[(groups.router == router) & (groups.group_kind == "background")].copy()
    frame["shared_two"] = (frame.detector_count == 2).astype(int)
    frame["shared_three"] = (frame.detector_count == 3).astype(int)
    frame["z_peak"] = (frame.max_peak - frame.max_peak.mean()) / max(frame.max_peak.std(), 1e-9)
    frame["log_duration"] = np.log1p(frame.group_duration_s)
    frame["z_log_duration"] = (frame.log_duration - frame.log_duration.mean()) / max(frame.log_duration.std(), 1e-9)
    # Cluster-robust logistic association; the patient is the uncertainty unit.
    fit = smf.glm(
        "retained ~ shared_two + shared_three + z_peak + z_log_duration + C(seed)",
        data=frame, family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": frame.patient})
    rows = []
    for term in fit.params.index:
        rows.append({
            "router": router, "term": term, "coefficient": float(fit.params[term]),
            "odds_ratio": float(np.exp(fit.params[term])),
            "ci025_or": float(np.exp(fit.conf_int().loc[term, 0])),
            "ci975_or": float(np.exp(fit.conf_int().loc[term, 1])),
            "p_value": float(fit.pvalues[term]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    h4 = load_module("h4_final_audit", ROOT / "scripts/84_run_groupkfold_h4_router.py")
    oracle_path = ROOT / "scripts/240_run_h4_protocol_matched_oracle.py"
    if not oracle_path.exists():
        oracle_path = Path("/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/scripts/240_run_h4_protocol_matched_oracle.py")
    oracle = load_module("oracle_final_audit", oracle_path)
    features = pd.read_parquet(SOURCE / "h4_groupkfold_event_features.parquet")
    validation = pd.read_csv(SOURCE / "h4_groupkfold_router_validation.csv")
    h1 = pd.read_csv(SOURCE / "h1_groupkfold_pairwise.csv")
    seizures = pd.read_parquet(ROOT / "manifests/seizures.parquet")
    recordings = pd.read_parquet(ROOT / "manifests/recordings.parquet")
    cv = json.loads((ROOT / "manifests/groupkfold_cv_v1/cv_manifest.json").read_text())
    durations = recordings.set_index("recording_id").duration_s.astype(float).to_dict()
    patient_by_recording = recordings.set_index("recording_id").patient_id.astype(str).to_dict()

    h1_sig: dict[tuple[int, str, str], bool] = {}
    for row in h1.itertuples(index=False):
        pair = tuple(sorted((str(row.left_model), str(row.right_model))))
        h1_sig[(int(row.seed), pair[0], pair[1])] = bool(float(row.q_fdr) < .05)

    group_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    decomposition_rows: list[dict[str, object]] = []
    test = features[features.partition.astype(str) == "test"]

    for (fold, seed), frame in test.groupby(["split", "seed"], sort=True):
        fold, seed = int(fold), int(seed)
        frame = frame.copy()
        starts, patients = h4._evaluation_meta(frame)
        actions = oracle.action_table(frame, h4, MODELS)
        full, _, assignments = oracle.protocol_oracle(frame, h4, seizures, starts, MODELS)
        forced = forced_selection(actions, full)
        for name, events in (("forced_selection_oracle", forced), ("full_abstaining_oracle", full)):
            score = h4._score_events(events, starts, patients, seizures, durations)
            policy_rows.append({"fold": fold, "seed": seed, "policy": name, **score})

        expected_patients = set(cv["folds"][fold]["test"])
        expected_recs = set(recordings[recordings.patient_id.astype(str).isin(expected_patients)].recording_id.astype(str))
        actual_recs = set(frame.recording.astype(str))
        missing = expected_recs - actual_recs
        missing_seizures = seizures[seizures.recording_id.astype(str).isin(missing)]
        eligible = 0
        reachable = 0
        for recording, start in starts.items():
            truth = h4._truths(seizures, recording, start)
            eligible += len(truth)
            rec_actions = actions[actions.recording.astype(str) == recording]
            reachable += sum(
                any(ov(float(a.event_start), float(a.event_end), s, e) >= MIN_OVERLAP for a in rec_actions.itertuples(index=False))
                for s, e in truth
            )
        manifest_truth = int(seizures[seizures.patient_id.astype(str).isin(expected_patients)].shape[0])
        coverage_rows.append({
            "fold": fold, "seed": seed, "expected_recordings": len(expected_recs),
            "recordings_with_candidates": len(actual_recs), "zero_candidate_recordings": len(missing),
            "zero_candidate_seizure_recordings": int(missing_seizures.recording_id.nunique()),
            "seizures_in_zero_candidate_recordings": int(len(missing_seizures)),
            "manifest_test_seizures": manifest_truth, "h4_eligible_truths": eligible,
            "truths_excluded_before_first_candidate_end": manifest_truth - eligible,
            "union_reachable_truths": reachable, "oracle_assignments": int(len(assignments)),
        })
        decomposition_rows.append({
            "fold": fold, "seed": seed, "eligible_truths": eligible,
            "union_reachable": reachable, "union_unreachable": eligible - reachable,
            "oracle_assigned": int(len(assignments)),
            "reachable_but_unassigned_group_conflict": reachable - int(len(assignments)),
            "background_groups": 0,
        })

        raw_members = raw_group_members(frame)
        group_kind: dict[str, str] = {}
        for group_id, group in actions.groupby("group_id", sort=True):
            recording = str(group.recording.iloc[0])
            truths = h4._truths(seizures, recording, starts[recording])
            hit = any(
                ov(float(a.event_start), float(a.event_end), s, e) >= MIN_OVERLAP
                for a in group.itertuples(index=False) for s, e in truths
            )
            group_kind[str(group_id)] = "seizure_reachable" if hit else "background"
        decomposition_rows[-1]["background_groups"] = sum(v == "background" for v in group_kind.values())

        assignment_truth_by_group = {
            str(x.group_id): (str(x.recording), int(x.truth_index)) for x in assignments.itertuples(index=False)
        }
        for method, aware in (("router_model_aware", True), ("router_no_model_id", False)):
            scored, selected, threshold = router_decisions(h4._candidate_groups(frame), fold, seed, method, aware, h4, validation)
            selected_ids = set(selected.group_id.astype(str)) if not selected.empty else set()
            matched_ids, matched_truths = actual_matches(selected, starts, seizures, h4)
            for group_id, group in actions.groupby("group_id", sort=True):
                group_id = str(group_id)
                raw = raw_members[group_id]
                recording = str(group.recording.iloc[0])
                active = set(group.selected_model.astype(str))
                pairs = [tuple(sorted((a, b))) for i, a in enumerate(active) for b in list(active)[i + 1:]]
                contributing = [h1_sig.get((seed, a, b), False) for a, b in pairs]
                candidate_scores = scored[scored.group_id.astype(str) == group_id]
                top = candidate_scores.sort_values(["candidate_router_score", "peak_probability"], ascending=False).iloc[0]
                retained = group_id in selected_ids
                assigned_truth = assignment_truth_by_group.get(group_id)
                kind = group_kind[group_id]
                if kind == "background":
                    regret = retained
                else:
                    regret = bool(assigned_truth is not None and assigned_truth not in matched_truths)
                rec_all = frame[frame.recording.astype(str) == recording]
                prox = proximity(raw, rec_all)
                group_rows.append({
                    "fold": fold, "seed": seed, "router": method,
                    "patient": str(group.patient.iloc[0]), "recording": recording, "group_id": group_id,
                    "group_kind": kind, "detector_count": len(active), "detector_mask": mask_name(active),
                    "pair_identity": mask_name(active) if len(active) == 2 else "NA",
                    "h1_local_shared_30s": len(active) >= 2,
                    "h1_all_contributing_pairs_fdr_significant": bool(contributing and all(contributing)),
                    "h1_any_contributing_pair_fdr_significant": bool(any(contributing)),
                    "retained": retained, "correctly_abstained": kind == "background" and not retained,
                    "router_false_alarm": kind == "background" and retained,
                    "router_true_positive_group": group_id in matched_ids,
                    "router_regret": regret, "oracle_selected_group": group_id in set(full.group_id.astype(str)),
                    "selected_model": str(top.source_model) if retained else "abstain",
                    "router_score": float(top.candidate_router_score), "router_threshold": threshold,
                    "max_peak": float(group.peak_probability.max()),
                    "mean_peak": float(group.peak_probability.mean()),
                    "group_duration_s": float(group.event_end.max() - group.event_start.min()),
                    **prox,
                })

    groups = pd.DataFrame(group_rows)
    policies = pd.DataFrame(policy_rows)
    coverage = pd.DataFrame(coverage_rows)
    decomposition = pd.DataFrame(decomposition_rows)
    groups.to_csv(OUT / "h1_h4_group_linkage.csv", index=False)
    policies.to_csv(OUT / "forced_and_full_oracle_per_run.csv", index=False)
    coverage.to_csv(OUT / "recording_truth_coverage.csv", index=False)
    decomposition.to_csv(OUT / "oracle_gap_decomposition.csv", index=False)

    summary = (
        groups[groups.group_kind == "background"]
        .groupby(["router", "detector_count"], as_index=False)
        .agg(groups=("group_id", "size"), retained_rate=("retained", "mean"),
             regret_rate=("router_regret", "mean"), mean_router_score=("router_score", "mean"))
    )
    summary.to_csv(OUT / "h1_h4_sharedness_summary.csv", index=False)
    mask_summary = (
        groups[groups.group_kind == "background"]
        .groupby(["router", "detector_mask"], as_index=False)
        .agg(groups=("group_id", "size"), retained_rate=("retained", "mean"), regret_rate=("router_regret", "mean"))
    )
    mask_summary.to_csv(OUT / "h1_h4_mask_summary.csv", index=False)
    boot = pd.concat([patient_category_bootstrap(groups, x) for x in ("router_model_aware", "router_no_model_id")], ignore_index=True)
    boot.to_csv(OUT / "h1_h4_patient_bootstrap.csv", index=False)
    gee_frames = []
    for router_name in ("router_model_aware", "router_no_model_id"):
        try:
            gee_frames.append(gee_association(groups, router_name))
        except Exception as exc:
            gee_frames.append(pd.DataFrame([{"router": router_name, "term": "ERROR", "error": repr(exc)}]))
    gee = pd.concat(gee_frames, ignore_index=True)
    gee.to_csv(OUT / "h1_h4_patient_clustered_gee.csv", index=False)

    forced_summary = policies.groupby("policy", as_index=False)[["sensitivity", "fa_per_24h"]].mean()
    forced_summary.to_csv(OUT / "forced_and_full_oracle_summary.csv", index=False)
    manifest = {
        "analysis_id": "h4-final-linkage-expert-audit-v1",
        "stage": "h1_h4_linkage_oracle_gap_coverage_complete",
        "source_protocol": RUN_ID,
        "runs": int(len(coverage)), "patient_bootstrap_replicates": N_BOOTSTRAP,
        "test_used_for_fitting_or_threshold_selection": False,
        "h1_mapping_note": "Local multi-detector status is group-level; H1 q-values are seed-level pair evidence and are not individual-group labels.",
        "forced_oracle_note": "Ground-truth-informed expert choice for reachable truths, but no abstention; every candidate group emits one existing action.",
        "outputs": sorted(p.name for p in OUT.iterdir() if p.is_file()),
    }
    (OUT / "stage1_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\nSHAREDNESS\n", summary.to_string(index=False))
    print("\nBOOTSTRAP\n", boot.to_string(index=False))
    print("\nGEE\n", gee.to_string(index=False))
    print("\nORACLES\n", forced_summary.to_string(index=False))
    print("\nCOVERAGE\n", coverage.to_string(index=False))
    print("\nDECOMPOSITION\n", decomposition.to_string(index=False))


if __name__ == "__main__":
    main()
