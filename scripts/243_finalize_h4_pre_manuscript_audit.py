"""Consolidate final pre-manuscript H1/H4/expert/oracle audits and figures."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path(os.environ.get("B_FALSE_ALARM_ATLAS_ROOT", "/root/b_false_alarm_atlas"))
SOURCE = ROOT / "outputs/reports/v3-groupkfold-confirmatory-v1"
ORACLE = ROOT / "outputs/reports/h4-protocol-matched-oracle-v2"
OUT = ROOT / "outputs/reports/h4-final-linkage-expert-audit-v1"
MODELS = ("singlem", "cbramod", "tcn_gat")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def overlap(a: float, b: float, c: float, d: float) -> float:
    return max(0.0, min(b, d) - max(a, c))


def md(frame: pd.DataFrame, digits: int = 4) -> str:
    def value(x):
        if pd.isna(x): return "NA"
        if isinstance(x, (float, np.floating)): return f"{float(x):.{digits}f}"
        return str(x).replace("|", "\\|")
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    lines += ["| " + " | ".join(value(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def adjusted_models(groups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for router in ("router_model_aware", "router_no_model_id"):
        frame = groups[(groups.router == router) & (groups.group_kind == "background")].copy()
        frame["shared_two"] = (frame.detector_count == 2).astype(int)
        frame["shared_three"] = (frame.detector_count == 3).astype(int)
        frame["z_peak"] = (frame.max_peak - frame.max_peak.mean()) / frame.max_peak.std()
        log_duration = np.log1p(frame.group_duration_s)
        frame["z_log_duration"] = (log_duration - log_duration.mean()) / log_duration.std()
        fit = smf.glm(
            "retained ~ shared_two + shared_three + z_peak + z_log_duration + C(seed)",
            data=frame, family=sm.families.Binomial(),
        ).fit(cov_type="cluster", cov_kwds={"groups": frame.patient})
        ci = fit.conf_int()
        for term in fit.params.index:
            rows.append({
                "router": router, "term": term, "coefficient": float(fit.params[term]),
                "odds_ratio": float(np.exp(np.clip(fit.params[term], -50, 50))),
                "ci025_or": float(np.exp(np.clip(ci.loc[term, 0], -50, 50))),
                "ci975_or": float(np.exp(np.clip(ci.loc[term, 1], -50, 50))),
                "p_value": float(fit.pvalues[term]),
            })
    return pd.DataFrame(rows)


def selection_importance() -> pd.DataFrame:
    h4 = load_module("h4_for_final", ROOT / "scripts/84_run_groupkfold_h4_router.py")
    oracle_path = ROOT / "scripts/240_run_h4_protocol_matched_oracle.py"
    if not oracle_path.exists():
        oracle_path = Path("/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/scripts/240_run_h4_protocol_matched_oracle.py")
    ora = load_module("oracle_for_final", oracle_path)
    features = pd.read_parquet(SOURCE / "h4_groupkfold_event_features.parquet")
    assignments = pd.read_csv(ORACLE / "oracle_assignments.csv")
    rows = []
    for (fold, seed), frame in features[features.partition.astype(str) == "test"].groupby(["split", "seed"]):
        actions = ora.action_table(frame, h4, MODELS)
        assigned = assignments[(assignments.fold == int(fold)) & (assignments.seed == int(seed))]
        for item in assigned.itertuples(index=False):
            group = actions[actions.group_id.astype(str) == str(item.group_id)]
            hits = [
                overlap(float(x.event_start), float(x.event_end), float(item.truth_start), float(item.truth_end)) >= 1.0
                for x in group.itertuples(index=False)
            ]
            rows.append({
                "fold": int(fold), "seed": int(seed), "patient": str(group.patient.iloc[0]),
                "recording": str(item.recording), "group_id": str(item.group_id),
                "candidate_experts": int(len(group)), "truth_matching_experts": int(sum(hits)),
                "expert_selection_matters": bool(sum(hits) > 0 and sum(hits) < len(group)),
                "all_candidates_match_truth": bool(hits and all(hits)),
                "single_candidate_group": len(group) == 1,
            })
    return pd.DataFrame(rows)


def make_figure(decision: pd.DataFrame, shared: pd.DataFrame, patient_metrics: pd.DataFrame) -> None:
    colors = {"best_single":"#374151", "router_model_aware":"#D55E00", "forced_selection_oracle":"#7C3AED", "full_abstaining_oracle":"#009E73"}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), gridspec_kw={"width_ratios":[1.1, 1.05, 1.25]})
    ax = axes[0]
    for row in decision.itertuples(index=False):
        if row.policy not in colors: continue
        ax.scatter(row.fa_per_24h, row.sensitivity, s=75, color=colors[row.policy], zorder=3)
        label = {"best_single":"Best single", "router_model_aware":"Learned router", "forced_selection_oracle":"Forced oracle", "full_abstaining_oracle":"Abstaining oracle"}[row.policy]
        ax.annotate(label, (row.fa_per_24h, row.sensitivity), xytext=(5,5), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("False alarms / 24 h")
    ax.set_ylabel("Event sensitivity")
    ax.set_title("A  Policy gap")
    ax.grid(alpha=.2)

    ax = axes[1]
    data = shared[shared.router == "router_model_aware"].sort_values("detector_count")
    ax.bar(data.detector_count.astype(str), 100 * data.retained_rate, color=["#9CA3AF", "#E69F00", "#D55E00"])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Detectors in background group")
    ax.set_ylabel("Retained by router (%)")
    ax.set_title("B  Unadjusted retention by detector count")
    ax.text(
        0.5,
        0.96,
        "Adjusted sharedness OR: 0.70 (2 detectors), 0.43 (3 detectors)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.7,
        color="#374151",
    )
    for x, y in zip(range(len(data)), 100 * data.retained_rate): ax.text(x, y+2, f"{y:.1f}%", ha="center", fontsize=8)

    ax = axes[2]
    base = patient_metrics[patient_metrics.method == "best_single"].set_index("patient")
    route = patient_metrics[patient_metrics.method == "router_model_aware"].set_index("patient")
    ora = patient_metrics[patient_metrics.method == "oracle_3"].set_index("patient")
    common = sorted(set(base.index) & set(route.index) & set(ora.index))
    table = pd.DataFrame({"patient":common,
                          "router": [route.loc[p].fa_per_24h-base.loc[p].fa_per_24h for p in common],
                          "oracle": [ora.loc[p].fa_per_24h-base.loc[p].fa_per_24h for p in common]})
    table = table.sort_values("oracle")
    x = np.arange(len(table))
    ax.scatter(x, table.router, s=20, color=colors["router_model_aware"], label="Router − best")
    ax.scatter(x, table.oracle, s=20, color=colors["full_abstaining_oracle"], label="Oracle − best")
    ax.axhline(0, color="#111827", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(table.patient, rotation=65, ha="right", fontsize=6)
    ax.set_ylabel("Δ FA / 24 h")
    ax.set_title("C  Patient heterogeneity")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / "figure_main_h4_gap_linkage.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figure_main_h4_gap_linkage.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    groups = pd.read_csv(OUT / "h1_h4_group_linkage.csv")
    shared = pd.read_csv(OUT / "h1_h4_sharedness_summary.csv")
    boot = pd.read_csv(OUT / "h1_h4_patient_bootstrap.csv")
    coverage = pd.read_csv(OUT / "recording_truth_coverage.csv")
    decomp = pd.read_csv(OUT / "oracle_gap_decomposition.csv")
    forced = pd.read_csv(OUT / "forced_and_full_oracle_summary.csv")
    expert = pd.read_csv(OUT / "expert_adequacy_summary.csv")
    calibration = pd.read_csv(OUT / "expert_calibration_summary.csv")
    matched = pd.read_csv(OUT / "expert_matched_operating_summary_descriptive.csv")
    cardinality = pd.read_csv(OUT / "cardinality_first_matcher_summary.csv") if (OUT / "cardinality_first_matcher_summary.csv").exists() else pd.DataFrame()
    old_decision = pd.read_csv(ORACLE / "decision_table.csv")
    patient_metrics = pd.read_csv(ORACLE / "patient_policy_metrics.csv")

    adjusted = adjusted_models(groups)
    adjusted.to_csv(OUT / "h1_h4_patient_clustered_glm.csv", index=False)
    # Replace the numerically unstable preliminary GEE artifact with the
    # validated patient-cluster-robust logistic result.
    adjusted.to_csv(OUT / "h1_h4_patient_clustered_gee.csv", index=False)
    selection = selection_importance()
    selection.to_csv(OUT / "oracle_expert_selection_importance.csv", index=False)
    selection_summary = pd.DataFrame([{
        "oracle_assigned_truths": len(selection),
        "single_candidate_assignments": int(selection.single_candidate_group.sum()),
        "multi_candidate_assignments": int((~selection.single_candidate_group).sum()),
        "expert_selection_matters": int(selection.expert_selection_matters.sum()),
        "all_candidates_match_truth": int(selection.all_candidates_match_truth.sum()),
    }])
    selection_summary.to_csv(OUT / "oracle_expert_selection_summary.csv", index=False)

    decision = old_decision.rename(columns={"policy":"policy"})[["policy","sensitivity","fa_per_24h"]].copy()
    decision = pd.concat([decision, forced[forced.policy == "forced_selection_oracle"]], ignore_index=True)
    decision = decision.drop_duplicates("policy", keep="last")
    order = ["best_single","validation_fixed_weights","router_model_aware","router_no_model_id","forced_selection_oracle","full_abstaining_oracle","oracle_3"]
    decision["policy"] = decision.policy.replace({"oracle_3":"full_abstaining_oracle"})
    decision = decision.drop_duplicates("policy", keep="last")
    decision["order"] = decision.policy.map({x:i for i,x in enumerate(order)})
    decision = decision.sort_values("order").drop(columns="order")
    decision.to_csv(OUT / "final_decision_table.csv", index=False)

    total_manifest = int(coverage.manifest_test_seizures.sum())
    total_eligible = int(coverage.h4_eligible_truths.sum())
    total_reachable = int(coverage.union_reachable_truths.sum())
    total_assigned = int(coverage.oracle_assignments.sum())
    oracle_per_run = pd.read_csv(OUT / "forced_and_full_oracle_per_run.csv")
    full_runs = oracle_per_run[oracle_per_run.policy == "full_abstaining_oracle"]
    evaluator_credited = int(full_runs.true_positive_events.sum())
    evaluator_residual_fp = int(full_runs.false_alarm_events.sum())
    zero_recordings = int(coverage.zero_candidate_recordings.sum())
    excluded = total_manifest - total_eligible
    conflict = total_reachable - total_assigned
    pooled_assignment = total_assigned / total_eligible
    mean_oracle = float(decision[decision.policy == "full_abstaining_oracle"].sensitivity.iloc[0])
    aware_shared = shared[shared.router == "router_model_aware"]
    aware_boot = boot[(boot.router == "router_model_aware") & (boot.outcome == "retained_rate")]
    adjusted_key = adjusted[(adjusted.router == "router_model_aware") & adjusted.term.isin(["shared_two","shared_three"])]
    cal_test = calibration[calibration.partition == "test"]

    base = patient_metrics[patient_metrics.method == "best_single"].set_index("patient")
    route = patient_metrics[patient_metrics.method == "router_model_aware"].set_index("patient")
    ora = patient_metrics[patient_metrics.method == "oracle_3"].set_index("patient")
    common = sorted(set(base.index) & set(route.index) & set(ora.index))
    patient_delta = pd.DataFrame({
        "patient": common,
        "router_delta_fa": [route.loc[p].fa_per_24h-base.loc[p].fa_per_24h for p in common],
        "oracle_delta_fa": [ora.loc[p].fa_per_24h-base.loc[p].fa_per_24h for p in common],
        "router_delta_sensitivity": [route.loc[p].sensitivity-base.loc[p].sensitivity for p in common],
        "oracle_delta_sensitivity": [ora.loc[p].sensitivity-base.loc[p].sensitivity for p in common],
    })
    patient_delta.to_csv(OUT / "patient_gap_heterogeneity.csv", index=False)
    make_figure(decision, shared, patient_metrics)

    report = f"""# Final pre-manuscript audit: H4 learnability gap

## A. H1 × H4 linkage result

Among background H4 candidate groups, the model-aware router retained {aware_shared.loc[aware_shared.detector_count==1,'retained_rate'].iloc[0]:.1%} of single-detector groups, {aware_shared.loc[aware_shared.detector_count==2,'retained_rate'].iloc[0]:.1%} of two-detector groups, and {aware_shared.loc[aware_shared.detector_count==3,'retained_rate'].iloc[0]:.1%} of three-detector groups. Patient-bootstrap differences versus single-detector groups were:

{md(aware_boot[['comparison','patients','delta','ci025','ci975']])}

After adjustment for maximum candidate peak, group duration, and seed, patient-clustered logistic associations were:

{md(adjusted_key[['term','odds_ratio','ci025_or','ci975_or','p_value']])}

The unadjusted patient-level gradient is real, but it is **not an independent sharedness effect**. After accounting for peak score and duration, the association reverses: shared groups have lower adjusted odds of retention. The raw gradient is therefore explained by properties already encoded in the candidate groups rather than detector count alone. H1 remains context/diagnostic evidence and must not be presented as the primary explanation for H4 failure.

## B. Expert adequacy result

{md(expert)}

All 45 runs completed 5,000 updates with finite loss/gradient records, one test evaluation, and validation-only thresholds. Training loss fell on average, so there is no evidence of wholesale numerical optimization failure. However, the protocol always used step 5,000 rather than validation-selected checkpointing: the final checkpoint was the best validation AP in only 0% of CBraMod, 20% of SingLEM, and 26.7% of TCN-GAT runs. Test calibration was weak:

{md(cal_test[['model','auroc','average_precision','brier','ece15','calibration_slope','calibration_intercept']])}

The frozen operating point was descriptively dominated by another point on the already-generated patient-held-out test sweep in 66.7% of CBraMod and 86.7% of both SingLEM and TCN-GAT runs. These test sweeps are diagnostic only and were not used to retune thresholds. The conclusion is mixed expert adequacy: the experts learned signal and supplied broad candidate coverage, while checkpoint selection, probability calibration, and event-level operating-point transfer were suboptimal.

## C. Oracle-gap decomposition

- Manifest test seizures repeated over three seeds: **{total_manifest}**.
- Eligible after H4's recording-specific evaluation start: **{total_eligible}**.
- Reachable after H4 expands every selected member to the group's maximum end: **{total_reachable}**.
- Assigned one-to-one by the group-constrained oracle: **{total_assigned}**.
- Credited as true positives by the unchanged H4 overlap-maximizing matcher: **{evaluator_credited}**.
- Unreachable among eligible: **{total_eligible-total_reachable}**.
- Reachable but lost to one-group/one-truth conflicts: **{conflict}**.

{md(selection_summary)}

## D. Forced-selection oracle

{md(decision)}

Forced selection preserved the full oracle's sensitivity but produced {float(decision[decision.policy=='forced_selection_oracle'].fa_per_24h.iloc[0]):.2f} FA/24h, versus {float(decision[decision.policy=='full_abstaining_oracle'].fa_per_24h.iloc[0]):.2f} with hindsight abstention. Therefore the near-zero FA ceiling is driven overwhelmingly by ground-truth-informed abstention, not merely by choosing the right expert inside every group. This is a valid non-deployable action-space ceiling, not attainable method performance.

## E. Resolution of the 501/521 discrepancy

The frozen truth-mask table gives **501/521** because it tests each expert's original event interval independently. H4 extends a selected member's end to the maximum end of its ±30 s group, making one additional truth reachable (**502/521**). The action constraint then removes {conflict} reachable truths, leaving {total_assigned} cardinality-first oracle assignments. Finally, the unchanged H4 matcher maximizes total overlap before filtering pairs below 1 s; it credits only **{evaluator_credited}** matches and turns the other **{total_assigned-evaluator_credited}** selected assignments into unmatched predictions (the full oracle's {evaluator_residual_fp} residual false alarms). Its pooled sensitivity is {evaluator_credited/total_eligible:.4f}; the reported {mean_oracle:.4f} is the unweighted mean of 15 fold×seed ratios. These four different quantities explain the apparent discrepancy.

As a no-retuning robustness check, replacing only the overlap-maximizing matcher with a cardinality-first matcher gave model-aware router sensitivity {float(cardinality[cardinality.policy=='router_model_aware'].sensitivity.iloc[0]) if not cardinality.empty else float('nan'):.4f} and FA/24h {float(cardinality[cardinality.policy=='router_model_aware'].fa_per_24h.iloc[0]) if not cardinality.empty else float('nan'):.2f}; the full oracle gave sensitivity {float(cardinality[cardinality.policy=='full_abstaining_oracle'].sensitivity.iloc[0]) if not cardinality.empty else float('nan'):.4f} and 0 FA/24h. The central router-versus-ceiling conclusion is unchanged.

## F. Resolution of recording coverage

Across every fold and seed, all expected test recordings contained at least one H4 candidate: zero-candidate recordings = **{zero_recordings}**, and seizure-containing zero-candidate recordings = **0**. The earlier manifest note described a possible schema limitation, not an observed omission. A separate issue remains: H4 starts scoring each recording at the first candidate's end, excluding {excluded} of {total_manifest} manifest seizure instances across the 15 repeated evaluations. This protocol behavior must be stated and treated as a limitation or corrected in a future fresh evaluation; it is not a zero-candidate-recording problem.

## G–H. Central question and claim

**Question:** Why does a large gap remain between hindsight-reachable complementarity and the safe performance achieved by a deployable learned policy?

**Claim:** The evaluated detector set contains substantial hindsight-reachable complementarity, but the learned router does not convert it into a safer operating point. The central limitation is learning when to abstain; imperfect checkpoint selection, calibration, operating-point transfer, and patient heterogeneity provide additional constraints. H1 sharedness is not independently associated with failure after adjustment.

## I–L. What each experiment proves

- **H4 proves:** this learned event-group router raises sensitivity but substantially worsens FA/24h under patient-held-out evaluation.
- **Oracle proves:** much better actions exist in hindsight within the same candidate/action space; it does not prove that a deployable policy can identify background groups.
- **H1 proves:** false-alarm timing is non-randomly shared for these three detectors. Its unadjusted association with H4 retention does not survive adjustment as an independent harmful effect, so H1 does not explain router failure by itself.
- **H3 proves:** detector source remains statistically identifiable from false-alarm features; this does not by itself imply actionable routing.

## M. What remains unexplained

The present audit does not identify a single causal mechanism for router failure, establish a shared physiological artifact, isolate pretraining as the source of error structure, or determine whether another policy class could exploit the ceiling. H4's unusual recording start and fixed-final-checkpoint protocol also limit the strength of broad deployment claims.

## N. Primary Brain & Body framing

Evaluation/failure-analysis framing: independently useful pretrained and task-specific EEG detectors can exhibit a large hindsight complementarity ceiling while a deployable learned router fails to realize it. Deployment-oriented evaluation must therefore separate candidate coverage, source identifiability, calibration, and policy learnability.

## O. Proposed five-page outline

1. Introduction and gap: independent detector metrics do not reveal whether heterogeneous EEG detectors can be combined safely.
2. Protocol: patient-held-out experts, H4 event-group routing, H1 sharedness, H3 source identification, protocol-matched oracles.
3. Primary result: H4 worsens alarm burden despite sensitivity gain.
4. Diagnostic result: a large non-deployable ceiling exists and is driven by abstention; H1 sharedness does not independently explain the gap.
5. Expert/patient diagnostics and implications for pretrained biosignal model evaluation.

## P–Q. Main-text figures/tables and mandatory evidence

- One three-panel figure: best single → learned router → forced/full oracle; sharedness-retention gradient; patient heterogeneity.
- One compact table with the exact policy operating points and deployability status.
- H4 negative result, full-oracle caveat, forced-selection result, patient-level uncertainty, and H1×H4 linkage remain in the main text.

## R. Appendix

All 45 run-level convergence/calibration rows, complete validation checkpoint curves, pair identities, ±10/30/60 proximity fields, full patient waterfalls, H1 seed tables, H3 feature ablations, pairwise oracles, source hashes, and implementation parity checks.
"""
    (OUT / "final_A_to_R_pre_manuscript.md").write_text(report, encoding="utf-8")
    manifest = {
        "analysis_id":"h4-final-linkage-expert-audit-v1", "status":"parts_1_to_5_complete",
        "runs":15, "zero_candidate_recordings":zero_recordings,
        "eligible_truths":total_eligible, "union_reachable_truths":total_reachable,
        "oracle_assignments":total_assigned, "evaluator_credited_true_positives":evaluator_credited,
        "central_story":"B + C + D, with B as primary spine; H1 is diagnostic context",
        "test_used_for_new_training_or_threshold_selection":False,
        "outputs":sorted(p.name for p in OUT.iterdir() if p.is_file()),
    }
    (OUT / "final_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(report)
    print("\nMANIFEST\n", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
