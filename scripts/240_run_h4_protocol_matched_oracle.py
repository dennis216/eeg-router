"""Protocol-matched hindsight oracle and complementarity audit for standard H4.

This script is deliberately downstream of the frozen
``v3-groupkfold-confirmatory-v1`` H4 evaluation. It never changes a detector,
threshold, split, router, or test prediction. It reconstructs the actual H4
action space: detector events are grouped by the frozen +/-30 s onset rule and
the policy may select at most one existing member of each group or abstain.

Ground truth is used only after the fact to select among those available
actions. The oracle cannot create a detector event, alter a validation-locked
expert threshold, or recover a seizure no candidate group can reach. Selected
actions are scored by the unchanged H4 event matcher.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


ROOT = Path("/root/b_false_alarm_atlas")
RUN_ID = "v3-groupkfold-confirmatory-v1"
OUT = ROOT / "outputs/reports/h4-protocol-matched-oracle-v2"
SOURCE = ROOT / "outputs/reports" / RUN_ID
EVENT_FEATURES = SOURCE / "h4_groupkfold_event_features.parquet"
H4_TEST = SOURCE / "h4_groupkfold_router_test.csv"
H4_VALIDATION = SOURCE / "h4_groupkfold_router_validation.csv"
H4_MANIFEST = SOURCE / "h4_groupkfold_router_manifest.json"
SPLIT_MANIFEST = ROOT / "manifests/groupkfold_cv_v1/cv_manifest.json"
SEIZURES = ROOT / "manifests/seizures.parquet"
RECORDINGS = ROOT / "manifests/recordings.parquet"
MODELS = ("singlem", "cbramod", "tcn_gat")
N_BOOTSTRAP = 2000
RNG_SEED = 20260904
MINIMUM_OVERLAP_S = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_h4_helpers():
    path = ROOT / "scripts/84_run_groupkfold_h4_router.py"
    spec = importlib.util.spec_from_file_location("h4_helpers_for_oracle_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def overlap(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "patient",
            "recording",
            "event_start",
            "event_end",
            "peak_probability",
            "selected_model",
            "group_id",
        ]
    )


def action_table(frame: pd.DataFrame, h4, models: tuple[str, ...]) -> pd.DataFrame:
    """Materialize every action the real H4 policy could take."""
    candidates = h4._candidate_groups(frame[frame.model.astype(str).isin(models)].copy())
    if candidates.empty:
        return empty_events()
    rows: list[dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        rows.append(
            {
                "patient": str(row.patient),
                "recording": str(row.recording),
                "event_start": float(row.start_s),
                # The real router uses the maximum end of the whole group.
                "event_end": float(row.group_decision_end),
                "peak_probability": float(row.peak_probability),
                "selected_model": str(row.source_model),
                "group_id": str(row.group_id),
                "source_event_start": float(row.start_s),
                "source_event_end": float(row.end_s),
                "source_event_type": str(row.event_type),
            }
        )
    return pd.DataFrame(rows)


def oracle_recording(
    actions: pd.DataFrame,
    truths: list[tuple[float, float]],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Maximize reachable truths with one member or abstention per group."""
    if actions.empty or not truths:
        return empty_events(), []
    groups = [group.copy() for _, group in actions.groupby("group_id", sort=True)]
    group_count = len(groups)
    truth_count = len(truths)
    # Truth columns plus a private abstention column for every group.
    costs = np.zeros((group_count, truth_count + group_count), dtype=float)
    costs[:, :truth_count] = 1e9
    best_member: dict[tuple[int, int], int] = {}
    cardinality_bonus = 1e6
    for group_index, group in enumerate(groups):
        for truth_index, (truth_start, truth_end) in enumerate(truths):
            choices: list[tuple[float, float, float, int]] = []
            for row_index, row in group.iterrows():
                amount = overlap(
                    float(row.event_start), float(row.event_end), truth_start, truth_end
                )
                if amount >= MINIMUM_OVERLAP_S:
                    delay = max(0.0, float(row.event_start) - truth_start)
                    choices.append((amount, -delay, float(row.peak_probability), int(row_index)))
            if choices:
                amount, negative_delay, peak, row_index = max(choices)
                best_member[(group_index, truth_index)] = row_index
                costs[group_index, truth_index] = -(
                    cardinality_bonus + amount + 1e-3 * negative_delay + 1e-6 * peak
                )
    assigned_rows, assigned_columns = linear_sum_assignment(costs)
    selected: list[dict[str, object]] = []
    assignments: list[dict[str, object]] = []
    for group_index, column in zip(assigned_rows, assigned_columns, strict=True):
        if column >= truth_count or (group_index, column) not in best_member:
            continue
        row = actions.loc[best_member[(group_index, column)]]
        truth_start, truth_end = truths[column]
        selected.append(
            {
                "patient": str(row.patient),
                "recording": str(row.recording),
                "event_start": float(row.event_start),
                "event_end": float(row.event_end),
                "peak_probability": float(row.peak_probability),
                "selected_model": str(row.selected_model),
                "group_id": str(row.group_id),
            }
        )
        assignments.append(
            {
                "recording": str(row.recording),
                "group_id": str(row.group_id),
                "truth_index": int(column),
                "truth_start": float(truth_start),
                "truth_end": float(truth_end),
                "selected_model": str(row.selected_model),
                "overlap_s": overlap(
                    float(row.event_start), float(row.event_end), truth_start, truth_end
                ),
            }
        )
    return pd.DataFrame(selected) if selected else empty_events(), assignments


def protocol_oracle(
    frame: pd.DataFrame,
    h4,
    seizures: pd.DataFrame,
    evaluation_starts: dict[str, float],
    models: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    actions = action_table(frame, h4, models)
    selected_frames: list[pd.DataFrame] = []
    assignment_rows: list[dict[str, object]] = []
    for recording, start in evaluation_starts.items():
        rec_actions = actions[actions.recording.astype(str) == str(recording)]
        truths = h4._truths(seizures, str(recording), float(start))
        selected, assignments = oracle_recording(rec_actions, truths)
        selected_frames.append(selected)
        assignment_rows.extend(assignments)
    selected_all = pd.concat(selected_frames, ignore_index=True) if selected_frames else empty_events()
    return selected_all, actions, pd.DataFrame(assignment_rows)


def load_probability_table(fold: int, seed: int, model: str) -> pd.DataFrame:
    path = ROOT / "runs" / RUN_ID / model / f"split{fold}_seed{seed}_main/test/probabilities.parquet"
    return pd.read_parquet(path)[["patient", "recording", "end", "probability"]].copy()


def validation_row(validation: pd.DataFrame, fold: int, seed: int, method: str) -> pd.Series:
    row = validation[
        (validation.fold == fold) & (validation.seed == seed) & (validation.method == method)
    ]
    if len(row) != 1:
        raise RuntimeError(f"expected one validation row fold={fold} seed={seed} method={method}")
    return row.iloc[0]


def materialize_frozen_policies(
    frame: pd.DataFrame,
    fold: int,
    seed: int,
    validation: pd.DataFrame,
    h4,
) -> dict[str, pd.DataFrame]:
    candidates = h4._candidate_groups(frame)
    best = validation_row(validation, fold, seed, "best_single")
    policies: dict[str, pd.DataFrame] = {
        "best_single": h4._event_rows_from_model(frame, str(best.selected_model))
    }
    tables = {model: load_probability_table(fold, seed, model) for model in MODELS}
    for method in ("equal_weight_average", "validation_fixed_weights"):
        row = validation_row(validation, fold, seed, method)
        weights = tuple(float(row[f"weight_{model}"]) for model in MODELS)
        policies[method] = h4._combined_events(tables, weights, float(row.threshold))
    for method, aware in (("router_model_aware", True), ("router_no_model_id", False)):
        row = validation_row(validation, fold, seed, method)
        model_path = SOURCE / f"h4_{method}_fold{fold}_seed{seed}.joblib"
        model = joblib.load(model_path)
        matrix, _ = h4._router_matrix(candidates, aware)
        scores = model.predict_proba(matrix)[:, 1]
        policies[method] = h4._router_select(candidates, scores, float(row.threshold))
    return policies


def mask_name(active: set[str]) -> str:
    return "+".join(model for model in MODELS if model in active) if active else "none"


def decompose_truths(
    frame: pd.DataFrame,
    evaluation_starts: dict[str, float],
    seizures: pd.DataFrame,
    h4,
    fold: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for recording, start in evaluation_starts.items():
        truth_intervals = h4._truths(seizures, str(recording), float(start))
        rec = frame[frame.recording.astype(str) == str(recording)]
        patient = str(rec.patient.iloc[0]) if not rec.empty else None
        for truth_index, (truth_start, truth_end) in enumerate(truth_intervals):
            detected: set[str] = set()
            for model in MODELS:
                member = rec[rec.model.astype(str) == model]
                if any(
                    overlap(float(row.start_s), float(row.end_s), truth_start, truth_end)
                    >= MINIMUM_OVERLAP_S
                    for row in member.itertuples(index=False)
                ):
                    detected.add(model)
            rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "patient": patient,
                    "recording": str(recording),
                    "truth_index": truth_index,
                    "detector_mask": mask_name(detected),
                    "detector_count": len(detected),
                    **{f"detected_{model}": model in detected for model in MODELS},
                }
            )
    return pd.DataFrame(rows)


def decompose_decision_groups(
    actions: pd.DataFrame,
    evaluation_starts: dict[str, float],
    seizures: pd.DataFrame,
    h4,
    fold: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_id, group in actions.groupby("group_id", sort=True):
        recording = str(group.recording.iloc[0])
        truths = h4._truths(seizures, recording, evaluation_starts[recording])
        reaches_truth = any(
            overlap(
                float(action.event_start),
                float(action.event_end),
                float(truth_start),
                float(truth_end),
            )
            >= MINIMUM_OVERLAP_S
            for action in group.itertuples(index=False)
            for truth_start, truth_end in truths
        )
        active = set(group.selected_model.astype(str))
        rows.append(
            {
                "fold": fold,
                "seed": seed,
                "patient": str(group.patient.iloc[0]),
                "recording": recording,
                "group_id": str(group_id),
                "group_kind": "seizure_reachable" if reaches_truth else "background",
                "detector_mask": mask_name(active),
                "detector_count": len(active),
                **{f"alarmed_{model}": model in active for model in MODELS},
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap(patient_average: pd.DataFrame) -> pd.DataFrame:
    baseline = patient_average[patient_average.method == "best_single"].set_index("patient")
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(RNG_SEED)
    for method in [m for m in patient_average.method.unique() if m != "best_single"]:
        comparison = patient_average[patient_average.method == method].set_index("patient")
        common = np.asarray(sorted(set(baseline.index) & set(comparison.index)))
        delta_sensitivity = np.empty(N_BOOTSTRAP)
        delta_fa = np.empty(N_BOOTSTRAP)
        for replicate in range(N_BOOTSTRAP):
            chosen = common[rng.integers(0, len(common), size=len(common))]
            delta_sensitivity[replicate] = float(
                (comparison.loc[chosen].sensitivity.to_numpy() - baseline.loc[chosen].sensitivity.to_numpy()).mean()
            )
            delta_fa[replicate] = float(
                (comparison.loc[chosen].fa_per_24h.to_numpy() - baseline.loc[chosen].fa_per_24h.to_numpy()).mean()
            )
        rows.append(
            {
                "method": method,
                "patients": len(common),
                "delta_sensitivity_mean": float(
                    (comparison.loc[common].sensitivity - baseline.loc[common].sensitivity).mean()
                ),
                "delta_sensitivity_q025": float(np.quantile(delta_sensitivity, 0.025)),
                "delta_sensitivity_q975": float(np.quantile(delta_sensitivity, 0.975)),
                "delta_fa_per_24h_mean": float(
                    (comparison.loc[common].fa_per_24h - baseline.loc[common].fa_per_24h).mean()
                ),
                "delta_fa_per_24h_q025": float(np.quantile(delta_fa, 0.025)),
                "delta_fa_per_24h_q975": float(np.quantile(delta_fa, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def make_figure(decision: pd.DataFrame, patient_average: pd.DataFrame) -> None:
    wanted = decision[
        decision.policy.isin(["best_single", "router_model_aware", "router_no_model_id", "oracle_3"])
    ].copy()
    colors = {
        "best_single": "#34495E",
        "router_model_aware": "#C44E52",
        "router_no_model_id": "#DD8452",
        "oracle_3": "#1B9E77",
    }
    labels = {
        "best_single": "Best single",
        "router_model_aware": "Model-aware router",
        "router_no_model_id": "No-model-ID router",
        "oracle_3": "Protocol-matched oracle",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1.05, 1.35]})
    ax = axes[0]
    for row in wanted.itertuples(index=False):
        ax.scatter(row.fa_per_24h, row.sensitivity, s=85, color=colors[row.policy], zorder=3)
        ax.annotate(labels[row.policy], (row.fa_per_24h, row.sensitivity), xytext=(6, 5), textcoords="offset points", fontsize=8)
    base = wanted[wanted.policy == "best_single"].iloc[0]
    for target in ("router_model_aware", "oracle_3"):
        row = wanted[wanted.policy == target].iloc[0]
        ax.annotate("", xy=(row.fa_per_24h, row.sensitivity), xytext=(base.fa_per_24h, base.sensitivity), arrowprops={"arrowstyle": "->", "color": colors[target], "lw": 1.5})
    ax.set_xlabel("False alarms / 24 h (lower is better)")
    ax.set_ylabel("Event sensitivity (higher is better)")
    ax.grid(alpha=0.2)
    ax.set_title("Policy operating points")

    ax = axes[1]
    base_patient = patient_average[patient_average.method == "best_single"].set_index("patient")
    oracle_patient = patient_average[patient_average.method == "oracle_3"].set_index("patient")
    common = sorted(set(base_patient.index) & set(oracle_patient.index))
    waterfall = pd.DataFrame(
        {
            "patient": common,
            "delta_fa": [oracle_patient.loc[p].fa_per_24h - base_patient.loc[p].fa_per_24h for p in common],
        }
    ).sort_values("delta_fa")
    bar_colors = ["#1B9E77" if value < 0 else "#C44E52" for value in waterfall.delta_fa]
    ax.bar(np.arange(len(waterfall)), waterfall.delta_fa, color=bar_colors, width=0.8)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(np.arange(len(waterfall)))
    ax.set_xticklabels(waterfall.patient, rotation=65, ha="right", fontsize=7)
    ax.set_ylabel("Oracle - best single FA / 24 h")
    ax.set_title("Patient-level actionable complementarity")
    ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Can a perfect H4-compatible selector exploit detector diversity?", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "figure_h4_best_router_oracle.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "figure_h4_best_router_oracle.pdf", bbox_inches="tight")
    plt.close(fig)


def frame_to_markdown(frame: pd.DataFrame, digits: int | None = None) -> str:
    """Small dependency-free Markdown renderer for audit tables."""
    def render(value: object) -> str:
        if pd.isna(value):
            return "NA"
        if digits is not None and isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value).replace("|", "\\|")

    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    decision: pd.DataFrame,
    pair_summary: pd.DataFrame,
    truth_summary: pd.DataFrame,
    background_summary: pd.DataFrame,
    patient_average: pd.DataFrame,
    boot: pd.DataFrame,
) -> None:
    by_policy = decision.set_index("policy")
    best = by_policy.loc["best_single"]
    oracle = by_policy.loc["oracle_3"]
    aware = by_policy.loc["router_model_aware"]
    no_id = by_policy.loc["router_no_model_id"]
    oracle_fa_reduction = float(best.fa_per_24h - oracle.fa_per_24h)
    oracle_sensitivity_gain = float(oracle.sensitivity - best.sensitivity)

    base_patient = patient_average[patient_average.method == "best_single"].set_index("patient")
    oracle_patient = patient_average[patient_average.method == "oracle_3"].set_index("patient")
    common = sorted(set(base_patient.index) & set(oracle_patient.index))
    patient_delta = pd.DataFrame(
        {
            "patient": common,
            "delta_sensitivity": [oracle_patient.loc[p].sensitivity - base_patient.loc[p].sensitivity for p in common],
            "delta_fa": [oracle_patient.loc[p].fa_per_24h - base_patient.loc[p].fa_per_24h for p in common],
        }
    )
    heterogeneous = bool(
        patient_delta.delta_sensitivity.std(ddof=0) >= 0.10
        or patient_delta.delta_fa.std(ddof=0) >= 0.25 * max(float(best.fa_per_24h), 1.0)
    )
    if oracle_fa_reduction < 0.10 * float(best.fa_per_24h) and oracle_sensitivity_gain < 0.03:
        case = "A — limited protocol-matched oracle gain"
        interpretation = "The evaluated detector events contain limited actionable complementarity under H4."
    elif oracle_sensitivity_gain >= 0.03 and oracle_fa_reduction < 0.10 * float(best.fa_per_24h):
        case = "C — sensitivity complementarity without material false-alarm reduction"
        interpretation = "Detector diversity adds reachable seizures but does not materially relieve alarm burden."
    elif heterogeneous and oracle_sensitivity_gain >= 0.03 and oracle_fa_reduction >= 0.10 * float(best.fa_per_24h):
        case = "B + D — large oracle ceiling, poor learned routing, and patient heterogeneity"
        interpretation = (
            "The frozen detectors contain substantial hindsight-selectable headroom, "
            "but the evaluated router fails to exploit it safely and the available "
            "headroom varies materially by patient."
        )
    elif heterogeneous:
        case = "D — strong patient heterogeneity"
        interpretation = "The amount of actionable complementarity varies materially by patient."
    else:
        case = "B — large oracle ceiling but poor learned routing"
        interpretation = "Useful actions exist in hindsight, but the evaluated H4 policy does not identify them reliably."

    deployable = decision.copy()
    deployable["deployable"] = deployable.deployable.map({True: "yes", False: "no"})
    decision_md = frame_to_markdown(deployable, digits=4)
    pair_md = frame_to_markdown(pair_summary, digits=4)
    truth_md = frame_to_markdown(truth_summary)
    background_md = frame_to_markdown(background_summary)
    patient_quantiles = patient_delta[["delta_sensitivity", "delta_fa"]].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).reset_index()
    patient_md = frame_to_markdown(patient_quantiles, digits=4)
    bootstrap_md = frame_to_markdown(boot, digits=4)

    report = f"""# H4 protocol-matched oracle and complementarity audit

## A. H4 protocol audit

- Decision unit: detector alarm events grouped within the frozen `+/-30 s` onset rule, not raw windows.
- Action: select at most one existing candidate member per group, or emit no event when every candidate score is below the validation-selected router threshold.
- The selected member supplies the event start; the maximum end time across the group supplies the event end.
- Expert events are already formed from validation-thresholded window probabilities using causal EMA `1/3`, minimum duration `4 s`, merge gap `10 s`, and refractory `30 s`.
- No second eventization or refractory pass is applied after router selection.
- Event matching is one-to-one Hungarian matching with at least `1 s` interval overlap.
- Sensitivity is matched seizure events / eligible truth events. FA/24h is unmatched predicted events x 24 / non-seizure monitoring hours.
- Outer test patients are held out by canonical-patient GroupKFold. Router fitting uses outer-train discovery; workpoint selection uses validation; test is evaluated once.

## B. Existing oracle audit

The prior `h4_groupkfold_validation_oracle.csv` is a validation-only grid search over fixed convex score combinations. It is a valid source for the deployable validation-fixed-weights baseline, but it is not a hindsight selector with the learned router's group-level action space. Compact/legacy oracle files are incompatible with the standard H4 protocol and remain excluded.

## C-D. New oracle definition and validity

For each event-candidate group, the oracle may select one existing member action or abstain, exactly like H4. Across a recording it computes a maximum-cardinality one-to-one assignment between candidate groups and reachable truth events. Unassigned groups abstain. It cannot create an event, alter an expert threshold, or recover an unreachable seizure. The resulting events pass through the unchanged H4 scorer.

Because H4 contains an abstention action, a hindsight oracle can reject every unassigned background group. Consequently, the oracle FA result is an action-space ceiling; the scientifically informative quantities are reachable seizure coverage, shared/unique event structure, patient heterogeneity, and the gap to learned routing. It is not deployable performance.

## E-G. Best single, learned routers, and oracle

{decision_md}

## H. Pairwise complementarity

{pair_md}

## I. Patient-level heterogeneity

Patient-level oracle-minus-best-single quantiles:

{patient_md}

Paired patient bootstrap:

{bootstrap_md}

## J. Shared-versus-unique seizure structure

{truth_md}

## K. Shared-versus-unique background alarm structure

{background_md}

## L. Relationship to H1

H1 and this oracle answer different questions. H1 measures whether false-alarm timing is more shared than an appropriate time-shift null. The oracle measures how much the frozen H4 action space could achieve under perfect hindsight selection. H1 cannot by itself establish that temporal sharing caused router failure.

## M. Relationship to H3

H3 source identifiability is not actionable complementarity. A candidate event can retain detector-source information while every available action at that time remains unsafe. The oracle gap determines whether source-dependent structure can, in principle, map to a better H4 action.

## N. Unsupported causal explanations

- Shared physiological artifact caused all detector errors.
- Pretraining caused the shared error structure.
- H1 temporal sharing is the sole cause of H4 failure.
- Source-identifiable errors are necessarily routable.
- All mixture-of-experts policies must fail.

## O. Supported interpretation

**Selected outcome: {case}.** {interpretation}

The H4 negative result remains primary. The oracle only determines whether the failure is principally an action-space ceiling or a failure of the evaluated features, objective, calibration, or policy to exploit available actions.

## P-Q. Brain & Body storyline and required main-text evidence

The Brain & Body framing should present pretrained and task-specific EEG detectors accurately and focus on patient-held-out, deployment-relevant evaluation. The five-page main text must retain: the H4 negative operating point, the protocol-matched oracle gap, the shared/unique decomposition needed to interpret that gap, and one compact patient-heterogeneity panel when heterogeneity is material.

## R. Appendix analyses

Full fold/seed tables, pairwise masks, all patient rows, bootstrap details, parity checks, source hashes, grouping implementation, and exhaustive H1/H3 control tables belong in the appendix. Central H4 negative evidence does not.
"""
    (OUT / "audit_A_to_R.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    h4 = load_h4_helpers()
    features = pd.read_parquet(EVENT_FEATURES)
    validation = pd.read_csv(H4_VALIDATION)
    frozen_test = pd.read_csv(H4_TEST)
    seizures = pd.read_parquet(SEIZURES)
    recordings = pd.read_parquet(RECORDINGS)
    durations = recordings.set_index("recording_id")["duration_s"].astype(float).to_dict()

    score_rows: list[dict[str, object]] = []
    patient_rows: list[pd.DataFrame] = []
    pair_rows: list[dict[str, object]] = []
    truth_frames: list[pd.DataFrame] = []
    decision_group_frames: list[pd.DataFrame] = []
    assignment_frames: list[pd.DataFrame] = []
    parity_rows: list[dict[str, object]] = []

    test_features = features[features.partition.astype(str) == "test"]
    for (fold, seed), frame in test_features.groupby(["split", "seed"], sort=True):
        fold = int(fold)
        seed = int(seed)
        frame = frame.copy()
        starts, patients = h4._evaluation_meta(frame)
        frozen_policies = materialize_frozen_policies(frame, fold, seed, validation, h4)
        for method, events in frozen_policies.items():
            score = h4._score_events(events, starts, patients, seizures, durations)
            score_rows.append({"fold": fold, "seed": seed, "policy": method, **score})
            patient = h4._patient_stats(events, starts, patients, seizures, durations)
            patient["fold"] = fold
            patient["seed"] = seed
            patient["method"] = method
            patient_rows.append(patient)
            frozen = frozen_test[(frozen_test.fold == fold) & (frozen_test.seed == seed) & (frozen_test.method == method)]
            if len(frozen) != 1:
                raise RuntimeError(f"missing frozen H4 result {fold=} {seed=} {method=}")
            frozen = frozen.iloc[0]
            parity_rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "method": method,
                    "sensitivity_abs_error": abs(float(score["sensitivity"]) - float(frozen.sensitivity)),
                    "fa_per_24h_abs_error": abs(float(score["fa_per_24h"]) - float(frozen.fa_per_24h)),
                }
            )

        all_oracle, all_actions, assignments = protocol_oracle(frame, h4, seizures, starts, MODELS)
        all_score = h4._score_events(all_oracle, starts, patients, seizures, durations)
        score_rows.append({"fold": fold, "seed": seed, "policy": "oracle_3", **all_score})
        patient = h4._patient_stats(all_oracle, starts, patients, seizures, durations)
        patient["fold"] = fold
        patient["seed"] = seed
        patient["method"] = "oracle_3"
        patient_rows.append(patient)
        assignments["fold"] = fold
        assignments["seed"] = seed
        assignment_frames.append(assignments)

        for left_index in range(len(MODELS)):
            for right_index in range(left_index + 1, len(MODELS)):
                pair = (MODELS[left_index], MODELS[right_index])
                selected, actions, _ = protocol_oracle(frame, h4, seizures, starts, pair)
                score = h4._score_events(selected, starts, patients, seizures, durations)
                pair_rows.append({"fold": fold, "seed": seed, "models": "+".join(pair), **score, "candidate_groups": int(actions.group_id.nunique())})

        truth_frames.append(decompose_truths(frame, starts, seizures, h4, fold, seed))
        decision_group_frames.append(decompose_decision_groups(all_actions, starts, seizures, h4, fold, seed))

    scores = pd.DataFrame(score_rows)
    patients_long = pd.concat(patient_rows, ignore_index=True)
    pairs = pd.DataFrame(pair_rows)
    truths = pd.concat(truth_frames, ignore_index=True)
    decision_groups = pd.concat(decision_group_frames, ignore_index=True)
    assignments = pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame()
    parity = pd.DataFrame(parity_rows)
    parity_max = float(parity[["sensitivity_abs_error", "fa_per_24h_abs_error"]].to_numpy().max())
    if parity_max > 1e-9:
        raise RuntimeError("reconstructed policy metrics do not match frozen H4 outputs")

    patient_average = (
        patients_long.groupby(["method", "patient"], as_index=False)[["sensitivity", "fa_per_24h", "mean_delay_s", "tp", "fp", "truth"]]
        .mean()
    )
    boot = paired_bootstrap(patient_average)

    decision_rows: list[dict[str, object]] = []
    baseline_mean = scores[scores.policy == "best_single"][["sensitivity", "fa_per_24h"]].mean()
    oracle_mean = scores[scores.policy == "oracle_3"][["sensitivity", "fa_per_24h"]].mean()
    for policy, group in scores.groupby("policy", sort=False):
        sensitivity = float(group.sensitivity.mean())
        fa = float(group.fa_per_24h.mean())
        decision_rows.append(
            {
                "policy": policy,
                "decision_granularity": "frozen detector" if policy == "best_single" else "window-score fusion" if policy in {"equal_weight_average", "validation_fixed_weights"} else "+/-30 s event-candidate group",
                "sensitivity": sensitivity,
                "fa_per_24h": fa,
                "delta_sensitivity_vs_best_pp": 100 * (sensitivity - float(baseline_mean.sensitivity)),
                "delta_fa_vs_best": fa - float(baseline_mean.fa_per_24h),
                "delta_sensitivity_vs_oracle_pp": 100 * (sensitivity - float(oracle_mean.sensitivity)),
                "delta_fa_vs_oracle": fa - float(oracle_mean.fa_per_24h),
                "deployable": policy != "oracle_3",
            }
        )
    decision = pd.DataFrame(decision_rows)

    truth_summary = truths.groupby("detector_mask", as_index=False).size().rename(columns={"size": "truth_events"})
    background_summary = (
        decision_groups[decision_groups.group_kind == "background"]
        .groupby("detector_mask", as_index=False)
        .size()
        .rename(columns={"size": "background_candidate_groups"})
    )
    pair_summary = pairs.groupby("models", as_index=False)[["sensitivity", "fa_per_24h"]].mean()

    scores.to_csv(OUT / "policy_per_run.csv", index=False)
    decision.to_csv(OUT / "decision_table.csv", index=False)
    pairs.to_csv(OUT / "pairwise_oracle_per_run.csv", index=False)
    pair_summary.to_csv(OUT / "pairwise_oracle_summary.csv", index=False)
    patients_long.to_csv(OUT / "patient_policy_metrics_seed.csv", index=False)
    patient_average.to_csv(OUT / "patient_policy_metrics.csv", index=False)
    boot.to_csv(OUT / "patient_paired_bootstrap.csv", index=False)
    truths.to_csv(OUT / "truth_event_detector_masks.csv", index=False)
    truth_summary.to_csv(OUT / "truth_event_detector_mask_summary.csv", index=False)
    decision_groups.to_csv(OUT / "decision_group_structure.csv", index=False)
    background_summary.to_csv(OUT / "background_detector_mask_summary.csv", index=False)
    assignments.to_csv(OUT / "oracle_assignments.csv", index=False)
    parity.to_csv(OUT / "frozen_metric_parity.csv", index=False)
    make_figure(decision, patient_average)
    write_report(decision, pair_summary, truth_summary, background_summary, patient_average, boot)

    manifest = {
        "analysis_id": "h4-protocol-matched-oracle-v2",
        "status": "complete_audit_only_non_deployable",
        "source_protocol": RUN_ID,
        "decision_unit": "+/-30 s onset-grouped detector events",
        "action_space": "select at most one existing member event per group or abstain",
        "oracle_objective": "maximize reachable truth-event cardinality; then overlap/delay/peak tie-breakers; abstain on unassigned groups",
        "ground_truth_use": "hindsight action selection only",
        "cannot_invent_predictions": True,
        "expert_threshold_source": "unchanged validation-locked thresholds",
        "test_used_for_training_or_threshold_selection": False,
        "event_evaluator": "unchanged H4 _score_events and match_events(minimum_overlap_s=1)",
        "policy_metric_parity_max_error": parity_max,
        "coverage_audit": {
            "h4_evaluated_recordings": int(test_features.recording.nunique()),
            "all_manifest_recordings": int(recordings.recording_id.nunique()),
            "note": "H4 metadata come from recordings having at least one event row; zero-candidate recordings are absent from H4 scoring.",
        },
        "source_hashes": {
            "event_features": sha256(EVENT_FEATURES),
            "h4_test": sha256(H4_TEST),
            "h4_validation": sha256(H4_VALIDATION),
            "h4_manifest": sha256(H4_MANIFEST),
            "split_manifest": sha256(SPLIT_MANIFEST),
            "seizures": sha256(SEIZURES),
            "recordings": sha256(RECORDINGS),
        },
        "runs": int(scores[scores.policy == "oracle_3"].shape[0]),
        "patient_bootstrap_replicates": N_BOOTSTRAP,
        "outputs": sorted(path.name for path in OUT.iterdir() if path.is_file()),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("\nDecision table:\n", decision.to_string(index=False))
    print("\nPairwise oracle:\n", pair_summary.to_string(index=False))
    print("\nTruth masks:\n", truth_summary.to_string(index=False))
    print("\nBackground masks:\n", background_summary.to_string(index=False))


if __name__ == "__main__":
    main()
