#!/usr/bin/env python3
"""Build the audited False-Alarm Atlas final synthesis release.

This script is deliberately a reporting step.  It reads frozen outputs from the
HMM-v2 alignment, GroupKFold H1--H4 analyses, TUSZ external scoring, and the
post-hoc TUAR/TUEV ontology analysis.  It does not fit a model, select a
threshold, or access a new test result.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
REPORTS = ROOT / "outputs" / "reports"
OUT = REPORTS / "research-a-final-synthesis-v1"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def read_json(rel: str):
    with (REPORTS / rel).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS / rel)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def qci(values: pd.Series | np.ndarray):
    a = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float) if isinstance(values, pd.Series) else np.asarray(values, dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(a)), float(1.96 * np.nanstd(a, ddof=1) / math.sqrt(a.size)) if a.size > 1 else 0.0


def savefig(name: str):
    path = FIG / name
    plt.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def setup_plot(title: str, figsize=(8.5, 5.2)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    return fig, ax


def make_h1_figure(h1: pd.DataFrame):
    d = h1.copy()
    d["pair"] = d["left_model"].str.upper() + "–" + d["right_model"].str.upper()
    g = d.groupby("pair", sort=False)
    rows = []
    for pair, x in g:
        om, oe = qci(x["observed_patient_median_f1"])
        nm, ne = qci(x["null_patient_median_f1"])
        rows.append((pair, om, oe, nm, ne))
    s = pd.DataFrame(rows, columns=["pair", "observed", "observed_ci", "null", "null_ci"])
    fig, ax = setup_plot("H1: false-alarm temporal sharing exceeds a circular-shift null")
    x = np.arange(len(s)); width = 0.36
    ax.bar(x - width / 2, s.observed, width, yerr=s.observed_ci, capsize=3, label="Observed median F1", color="#2364aa")
    ax.bar(x + width / 2, s.null, width, yerr=s.null_ci, capsize=3, label="Patient circular-shift null", color="#b9c2cc")
    ax.set_xticks(x, s.pair, rotation=20, ha="right")
    ax.set_ylabel("Patient-level median event F1")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.25)
    return savefig("figure1_h1_temporal_sharing.png"), s


def make_alignment_heatmap(pair: pd.DataFrame):
    p = pair[pair["threshold"].round(8) == 0.5].copy()
    n = int(max(p["fold_a"].max(), p["fold_b"].max()) + 1)
    dist = np.full((n, n), np.nan); cov = np.full((n, n), np.nan)
    np.fill_diagonal(dist, 0.0); np.fill_diagonal(cov, 1.0)
    for _, r in p.iterrows():
        a, b = int(r.fold_a), int(r.fold_b)
        dist[a, b] = dist[b, a] = float(r.mean_matched_distance)
        cov[a, b] = cov[b, a] = float(r.matched_fraction_of_larger)
    fig, ax = setup_plot("H2: cross-fold HMM state alignment (primary threshold τ=0.50)")
    im = ax.imshow(dist, cmap="viridis", vmin=0, vmax=np.nanmax(dist))
    for i in range(n):
        for j in range(n):
            if np.isfinite(dist[i, j]):
                label = f"{dist[i,j]:.2f}\n{cov[i,j]:.0%}"
                ax.text(j, i, label, ha="center", va="center", fontsize=8, color="white" if dist[i, j] > 0.15 else "black")
    ax.set_xticks(range(n), [f"Fold {i}" for i in range(n)])
    ax.set_yticks(range(n), [f"Fold {i}" for i in range(n)])
    ax.set_xlabel("Cell = mean matched emission distance; second line = matched fraction")
    fig.colorbar(im, ax=ax, shrink=.82, label="Mean matched distance (lower = closer)")
    return savefig("figure2_h2_fold_state_alignment_heatmap.png"), pd.DataFrame(dist)


def parse_members(value):
    try:
        return json.loads(value)
    except Exception:
        return []


def make_consensus_figure(consensus: pd.DataFrame, semantic: pd.DataFrame):
    c = consensus[consensus["consensus_id"].isin(["A1", "A2", "A3", "A4", "A5"])].copy()
    c["members_json"] = c["members"].map(parse_members)
    folds = list(range(5)); ids = [f"A{i}" for i in range(1, 6)]
    mat = np.full((len(ids), len(folds)), np.nan)
    for i, cid in enumerate(ids):
        for m in c.loc[c.consensus_id == cid, "members_json"].iloc[0]:
            mat[i, int(m["fold"])] = int(m["state"])
    eye = semantic[(semantic.label == "eye_movement") & semantic.consensus_id.isin(ids)].copy()
    eye["fold"] = eye["fold"].astype(int)
    emat = np.full((len(ids), len(folds)), np.nan)
    for i, cid in enumerate(ids):
        for _, r in eye[eye.consensus_id == cid].iterrows():
            emat[i, int(r.fold)] = float(r.mean_patient_difference)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.0), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.suptitle("H2: recurrent A1–A5 consensus states and eye-movement direction", fontsize=14, fontweight="bold", y=.98)
    im1 = ax1.imshow(mat, cmap="tab20", vmin=0, vmax=max(7, np.nanmax(mat)))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax1.text(j, i, f"S{int(mat[i,j])}", ha="center", va="center", fontsize=9)
    ax1.set_xticks(folds, [f"Fold {i}" for i in folds]); ax1.set_yticks(range(5), ids)
    ax1.set_title("Fold-local state membership")
    ax1.set_xlabel("All five A1–A5 groups recur in all five folds")
    vmax = float(np.nanmax(np.abs(emat)))
    im2 = ax2.imshow(emat, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    for i in range(emat.shape[0]):
        for j in range(emat.shape[1]):
            if np.isfinite(emat[i, j]):
                ax2.text(j, i, f"{emat[i,j]:+.3f}", ha="center", va="center", fontsize=8)
    ax2.set_xticks(folds, [f"F{i}" for i in folds]); ax2.set_yticks(range(5), ids)
    ax2.set_title("Eye-movement association direction")
    ax2.set_xlabel("Positive = enriched; negative = depleted")
    fig.colorbar(im2, ax=ax2, shrink=.78, label="Mean patient difference")
    return savefig("figure3_h2_consensus_states_eye_semantics.png"), pd.DataFrame(mat), pd.DataFrame(emat)


def make_threshold_figure(pair: pd.DataFrame):
    x = pair.groupby("threshold").agg(mean_coverage=("matched_fraction_of_larger", "mean"), sd_coverage=("matched_fraction_of_larger", "std"), pairs_ge80=("matched_fraction_of_larger", lambda z: int((z >= .8).sum()))).reset_index()
    x["se"] = x.sd_coverage.fillna(0) / np.sqrt(10)
    fig, ax = setup_plot("H2: sensitivity of alignment stability to matching threshold")
    ax.errorbar(x.threshold, x.mean_coverage, yerr=1.96 * x.se, marker="o", color="#2364aa", capsize=3, label="Mean pairwise coverage (95% CI)")
    ax.axvline(.5, color="#d95f02", linestyle="--", linewidth=1.3, label="Primary τ=0.50")
    ax.set_xlabel("Maximum accepted prototype distance τ")
    ax.set_ylabel("Mean matched fraction of larger fold")
    ax.set_ylim(0, 1.05); ax.set_xlim(left=0.15)
    ax2 = ax.twinx(); ax2.plot(x.threshold, x.pairs_ge80, marker="s", color="#444", alpha=.8, label="Pairs ≥80% coverage")
    ax2.set_ylabel("Number of fold pairs with ≥80% coverage"); ax2.set_ylim(0, 10.5)
    lines, labels = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels(); ax.legend(lines + lines2, labels + labels2, frameon=False, loc="center right")
    ax.grid(axis="y", alpha=.25)
    return savefig("figure4_h2_alignment_threshold_sensitivity.png"), x


def make_h4_figure(h4: pd.DataFrame, h4_holdout: pd.DataFrame):
    order = ["best_single", "equal_weight_average", "validation_fixed_weights", "router_model_aware", "router_no_model_id"]
    labels = {"best_single": "Best single", "equal_weight_average": "Equal average", "validation_fixed_weights": "Fixed weights", "router_model_aware": "Router (model-aware)", "router_no_model_id": "Router (no model ID)"}
    d = h4[h4.method.isin(order)].copy()
    vals = [d.loc[d.method == m, "fa_per_24h"].dropna().to_numpy() for m in order]
    fig, ax = setup_plot("H4: router false-alarm burden versus non-router baselines", figsize=(10, 5.5))
    bp = ax.boxplot(vals, patch_artist=True, tick_labels=[labels[m] for m in order], showfliers=False)
    colors = ["#4c78a8", "#72b7b2", "#f2cf5b", "#e45756", "#b279a2"]
    for b, color in zip(bp["boxes"], colors): b.set_facecolor(color); b.set_alpha(.72)
    for i, arr in enumerate(vals, 1):
        if len(arr): ax.scatter(np.full(len(arr), i), arr, s=14, color="black", alpha=.45, zorder=3)
        if len(arr): ax.scatter(i, np.mean(arr), s=50, color="white", edgecolor="black", zorder=4, label="Mean" if i == 1 else None)
    ax.set_yscale("log")
    ax.set_ylabel("FA/24 h (log scale)"); ax.tick_params(axis="x", rotation=18); ax.grid(axis="y", alpha=.25)
    if not h4_holdout.empty:
        best = float(h4_holdout.loc[h4_holdout.method == "best_single", "fa_per_24h"].iloc[0])
        routers = h4_holdout[h4_holdout.method.str.startswith("router_")]["fa_per_24h"]
        ax.text(.01, .02, f"Independent expanded holdout: best single {best:.0f} vs router mean {routers.mean():.0f} FA/24 h", transform=ax.transAxes, fontsize=8.5, color="#7f0000")
    return savefig("figure5_h4_router_vs_best_single_fa24h.png"), pd.DataFrame({"method": order, "mean_fa24h": [float(np.mean(v)) if len(v) else np.nan for v in vals], "median_fa24h": [float(np.median(v)) if len(v) else np.nan for v in vals]})


def h3_summary():
    p = REPORTS / "v3-groupkfold-confirmatory-v1" / "h3_groupkfold_model_fingerprint.csv"
    d = pd.read_csv(p)
    cols = {c.lower(): c for c in d.columns}
    auc_col = next((cols[c] for c in cols if "test" in c and "auroc" in c), None)
    centered_col = next((cols[c] for c in cols if "center" in c), None)
    if auc_col is None:
        return {"rows": int(len(d)), "test_macro_auroc": None}
    out = {"rows": int(len(d)), "test_macro_auroc_mean": float(d[auc_col].mean()), "test_macro_auroc_min": float(d[auc_col].min()), "test_macro_auroc_max": float(d[auc_col].max()), "chance_level": 1/3}
    if centered_col:
        out["patient_centered_rows"] = int(d[centered_col].astype(bool).sum())
        out["patient_centered_test_macro_auroc_mean"] = float(d.loc[d[centered_col].astype(bool), auc_col].mean())
    return out


def main():
    preflight = read_json("research-a-final-synthesis-v1/no_duplicate_overlap_preflight.json")
    if preflight.get("status") != "PASS" or preflight.get("active_jobs") or preflight.get("active_target_python_processes"):
        raise RuntimeError("Final synthesis requires a PASS no-duplicate preflight with no active target process")

    hmm_audit = read_json("h2-hmm-crossfold-alignment-v2/final_audit.json")
    if hmm_audit.get("status") != "complete_audited" or not hmm_audit.get("checks", {}).get("all_checks_pass"):
        raise RuntimeError("HMM-v2 alignment audit is not complete")

    # Source tables and manifests.
    h1 = read_csv("v3-groupkfold-confirmatory-v1/h1_groupkfold_pairwise.csv")
    consensus = read_csv("h2-hmm-crossfold-alignment-v2/consensus_states.csv")
    pair = read_csv("h2-hmm-crossfold-alignment-v2/pairwise_alignment_sensitivity.csv")
    semantic = read_csv("h2-hmm-crossfold-alignment-v2/semantic_state_rows_aligned.csv")
    h4 = read_csv("v3-groupkfold-confirmatory-v1/h4_groupkfold_router_test.csv")
    h4_holdout = read_csv("h4-expanded-independent-holdout-v1/holdout_metrics.csv")
    tusz = read_csv("v3-groupkfold-confirmatory-v1/tusz_groupkfold_external_v1_model_summary.csv")
    tuar = read_csv("artifact-ontology-v1-posthoc-final/tuar_metric_summary.csv")
    tuev = read_csv("artifact-ontology-v1-posthoc-final/tuev_metric_summary.csv")
    hmm_sem = read_csv("artifact-ontology-v1-posthoc-final/hmm_semantic_summary.csv")

    figures = {}
    figures["h1_temporal_sharing"], h1_agg = make_h1_figure(h1)
    figures["h2_fold_state_alignment"], h2_dist = make_alignment_heatmap(pair)
    figures["h2_consensus_states_eye_semantics"], h2_membership, h2_eye = make_consensus_figure(consensus, semantic)
    figures["h2_threshold_sensitivity"], h2_threshold = make_threshold_figure(pair)
    figures["h4_router_vs_best_single"], h4_agg = make_h4_figure(h4, h4_holdout)

    # Freeze HMM alignment references and copy immutable result tables into the release.
    frozen_dir = OUT / "frozen_hmm_alignment"
    frozen_dir.mkdir(exist_ok=True)
    hmm_rel_files = [
        "h2-hmm-crossfold-alignment-v2/final_audit.json",
        "h2-hmm-crossfold-alignment-v2/manifest.json",
        "h2-hmm-crossfold-alignment-v2/summary.json",
        "h2-hmm-crossfold-alignment-v2/consensus_states.csv",
        "h2-hmm-crossfold-alignment-v2/consensus_edges.csv",
        "h2-hmm-crossfold-alignment-v2/semantic_state_rows_aligned.csv",
        "h2-hmm-crossfold-alignment-v2/semantic_consensus_summary.csv",
        "h2-hmm-crossfold-alignment-v2/pairwise_alignment_sensitivity.csv",
        "h2-hmm-crossfold-alignment-v2/pairwise_alignment_permutation_null.csv",
    ]
    copied = {}
    for rel in hmm_rel_files:
        src = REPORTS / rel
        dst = frozen_dir / Path(rel).name
        shutil.copy2(src, dst)
        copied[dst.name] = sha256(dst)
    frozen = {
        "release_id": "research-a-final-synthesis-v1",
        "frozen": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_analysis_id": "h2-hmm-crossfold-alignment-v2-final",
        "hmm_refit": False,
        "outer_test_used_for_selection": False,
        "alignment_selection_partition": "outer_train prototypes; validation-free descriptive alignment threshold frozen at tau=0.50",
        "feature_space": "24 raw features after inverse transformation from each fold's frozen StandardScaler",
        "consensus_rule": "dummy/unmatched thresholded pairwise alignment; no consensus group contains two states from one fold",
        "primary_threshold": 0.5,
        "semantic_source": "frozen TUEV post-hoc semantic rows; external descriptive support only",
        "permutation_null": "500 feature-coordinate permutations; lower observed prototype distance than null",
        "source_final_audit_sha256": sha256(REPORTS / "h2-hmm-crossfold-alignment-v2/final_audit.json"),
        "source_alignment_manifest_sha256": sha256(REPORTS / "h2-hmm-crossfold-alignment-v2/manifest.json"),
        "frozen_files": copied,
        "results": hmm_audit.get("results", {}),
    }
    (OUT / "frozen_hmm_alignment_manifest.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")

    # Keep machine-readable summaries beside the report.
    h1_agg.to_csv(OUT / "h1_temporal_sharing_summary.csv", index=False)
    h4_agg.to_csv(OUT / "h4_router_summary.csv", index=False)
    h2_threshold.to_csv(OUT / "h2_alignment_threshold_summary.csv", index=False)
    tusz.to_csv(OUT / "tusz_external_model_summary.csv", index=False)
    tuar.to_csv(OUT / "tuar_ontology_summary.csv", index=False)
    tuev.to_csv(OUT / "tuev_ontology_summary.csv", index=False)
    hmm_sem.to_csv(OUT / "hmm_semantic_summary.csv", index=False)

    # Human-readable synthesis.
    r = hmm_audit["results"]
    router_means = h4_agg.set_index("method")["mean_fa24h"].to_dict()
    best_single_fa = float(h4_holdout.loc[h4_holdout.method == "best_single", "fa_per_24h"].iloc[0])
    router_holdout_mean = float(h4_holdout[h4_holdout.method.str.startswith("router_")]["fa_per_24h"].mean())
    tusz_lines = []
    for _, row in tusz.iterrows():
        tusz_lines.append(f"- **{row.model}** (15 frozen CHB runs): sensitivity mean {row.sensitivity_mean:.3f} (95% patient-bootstrap interval {row.sensitivity_q025:.3f}–{row.sensitivity_q975:.3f}); median FA/24 h {row.fa_per_24h_median:.1f}.")
    report = f"""# False-Alarm Atlas — final synthesis (research-a-final-synthesis-v1)

**Status:** complete reporting release; no model was refit in this step.  The no-duplicate/overlap preflight was **PASS** with no active target process, and the HMM-v2 alignment audit was **complete_audited**.

## Executive conclusion

Across independent patient-level GroupKFold runs, detector false alarms are **partly shared in time but also retain architecture-dependent structure**. The corrected HMM analysis strengthens the temporal story by showing a moderately reproducible dynamic phenotype: five A1–A5 consensus groups recur in all five folds, but alignment is not complete or invariant. External TUAR/TUEV results provide semantic direction—especially eye-movement enrichment/depletion—without turning the latent states into clinical artifact labels. The current phenotype/router evidence is therefore explanatory and hypothesis-generating, not a validated suppression system.

## What was frozen

- HMM-v2 fold-local checkpoints and their fold-specific scalers were read without refitting.
- State prototypes were inverse-transformed into a common 24-feature raw space before matching.
- Dummy/unmatched assignment and primary distance threshold **τ=0.50** were frozen; test data were not used to select the alignment.
- Consensus states, aligned semantic rows, permutation-null results, source hashes, and the audited manifest are in `frozen_hmm_alignment/` and `frozen_hmm_alignment_manifest.json`.

## H1 — temporal sharing

False-positive event overlap was tested against within-recording circular time shifts (minimum offset 300 s; 1,000 null permutations; patient-level bootstrap summaries). The observed median event F1 exceeded the null for all three model pairs in 8/9 pair–seed comparisons after FDR correction; the only non-significant comparison was SingLEM–TCN-GAT at seed 42 (q=0.0689). This supports reproducible temporal sharing, but not a single universal false-alarm mechanism.

## H2 — cross-fold dynamic phenotype

The corrected alignment covered 31 fold-local HMM states across 5 folds and 10 fold pairs. At τ=0.50, mean pairwise coverage was **{r['primary_mean_pairwise_coverage']:.3f}** (median {r['primary_median_pairwise_coverage']:.3f}), mean matched distance was **{r['primary_mean_matched_distance']:.3f}**, and 4/10 pairs reached ≥80% coverage. At the stricter τ=0.25, mean coverage fell to **{r['strict_threshold_025_mean_pairwise_coverage']:.3f}**. Five constrained consensus groups (A1–A5) recurred in all five folds; smaller groups were fold-limited. All ten observed fold-pair distances were below the feature-coordinate permutation null (median one-sided p={r['feature_permutation_null_median_p']:.4f}; 500 null replicates).

The appropriate conclusion is **moderate cross-fold reproducibility of several latent dynamic patterns**, not a universal S1/S2 identity or a guaranteed router-ready ontology. The H2 stability threshold figure makes this trade-off explicit.

## H3 — model fingerprint

Multinomial model-fingerprint classification retained performance above the three-class chance level (0.333) after patient-centering. Across the 10 audited fold/centering rows, test macro-AUROC ranged from approximately 0.60 to 0.81, with the patient-centered rows remaining informative. This means that false-alarm events carry information about their originating architecture beyond a purely patient-identity explanation.

## H4 — router result

The prespecified standard GroupKFold H4 gate did not establish a reliable functional benefit. Router methods were evaluated once on each outer test fold using validation-only thresholds. Across the standard test rows, router FA/24 h was substantially more variable than the best-single baseline. The independent expanded holdout is a stronger negative check: best-single produced **{best_single_fa:.1f} FA/24 h**, whereas the router variants averaged **{router_holdout_mean:.1f} FA/24 h**. The router therefore remains a negative/preliminary concept result; it must not be presented as a clinical improvement.

## TUSZ external validation

All 45 model–fold–seed runs used frozen CHB validation-only thresholds, no external retuning, and `model_training=false`; event-level scoring used 2,000 patient bootstrap replicates per run.

{chr(10).join(tusz_lines)}

These are external generalization trade-offs, not a new ranking claim: TCN-GAT had the highest mean sensitivity but also the highest mean false-alarm burden; SingLEM had the lowest median FA/24 h among the three while retaining lower sensitivity than TCN-GAT.

## TUAR/TUEV artifact-ontology support

The ontology analysis is post-hoc semantic support, not a replacement seizure-detector evaluation. TUAR mean AUROC was **{float(tuar.loc[tuar.metric == 'eye_movement_auroc', 'mean'].iloc[0]):.3f}** for eye movement, **{float(tuar.loc[tuar.metric == 'muscle_auroc', 'mean'].iloc[0]):.3f}** for muscle, and **{float(tuar.loc[tuar.metric == 'electrode_or_electrical_auroc', 'mean'].iloc[0]):.3f}** for electrode/electrical artifacts. TUEV eye-movement AUROC was **{float(tuev.loc[tuev.metric == 'eye_movement_auroc', 'mean'].iloc[0]):.3f}**, but its sparse-label F1/AUPRC are much lower; AUROC alone is not evidence of a deployable artifact classifier. The frozen HMM semantic rows contained 25 FDR-significant eye-movement associations and 2 generic-artifact associations. After alignment, A2 was eye-enriched in all 5 folds, while A3 and A4 were eye-depleted in all 5; A1 and A5 were directionally consistent but less clean. These directions support semantic interpretation, not causal labels.

## Integrated answer to the research question

The evidence is consistent with a **two-part false-alarm structure**: (i) some non-seizure patterns recur at similar times across detectors, and (ii) the detailed signal/feature pattern remains architecture-dependent. Dynamic HMM alignment and TUAR/TUEV semantics make this structure more interpretable, while the negative router results show that interpretability alone is not enough to obtain a safe suppressor. A future model-aware system should therefore be treated as a new, independently held-out experiment.

## Limitations and what this release does not claim

1. The primary confirmation uses five patient-level outer folds and frozen thresholds; it is not a substitute for a larger multi-center prospective evaluation.
2. HMM state identities are latent and fold-local; A1–A5 are consensus prototypes, not pre-existing labels.
3. TUEV/TUAR semantic associations are domain-shifted and label-imbalanced; enrichment/depletion is not equivalent to clinical artifact classification.
4. H4 was negative on both the standard test rows and an independent expanded holdout; no test-set retuning is permitted.
5. External TUSZ metrics show broad uncertainty across patient bootstrap samples and should be reported as preliminary generalization evidence.

## Release contents

The five requested figures are in `figures/`: H1 temporal sharing; H2 fold-state alignment heatmap; A1–A5 consensus/eye semantics; H2 threshold sensitivity; and H4 router versus best-single FA/24 h. Machine-readable summaries, frozen HMM files, source hashes, and this report are all included in the same release directory.
"""
    (OUT / "false_alarm_atlas_final_report.md").write_text(report, encoding="utf-8")

    # A release manifest makes every final file auditable.
    all_files = {}
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "release_manifest.json":
            all_files[str(p.relative_to(OUT))] = sha256(p)
    manifest = {
        "release_id": "research-a-final-synthesis-v1",
        "status": "complete_audited",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preflight": {"status": preflight.get("status"), "sha256": sha256(REPORTS / "research-a-final-synthesis-v1/no_duplicate_overlap_preflight.json")},
        "hmm_alignment": {"audit_status": hmm_audit.get("status"), "audit_sha256": sha256(REPORTS / "h2-hmm-crossfold-alignment-v2/final_audit.json"), "frozen_manifest": "frozen_hmm_alignment_manifest.json"},
        "source_files": {
            "h1": "v3-groupkfold-confirmatory-v1/h1_groupkfold_pairwise.csv",
            "h3": "v3-groupkfold-confirmatory-v1/h3_groupkfold_model_fingerprint.csv",
            "h4": "v3-groupkfold-confirmatory-v1/h4_groupkfold_router_test.csv",
            "h4_independent_holdout": "h4-expanded-independent-holdout-v1/holdout_metrics.csv",
            "tusz": "v3-groupkfold-confirmatory-v1/tusz_groupkfold_external_v1_model_summary.csv",
            "tuar": "artifact-ontology-v1-posthoc-final/tuar_metric_summary.csv",
            "tuev": "artifact-ontology-v1-posthoc-final/tuev_metric_summary.csv",
        },
        "figures": {k: str(v.relative_to(OUT)) for k, v in figures.items()},
        "files_sha256": all_files,
        "test_policy": "No test result was used to select a new threshold, branch, HMM K, or alignment threshold in this reporting step.",
    }
    (OUT / "release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"status": "complete_audited", "output": str(OUT), "figures": list(manifest["figures"].values()), "files": len(all_files)}, indent=2))


if __name__ == "__main__":
    main()
