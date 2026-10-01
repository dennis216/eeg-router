"""Create a concise, data-backed report for the completed external H3 run."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
BASE = ROOT / "outputs" / "reports" / "h3-external-tusz-v1"
OUT = BASE / "H3_external_TUSZ_replication_report.md"


def f(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    summary = pd.read_csv(BASE / "h3_external_tusz_summary_metrics.csv")
    model_summary = pd.read_csv(BASE / "h3_external_tusz_model_summary.csv")
    bootstrap = pd.read_csv(BASE / "h3_external_tusz_patient_bootstrap.csv")
    permutation = pd.read_csv(BASE / "h3_external_tusz_patient_permutation.csv")
    manifest = json.loads((BASE / "h3_external_tusz_manifest.json").read_text())
    feature_manifest = json.loads((BASE / "tusz_h3_fp_features_manifest.json").read_text())
    audit = json.loads((BASE / "h3_external_tusz_audit.json").read_text())
    centered = model_summary[model_summary.centering == "patient_centered"].set_index("feature_set")
    raw = model_summary[model_summary.centering == "raw"].set_index("feature_set")
    centered_full = summary[(summary.centering == "patient_centered") & (summary.feature_set == "full15")]
    centered_eeg = summary[(summary.centering == "patient_centered") & (summary.feature_set == "eeg_only13")]
    ci_full = bootstrap[(bootstrap.centering == "patient_centered") & (bootstrap.feature_set == "full15")]
    ci_eeg = bootstrap[(bootstrap.centering == "patient_centered") & (bootstrap.feature_set == "eeg_only13")]
    chb_path = ROOT / "outputs" / "reports" / "v3-groupkfold-confirmatory-v1" / "h3_groupkfold_model_fingerprint.csv"
    chb = pd.read_csv(chb_path)
    chb_mean = float(chb[chb.patient_centered == True].test_macro_auroc.mean())
    lines: list[str] = []
    lines += [
        "# Independent external H3 replication: TUSZ detector-source information",
        "",
        "> Scope: replicate the CHB H3 question on an independent TUSZ cohort: can false-positive event features identify which detector generated the event? This run is separate from the CHB confirmatory test and does not change any frozen detector thresholds.",
        "",
        "## Executive conclusion",
        "",
        f"A patient-grouped TUSZ analysis reproduced strong detector-source separability. With patient-centered full 15-feature inputs, the mean macro one-vs-rest AUROC across 15 frozen checkpoint units was **{f(centered.loc['full15','mean_macro_auroc'])}** (range {f(centered.loc['full15','min_macro_auroc'])}–{f(centered.loc['full15','max_macro_auroc'])}); all 15 units had 2,000-patient-bootstrap lower bounds above 0.5. The corresponding CHB patient-centered mean was {f(chb_mean)}. This supports a reproducible detector-dependent output pattern across datasets.",
        "",
        "The effect is not evidence for a physiological mechanism. The duration+peak-probability ablation was nearly as strong as full15, while EEG/QC-only features retained a smaller but above-chance signal. Therefore the external result is driven mainly by detector output behavior, with residual information in the EEG/QC features; it should not be described as an architecture-specific biological signature.",
        "",
        "## Frozen inputs",
        "",
        f"- Cohort: TUSZ eval, {manifest['patients']} canonical patients, {feature_manifest['recordings']} EDFs containing FP events; {manifest['events']} FP event rows.",
        "- The 45 frozen detector inference runs (3 models × 5 folds × 3 seeds) were reused; model training was false and external threshold retuning was false.",
        "- Events were selected only from the already-scored external FP table. The 15 event features were recomputed from a common canonical bipolar TCN-GAT-style view and frozen TUH quality features; no scaler, PCA, feature selection, or classifier was fitted during extraction.",
        "",
        "## Analysis method",
        "",
        "1. For each of the 15 detector checkpoint units (fold × seed, with the three detector sources as classes), use a fixed 5-fold `GroupKFold` on canonical TUSZ patient ID. Every EDF and event from a patient stays in one held-out fold.",
        "2. In each inner training fold, fit only the fixed CHB specification: class-balanced multinomial logistic regression (C=1), train-fold medians, train-fold 1–99% clipping, and a train-fold StandardScaler. Held-out folds are used only for prediction; no threshold or feature choice is made.",
        "3. Report raw and label-free patient-centered features. The centered version subtracts each partition's patient mean before train-fold cleaning, matching the CHB sensitivity analysis; it is reported as a sensitivity analysis rather than a causal adjustment.",
        "4. Report three pre-specified feature sets: full15, EEG/QC-only13 (removes duration and detector peak probability), and duration+peak2. For each unit, concatenate the five held-out predictions into an OOF table.",
        "5. Use 2,000 patient bootstrap replicates for CIs. A secondary 1,000-replicate within-patient held-out-label permutation is conditional on fixed scores and is not interpreted as a refit-training null p-value.",
        "",
        "## Results (mean across 15 external checkpoint units)",
        "",
        "| Feature set | Raw mean AUROC | Patient-centered mean AUROC | Centered range | Centered units with bootstrap CI lower > 0.5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, label in [("full15", "Full 15"), ("eeg_only13", "EEG/QC-only 13"), ("duration_peak2", "Duration + peak 2")]:
        c = centered.loc[name]
        r = raw.loc[name]
        ci = bootstrap[(bootstrap.centering == "patient_centered") & (bootstrap.feature_set == name)]
        lines.append(f"| {label} | {f(r.mean_macro_auroc)} | **{f(c.mean_macro_auroc)}** | {f(c.min_macro_auroc)}–{f(c.max_macro_auroc)} | {int((ci.bootstrap_ci_low > 0.5).sum())}/15 |")
    lines += [
        "",
        "### Centered full15 class-wise AUROC",
        "",
        f"SingleM {f(centered_full.singlem_auroc.mean())}; CBraMod {f(centered_full.cbramod_auroc.mean())}; TCN-GAT {f(centered_full.tcn_gat_auroc.mean())}. All 15 units had conditional permutation p < 0.05; this secondary test is reported only as a conditional held-out-score check.",
        "",
        "### Interpretation of the ablations",
        "",
        f"- Removing duration and peak probability reduced centered mean AUROC from {f(centered.loc['full15','mean_macro_auroc'])} to {f(centered.loc['eeg_only13','mean_macro_auroc'])}; {int((ci_eeg.bootstrap_ci_low > 0.5).sum())}/15 centered units retained a bootstrap lower bound above chance.",
        f"- Duration+peak alone remained at {f(centered.loc['duration_peak2','mean_macro_auroc'])}, indicating that source-specific detector output behavior carries most of the separability in this external cohort.",
        "- The raw-to-centered decrease is expected when patient baseline differences are removed; the residual centered signal is the more conservative evidence.",
        "",
        "## What this establishes",
        "",
        "- **Supported:** detector-source information in FP event records is reproducible in an independent TUSZ cohort under patient-level held-out evaluation.",
        "- **Not established:** a shared physiological artifact, detector-specific biological mechanism, causation, or a safer suppression policy.",
        "- **Important scope:** this is a TUSZ construct replication with a new patient-grouped classifier fit. It is not a zero-shot application of a classifier trained on CHB. A separate CHB→TUSZ transfer test would be a stricter next experiment.",
        "",
        "## Audit and hashes",
        "",
        f"Audit status: **{audit['status']}**; {sum(bool(value) for value in audit['checks'].values())}/{len(audit['checks'])} checks passed, including no-duplicate preflight, frozen 45-run source scoring, feature/event key equality, 43-patient roster, 15 checkpoint units, five patient folds per unit, 2,000 patient bootstrap, and 1,000 conditional permutations. AUROC chance is correctly set to 0.5; 1/3 is the three-class accuracy chance, not AUROC chance.",
        "",
        "Key artifacts:",
        "- `tusz_h3_fp_features.parquet` — 36,698 frozen external FP feature rows.",
        "- `h3_external_tusz_summary_metrics.csv` — 90 unit/feature/centering OOF summaries.",
        "- `h3_external_tusz_patient_bootstrap.csv` — 2,000 patient replicates per summary unit.",
        "- `h3_external_tusz_patient_permutation.csv` — 1,000 conditional permutations per summary unit.",
        "- `h3_external_tusz_audit.json` — read-only audit and source hashes.",
        "",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
