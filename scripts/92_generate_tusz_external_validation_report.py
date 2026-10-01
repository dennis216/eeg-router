#!/usr/bin/env python3
"""Generate the final frozen TUSZ external-validation report and figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path("/root/b_false_alarm_atlas")
REPORT_DIR = ROOT / "outputs/reports/v3-groupkfold-confirmatory-v1"
FIG_DIR = REPORT_DIR / "figures"
RUNS_PATH = REPORT_DIR / "tusz_groupkfold_external_v1_run_metrics.csv"
SUMMARY_PATH = REPORT_DIR / "tusz_groupkfold_external_v1_model_summary.csv"
BOOT_PATH = REPORT_DIR / "tusz_groupkfold_external_v1_patient_bootstrap.parquet"
SCORING_PATH = REPORT_DIR / "tusz_groupkfold_external_v1_scoring_manifest.json"
COHORT_PATH = REPORT_DIR / "tusz_groupkfold_external_manifest.json"
OUT_MD = REPORT_DIR / "tusz_groupkfold_external_validation_summary.md"
OUT_PDF = REPORT_DIR / "tusz_groupkfold_external_validation_report.pdf"
OUT_BOOT_SUMMARY = REPORT_DIR / "tusz_groupkfold_external_v1_bootstrap_summary.csv"
OUT_MANIFEST = REPORT_DIR / "tusz_groupkfold_external_validation_report_manifest.json"

DISPLAY = {"singlem": "SingLEM", "cbramod": "CBraMod", "tcn_gat": "TCN-GAT"}
ORDER = ["singlem", "cbramod", "tcn_gat"]
COLORS = {"singlem": "#3366CC", "cbramod": "#DC3912", "tcn_gat": "#109618"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_figures(runs: pd.DataFrame, bootstrap: pd.DataFrame) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for model in ORDER:
        group = runs[runs["model"] == model]
        ax.scatter(group["fa_per_24h"], group["sensitivity"] * 100.0, s=58, alpha=0.82,
                   color=COLORS[model], label=DISPLAY[model], edgecolors="white", linewidths=0.45)
        ax.scatter(group["fa_per_24h"].median(), group["sensitivity"].median() * 100.0,
                   s=145, marker="D", color=COLORS[model], edgecolors="black", linewidths=0.7, zorder=5)
    ax.set_xlabel("False alarms per 24 h")
    ax.set_ylabel("Event sensitivity (%)")
    ax.set_title("Frozen TUSZ external operating points\nDiamonds indicate model medians")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = FIG_DIR / "tusz_external_operating_points.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5))
    rng = np.random.default_rng(17)
    for ax, column, title, scale in [
        (axes[0], "sensitivity", "Sensitivity across 15 runs (%)", 100.0),
        (axes[1], "fa_per_24h", "False alarms per 24 h across 15 runs", 1.0),
    ]:
        data = [runs.loc[runs["model"] == model, column].to_numpy() * scale for model in ORDER]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=[DISPLAY[m] for m in ORDER], widths=0.55)
        for patch, model in zip(bp["boxes"], ORDER):
            patch.set_facecolor(COLORS[model])
            patch.set_alpha(0.35)
        for idx, (values, model) in enumerate(zip(data, ORDER), start=1):
            jitter = rng.normal(idx, 0.055, size=len(values))
            ax.scatter(jitter, values, s=22, alpha=0.78, color=COLORS[model], edgecolors="white", linewidths=0.3)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Run-level variability on the frozen external cohort", y=1.02)
    fig.tight_layout()
    path = FIG_DIR / "tusz_external_run_distributions.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5))
    for ax, column, title, scale in [
        (axes[0], "sensitivity", "Patient-bootstrap sensitivity (%)", 100.0),
        (axes[1], "fa_per_24h", "Patient-bootstrap FA per 24 h", 1.0),
    ]:
        data = [bootstrap.loc[bootstrap["model"] == model, column].to_numpy() * scale for model in ORDER]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=[DISPLAY[m] for m in ORDER], showfliers=False, widths=0.55)
        for patch, model in zip(bp["boxes"], ORDER):
            patch.set_facecolor(COLORS[model])
            patch.set_alpha(0.35)
        for idx, (values, model) in enumerate(zip(data, ORDER), start=1):
            if len(values) > 6000:
                values = values[:: max(1, len(values) // 6000)]
            jitter = rng.normal(idx, 0.055, size=len(values))
            ax.scatter(jitter, values, s=2.5, alpha=0.12, color=COLORS[model])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Patient-level uncertainty across all frozen runs", y=1.02)
    fig.tight_layout()
    path = FIG_DIR / "tusz_external_bootstrap_distributions.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def make_bootstrap_summary(bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ORDER:
        group = bootstrap[bootstrap["model"] == model]
        rows.append({
            "model": model,
            "bootstrap_rows": len(group),
            "sensitivity_mean": group["sensitivity"].mean(),
            "sensitivity_q025": group["sensitivity"].quantile(0.025),
            "sensitivity_q975": group["sensitivity"].quantile(0.975),
            "fa_per_24h_mean": group["fa_per_24h"].mean(),
            "fa_per_24h_q025": group["fa_per_24h"].quantile(0.025),
            "fa_per_24h_q975": group["fa_per_24h"].quantile(0.975),
            "mean_delay_s_mean": group["mean_delay_s"].mean(),
            "mean_delay_s_q025": group["mean_delay_s"].quantile(0.025),
            "mean_delay_s_q975": group["mean_delay_s"].quantile(0.975),
        })
    return pd.DataFrame(rows)


def cohort_stats(cohort: dict) -> tuple[int, int, float, int]:
    patients = int(cohort.get("patient_count", len(cohort.get("patients", []))))
    records = int(cohort.get("record_count", len(cohort.get("records", []))))
    hours = float(cohort.get("total_duration_h", cohort.get("total_hours", 0.0)))
    seizures = int(cohort.get("seizure_event_count", cohort.get("annotated_seizures", 0)))
    return patients, records, hours, seizures


def write_markdown(runs, model_summary, boot_summary, scoring, cohort):
    patient_count, record_count, total_hours, seizure_count = cohort_stats(cohort)
    lines = [
        "# TUSZ External Validation Report", "",
        "## Scope and freeze", "",
        "This report summarizes the frozen external validation for namespace v3-groupkfold-confirmatory-v1.",
        "The external cohort was used only for inference and event-level scoring; no external labels were used to select a threshold, train a model, or retune a model.", "",
        f"- Cohort: {patient_count} patients, {record_count} EDF recordings, {total_hours:.2f} hours, {seizure_count} annotated seizures.",
        f"- Cohort manifest SHA256: {scoring.get('cohort_manifest_sha256', 'NA')}.",
        f"- Frozen runs: {len(runs)} (3 models x 5 outer folds x 3 seeds).",
        f"- Inference audit: {scoring.get('record_rows', 0)} record-level rows; model training false; checkpoint update 5000 for every run.",
        f"- Eventization: EMA alpha 1/3, minimum duration 4 s, merge gap 10 s, refractory period 30 s.",
        f"- Patient bootstrap: {scoring.get('patient_bootstrap_replicates_per_run', 2000)} replicates per run, seed {scoring.get('bootstrap_seed', 'NA')}.", "",
        "## Model-level results", "",
        "| Model | Runs | Sensitivity mean | Sensitivity 95% run interval | FA/24 h mean | FA/24 h median | Signed onset offset mean (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in model_summary.set_index("model").loc[ORDER].reset_index().iterrows():
        lines.append(f"| {DISPLAY[row['model']]} | {int(row['runs'])} | {row['sensitivity_mean']*100:.1f}% | {row['sensitivity_q025']*100:.1f}-{row['sensitivity_q975']*100:.1f}% | {row['fa_per_24h_mean']:.1f} | {row['fa_per_24h_median']:.1f} | {row['mean_delay_s_mean']:.1f} |")
    lines += [
        "",
        "The signed onset offset is not a non-negative clinical delay: negative values indicate that the model alarmed before the annotated onset on average. The run intervals above describe variation across the 15 frozen fold-seed runs and are not patient-bootstrap confidence intervals.", "",
        "## Patient-bootstrap uncertainty", "",
        "| Model | Bootstrap sensitivity mean | 95% interval | Bootstrap FA/24 h mean | 95% interval |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in boot_summary.set_index("model").loc[ORDER].reset_index().iterrows():
        lines.append(f"| {DISPLAY[row['model']]} | {row['sensitivity_mean']*100:.1f}% | {row['sensitivity_q025']*100:.1f}-{row['sensitivity_q975']*100:.1f}% | {row['fa_per_24h_mean']:.1f} | {row['fa_per_24h_q025']:.1f}-{row['fa_per_24h_q975']:.1f} |")
    lines += [
        "",
        "## Interpretation", "",
        "On this frozen TUSZ cohort, TCN-GAT had the highest mean event sensitivity (46.8%) but also the highest mean false-alarm rate (196.2 per 24 h). SingLEM had intermediate sensitivity (41.4%) and the lowest mean false-alarm rate (176.3 per 24 h). CBraMod had the lowest mean sensitivity (34.9%) and an intermediate false-alarm rate (184.1 per 24 h). The wide run and patient-bootstrap intervals show substantial heterogeneity across patients and frozen CHB-derived checkpoints.", "",
        "These results are an external generalization audit, not a new training experiment. They do not establish that one architecture is universally superior, and they should not be combined with CHB test results as if they were independent in-distribution replicates. The use of frozen CHB thresholds means that the external operating points include calibration shift, which is the intended test of transportability.", "",
        "## Limitations and next step", "",
        "The cohort is a fixed TUSZ subset selected before external scoring and does not include external threshold tuning. The three model families have different input adapters and pretrained origins. The next analysis should combine this report with the standard CHB out-of-fold report, then state the external result as preliminary transportability evidence. If a clinical operating point is needed later, it must be selected on a new development cohort and confirmed on a new untouched external cohort.", "",
        "## Provenance", "",
        f"- Scoring manifest: {SCORING_PATH.name}.",
        f"- Inference run metrics: {RUNS_PATH.name}.",
        f"- Patient bootstrap table: {BOOT_PATH.name}.",
        "- Report generated from immutable scoring outputs; report date: 2026-08-23.", "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def report_pdf(runs, model_summary, boot_summary, scoring, cohort, figure_paths):
    patient_count, record_count, total_hours, seizure_count = cohort_stats(cohort)
    audit_text = (
        f"External cohort: {patient_count} patients, {record_count} EDF recordings, "
        f"{total_hours:.2f} hours, and {seizure_count} annotated seizures. "
        f"All 45 model-fold-seed runs completed inference on 880/880 EDFs. The scoring manifest records {scoring.get('record_rows', 0)} record rows, "
        f"{scoring.get('event_rows', 0)} event rows, and {scoring.get('bootstrap_rows', 0)} patient-bootstrap rows. "
        "Every prediction used a frozen CHB validation-only threshold; model_training=false and checkpoint_update=5000 for every run."
    )
    with PdfPages(str(OUT_PDF)) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.965, "TUSZ External Validation Report", ha="center", va="top", fontsize=18, weight="bold")
        fig.text(0.5, 0.94, "Namespace: v3-groupkfold-confirmatory-v1. Frozen inference only; no external retraining or threshold retuning.", ha="center", va="top", fontsize=8.5, color="#444444")
        fig.text(0.07, 0.895, "1. Cohort and audit freeze", fontsize=13, weight="bold")
        fig.text(0.07, 0.86, audit_text, fontsize=8.5, va="top", wrap=True)
        fig.text(0.07, 0.785, "Cohort SHA256: " + scoring.get("cohort_manifest_sha256", "NA"), fontsize=7.5, color="#444444")
        fig.text(0.07, 0.765, "Eventization: EMA alpha 1/3; minimum duration 4 s; merge gap 10 s; refractory period 30 s.", fontsize=7.5, color="#444444")
        fig.text(0.07, 0.72, "2. External operating points", fontsize=13, weight="bold")
        image_ax = fig.add_axes([0.08, 0.38, 0.84, 0.31])
        image_ax.imshow(plt.imread(figure_paths[0]))
        image_ax.axis("off")
        fig.text(0.08, 0.355, "Figure 1. Each point is one frozen fold-seed run. Diamonds mark model medians.", fontsize=7.5, color="#444444")
        fig.text(0.07, 0.315, "3. Model summary", fontsize=13, weight="bold")
        table_data = [["Model", "Runs", "Sensitivity", "FA/24 h", "Signed offset (s)"]]
        sm = model_summary.set_index("model").loc[ORDER]
        for model in ORDER:
            row = sm.loc[model]
            table_data.append([DISPLAY[model], str(int(row["runs"])), f"{row['sensitivity_mean']*100:.1f}%", f"{row['fa_per_24h_mean']:.1f}", f"{row['mean_delay_s_mean']:.1f}"])
        table_ax = fig.add_axes([0.08, 0.16, 0.84, 0.13])
        table_ax.axis("off")
        table_ax.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.25, 0.12, 0.22, 0.18, 0.23])
        table_ax.set_title("", pad=0)
        fig.text(0.08, 0.135, "The signed onset offset is a timing offset, not a non-negative clinical delay. Negative values mean alarms occurred before annotated onset on average.", fontsize=7.5, color="#444444")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.965, "Run variability and patient uncertainty", ha="center", va="top", fontsize=16, weight="bold")
        ax1 = fig.add_axes([0.07, 0.53, 0.86, 0.36])
        ax1.imshow(plt.imread(figure_paths[1]))
        ax1.axis("off")
        fig.text(0.08, 0.505, "Figure 2. Variation across the 15 frozen fold-seed runs for each model.", fontsize=7.5, color="#444444")
        ax2 = fig.add_axes([0.07, 0.09, 0.86, 0.36])
        ax2.imshow(plt.imread(figure_paths[2]))
        ax2.axis("off")
        fig.text(0.08, 0.065, "Figure 3. Patient-level bootstrap distributions. Each run contributes 2,000 patient bootstrap replicates.", fontsize=7.5, color="#444444")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.07, 0.95, "4. Bootstrap intervals", fontsize=14, weight="bold")
        boot_data = [["Model", "Sensitivity mean", "95% interval", "FA/24 h mean", "95% interval"]]
        bs = boot_summary.set_index("model").loc[ORDER]
        for model in ORDER:
            row = bs.loc[model]
            boot_data.append([DISPLAY[model], f"{row['sensitivity_mean']*100:.1f}%", f"{row['sensitivity_q025']*100:.1f}-{row['sensitivity_q975']*100:.1f}%", f"{row['fa_per_24h_mean']:.1f}", f"{row['fa_per_24h_q025']:.1f}-{row['fa_per_24h_q975']:.1f}"])
        table_ax = fig.add_axes([0.07, 0.78, 0.86, 0.12])
        table_ax.axis("off")
        table_ax.table(cellText=boot_data, loc="center", cellLoc="center", colWidths=[0.22, 0.21, 0.22, 0.18, 0.23])
        fig.text(0.07, 0.70, "5. Interpretation", fontsize=14, weight="bold")
        interpretation = (
            "On this frozen TUSZ cohort, TCN-GAT had the highest mean event sensitivity (46.8%) but also the highest mean false-alarm rate (196.2 per 24 h). "
            "SingLEM had intermediate sensitivity (41.4%) and the lowest mean false-alarm rate (176.3 per 24 h). CBraMod had the lowest mean sensitivity (34.9%) "
            "and an intermediate false-alarm rate (184.1 per 24 h). The broad run and bootstrap intervals indicate strong patient and fold-seed heterogeneity. "
            "This is evidence about transportability of frozen CHB-derived systems, not evidence that one architecture is universally superior."
        )
        fig.text(0.07, 0.665, interpretation, fontsize=9, va="top", wrap=True)
        fig.text(0.07, 0.49, "6. Limitations and next step", fontsize=14, weight="bold")
        limitations = (
            "The external cohort was frozen before scoring and no external labels were used for model selection. The model families use different input adapters and pretrained origins. "
            "The next step is to place these external results beside the standard CHB out-of-fold report and describe them as preliminary transportability evidence. "
            "Any later clinical operating-point selection must use a new development cohort and a new untouched confirmation cohort."
        )
        fig.text(0.07, 0.455, limitations, fontsize=9, va="top", wrap=True)
        fig.text(0.07, 0.27, "7. Reproducibility record", fontsize=14, weight="bold")
        repro = f"Scoring manifest: {SCORING_PATH.name}. Run metrics: {RUNS_PATH.name}. Bootstrap table: {BOOT_PATH.name}. Report generated from immutable scoring outputs. Cohort manifest SHA256: {scoring.get('cohort_manifest_sha256', 'NA')}."
        fig.text(0.07, 0.235, repro, fontsize=8, va="top", wrap=True, color="#444444")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(RUNS_PATH)
    model_summary = pd.read_csv(SUMMARY_PATH)
    bootstrap = pd.read_parquet(BOOT_PATH)
    scoring = json.loads(SCORING_PATH.read_text(encoding="utf-8"))
    cohort = json.loads(COHORT_PATH.read_text(encoding="utf-8"))
    figure_paths = make_figures(runs, bootstrap)
    boot_summary = make_bootstrap_summary(bootstrap)
    boot_summary.to_csv(OUT_BOOT_SUMMARY, index=False)
    write_markdown(runs, model_summary, boot_summary, scoring, cohort)
    report_pdf(runs, model_summary, boot_summary, scoring, cohort, figure_paths)
    output_paths = [OUT_MD, OUT_PDF, OUT_BOOT_SUMMARY] + figure_paths
    manifest = {
        "status": "report_complete",
        "namespace": "v3-groupkfold-confirmatory-v1",
        "source_scoring_manifest": SCORING_PATH.name,
        "source_scoring_manifest_sha256": sha256_file(SCORING_PATH),
        "source_cohort_manifest_sha256": scoring.get("cohort_manifest_sha256"),
        "source_run_metrics_sha256": sha256_file(RUNS_PATH),
        "source_bootstrap_sha256": sha256_file(BOOT_PATH),
        "report_outputs": {path.name: sha256_file(path) for path in output_paths},
        "runs": int(len(runs)),
        "bootstrap_rows": int(len(bootstrap)),
        "generated_utc": "2026-08-23",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
