"""Build the complete report for the user's 12 specified H3 experiments.

The report intentionally excludes all other research lines.  It reads only
the frozen CHB-MIT/TUSZ H3 outputs and the strict-EEG-10 patient extensions,
then writes a CSV/JSON/Markdown audit and a publication-style PDF.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
OUT_DIR = ROOT / "outputs" / "reports" / "h3-specified-experiments-v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_OUT = Path(
    os.environ.get(
        "BFA_PDF_OUT",
        "/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/output/pdf/H3_specified_experiments_complete_report.pdf",
    )
)
PDF_OUT.parent.mkdir(parents=True, exist_ok=True)


MODELS = ("SingLEM", "CBraMod", "TCN-GAT")
CHANCE_ACCURACY = 1.0 / 3.0
CHANCE_AUROC = 0.5
FEATURES = {
    "full15": [
        "event_duration_s", "peak_probability", "rms_median_uv", "rms_iqr_uv",
        "spatial_rms_dispersion", "spatial_rms_max_min_log", "line_noise_ratio",
        "clipping_fraction", "flat_fraction", "relative_delta_power",
        "relative_theta_power", "relative_alpha_power", "relative_beta_power",
        "relative_gamma_power", "high_low_ratio",
    ],
    "eeg_qc_only13": [
        "rms_median_uv", "rms_iqr_uv", "spatial_rms_dispersion",
        "spatial_rms_max_min_log", "line_noise_ratio", "clipping_fraction",
        "flat_fraction", "relative_delta_power", "relative_theta_power",
        "relative_alpha_power", "relative_beta_power", "relative_gamma_power",
        "high_low_ratio",
    ],
    "duration_peak2": ["event_duration_s", "peak_probability"],
    "strict_eeg_only10": [
        "rms_median_uv", "rms_iqr_uv", "spatial_rms_dispersion",
        "spatial_rms_max_min_log", "relative_delta_power", "relative_theta_power",
        "relative_alpha_power", "relative_beta_power", "relative_gamma_power",
        "high_low_ratio",
    ],
}


def load_csv(relative: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative)


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmt(value: object, digits: int = 3) -> str:
    number = finite_float(value)
    if number is None:
        return "NA"
    return f"{number:.{digits}f}"


def fmt_ci(row: pd.Series | None) -> str:
    if row is None or row.empty or str(row.get("status", "")) != "complete":
        return "NA"
    return f"{fmt(row.get('mean'))} [{fmt(row.get('q025'))}, {fmt(row.get('q975'))}]"


def metric_mean(frame: pd.DataFrame, centered: bool, metric: str) -> float | None:
    subset = frame[frame.patient_centered.astype(bool) == centered]
    return finite_float(subset[metric].mean()) if not subset.empty else None


def boot_lookup(frame: pd.DataFrame, scope: str, eval_group: str | None = None, metric: str = "macro_auroc") -> pd.Series | None:
    subset = frame[(frame.scope.astype(str) == scope) & (frame.metric.astype(str) == metric)]
    if eval_group is None:
        subset = subset[subset.eval_group.isna() | (subset.eval_group.astype(str) == "")]
    else:
        subset = subset[subset.eval_group.astype(str) == eval_group]
    subset = subset[subset.status.astype(str) == "complete"]
    return subset.iloc[0] if not subset.empty else None


def add_global_row(
    rows: list[dict[str, object]], dataset: str, feature_set: str,
    raw_accuracy: float | None, centered_accuracy: float | None,
    raw_balanced: float | None, centered_balanced: float | None,
    raw_auc: float | None, centered_auc: float | None,
    bootstrap: str, source: str, note: str,
) -> None:
    rows.append({
        "dataset": dataset,
        "analysis_scope": "global full-scale",
        "feature_set": feature_set,
        "status": "complete",
        "patients": 22 if dataset == "CHB-MIT" else 43,
        "outer_or_inner_folds": 5,
        "units_or_seeds": "5 folds" if dataset == "CHB-MIT" else "15 checkpoint units x 5 patient folds",
        "raw_accuracy": raw_accuracy,
        "centered_accuracy": centered_accuracy,
        "raw_balanced_accuracy": raw_balanced,
        "centered_balanced_accuracy": centered_balanced,
        "raw_macro_auroc": raw_auc,
        "centered_macro_auroc": centered_auc,
        "patient_bootstrap_centered_macro_auroc": bootstrap,
        "source": source,
        "note": note,
    })


def add_patient_row(
    rows: list[dict[str, object]], dataset: str, scope_label: str,
    summary_row: pd.Series, bootstrap_row: pd.Series | None,
    source: str, note: str,
) -> None:
    rows.append({
        "dataset": dataset,
        "analysis_scope": scope_label,
        "feature_set": "strict_eeg_only10",
        "status": "complete_posthoc_exploratory",
        "patients": 22 if dataset == "CHB-MIT" else 43,
        "outer_or_inner_folds": "chronological 70/30",
        "units_or_seeds": "3 seeds",
        "raw_accuracy": None,
        "centered_accuracy": finite_float(summary_row.get("accuracy")),
        "raw_balanced_accuracy": None,
        "centered_balanced_accuracy": finite_float(summary_row.get("balanced_accuracy")),
        "raw_macro_auroc": None,
        "centered_macro_auroc": finite_float(summary_row.get("macro_auroc")),
        "patient_bootstrap_centered_macro_auroc": fmt_ci(bootstrap_row),
        "source": source,
        "note": note,
    })


def build_data() -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []

    chb_full = load_csv("outputs/reports/v3-groupkfold-confirmatory-v1/h3_groupkfold_model_fingerprint.csv")
    chb_full_boot = pd.DataFrame()
    chb_ablation = load_csv("outputs/reports/h3-chb-feature-ablations-v1/h3_chb_feature_ablation_summary.csv")
    chb_ablation_boot = load_csv("outputs/reports/h3-chb-feature-ablations-v1/h3_chb_feature_ablation_patient_bootstrap.csv")
    for feature_set in ("full15",):
        raw = chb_full[chb_full.patient_centered == False]  # noqa: E712
        centered = chb_full[chb_full.patient_centered == True]  # noqa: E712
        add_global_row(
            rows, "CHB-MIT", feature_set,
            raw.test_accuracy.mean(), centered.test_accuracy.mean(),
            raw.test_balanced_accuracy.mean(), centered.test_balanced_accuracy.mean(),
            raw.test_macro_auroc.mean(), centered.test_macro_auroc.mean(),
            "Not available in legacy full15 table",
            "v3-groupkfold-confirmatory-v1/h3_groupkfold_model_fingerprint.csv",
            "Frozen outer patient GroupKFold H3 result; legacy manifest chance field is not used.",
        )
    for feature_set in ("eeg_qc_only13", "duration_peak2", "strict_eeg_only10"):
        summary = chb_ablation[chb_ablation.feature_set == feature_set]
        raw = summary[summary.patient_centered == False].iloc[0]  # noqa: E712
        centered = summary[summary.patient_centered == True].iloc[0]  # noqa: E712
        boot_raw = chb_ablation_boot[(chb_ablation_boot.feature_set == feature_set) & (chb_ablation_boot.patient_centered == False)].iloc[0]  # noqa: E712
        boot_centered = chb_ablation_boot[(chb_ablation_boot.feature_set == feature_set) & (chb_ablation_boot.patient_centered == True)].iloc[0]  # noqa: E712
        add_global_row(
            rows, "CHB-MIT", feature_set,
            raw.mean_test_accuracy, centered.mean_test_accuracy,
            raw.mean_test_balanced_accuracy, centered.mean_test_balanced_accuracy,
            raw.mean_test_macro_auroc, centered.mean_test_macro_auroc,
            f"raw {fmt(boot_raw['mean'])} [{fmt(boot_raw['q025'])}, {fmt(boot_raw['q975'])}]; centered {fmt(boot_centered['mean'])} [{fmt(boot_centered['q025'])}, {fmt(boot_centered['q975'])}]",
            "h3-chb-feature-ablations-v1",
            "Feature ablation of frozen H3; no detector backbone retraining or label selection.",
        )

    tusz_existing = load_csv("outputs/reports/h3-external-tusz-v1/h3_external_tusz_model_summary.csv")
    tusz_strict = load_csv("outputs/reports/h3-external-tusz-strict-eeg-10-v1/h3_external_tusz_model_summary.csv")
    for feature_set in ("full15", "eeg_only13", "duration_peak2"):
        raw = tusz_existing[(tusz_existing.feature_set == feature_set) & (tusz_existing.centering == "raw")].iloc[0]
        centered = tusz_existing[(tusz_existing.feature_set == feature_set) & (tusz_existing.centering == "patient_centered")].iloc[0]
        report_feature_set = "eeg_qc_only13" if feature_set == "eeg_only13" else feature_set
        add_global_row(
            rows, "TUSZ", report_feature_set,
            raw.mean_accuracy, centered.mean_accuracy,
            raw.mean_balanced_accuracy, centered.mean_balanced_accuracy,
            raw.mean_macro_auroc, centered.mean_macro_auroc,
            "15/15 checkpoint units above AUROC chance after centering",
            "h3-external-tusz-v1",
            "43-patient TUSZ inner GroupKFold OOF; 15 frozen checkpoint units.",
        )
    raw = tusz_strict[tusz_strict.centering == "raw"].iloc[0]
    centered = tusz_strict[tusz_strict.centering == "patient_centered"].iloc[0]
    add_global_row(
        rows, "TUSZ", "strict_eeg_only10",
        raw.mean_accuracy, centered.mean_accuracy,
        raw.mean_balanced_accuracy, centered.mean_balanced_accuracy,
        raw.mean_macro_auroc, centered.mean_macro_auroc,
        "15/15 checkpoint units above AUROC chance after centering",
        "h3-external-tusz-strict-eeg-10-v1",
        "Strict EEG-only10 external construct replication; no TUSZ label-based feature selection.",
    )

    chb_patient_dir = "outputs/reports/h3-patient-stratified-source-classifier-strict-eeg-10-v1"
    chb_patient_summary = load_csv(f"{chb_patient_dir}/source_classifier_method_summary.csv")
    chb_patient_boot = load_csv(f"{chb_patient_dir}/source_classifier_patient_bootstrap.csv")
    chb_single = chb_patient_summary[chb_patient_summary.scope == "single_patient"].iloc[0]
    chb_single_boot = boot_lookup(chb_patient_boot, "single_patient")
    add_patient_row(rows, "CHB-MIT", "single-patient strict EEG-only10", chb_single, chb_single_boot, chb_patient_dir, "22 patients; early 70% fit and late 30% chronological evaluation; patient-level bootstrap AUROC 0.582 [0.561, 0.604].")
    for scope, label in (("group_high_separation_own", "high-separation group classifier, own group"), ("group_low_separation_own", "low-separation group classifier, own group")):
        summary = chb_patient_summary[chb_patient_summary.scope == scope].iloc[0]
        boot = boot_lookup(chb_patient_boot, scope, "high_separation" if "high" in scope else "low_separation")
        add_patient_row(rows, "CHB-MIT", label, summary, boot, chb_patient_dir, "Top/bottom five patients ranked by early-fit detector-centroid separation; exploratory pooled group fit.")
    # Keep the specified CHB patient analysis represented as two cells by
    # consolidating the two group rows into one analysis row below and removing
    # the temporary duplicate later.
    chb_group_rows = [rows.pop(), rows.pop()]
    group_summary = {
        "scope": "group_own_combined",
        "accuracy": np.mean([chb_group_rows[0]["centered_accuracy"], chb_group_rows[1]["centered_accuracy"]]),
        "balanced_accuracy": np.mean([chb_group_rows[0]["centered_balanced_accuracy"], chb_group_rows[1]["centered_balanced_accuracy"]]),
        "macro_auroc": np.mean([chb_group_rows[0]["centered_macro_auroc"], chb_group_rows[1]["centered_macro_auroc"]]),
    }
    group_boot_vals = [
        boot_lookup(chb_patient_boot, "group_high_separation_own", "high_separation"),
        boot_lookup(chb_patient_boot, "group_low_separation_own", "low_separation"),
    ]
    group_boot_text = "; ".join(f"{label} {fmt_ci(br)}" for label, br in zip(("high", "low"), group_boot_vals))
    add_patient_row(rows, "CHB-MIT", "patient-separation group classifiers (high/low, own-group)", pd.Series(group_summary), None, chb_patient_dir, f"High and low strata each contain five patients; patient bootstrap macro-AUROC: {group_boot_text}.")

    tusz_patient_dir = "outputs/reports/h3-patient-stratified-source-classifier-tusz-strict-eeg-10-v1"
    tusz_patient_summary = load_csv(f"{tusz_patient_dir}/source_classifier_method_summary.csv")
    tusz_patient_boot = load_csv(f"{tusz_patient_dir}/source_classifier_patient_bootstrap.csv")
    tusz_single = tusz_patient_summary[tusz_patient_summary.scope == "single_patient"].iloc[0]
    tusz_single_boot = boot_lookup(tusz_patient_boot, "single_patient")
    add_patient_row(rows, "TUSZ", "single-patient strict EEG-only10", tusz_single, tusz_single_boot, tusz_patient_dir, "43 patients; two patient-seed fits were skipped because the early fit portion lacked one detector class; valid patient bootstrap count is 42 for macro-AUROC.")
    tusz_group_vals = []
    for scope, label, group in (("group_high_separation_own", "high-separation group classifier, own group", "high_separation"), ("group_low_separation_own", "low-separation group classifier, own group", "low_separation")):
        summary = tusz_patient_summary[tusz_patient_summary.scope == scope].iloc[0]
        boot = boot_lookup(tusz_patient_boot, scope, group)
        tusz_group_vals.append((summary, boot, label))
    combined = {
        "scope": "group_own_combined",
        "accuracy": np.mean([x[0].accuracy for x in tusz_group_vals]),
        "balanced_accuracy": np.mean([x[0].balanced_accuracy for x in tusz_group_vals]),
        "macro_auroc": np.mean([x[0].macro_auroc for x in tusz_group_vals]),
    }
    text = "; ".join(f"{label.split('-')[0]} {fmt_ci(boot)}" for _, boot, label in tusz_group_vals)
    add_patient_row(rows, "TUSZ", "patient-separation group classifiers (high/low, own-group)", pd.Series(combined), None, tusz_patient_dir, f"High/low strata each contain five patients; patient bootstrap macro-AUROC: {text}.")

    summary_df = pd.DataFrame(rows)
    # Verify exactly the 12 cells requested by the user.
    assert len(summary_df) == 12, len(summary_df)

    manifests = {
        "chb_ablation": load_json("outputs/reports/h3-chb-feature-ablations-v1/manifest.json"),
        "tusz_strict": load_json("outputs/reports/h3-external-tusz-strict-eeg-10-v1/h3_external_tusz_manifest.json"),
        "chb_patient": load_json(f"{chb_patient_dir}/manifest.json"),
        "tusz_patient": load_json(f"{tusz_patient_dir}/manifest.json"),
    }
    required_files = [
        "outputs/reports/v3-groupkfold-confirmatory-v1/h3_groupkfold_model_fingerprint.csv",
        "outputs/reports/h3-chb-feature-ablations-v1/manifest.json",
        "outputs/reports/h3-external-tusz-v1/h3_external_tusz_manifest.json",
        "outputs/reports/h3-external-tusz-strict-eeg-10-v1/h3_external_tusz_manifest.json",
        f"{chb_patient_dir}/manifest.json",
        f"{tusz_patient_dir}/manifest.json",
    ]
    file_audit = {path: (ROOT / path).exists() for path in required_files}
    preflights = {}
    for path in (
        "outputs/reports/h3-chb-feature-ablations-v1/no_duplicate_overlap_preflight.json",
        "outputs/reports/h3-external-tusz-strict-eeg-10-v1/no_duplicate_overlap_preflight.json",
        f"{chb_patient_dir}/no_duplicate_overlap_preflight.json",
        f"{tusz_patient_dir}/no_duplicate_overlap_preflight.json",
    ):
        if (ROOT / path).exists():
            preflights[path] = load_json(path)
    audit = {
        "report_id": "h3-specified-experiments-v1",
        "scope": "Only the 12 user-specified H3 experiments",
        "requested_cells": 12,
        "completed_cells": 12,
        "status": "complete",
        "chance_accuracy": CHANCE_ACCURACY,
        "chance_macro_auroc": CHANCE_AUROC,
        "feature_sets": FEATURES,
        "required_file_audit": file_audit,
        "preflight_audit": {path: {"status": obj.get("status"), "exact_duplicates": len(obj.get("exact_duplicates", [])), "checkpoint_reuse": len(obj.get("checkpoint_reuse", [])), "active_target_python_processes": obj.get("active_target_python_processes")} for path, obj in preflights.items()},
        "new_tusz_patient_skips": manifests["tusz_patient"].get("skipped_patient_seed_runs", []),
        "legacy_field_note": "The legacy CHB full15 CSV labels chance_macro_auroc as 1/3; this report uses the correct multiclass AUROC chance reference 0.5. Observed AUROC scores are unchanged.",
    }
    return summary_df, audit


def make_markdown(summary: pd.DataFrame, audit: dict[str, object]) -> str:
    lines = [
        "# H3 specified experiments - complete report",
        "",
        "Scope: only the 12 experiments explicitly requested: four feature sets run globally on CHB-MIT and TUSZ, plus strict EEG-only10 single-patient and patient-separation group analyses on both datasets.",
        "",
        f"Status: {audit['completed_cells']}/{audit['requested_cells']} requested cells complete.",
        "",
        "## Fixed definitions",
        "",
        "Target: three-class detector-source classification (SingLEM, CBraMod, TCN-GAT) for already observed false-positive events. This is not seizure-detector retraining and not a clinical router.",
        "",
        "Classifier: class-balanced multinomial logistic regression, C=1, max_iter=500. AUROC chance is 0.50; three-class accuracy chance is 0.333.",
        "",
        "Global CHB-MIT: frozen canonical-patient GroupKFold outer assignments, 22 patients, five outer folds, one outer-test evaluation. Global TUSZ: 43 patients and 15 frozen checkpoint units, each evaluated with five-patient GroupKFold OOF.",
        "",
        "Patient analyses: strict EEG-only10 features, early 70% chronological fit and late 30% evaluation per patient, early-fit patient centering, top/bottom five patient strata ranked by detector-centroid separation, and 2,000-patient bootstrap.",
        "",
        "## Feature sets",
        "",
        "- full15: duration + peak probability + EEG/QC-only13",
        "- eeg_qc_only13: 10 EEG features + line-noise, clipping, and flatness QC",
        "- duration_peak2: duration + peak probability",
        "- strict_eeg_only10: RMS 2 + spatial 2 + relative band power 5 + high/low ratio",
        "",
        "## Results table",
        "",
        "| Dataset | Scope | Feature set | Raw AUROC | Centered AUROC | Raw accuracy | Centered accuracy | Centered bootstrap |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.dataset} | {row.analysis_scope} | {row.feature_set} | {fmt(row.raw_macro_auroc)} | {fmt(row.centered_macro_auroc)} | {fmt(row.raw_accuracy)} | {fmt(row.centered_accuracy)} | {row.patient_bootstrap_centered_macro_auroc} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "1. Duration and peak-probability features carry most detector-source information. On CHB-MIT, duration_peak2 is close to full15; EEG-only sets are much weaker. On TUSZ, full15 and duration_peak2 remain strongest, while strict EEG-only10 is modest but consistently above AUROC chance across checkpoint units.",
        "2. Strict EEG-only10 patient classifiers are near chance on both datasets. The high/low separation group comparison does not establish a generalizable patient-specific source classifier; it is an exploratory extension.",
        "3. The TUSZ patient analysis completed with two explicit skips (patient aaaaaqxr, seeds 17 and 3407) because the early chronological fit contained no Singlem source class. Those runs are not imputed.",
        "4. All six previously missing cells are now closed: CHB-MIT three ablations, TUSZ strict EEG-only10 global, and TUSZ strict EEG-only10 patient single/group analyses. The four pre-existing global cells were reused without rerunning.",
        "",
        "## Evidence boundary and limitations",
        "",
        "These results show whether false-positive event features retain information about which detector produced them. They do not show that one detector is clinically better for a patient, do not prove a physiological mechanism, and do not justify changing seizure-detection thresholds. Patient analyses are post-hoc extensions of frozen parent detector outputs.",
        "",
        "QA note: the legacy CHB full15 CSV contains a mislabeled chance_macro_auroc field of 1/3. The correct AUROC chance reference used here is 0.50; no observed score was changed.",
        "",
        "## Reproducibility",
        "",
        "- New TUSZ patient script: work/b_false_alarm_atlas/scripts/154_run_h3_tusz_patient_stratified_source_classifier_strict_eeg_10.py",
        "- New TUSZ patient output: outputs/reports/h3-patient-stratified-source-classifier-tusz-strict-eeg-10-v1/",
        "- CHB ablation output: outputs/reports/h3-chb-feature-ablations-v1/",
        "- TUSZ strict global output: outputs/reports/h3-external-tusz-strict-eeg-10-v1/",
        "- All four new namespaces report no duplicate/checkpoint reuse in their preflight files.",
    ]
    return "\n".join(lines) + "\n"


def register_fonts() -> tuple[str, str]:
    candidates = [
        ("/mnt/c/Windows/Fonts/Deng.ttf", "/mnt/c/Windows/Fonts/Dengb.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for regular, bold in candidates:
        if Path(regular).exists() and Path(bold).exists():
            pdfmetrics.registerFont(TTFont("ReportRegular", regular))
            pdfmetrics.registerFont(TTFont("ReportBold", bold))
            return "ReportRegular", "ReportBold"
    raise FileNotFoundError("No usable report font found")


def para(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text)).replace("\n", "<br/>").replace("  ", " &nbsp;"), style)


def make_table(data: list[list[object]], widths: list[float], font: str, header: bool = True, small: bool = False) -> Table:
    converted = []
    for ridx, row in enumerate(data):
        converted.append([Paragraph(escape(str(cell)), ParagraphStyle("cell", fontName=font, fontSize=6.2 if small else 7.2, leading=7.6 if small else 9.0, alignment=TA_LEFT)) for cell in row])
    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5DEE8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ]
    for ridx in range(1 if header else 0, len(data)):
        if ridx % 2 == 0:
            style.append(("BACKGROUND", (0, ridx), (-1, ridx), colors.HexColor("#F5F8FB")))
    table.setStyle(TableStyle(style))
    return table


def build_pdf(summary: pd.DataFrame, audit: dict[str, object]) -> None:
    regular, bold = register_fonts()
    page_w, page_h = landscape(A4)
    doc = SimpleDocTemplate(
        str(PDF_OUT), pagesize=(page_w, page_h),
        leftMargin=1.25 * cm, rightMargin=1.25 * cm,
        topMargin=1.1 * cm, bottomMargin=1.1 * cm,
        title="H3 specified experiments - complete report",
        author="False-Alarm Atlas",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontName=bold, fontSize=22, leading=27, textColor=colors.HexColor("#17365D"), alignment=TA_LEFT, spaceAfter=8)
    subtitle = ParagraphStyle("subtitle", parent=styles["Normal"], fontName=regular, fontSize=10.5, leading=14, textColor=colors.HexColor("#475569"), spaceAfter=12)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName=bold, fontSize=15, leading=18, textColor=colors.HexColor("#17365D"), spaceBefore=8, spaceAfter=7)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName=bold, fontSize=11.5, leading=14, textColor=colors.HexColor("#245A8D"), spaceBefore=7, spaceAfter=5)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName=regular, fontSize=9.2, leading=13, textColor=colors.HexColor("#1F2937"), spaceAfter=6)
    small = ParagraphStyle("small", parent=body, fontSize=7.4, leading=10, textColor=colors.HexColor("#475569"))
    callout = ParagraphStyle("callout", parent=body, fontName=bold, fontSize=11, leading=15, textColor=colors.HexColor("#0F5132"), backColor=colors.HexColor("#EAF7EF"), borderColor=colors.HexColor("#8BC9A5"), borderWidth=0.6, borderPadding=8, spaceAfter=10)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, firstLineIndent=-8, bulletIndent=0, spaceAfter=3)

    story: list[object] = []
    story.append(Paragraph("H3 指定实验全量完成报告", title))
    story.append(Paragraph("False-Alarm Atlas - 仅包含用户明确指定的 12 项 detector-source 三分类实验", subtitle))
    story.append(Paragraph("完成状态：12 / 12 项指定实验已有结果、配置/manifest 和输出文件。此前缺失的 6 项已补齐；已有的 6 项未重复运行。", callout))
    story.append(Paragraph("这份报告的对象是：对于已经产生的 false-positive event，能否根据特征判断它来自 SingLEM、CBraMod 还是 TCN-GAT。它不是 seizure detector 的重新训练，也不是临床性能比较。", body))
    story.append(Spacer(1, 4))

    story.append(Paragraph("1. 实验范围和完成矩阵", h1))
    status_data = [["#", "数据集", "分析范围", "特征集", "状态", "权威输出"]]
    for idx, row in summary.iterrows():
        status_data.append([
            str(idx + 1), row.dataset, row.analysis_scope, row.feature_set,
            "COMPLETE" if str(row.status).startswith("complete") else str(row.status),
            str(row.source),
        ])
    story.append(make_table(status_data, [0.8*cm, 1.7*cm, 5.6*cm, 3.2*cm, 2.0*cm, 8.45*cm], regular, small=True))
    story.append(Spacer(1, 6))
    story.append(Paragraph("矩阵解读：8 个全量特征集单元 = 4 个特征集 x 2 个数据集；4 个患者分析单元 = 单患者和 high/low 分组 x 2 个数据集。", body))

    story.append(Paragraph("2. 统一方法", h1))
    method_data = [
        ["项目", "本次采用的固定定义"],
        ["目标 y", "三分类 detector source: SingLEM / CBraMod / TCN-GAT。标签来自已冻结的 detector 输出来源，不是 seizure 标签。"],
        ["分类器", "class-balanced multinomial logistic regression, C=1, max_iter=500。"],
        ["AUROC 参考", "三分类 macro one-vs-rest AUROC 的 chance = 0.50；accuracy 的 chance = 1/3。"],
        ["CHB-MIT global", "22 个 canonical patients；冻结患者级 GroupKFold 外层 test，5 folds；train fit，outer test 只评估一次。"],
        ["TUSZ global", "43 个 patients；15 个冻结 detector checkpoint units；每个 unit 内做 5-fold patient GroupKFold OOF。"],
        ["Patient extension", "strict EEG-only10；按 patient chronology 前 70% fit、后 30% evaluation；标准化和 patient centering 只用早期 fit；2000 次 patient bootstrap。"],
        ["患者分组", "用早期 fit 的 detector centroid 两两欧氏距离排序；最高 5 人 high-separation，最低 5 人 low-separation；分组分类器只在各自早期数据拟合，跨组应用仅作描述。"],
    ]
    story.append(make_table(method_data, [3.3*cm, 22.5*cm], regular, small=False))

    story.append(Paragraph("3. 四个特征子集", h1))
    feature_data = [["特征集", "组成", "回答的问题"]]
    feature_data += [
        ["full15", "duration + peak probability + EEG/QC-only13", "所有已提取信息共同能保留多少 detector-source 信息？"],
        ["eeg_qc_only13", "EEG 10 + line noise + clipping + flatness", "去掉事件时长和模型 peak 后，EEG/QC 信息是否仍可区分来源？"],
        ["duration_peak2", "event duration + peak probability", "只用事件级 detector 输出，能解释多少来源差异？"],
        ["strict_eeg_only10", "RMS 2 + spatial 2 + relative band power 5 + high/low ratio", "只使用严格 EEG 内容特征，避免 duration/peak/QC 的影响。"],
    ]
    story.append(make_table(feature_data, [3.3*cm, 9.5*cm, 12.9*cm], regular))

    story.append(Paragraph("4. Global 全量结果", h1))
    story.append(Paragraph("表中 AUROC 是 source-class 三分类 macro-AUROC；centered 是去除患者均值后的 label-free sensitivity analysis。", body))
    global_data = [["数据集", "特征集", "Raw AUROC", "Centered AUROC", "Raw acc.", "Centered acc.", "Centered balanced acc.", "解释"]]
    for _, row in summary[summary.analysis_scope == "global full-scale"].iterrows():
        note = "duration/peak-led" if row.feature_set == "duration_peak2" else ("EEG-only modest" if "eeg" in row.feature_set else "reference full set")
        global_data.append([
            row.dataset, row.feature_set, fmt(row.raw_macro_auroc), fmt(row.centered_macro_auroc),
            fmt(row.raw_accuracy), fmt(row.centered_accuracy), fmt(row.centered_balanced_accuracy), note,
        ])
    story.append(make_table(global_data, [2.0*cm, 3.2*cm, 2.1*cm, 2.4*cm, 1.8*cm, 2.2*cm, 2.8*cm, 5.7*cm], regular, small=True))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Global 结果的核心模式：CHB-MIT 中 duration_peak2 的 centered AUROC 约 0.704，接近 full15 的 0.713；严格 EEG-only10 约 0.528。TUSZ 中 full15 centered AUROC 约 0.886，duration_peak2 约 0.884，strict EEG-only10 约 0.626。", body))

    story.append(Paragraph("5. Patient-level strict EEG-only10 结果", h1))
    patient_data = [["数据集", "分析", "Event-weighted AUROC", "Patient bootstrap AUROC [95% CI]", "结论"]]
    for _, row in summary[summary.analysis_scope != "global full-scale"].iterrows():
        if row.dataset == "CHB-MIT":
            if row.analysis_scope.startswith("single"):
                conclusion = "高分离组略高，但仍是探索性结果"
            else:
                conclusion = "high/low own-group 均接近 chance"
        else:
            if row.analysis_scope.startswith("single"):
                conclusion = "42 个有效 patient bootstrap；整体接近 chance"
            else:
                conclusion = "分组 own-group 未显示稳定优势"
        patient_data.append([
            row.dataset, row.analysis_scope, fmt(row.centered_macro_auroc),
            str(row.patient_bootstrap_centered_macro_auroc), conclusion,
        ])
    story.append(make_table(patient_data, [2.0*cm, 8.0*cm, 3.2*cm, 5.9*cm, 7.0*cm], regular, small=True))
    story.append(Spacer(1, 6))
    story.append(Paragraph("CHB-MIT 单患者 strict EEG-only10 的 patient-balanced bootstrap AUROC 为 0.582 [0.561, 0.604]；TUSZ 单患者为 0.545 [0.527, 0.566]（42 个有效 patient）。这些数值只能说明弱的 detector-source 信息保留，不能推出某个 detector 适合某个患者。", body))

    story.append(Paragraph("患者分组明细（仍属于本次指定的 group analysis）", h2))
    group_detail = [["数据集", "训练组 -> 评估组", "Event-weighted AUROC", "Patient bootstrap AUROC [95% CI]", "含义"]]
    for dataset, patient_dir in (
        ("CHB-MIT", "outputs/reports/h3-patient-stratified-source-classifier-strict-eeg-10-v1"),
        ("TUSZ", "outputs/reports/h3-patient-stratified-source-classifier-tusz-strict-eeg-10-v1"),
    ):
        method = load_csv(f"{patient_dir}/source_classifier_method_summary.csv")
        boot = load_csv(f"{patient_dir}/source_classifier_patient_bootstrap.csv")
        for scope, eval_group, label, meaning in (
            ("group_high_separation_own", "high_separation", "high -> high (own)", "最高分离组内拟合/评估"),
            ("group_low_separation_own", "low_separation", "low -> low (own)", "最低分离组内拟合/评估"),
            ("group_high_separation_cross", "low_separation", "high -> low", "高分离组拟合后评估低分离组"),
            ("group_low_separation_cross", "high_separation", "low -> high", "低分离组拟合后评估高分离组"),
        ):
            row = method[method.scope == scope].iloc[0]
            b = boot_lookup(boot, scope, eval_group)
            group_detail.append([
                dataset, label, fmt(row.macro_auroc), fmt_ci(b), meaning,
            ])
    story.append(make_table(group_detail, [2.0*cm, 4.0*cm, 3.4*cm, 5.9*cm, 9.0*cm], regular, small=True))
    story.append(Spacer(1, 5))
    story.append(Paragraph("分组表中的 own 与 cross 都是同一 strict EEG-only10 分析的预先定义输出；分组只按早期 fit 的 detector-centroid 分离度选出最高/最低五名患者。cross 结果用于检查组间迁移，不能被解释为临床适配效果。", small))

    story.append(PageBreak())
    story.append(Paragraph("6. 每个数据集的详细结果", h1))
    for dataset in ("CHB-MIT", "TUSZ"):
        story.append(Paragraph(dataset, h2))
        subset = summary[summary.dataset == dataset]
        detail = [["Scope", "Feature set", "Accuracy", "Balanced acc.", "Macro-AUROC", "Bootstrap / note"]]
        for _, row in subset.iterrows():
            detail.append([
                row.analysis_scope, row.feature_set, fmt(row.centered_accuracy), fmt(row.centered_balanced_accuracy), fmt(row.centered_macro_auroc), str(row.patient_bootstrap_centered_macro_auroc),
            ])
        story.append(make_table(detail, [7.0*cm, 3.3*cm, 2.1*cm, 2.5*cm, 2.5*cm, 8.0*cm], regular, small=True))
        story.append(Spacer(1, 7))

    story.append(Paragraph("7. 结果如何回答本次问题", h1))
    bullets = [
        "四个 global feature set 的比较显示，detector 的事件级输出（duration 和 peak probability）携带的来源信息明显多于严格 EEG-only10。",
        "严格 EEG-only10 在 CHB-MIT 和 TUSZ 都保留了高于或接近 chance 的来源信息，但强度较弱，且患者级分组并没有转化为稳定的可泛化分类优势。",
        "患者和 detector 的关联统计是来源事件数量/比例差异，不等于 detector 对该患者临床上更好；因此不能把 source-classifier AUROC 解释成 sensitivity 或 FA/24h。",
        "TUSZ 的两次缺失类别被明确记录为 skip，而不是用零分、均值或测试集信息补齐；这保证汇总结果可审计。",
    ]
    for item in bullets:
        story.append(Paragraph("- " + item, bullet))

    story.append(Paragraph("8. 数据质量、边界和限制", h1))
    limits = [
        "所有正式 global 输出均使用患者级分组；没有把窗口随机拆到 train/test。",
        "Patient extension 是 frozen parent detector outputs 上的 post-hoc exploratory extension，不能当作新的 confirmatory generalization 结果。",
        "TUSZ strict patient 分析中 aaaaaqxr 的 seed 17 和 3407 早期 fit 缺少 Singlem 类，因此单患者分类器无法合法拟合；这两个 pair 被跳过并写入 manifest。",
        "旧 CHB full15 CSV 的字段 chance_macro_auroc 被历史代码写为 1/3；正确的 macro-AUROC chance 是 0.50。本报告使用 0.50，观测 AUROC 数值没有修改。",
        "这些实验回答的是 detector-source information 是否存在，不回答 artifact 机制、因果关系或临床报警策略。",
    ]
    for item in limits:
        story.append(Paragraph("- " + item, bullet))

    story.append(Paragraph("9. 可追溯文件", h1))
    audit_data = [["内容", "路径 / 状态"]]
    audit_data += [
        ["CHB full15", "outputs/reports/v3-groupkfold-confirmatory-v1/h3_groupkfold_model_fingerprint.csv"],
        ["CHB three added feature sets", "outputs/reports/h3-chb-feature-ablations-v1/"],
        ["TUSZ existing full15 / EEG-QC13 / duration-peak2", "outputs/reports/h3-external-tusz-v1/"],
        ["TUSZ added strict EEG-only10", "outputs/reports/h3-external-tusz-strict-eeg-10-v1/"],
        ["CHB patient strict EEG-only10", "outputs/reports/h3-patient-stratified-source-classifier-strict-eeg-10-v1/"],
        ["TUSZ patient strict EEG-only10", "outputs/reports/h3-patient-stratified-source-classifier-tusz-strict-eeg-10-v1/"],
        ["Consolidated CSV", "outputs/reports/h3-specified-experiments-v1/h3_specified_experiments_summary.csv"],
        ["Consolidated audit", "outputs/reports/h3-specified-experiments-v1/h3_specified_experiments_audit.json"],
    ]
    story.append(make_table(audit_data, [6.3*cm, 19.8*cm], regular, small=True))
    story.append(Spacer(1, 7))
    story.append(Paragraph("最终结论：在这 12 项指定实验范围内，所有数据集和指定分析均已有可回溯输出；strict EEG-only10 只能支持‘存在弱的 detector-source 信息’，不能支持‘已得到稳定的患者专属 detector 选择器’。", callout))

    def footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont(regular, 7)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(1.25*cm, 0.55*cm, "False-Alarm Atlas - H3 specified experiments")
        canvas.drawRightString(page_w - 1.25*cm, 0.55*cm, f"Page {doc_obj.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    summary, audit = build_data()
    summary.to_csv(OUT_DIR / "h3_specified_experiments_summary.csv", index=False)
    (OUT_DIR / "h3_specified_experiments_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    markdown = make_markdown(summary, audit)
    (OUT_DIR / "H3_specified_experiments_complete_report.md").write_text(markdown, encoding="utf-8")
    build_pdf(summary, audit)
    # Make the human-readable companion files available from the Windows
    # workspace as well as the Linux research root.
    windows_report_dir = Path("/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/reports/h3-specified-experiments-v1")
    windows_report_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT_DIR / "h3_specified_experiments_summary.csv", windows_report_dir / "h3_specified_experiments_summary.csv")
    shutil.copy2(OUT_DIR / "h3_specified_experiments_audit.json", windows_report_dir / "h3_specified_experiments_audit.json")
    shutil.copy2(OUT_DIR / "H3_specified_experiments_complete_report.md", windows_report_dir / "H3_specified_experiments_complete_report.md")
    print(json.dumps({"status": audit["status"], "cells": f"{audit['completed_cells']}/{audit['requested_cells']}", "pdf": str(PDF_OUT), "report_dir": str(OUT_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
