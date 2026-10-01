from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "continuous_alarm_duration_chb_tusz_v1"
OUT.mkdir(parents=True, exist_ok=True)

CHB_EVENTS = Path(
    r"\\wsl.localhost\Ubuntu-24.04\root\b_false_alarm_atlas\outputs\reports\v3-groupkfold-confirmatory-v1\oof_events.parquet"
)
CHB_RECORDINGS = Path(
    r"\\wsl.localhost\Ubuntu-24.04\root\b_false_alarm_atlas\manifests\recordings.parquet"
)
CHB_SEIZURES = Path(
    r"\\wsl.localhost\Ubuntu-24.04\root\b_false_alarm_atlas\manifests\seizures.parquet"
)
CHB_RUN_METRICS = Path(
    r"\\wsl.localhost\Ubuntu-24.04\root\b_false_alarm_atlas\outputs\reports\v3-groupkfold-confirmatory-v1\oof_run_metrics.csv"
)
TUSZ_EVENTS = Path(
    r"\\wsl.localhost\Ubuntu-24.04\root\b_false_alarm_atlas\outputs\reports\v3-compact-extension-v1\tusz_eval_full_v1_events.parquet"
)
TUSZ_MANIFEST = Path(
    r"\\wsl.localhost\Ubuntu-24.04\root\b_false_alarm_atlas\outputs\reports\v3-compact-extension-v1\tusz_eval_full_v1_manifest.json"
)

MODEL_ORDER = ["singlem", "cbramod", "tcn_gat"]
MODEL_LABELS = {"singlem": "SingLEM", "cbramod": "CBraMod", "tcn_gat": "TCN-GAT"}
MODEL_COLORS = {"singlem": "#386cb0", "cbramod": "#f0027f", "tcn_gat": "#1b9e77"}


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    values = sorted((float(s), float(e)) for s, e in intervals if float(e) > float(s))
    if not values:
        return []
    merged: list[list[float]] = [[values[0][0], values[0][1]]]
    for start, end in values[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def interval_length(intervals: Iterable[tuple[float, float]]) -> float:
    return float(sum(e - s for s, e in merge_intervals(intervals)))


def subtract_intervals(
    intervals: Iterable[tuple[float, float]], subtract: Iterable[tuple[float, float]]
) -> float:
    base = merge_intervals(intervals)
    cuts = merge_intervals(subtract)
    if not base:
        return 0.0
    if not cuts:
        return interval_length(base)
    total = 0.0
    for start, end in base:
        cursor = start
        for cut_start, cut_end in cuts:
            if cut_end <= cursor:
                continue
            if cut_start >= end:
                break
            if cut_start > cursor:
                total += max(0.0, min(cut_start, end) - cursor)
            cursor = max(cursor, cut_end)
            if cursor >= end:
                break
        if cursor < end:
            total += end - cursor
    return float(total)


def format_seconds(value: float) -> str:
    return f"{float(value):,.0f} s"


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / len(x) * 100.0
    return x, y


def build_chb() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    events = pd.read_parquet(CHB_EVENTS).copy()
    events["duration_s"] = (events["end_s"] - events["start_s"]).clip(lower=0)
    recordings = pd.read_parquet(CHB_RECORDINGS).copy()
    seizures = pd.read_parquet(CHB_SEIZURES).copy()
    run_metrics = pd.read_csv(CHB_RUN_METRICS)
    truth_map: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for (patient, recording), group in seizures.groupby(["patient_id", "recording_id"], sort=False):
        truth_map[(str(patient), str(recording))] = list(
            zip(group["start_s"].astype(float), group["end_s"].astype(float))
        )

    events["off_seizure_duration_s"] = [
        subtract_intervals(
            [(row.start_s, row.end_s)],
            truth_map.get((str(row.patient), str(row.recording)), []),
        )
        for row in events.itertuples()
    ]
    events["dataset"] = "CHB-MIT"
    events["event_type_display"] = np.where(events["event_type"].eq("fp"), "false alarm", "seizure-associated")

    raw_total_s = float(recordings["duration_s"].sum())
    seed_count = int(run_metrics["seed"].nunique())
    summary_rows: list[dict] = []
    for model in MODEL_ORDER:
        g = events[events["model"].eq(model)]
        groups = g.groupby(["model", "fold", "seed", "patient", "recording"], sort=False)
        all_union_s = 0.0
        off_union_s = 0.0
        for _, h in groups:
            predicted = list(zip(h["start_s"], h["end_s"]))
            truth = truth_map.get((str(h.iloc[0]["patient"]), str(h.iloc[0]["recording"])), [])
            all_union_s += interval_length(predicted)
            off_union_s += subtract_intervals(predicted, truth)
        event_sum_s = float(g["duration_s"].sum())
        fp_event_sum_s = float(g.loc[g["event_type"].eq("fp"), "duration_s"].sum())
        summary_rows.append(
            {
                "dataset": "CHB-MIT",
                "model": model,
                "aggregation": "pooled 15 held-out run-evaluations (5 folds x 3 seeds)",
                "run_evaluations": int(g[["fold", "seed"]].drop_duplicates().shape[0]),
                "unique_recordings": int(recordings.shape[0]),
                "monitoring_s_raw": raw_total_s * seed_count,
                "monitoring_s_raw_per_seed": raw_total_s,
                "alarm_s_continuous_union": all_union_s,
                "alarm_fraction_raw_pct": all_union_s / (raw_total_s * seed_count) * 100.0,
                "off_seizure_alarm_s_continuous_union": off_union_s,
                "off_seizure_alarm_fraction_raw_pct": off_union_s / (raw_total_s * seed_count) * 100.0,
                "event_duration_sum_s": event_sum_s,
                "false_alarm_event_duration_sum_s": fp_event_sum_s,
                "alarm_events": int(len(g)),
                "false_alarm_events": int((g["event_type"] == "fp").sum()),
                "seizure_associated_events": int((g["event_type"] == "tp").sum()),
                "median_event_duration_s": float(g["duration_s"].median()),
                "p90_event_duration_s": float(g["duration_s"].quantile(0.90)),
                "p95_event_duration_s": float(g["duration_s"].quantile(0.95)),
                "max_event_duration_s": float(g["duration_s"].max()),
                "continuous_union_overlap_check": bool(abs(all_union_s - event_sum_s) < 1e-6),
            }
        )
        # Add complete-pass values by seed to avoid hiding seed heterogeneity.
        seed_rows: list[dict] = []
        for seed, h in g.groupby("seed", sort=True):
            seed_row = {
                    "dataset": "CHB-MIT",
                    "model": model,
                    "aggregation": f"complete dataset pass, seed {int(seed)}",
                    "run_evaluations": int(h["fold"].nunique()),
                    "unique_recordings": int(recordings.shape[0]),
                    "monitoring_s_raw": raw_total_s,
                    "monitoring_s_raw_per_seed": raw_total_s,
                    "alarm_s_continuous_union": float(h["duration_s"].sum()),
                    "alarm_fraction_raw_pct": float(h["duration_s"].sum()) / raw_total_s * 100.0,
                    "off_seizure_alarm_s_continuous_union": float(h["off_seizure_duration_s"].sum()),
                    "off_seizure_alarm_fraction_raw_pct": float(h["off_seizure_duration_s"].sum()) / raw_total_s * 100.0,
                    "event_duration_sum_s": float(h["duration_s"].sum()),
                    "false_alarm_event_duration_sum_s": float(h.loc[h["event_type"].eq("fp"), "duration_s"].sum()),
                    "alarm_events": int(len(h)),
                    "false_alarm_events": int((h["event_type"] == "fp").sum()),
                    "seizure_associated_events": int((h["event_type"] == "tp").sum()),
                    "median_event_duration_s": float(h["duration_s"].median()),
                    "p90_event_duration_s": float(h["duration_s"].quantile(0.90)),
                    "p95_event_duration_s": float(h["duration_s"].quantile(0.95)),
                    "max_event_duration_s": float(h["duration_s"].max()),
                    "continuous_union_overlap_check": True,
                }
            seed_rows.append(seed_row)
            summary_rows.append(seed_row)
        # A single complete-pass estimate, averaged over the three pre-registered
        # seeds, is easier to interpret than a denominator repeated three times.
        if seed_rows:
            numeric_mean = lambda key: float(np.mean([row[key] for row in seed_rows]))
            summary_rows.append(
                {
                    "dataset": "CHB-MIT",
                    "model": model,
                    "aggregation": "mean across 3 complete dataset passes",
                    "run_evaluations": 5,
                    "unique_recordings": int(recordings.shape[0]),
                    "monitoring_s_raw": raw_total_s,
                    "monitoring_s_raw_per_seed": raw_total_s,
                    "alarm_s_continuous_union": numeric_mean("alarm_s_continuous_union"),
                    "alarm_fraction_raw_pct": numeric_mean("alarm_fraction_raw_pct"),
                    "off_seizure_alarm_s_continuous_union": numeric_mean("off_seizure_alarm_s_continuous_union"),
                    "off_seizure_alarm_fraction_raw_pct": numeric_mean("off_seizure_alarm_fraction_raw_pct"),
                    "event_duration_sum_s": numeric_mean("event_duration_sum_s"),
                    "false_alarm_event_duration_sum_s": numeric_mean("false_alarm_event_duration_sum_s"),
                    "alarm_events": numeric_mean("alarm_events"),
                    "false_alarm_events": numeric_mean("false_alarm_events"),
                    "seizure_associated_events": numeric_mean("seizure_associated_events"),
                    "median_event_duration_s": numeric_mean("median_event_duration_s"),
                    "p90_event_duration_s": numeric_mean("p90_event_duration_s"),
                    "p95_event_duration_s": numeric_mean("p95_event_duration_s"),
                    "max_event_duration_s": numeric_mean("max_event_duration_s"),
                    "continuous_union_overlap_check": True,
                }
            )

    # Event-level rows are retained for reproducible distribution plots.
    event_rows = events[
        [
            "dataset",
            "model",
            "fold",
            "seed",
            "patient",
            "recording",
            "event_type",
            "start_s",
            "end_s",
            "duration_s",
            "off_seizure_duration_s",
        ]
    ].copy()
    meta = {
        "source_events": str(CHB_EVENTS),
        "source_recordings": str(CHB_RECORDINGS),
        "source_seizures": str(CHB_SEIZURES),
        "definition": "Each event interval is end_s - start_s from canonical v3 GroupKFold OOF event output. Continuous totals are interval unions within model/fold/seed/patient/recording.",
        "raw_monitoring_s_per_seed": raw_total_s,
        "raw_monitoring_h_per_seed": raw_total_s / 3600.0,
        "seed_count": seed_count,
        "pooled_denominator": "raw monitoring seconds repeated once per held-out seed (15 runs per model)",
    }
    return event_rows, pd.DataFrame(summary_rows), meta


def build_tusz() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    events = pd.read_parquet(TUSZ_EVENTS).copy()
    events["event_type"] = np.where(events["matched_truth"].astype(bool), "tp", "fp")
    events["duration_s"] = (events["end_s"] - events["start_s"]).clip(lower=0)
    manifest = json.loads(TUSZ_MANIFEST.read_text())
    truth_map: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for rec in manifest["records"]:
        truth_map[(str(rec["patient_id"]), str(rec["relative_edf"]))] = [
            (float(x["start_s"]), float(x["end_s"])) for x in rec.get("seizure_events", [])
        ]
    events["off_seizure_duration_s"] = [
        subtract_intervals(
            [(row.start_s, row.end_s)],
            truth_map.get((str(row.patient_id), str(row.recording)), []),
        )
        for row in events.itertuples()
    ]
    events["dataset"] = "TUSZ"
    event_rows = events[
        [
            "dataset",
            "model",
            "patient_id",
            "recording",
            "event_index",
            "event_type",
            "start_s",
            "end_s",
            "duration_s",
            "off_seizure_duration_s",
        ]
    ].copy()
    raw_total_s = float(manifest["total_duration_s"])
    summary_rows: list[dict] = []
    for model in MODEL_ORDER:
        g = events[events["model"].eq(model)]
        groups = g.groupby(["model", "patient_id", "recording"], sort=False)
        all_union_s = 0.0
        off_union_s = 0.0
        for _, h in groups:
            predicted = list(zip(h["start_s"], h["end_s"]))
            truth = truth_map.get((str(h.iloc[0]["patient_id"]), str(h.iloc[0]["recording"])), [])
            all_union_s += interval_length(predicted)
            off_union_s += subtract_intervals(predicted, truth)
        summary_rows.append(
            {
                "dataset": "TUSZ",
                "model": model,
                "aggregation": "full eval v1, one frozen output per detector",
                "run_evaluations": 1,
                "unique_recordings": int(manifest["record_count"]),
                "monitoring_s_raw": raw_total_s,
                "monitoring_s_raw_per_seed": raw_total_s,
                "alarm_s_continuous_union": all_union_s,
                "alarm_fraction_raw_pct": all_union_s / raw_total_s * 100.0,
                "off_seizure_alarm_s_continuous_union": off_union_s,
                "off_seizure_alarm_fraction_raw_pct": off_union_s / raw_total_s * 100.0,
                "event_duration_sum_s": float(g["duration_s"].sum()),
                "false_alarm_event_duration_sum_s": float(g.loc[g["event_type"].eq("fp"), "duration_s"].sum()),
                "alarm_events": int(len(g)),
                "false_alarm_events": int((g["event_type"] == "fp").sum()),
                "seizure_associated_events": int((g["event_type"] == "tp").sum()),
                "median_event_duration_s": float(g["duration_s"].median()),
                "p90_event_duration_s": float(g["duration_s"].quantile(0.90)),
                "p95_event_duration_s": float(g["duration_s"].quantile(0.95)),
                "max_event_duration_s": float(g["duration_s"].max()),
                "continuous_union_overlap_check": bool(abs(all_union_s - g["duration_s"].sum()) < 1e-6),
            }
        )
    meta = {
        "source_events": str(TUSZ_EVENTS),
        "source_manifest": str(TUSZ_MANIFEST),
        "definition": "Each event interval is end_s - start_s from the frozen full-eval event output. Continuous totals are interval unions within model/patient/recording.",
        "raw_monitoring_s": raw_total_s,
        "raw_monitoring_h": raw_total_s / 3600.0,
        "pooled_denominator": "all 880 frozen eval recordings",
    }
    return event_rows, pd.DataFrame(summary_rows), meta


def plot_duration_figure(chb_events: pd.DataFrame, tusz_events: pd.DataFrame, summary: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(15, 15), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.12, hspace=0.48, wspace=0.22)
    fig.suptitle(
        "Continuous alarm duration and alarm-time burden\n"
        "Frozen detector outputs from CHB-MIT and TUSZ",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )

    def plot_ecdf(ax: plt.Axes, data: pd.DataFrame, dataset: str, false_only: bool) -> None:
        for model in MODEL_ORDER:
            g = data[data["model"].eq(model)]
            if false_only:
                vals = g.loc[g["event_type"].eq("fp"), "duration_s"].to_numpy(float)
            else:
                vals = g["duration_s"].to_numpy(float)
            x, y = ecdf(vals)
            ax.step(x, y, where="post", color=MODEL_COLORS[model], linewidth=2.0, label=MODEL_LABELS[model])
        ax.set_xscale("log")
        ax.set_xlim(3.5, max(float(data["duration_s"].max()) * 1.12, 10.0))
        ax.set_ylim(0, 100)
        ax.set_xlabel("Continuous alarm duration (s; log scale)")
        ax.set_ylabel("Alarm intervals ≤ duration (%)")
        ax.grid(True, which="both", color="#d9d9d9", linewidth=0.6, alpha=0.65)
        ax.set_title(
            f"{dataset}: {'false-alarm intervals' if false_only else 'all alarm intervals'}",
            loc="left",
            fontweight="bold",
        )

    plot_ecdf(axes[0, 0], chb_events, "CHB-MIT", False)
    plot_ecdf(axes[0, 1], tusz_events, "TUSZ", False)
    plot_ecdf(axes[1, 0], chb_events, "CHB-MIT", True)
    plot_ecdf(axes[1, 1], tusz_events, "TUSZ", True)
    axes[0, 0].legend(loc="lower right", frameon=False)

    def plot_burden(ax: plt.Axes, dataset: str) -> None:
        if dataset == "CHB-MIT":
            aggregation = "mean across 3 complete dataset passes"
        else:
            aggregation = "full eval v1, one frozen output per detector"
        sub = summary[(summary["dataset"] == dataset) & (summary["aggregation"] == aggregation)].copy()
        sub = sub.set_index("model").loc[MODEL_ORDER].reset_index()
        x = np.arange(len(MODEL_ORDER))
        width = 0.36
        all_pct = sub["alarm_fraction_raw_pct"].to_numpy(float)
        off_pct = sub["off_seizure_alarm_fraction_raw_pct"].to_numpy(float)
        b1 = ax.bar(x - width / 2, all_pct, width, color="#6a3d9a", label="all predicted alarm time")
        b2 = ax.bar(x + width / 2, off_pct, width, color="#ff7f00", label="outside seizure intervals")
        ax.set_xticks(x, [MODEL_LABELS[m] for m in MODEL_ORDER])
        ax.set_ylabel("Share of raw monitoring time (%)")
        ax.set_title(f"{dataset}: total alarm-time share", loc="left", fontweight="bold")
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.65)
        ax.set_axisbelow(True)
        ymax = max(float(max(all_pct.max(), off_pct.max())) * 1.38, 1.0)
        ax.set_ylim(0, ymax)
        for bars, col in [(b1, sub["alarm_s_continuous_union"].to_numpy(float)), (b2, sub["off_seizure_alarm_s_continuous_union"].to_numpy(float))]:
            for bar, seconds, pct in zip(bars, col, (all_pct if bars is b1 else off_pct)):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax * 0.025,
                    f"{format_seconds(seconds)}\n({pct:.2f}%)",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                )
        ax.legend(loc="upper left", frameon=False)
        hours = float(sub.iloc[0]["monitoring_s_raw"] / 3600.0)
        ax.text(
            0.99,
            0.96,
            f"denominator = {hours:,.1f} h" + (" per complete pass" if dataset == "CHB-MIT" else ""),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#555555",
        )

    plot_burden(axes[2, 0], "CHB-MIT")
    plot_burden(axes[2, 1], "TUSZ")
    fig.text(
        0.08,
        0.035,
        "CHB-MIT curves pool the 15 held-out run-evaluations per detector (5 patient folds × 3 seeds); bars show the mean seconds per complete pass. TUSZ uses one frozen full-eval output per detector.\n"
        "Alarm-time totals use interval unions within each run/recording. Percentages use raw monitoring time; the orange series removes seizure intervals.",
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#444444",
    )
    fig.savefig(OUT / "continuous_alarm_duration_chb_tusz.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "continuous_alarm_duration_chb_tusz.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    chb_events, chb_summary, chb_meta = build_chb()
    tusz_events, tusz_summary, tusz_meta = build_tusz()
    summary = pd.concat([chb_summary, tusz_summary], ignore_index=True)
    all_events = pd.concat([chb_events, tusz_events], ignore_index=True)
    summary.to_csv(OUT / "continuous_alarm_duration_summary.csv", index=False)
    all_events.to_csv(OUT / "continuous_alarm_duration_event_rows.csv", index=False)
    manifest = {
        "analysis_id": "continuous-alarm-duration-chb-tusz-v1",
        "status": "complete_read_only_summary",
        "sources": {"CHB-MIT": chb_meta, "TUSZ": tusz_meta},
        "event_definition": "duration_s = end_s - start_s from frozen event tables; no threshold or eventization changes",
        "continuous_total_definition": "union of predicted intervals within model and recording; CHB also separates fold and seed",
        "false_alarm_time_definition": "predicted interval time outside all seizure annotation intervals",
        "outputs": [
            "continuous_alarm_duration_summary.csv",
            "continuous_alarm_duration_event_rows.csv",
            "continuous_alarm_duration_chb_tusz.png",
            "continuous_alarm_duration_chb_tusz.pdf",
            "manifest.json",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_duration_figure(chb_events, tusz_events, summary)
    # Print only the headline pooled rows for easy inspection.
    pooled = summary[summary["aggregation"].str.startswith("pooled") | summary["aggregation"].str.startswith("full eval")]
    print(pooled[[
        "dataset", "model", "monitoring_s_raw", "alarm_s_continuous_union", "alarm_fraction_raw_pct",
        "off_seizure_alarm_s_continuous_union", "off_seizure_alarm_fraction_raw_pct", "alarm_events",
        "false_alarm_events", "median_event_duration_s", "p90_event_duration_s", "max_event_duration_s",
    ]].to_string(index=False))
    print(f"\nWrote outputs to {OUT}")


if __name__ == "__main__":
    main()
