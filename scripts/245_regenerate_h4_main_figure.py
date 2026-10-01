"""Regenerate the locked H4 diagnostic figure without recomputing analyses."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path("/root/b_false_alarm_atlas/outputs/reports/h4-final-linkage-expert-audit-v1")


def main() -> None:
    decision = pd.read_csv(OUT / "final_decision_table.csv")
    shared = pd.read_csv(OUT / "h1_h4_sharedness_summary.csv")
    patient_metrics = pd.read_csv(
        "/root/b_false_alarm_atlas/outputs/reports/h4-protocol-matched-oracle-v2/patient_policy_metrics.csv"
    )
    colors = {
        "best_single": "#374151",
        "router_model_aware": "#D55E00",
        "forced_selection_oracle": "#7C3AED",
        "full_abstaining_oracle": "#009E73",
    }
    fig, axes = plt.subplots(
        1, 3, figsize=(13.2, 4.0), gridspec_kw={"width_ratios": [1.1, 1.05, 1.25]}
    )

    ax = axes[0]
    for row in decision.itertuples(index=False):
        if row.policy not in colors:
            continue
        ax.scatter(row.fa_per_24h, row.sensitivity, s=75, color=colors[row.policy], zorder=3)
        label = {
            "best_single": "Best single",
            "router_model_aware": "Learned router",
            "forced_selection_oracle": "Forced oracle",
            "full_abstaining_oracle": "Abstaining oracle",
        }[row.policy]
        ax.annotate(label, (row.fa_per_24h, row.sensitivity), xytext=(5, 5), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("False alarms / 24 h")
    ax.set_ylabel("Event sensitivity")
    ax.set_title("A  Policy gap")
    ax.grid(alpha=0.2)

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
    for x, y in zip(range(len(data)), 100 * data.retained_rate):
        ax.text(x, y + 2, f"{y:.1f}%", ha="center", fontsize=8)

    ax = axes[2]
    base = patient_metrics[patient_metrics.method == "best_single"].set_index("patient")
    route = patient_metrics[patient_metrics.method == "router_model_aware"].set_index("patient")
    ora = patient_metrics[patient_metrics.method == "oracle_3"].set_index("patient")
    common = sorted(set(base.index) & set(route.index) & set(ora.index))
    table = pd.DataFrame(
        {
            "patient": common,
            "router": [route.loc[p].fa_per_24h - base.loc[p].fa_per_24h for p in common],
            "oracle": [ora.loc[p].fa_per_24h - base.loc[p].fa_per_24h for p in common],
        }
    ).sort_values("oracle")
    x = np.arange(len(table))
    ax.scatter(x, table.router, s=20, color=colors["router_model_aware"], label="Router − best")
    ax.scatter(x, table.oracle, s=20, color=colors["full_abstaining_oracle"], label="Oracle − best")
    ax.axhline(0, color="#111827", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(table.patient, rotation=65, ha="right", fontsize=6)
    ax.set_ylabel("Δ FA / 24 h")
    ax.set_title("C  Patient heterogeneity")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUT / "figure_main_h4_gap_linkage.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figure_main_h4_gap_linkage.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
