"""Make the BSN pilot performance and H1 summary figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs/figures/BSN-pilot-v1"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    performance = pd.read_csv(ROOT / "outputs/reports/BSN-pilot-v1/model_summary_3seeds.csv")
    h1 = pd.read_csv(ROOT / "outputs/reports/bsn_pilot_v1_h1_pairwise.csv")
    colors = {"singlem": "#35608d", "cbramod": "#d07a2d", "tcn_gat": "#248277"}
    labels = {"singlem": "SingLEM", "cbramod": "CBraMod", "tcn_gat": "TCN-GAT"}
    pair_labels = {
        ("singlem", "cbramod"): "SingLEM–CBraMod",
        ("singlem", "tcn_gat"): "SingLEM–TCN-GAT",
        ("cbramod", "tcn_gat"): "CBraMod–TCN-GAT",
    }

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)

    ax = axes[0]
    for model, group in performance.groupby("model", sort=False):
        ax.scatter(
            group.test_fa_per_24h,
            group.test_event_sensitivity * 100,
            s=75,
            color=colors[model],
            label=labels[model],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for row in group.itertuples(index=False):
            ax.annotate(str(row.seed), (row.test_fa_per_24h, row.test_event_sensitivity * 100), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Test false alarms per 24 h")
    ax.set_ylabel("Test event sensitivity (%)")
    ax.set_title("A  Fixed-budget detector performance")
    ax.grid(axis="both", alpha=0.18)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    x_positions = list(range(len(h1)))
    observed = h1.observed_patient_median_f1.to_numpy()
    null = h1.null_patient_median_f1.to_numpy()
    q025 = h1.null_f1_q025.to_numpy()
    q975 = h1.null_f1_q975.to_numpy()
    pair_colors = [colors["singlem"] if r.left_model == "singlem" and r.right_model == "cbramod" else colors["cbramod"] if r.left_model == "cbramod" else "#6d597a" for r in h1.itertuples()]
    ax.errorbar(x_positions, null, yerr=[null - q025, q975 - null], fmt="_", color="#858585", capsize=3, linewidth=1.3, label="Circular-shift null")
    ax.scatter(x_positions, observed, c=pair_colors, s=58, edgecolor="white", linewidth=0.8, zorder=3, label="Observed FP match F1")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{pair_labels[(r.left_model, r.right_model)]}\nseed {r.seed}" for r in h1.itertuples()], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Patient-median FP onset-match F1")
    ax.set_title("B  Temporal sharing of false alarms")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle("BSN-pilot-v1: preliminary false-alarm structure", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "bsn_pilot_v1_main.png", dpi=220, facecolor="white")
    fig.savefig(OUT / "bsn_pilot_v1_main.pdf", facecolor="white")
    print(OUT / "bsn_pilot_v1_main.png")


if __name__ == "__main__":
    main()
