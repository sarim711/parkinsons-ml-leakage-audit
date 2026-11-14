"""
Draw the paper's figures from the CSVs written by run.py and analyses.py.

Usage:
    python figures.py [--results results] [--out figures]

Needs: results/leakage_comparison.csv (run.py leakage-demo),
       results/baselines.csv, feature_groups_folds.csv, patients_vs_recordings.csv,
       calibration.csv (analyses.py all).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Categorical slots 1-3 of the reference palette (validated all-pairs in light
# mode); every series also has its own marker so identity never rests on colour.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "legend.frameon": False, "legend.fontsize": 8, "savefig.dpi": 300,
})


def _grid(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)


DOTTED, DASHDOT = (0, (1, 2)), (0, (5, 2, 1, 2))


def _ref(ax, y, text=None, ha="right", va="bottom", x=None, style=DOTTED, label=None):
    """Reference line; labelled in place (text) or through the legend (label)."""
    ax.axhline(y, color=MUTED, linewidth=0.9, linestyle=style, zorder=1, label=label)
    if text:
        xmin, xmax = ax.get_xlim()
        ax.text(x if x is not None else (xmax if ha == "right" else xmin), y + (0.01 if va == "bottom" else -0.01),
                text, color=MUTED, fontsize=7.5, ha=ha, va=va)


def figure1(res, out):
    """Same model, two ways of splitting the data."""
    leak = pd.read_csv(os.path.join(res, "leakage_comparison.csv")).set_index("protocol")
    base = pd.read_csv(os.path.join(res, "baselines.csv")).set_index("name")
    rows = [("Split by\nrecording", "row_level_stratified_kfold", ORANGE),
            ("Split by\npatient", "patient_disjoint_stratified_group_kfold", BLUE)]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    for ax, metric, label in ((axes[0], "mae", "Motor-UPDRS MAE (points)"), (axes[1], "auc", "AUC")):
        for i, (name, key, colour) in enumerate(rows):
            v, sd = leak.loc[key, metric], leak.loc[key, f"{metric}_std"]
            ax.bar(i, v, width=0.55, color=colour, yerr=sd, capsize=3,
                   error_kw={"elinewidth": 0.8, "ecolor": INK})
            ax.text(i, v + sd + (0.15 if metric == "mae" else 0.02), f"{v:.2f}" if metric == "mae" else f"{v:.3f}",
                    ha="center", va="bottom", color=INK, fontsize=8)
        ax.set_xticks([0, 1], [r[0] for r in rows]); ax.set_xlim(-0.6, 1.6); ax.set_ylabel(label); _grid(ax)
    axes[0].set_ylim(0, 10.5)
    _ref(axes[0], base.loc["overall_average_training_mean", "mae"], style=DOTTED,
         label=f"Overall-average baseline, MAE {base.loc['overall_average_training_mean', 'mae']:.2f}")
    _ref(axes[0], base.loc["patient_average_baseline", "mae"], style=DASHDOT,
         label=f"Patient-average baseline, MAE {base.loc['patient_average_baseline', 'mae']:.2f}")
    axes[1].set_ylim(0, 1.12)
    _ref(axes[1], 0.5, style=(0, (4, 2)), label="Chance, AUC 0.5")
    handles = axes[0].get_legend_handles_labels()[0] + axes[1].get_legend_handles_labels()[0]
    fig.legend(handles=handles, loc="lower center", ncol=3, handlelength=2.8, bbox_to_anchor=(0.5, 0.0))
    axes[0].set_title("a  Error (lower is better)", loc="left")
    axes[1].set_title("b  Discrimination at the cut-off of 20", loc="left")
    fig.set_size_inches(6.5, 3.1)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(out, "figure1.png")); plt.close(fig)


def figure2(res, out):
    """Per-fold AUC of the final model with all features, demographics only, voice only."""
    f = pd.read_csv(os.path.join(res, "feature_groups_folds.csv"))
    f = f[f["fold_seed"] == 42]
    series = [("all", "All 39 features", BLUE, "o", -0.2),
              ("demographics", "Age, sex and test time only", ORANGE, "s", 0.0),
              ("voice", "Voice measures only", AQUA, "^", 0.2)]
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    for key, label, colour, marker, dx in series:
        sub = f[f["group"] == key].sort_values("fold")
        ax.scatter(sub["fold"] + 1 + dx, sub["auc"], s=42, color=colour, marker=marker,
                   edgecolor="white", linewidth=1.2, zorder=3, label=f"{label} (mean {sub['auc'].mean():.3f})")
    ax.set_xticks(range(1, 6), [f"Fold {i}" for i in range(1, 6)]); ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.2, 1.0); ax.set_ylabel("AUC on the fold's test patients"); _grid(ax)
    _ref(ax, 0.5, "chance (0.5)")
    ax.legend(loc="upper right", ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figure2.png")); plt.close(fig)


def figure3(res, out):
    """Patients versus recordings."""
    p = pd.read_csv(os.path.join(res, "patients_vs_recordings.csv"))
    rec = p[p["experiment"] == "recordings_per_patient"]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)
    labels = [str(v) if v != "all" else "all\n(~140)" for v in rec["level"]]
    axes[0].plot(range(len(rec)), rec["auc"], color=BLUE, linewidth=1.6, marker="o", markersize=6,
                 markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    axes[0].set_xticks(range(len(rec)), labels)
    axes[0].set_xlabel("Recordings per training patient")
    axes[0].set_ylabel("AUC")
    # Two fixed training volumes; the points are nudged sideways so the error bars do not overlap.
    series = [("patients_volume_2000", "About 2,000 training recordings", BLUE, "o", -0.45),
              ("patients_volume_800", "About 800 training recordings", ORANGE, "s", 0.45)]
    for key, label, colour, marker, dx in series:
        pat = p[p["experiment"] == key]
        x = pat["level"].astype(int).values
        axes[1].errorbar(x + dx, pat["auc"], yerr=pat["auc_sd_over_draws"], color=colour, linewidth=1.6,
                         marker=marker, markersize=6, markeredgecolor="white", markeredgewidth=1.2,
                         capsize=3, elinewidth=0.8, zorder=3, label=label)
    axes[1].set_xticks(x, [str(xi) for xi in x])
    axes[1].set_xlim(4, 37)
    axes[1].set_xlabel("Training patients")
    for ax in axes:
        ax.set_ylim(0.4, 0.8); _grid(ax)
    _ref(axes[0], 0.5, "chance (0.5)")
    _ref(axes[1], 0.5, "chance (0.5)")
    axes[1].legend(loc="upper left", handlelength=1.6)
    axes[0].set_title("a  More recordings per patient", loc="left")
    axes[1].set_title("b  More patients, fixed training volume", loc="left")
    fig.tight_layout()
    fig.align_xlabels(axes)      # panel a has two-line tick labels; keep the axis titles level
    fig.savefig(os.path.join(out, "figure3.png")); plt.close(fig)


def figure4(res, out):
    """Calibration with a new patient's own recordings."""
    c = pd.read_csv(os.path.join(res, "calibration.csv"))
    base = pd.read_csv(os.path.join(res, "baselines.csv")).set_index("name")
    pos = np.arange(len(c))
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.set_xlim(-0.4, len(c) - 0.6)
    series = [("random_model_offset", "Patient's own recordings: k chosen at random", BLUE, "o"),
              ("first_k_model_offset", "Patient's own recordings: the first k", ORANGE, "s"),
              ("random_calibration_mean", "Mean score of k random recordings (no model)", AQUA, "^")]
    for key, label, colour, marker in series:
        ok = c[key].notna().values
        ax.plot(pos[ok], c.loc[ok, key], color=colour, linewidth=1.6, marker=marker, markersize=6,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3, label=label)
    ax.set_xticks(pos, c["k"].astype(int).astype(str))
    ax.set_xlabel("Calibration recordings with known scores (k)")
    ax.set_ylabel("Motor-UPDRS MAE (points)")
    ax.set_ylim(0, 8.5); _grid(ax)
    _ref(ax, base.loc["overall_average_training_mean", "mae"], "overall-average baseline")
    _ref(ax, base.loc["patient_average_baseline", "mae"], "patient-average baseline", ha="left", va="top", x=-0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.78))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figure4.png")); plt.close(fig)


def main():
    res = sys.argv[sys.argv.index("--results") + 1] if "--results" in sys.argv else "results"
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "figures"
    os.makedirs(out, exist_ok=True)
    for fn in (figure1, figure2, figure3, figure4):
        fn(res, out)
        print(f"  -> saved {os.path.join(out, fn.__name__ + '.png')}")


if __name__ == "__main__":
    main()
