"""
Stage 11 — Figures for the mechanism experiments (stages 08, 09, 10).

Stage 07 produced figures for the primary comparison, context-equalized
ablation, multi-seed stability, and confusion matrices. This script adds
the four figures covering the experiments that identified the mechanism:

  Fig 5  Within-window shuffle test        (stage 08)
  Fig 6  Aggregation decomposition         (stage 09, experiment A)
  Fig 7  Window-size sweep                 (stage 09, experiment B)
  Fig 8  Variance decomposition            (stage 10)

Reads the saved CSVs so figures always match the reported numbers.
Run AFTER stages 08, 09, and 10.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ============================== CONFIG ==============================
SHUFFLE_DIR = Path("data/results/shuffle_test")
MEANPOOL_DIR = Path("data/results/mean_pooling")
SPLITVAR_DIR = Path("data/results/split_variance")
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
BLUE, ORANGE, GREEN, RED = "#4C72B0", "#DD8452", "#55A868", "#C44E52"
# ======================================================================


def fig5_shuffle_test():
    df = pd.read_csv(SHUFFLE_DIR / "shuffle_test_summary.csv")
    df.columns = [c.strip() for c in df.columns]
    piv = df.pivot(index="model", columns="order",
                    values=["acc_noisy_mean", "acc_noisy_std"])

    models = piv.index.tolist()
    ordered = piv[("acc_noisy_mean", "ordered")].values * 100
    shuffled = piv[("acc_noisy_mean", "shuffled")].values * 100
    ord_err = piv[("acc_noisy_std", "ordered")].values * 100
    shf_err = piv[("acc_noisy_std", "shuffled")].values * 100

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, ordered, w, yerr=ord_err, capsize=5, label="Ordered",
                 color=BLUE, edgecolor="black")
    b2 = ax.bar(x + w/2, shuffled, w, yerr=shf_err, capsize=5,
                 label="Shuffled within window", color=ORANGE, hatch="//",
                 edgecolor="black")

    ax.set_ylabel("Noisy-Test Accuracy (%)")
    ax.set_title("Within-Window Shuffle Test\n"
                 "Destroying packet order does not degrade performance")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(85, 102)
    ax.legend(loc="lower right")
    ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=8)
    ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_shuffle_test.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig5_shuffle_test.png")


def fig6_aggregation_decomposition():
    df = pd.read_csv(MEANPOOL_DIR / "experiment_a_summary.csv", index_col=0)
    order = ["single_packet", "mean_pooled", "flattened"]
    df = df.loc[[r for r in order if r in df.index]]

    labels = {
        "single_packet": "Single packet\n(18 dims)",
        "mean_pooled": "Mean-pooled window\n(18 dims)",
        "flattened": "Flattened window\n(180 dims)",
    }
    names = [labels.get(i, i) for i in df.index]
    vals = df["acc_noisy_mean"].values * 100
    errs = df["acc_noisy_std"].values * 100

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(names, vals, yerr=errs, capsize=6,
                   color=[RED, ORANGE, GREEN], edgecolor="black")
    for b, h in zip(bars, ["..", "//", ""]):
        b.set_hatch(h)

    ax.set_ylabel("Noisy-Test Accuracy (%)")
    ax.set_title("Where the Robustness Comes From\n"
                 "(Random Forest, identical underlying windows)")
    ax.set_ylim(70, 112)
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i, v + e + 0.6, f"{v:.2f}", ha="center", fontsize=9,
                fontweight="bold")

    # contribution brackets drawn above the bars so nothing overlaps
    if len(vals) == 3:
        y1 = max(vals[0], vals[1]) + 5.0
        ax.annotate("", xy=(1, y1), xytext=(0, y1),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1.4))
        ax.text(0.5, y1 + 0.8,
                f"+{vals[1]-vals[0]:.1f} pts  \u2014  aggregation\n(noise averaging)",
                ha="center", fontsize=8.5)

        y2 = max(vals[1], vals[2]) + 5.0
        ax.annotate("", xy=(2, y2), xytext=(1, y2),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1.4))
        ax.text(1.5, y2 + 0.8,
                f"+{vals[2]-vals[1]:.1f} pts  \u2014  retaining packets\n(order-independent)",
                ha="center", fontsize=8.5)

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig6_aggregation_decomposition.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig6_aggregation_decomposition.png")


def fig7_window_sweep():
    df = pd.read_csv(MEANPOOL_DIR / "experiment_b_summary.csv", index_col=0)
    W = df.index.values.astype(float)
    noisy = df["acc_noisy_mean"].values * 100
    noisy_err = df["acc_noisy_std"].values * 100
    clean = df["acc_clean_mean"].values * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(W, noisy, yerr=noisy_err, marker="o", markersize=8,
                 linewidth=2, capsize=5, color=BLUE, label="Noisy test")
    ax.plot(W, clean, marker="s", markersize=7, linewidth=2, linestyle="--",
            color=GREEN, label="Clean test")

    # sqrt-shaped reference anchored at the endpoints of the noisy curve
    if len(W) > 1:
        ref = noisy[-1] - (noisy[-1] - noisy[0]) * (
            (1 / np.sqrt(W)) - (1 / np.sqrt(W[-1]))
        ) / ((1 / np.sqrt(W[0])) - (1 / np.sqrt(W[-1])))
        ax.plot(W, ref, linestyle=":", color="gray", linewidth=1.8,
                label=r"$1/\sqrt{W}$ reference")

    ax.set_xlabel("Window size W (packets averaged)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Window-Size Sweep (mean-pooled, input dimensionality fixed)\n"
                 "Aggregation helps only under noise, with diminishing returns")
    ax.set_xticks(W)
    ax.set_ylim(75, 102)
    ax.legend(loc="center right")
    ax.grid(linestyle="--", alpha=0.4)
    for w_, v in zip(W, noisy):
        ax.annotate(f"{v:.1f}", (w_, v), textcoords="offset points",
                     xytext=(0, -16), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig7_window_size_sweep.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig7_window_size_sweep.png")


def fig8_variance_decomposition():
    raw = pd.read_csv(SPLITVAR_DIR / "split_variance_raw.csv")

    models = sorted(raw["model"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # left: per-split means, showing the gap holds on every split
    ax = axes[0]
    split_seeds = sorted(raw["split_seed"].unique())
    x = np.arange(len(split_seeds))
    w = 0.35
    for i, (m, c, h) in enumerate(zip(models, [BLUE, RED], ["", ".."])):
        means = [raw[(raw["model"] == m) & (raw["split_seed"] == s)]
                 ["accuracy_noisy"].mean() * 100 for s in split_seeds]
        b = ax.bar(x + (i - 0.5) * w, means, w, label=m, color=c,
                    edgecolor="black", hatch=h)
        ax.bar_label(b, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"split {s}" for s in split_seeds])
    ax.set_ylabel("Noisy-Test Accuracy (%)")
    ax.set_title("Per-Split Performance\n(gap holds on every partition)")
    ax.set_ylim(85, 102)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # right: variance sources
    ax = axes[1]
    x2 = np.arange(len(models))
    within, across = [], []
    for m in models:
        sub = raw[raw["model"] == m]
        within.append(sub.groupby("split_seed")["accuracy_noisy"].std().mean() * 100)
        across.append(sub.groupby("split_seed")["accuracy_noisy"].mean().std() * 100)
    b1 = ax.bar(x2 - 0.18, within, 0.36, label="Model-seed std",
                 color=GREEN, edgecolor="black")
    b2 = ax.bar(x2 + 0.18, across, 0.36, label="Data-split std",
                 color=ORANGE, edgecolor="black", hatch="//")
    ax.set_xticks(x2)
    ax.set_xticklabels(models)
    ax.set_ylabel("Standard deviation (percentage points)")
    ax.set_title("Variance Sources\n(split variance exceeds seed variance)")
    ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=8)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig8_variance_decomposition.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig8_variance_decomposition.png")


def main():
    for fn, name in [(fig5_shuffle_test, "fig5"),
                      (fig6_aggregation_decomposition, "fig6"),
                      (fig7_window_sweep, "fig7"),
                      (fig8_variance_decomposition, "fig8")]:
        try:
            fn()
        except FileNotFoundError as e:
            print(f"  {name} skipped - missing input: {e}")
        except Exception as e:
            print(f"  {name} failed: {e}")
    print(f"\nFigures written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
