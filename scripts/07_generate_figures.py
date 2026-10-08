"""
Stage 7 — Generate publication figures from saved result files.

Reads the CSV/summary files already produced by 03/04/05/06, and
produces PNG figures ready to insert into the paper. Run this AFTER
all training scripts have completed.

Output: figures/*.png (300 DPI, grayscale-friendly via hatching so they
remain legible if printed without color, per the template's
accessibility guidance).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ============================== CONFIG ==============================
BASELINES_DIR = Path("data/results/baselines")
TRANSFORMER_DIR = Path("data/results/transformer")
FLATTENED_DIR = Path("data/results/flattened_ablation")
MULTISEED_DIR = Path("data/results/multiseed")
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
HATCHES = ["", "//", "xx", "..", "\\\\", "++"]
# ======================================================================


def load_pivot(path):
    """Load a summary_clean_vs_noisy.csv (multi-index columns) back into
    a usable flat DataFrame."""
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    df.columns = ["_".join(c).strip() for c in df.columns]
    return df


def fig1_primary_comparison():
    baselines = load_pivot(BASELINES_DIR / "summary_clean_vs_noisy.csv")
    transformer = load_pivot(TRANSFORMER_DIR / "summary_clean_vs_noisy.csv")

    # keep only the tuned transformer for the primary comparison
    transformer = transformer.loc[["Transformer_Tuned"]]
    transformer.index = ["Transformer (BBOT-Tuned)"]

    combined = pd.concat([baselines[["accuracy_clean", "accuracy_noisy"]],
                           transformer[["accuracy_clean", "accuracy_noisy"]]])
    combined = combined.sort_values("accuracy_noisy", ascending=False)

    models = combined.index.tolist()
    clean = combined["accuracy_clean"].values * 100
    noisy = combined["accuracy_noisy"].values * 100

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width/2, clean, width, label="Clean", hatch=HATCHES[0],
                edgecolor="black", color="#4C72B0")
    b2 = ax.bar(x + width/2, noisy, width, label="Noisy", hatch=HATCHES[1],
                edgecolor="black", color="#DD8452")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Primary Comparison: Single-Packet Baselines vs. Windowed Transformer")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=7)
    ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_primary_comparison.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig1_primary_comparison.png")


def fig2_context_equalized_ablation():
    flat = load_pivot(FLATTENED_DIR / "summary_clean_vs_noisy.csv")
    transformer = load_pivot(TRANSFORMER_DIR / "summary_clean_vs_noisy.csv")
    transformer = transformer.loc[["Transformer_Tuned"]]
    transformer.index = ["Transformer (BBOT-Tuned)"]

    combined = pd.concat([flat[["accuracy_clean", "accuracy_noisy"]],
                           transformer[["accuracy_clean", "accuracy_noisy"]]])
    combined = combined.sort_values("accuracy_noisy", ascending=False)

    models = [m.replace("_Flattened", "") for m in combined.index]
    clean = combined["accuracy_clean"].values * 100
    noisy = combined["accuracy_noisy"].values * 100

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width/2, clean, width, label="Clean", hatch=HATCHES[0],
                edgecolor="black", color="#55A868")
    b2 = ax.bar(x + width/2, noisy, width, label="Noisy", hatch=HATCHES[2],
                edgecolor="black", color="#C44E52")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Context-Equalized Ablation: All Models Given Identical 10-Packet Context")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=7)
    ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_context_equalized_ablation.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig2_context_equalized_ablation.png")


def fig3_multiseed_stability():
    df = pd.read_csv(MULTISEED_DIR / "multiseed_summary.csv", index_col=0)
    df = df.sort_values("accuracy_noisy_mean", ascending=False)

    models = [m.replace("_Flattened", "").replace("_Tuned", " (BBOT-Tuned)")
              for m in df.index]
    means = df["accuracy_noisy_mean"].values * 100
    stds = df["accuracy_noisy_std"].values * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars = ax.bar(models, means, yerr=stds, capsize=6,
                   color=colors[:len(models)], edgecolor="black")
    for i, bar in enumerate(bars):
        bar.set_hatch(HATCHES[i % len(HATCHES)])

    ax.set_ylabel("Noisy-Test Accuracy (%)")
    ax.set_title("Multi-Seed Stability (n=3 seeds, mean \u00B1 std)")
    ax.set_ylim(85, 102)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.3, f"{m:.2f}\u00B1{s:.2f}", ha="center", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_multiseed_stability.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig3_multiseed_stability.png")


def fig4_confusion_matrix(model_name, condition, source_dir, class_names=None):
    """model_name should match the filename prefix used when saving,
    e.g. 'RandomForest_Flattened'."""
    cm_path = source_dir / f"{model_name}_{condition}_confusion_matrix.csv"
    if not cm_path.exists():
        print(f"  skipping confusion matrix for {model_name} [{condition}] "
              f"— file not found: {cm_path}")
        return
    cm = np.loadtxt(cm_path, delimiter=",")
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid div-by-zero on empty rows
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"{model_name.replace('_', ' ')} \u2014 {condition.capitalize()} "
                 f"Test Confusion Matrix (row-normalized)")

    if class_names is not None and len(class_names) == cm.shape[0]:
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)
    else:
        ax.set_xlabel("Predicted class index")
        ax.set_ylabel("True class index")

    fig.colorbar(im, ax=ax, label="Proportion")
    fig.tight_layout()
    outname = f"fig4_confusion_{model_name}_{condition}.png"
    fig.savefig(OUTPUT_DIR / outname, dpi=DPI)
    plt.close(fig)
    print(f"Saved {outname}")


def load_class_names():
    try:
        import pickle
        with open(Path("data/prepared") / "label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        return list(le.classes_)
    except Exception as e:
        print(f"  couldn't load class names from label_encoder.pkl: {e}")
        return None


def main():
    print("Generating Figure 1: primary comparison...")
    fig1_primary_comparison()

    print("Generating Figure 2: context-equalized ablation...")
    fig2_context_equalized_ablation()

    print("Generating Figure 3: multi-seed stability...")
    fig3_multiseed_stability()

    print("Generating Figure 4: confusion matrices (best and worst models)...")
    class_names = load_class_names()
    try:
        fig4_confusion_matrix("RandomForest_Flattened", "noisy", FLATTENED_DIR, class_names)
    except Exception as e:
        print(f"  RandomForest_Flattened confusion matrix skipped: {e}")
    try:
        fig4_confusion_matrix("Transformer_Tuned", "noisy", TRANSFORMER_DIR, class_names)
    except Exception as e:
        print(f"  Transformer_Tuned confusion matrix skipped: {e}")

    print(f"\nAll figures saved to {OUTPUT_DIR}/ — insert into the paper "
          f"under the corresponding results subsections.")


if __name__ == "__main__":
    main()
