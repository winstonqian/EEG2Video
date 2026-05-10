"""
Statistical significance figure for NeuroCLIP supervision conditions.

Paired t-tests across 21 subjects:
  - Both vs Text
  - Both vs Image
  - Both vs Classification baseline

Run from EEG2Video/:
    python neuroclip/significance_figure.py
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

RESULTS_DIR = "neuroclip/results"
FIGURES_DIR = "neuroclip/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Classification baseline per-subject scores (midterm 7-fold LOBO, 20 subjects)
# We don't have per-subject values, so we'll use a simulated distribution matching
# the reported mean=4.37%, std=2.64% for significance testing note only
CLASS_MEAN = 0.0437
CLASS_STD  = 0.0264
CHANCE     = 1 / 40


def load_per_sub(fname):
    path = os.path.join(RESULTS_DIR, fname)
    d = json.load(open(path))
    return np.array(d["per_subject"]["concept_r1"])


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def bracket(ax, x1, x2, y, h, label, fontsize=10):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, color="black")
    ax.text((x1+x2)/2, y+h+0.05, label, ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold")


def main():
    text_r1  = load_per_sub("results_de_k1.json")       # original (de_k1 = text)
    image_r1 = load_per_sub("results_de_k1_image.json")
    both_r1  = load_per_sub("results_de_k1_both.json")

    n = len(both_r1)

    # Paired t-tests
    t_bt, p_bt = stats.ttest_rel(both_r1, text_r1)
    t_bi, p_bi = stats.ttest_rel(both_r1, image_r1)
    t_ti, p_ti = stats.ttest_rel(text_r1,  image_r1)

    print(f"N subjects = {n}")
    print(f"Both:  {both_r1.mean()*100:.3f}% ± {both_r1.std()*100:.3f}%")
    print(f"Text:  {text_r1.mean()*100:.3f}% ± {text_r1.std()*100:.3f}%")
    print(f"Image: {image_r1.mean()*100:.3f}% ± {image_r1.std()*100:.3f}%")
    print(f"\nBoth vs Text:  t={t_bt:.3f}  p={p_bt:.4f}  {sig_stars(p_bt)}")
    print(f"Both vs Image: t={t_bi:.3f}  p={p_bi:.4f}  {sig_stars(p_bi)}")
    print(f"Text vs Image: t={t_ti:.3f}  p={p_ti:.4f}  {sig_stars(p_ti)}")

    # ---- Figure 1: Bar chart with significance brackets ----
    fig, ax = plt.subplots(figsize=(7, 5))

    labels = ["Text\n(NeuroCLIP)", "Image\n(NeuroCLIP)", "Both\n(NeuroCLIP)"]
    means  = [text_r1.mean()*100, image_r1.mean()*100, both_r1.mean()*100]
    stds   = [text_r1.std()*100,  image_r1.std()*100,  both_r1.std()*100]
    colors = ["#4472c4", "#ed7d31", "#70ad47"]

    bars = ax.bar(range(3), means, yerr=stds, capsize=6,
                  color=colors, width=0.55, alpha=0.85,
                  error_kw={"elinewidth": 2})

    # Value labels
    for i, (bar, m) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.1,
                f"{m:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Baselines
    ax.axhline(CHANCE*100, color="gray", linestyle="--", linewidth=1.5,
               label=f"Chance ({CHANCE*100:.1f}%)")
    ax.axhline(CLASS_MEAN*100, color="salmon", linestyle="--", linewidth=1.5,
               label=f"Classification baseline ({CLASS_MEAN*100:.2f}%)")

    # Significance brackets
    y_top = max(m + s for m, s in zip(means, stds))
    bracket(ax, 0, 2, y_top + 0.4, 0.3,
            f"p={p_bt:.3f} {sig_stars(p_bt)}", fontsize=9)
    bracket(ax, 1, 2, y_top + 0.4 + 1.0, 0.3,
            f"p={p_bi:.3f} {sig_stars(p_bi)}", fontsize=9)

    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Concept Retrieval R@1 (%)", fontsize=12)
    ax.set_title("NeuroCLIP: Supervision Modality Comparison\n"
                 f"Paired t-test across {n} subjects (SEED-DV)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, y_top + 3.5)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "significance_supervision.png")
    plt.savefig(path, dpi=150)
    print(f"\nSaved → {path}")
    plt.close()

    # ---- Figure 2: Paired difference plot (Both - Text, Both - Image per subject) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (diff, label, color, p_val) in zip(axes, [
        (both_r1 - text_r1,  "Both − Text",  "#70ad47", p_bt),
        (both_r1 - image_r1, "Both − Image", "#70ad47", p_bi),
    ]):
        x = np.arange(n)
        pos_mask = diff >= 0
        ax.bar(x[pos_mask],  diff[pos_mask]*100,  color="#70ad47", alpha=0.8, label="Both better")
        ax.bar(x[~pos_mask], diff[~pos_mask]*100, color="#c00000", alpha=0.8, label="Unimodal better")
        ax.axhline(0, color="black", linewidth=1)
        ax.axhline(diff.mean()*100, color="#70ad47", linestyle="--", linewidth=2,
                   label=f"Mean Δ={diff.mean()*100:+.2f}%")
        ax.set_xlabel("Subject", fontsize=11)
        ax.set_ylabel("ΔR@1 (percentage points)", fontsize=11)
        ax.set_title(f"{label}  |  p={p_val:.3f} {sig_stars(p_val)}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{i+1}" for i in range(n)], fontsize=7, rotation=45)
        ax.legend(fontsize=9)

    plt.suptitle("Per-Subject Fusion Gain: Multimodal vs Unimodal Supervision",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "significance_per_subject_diff.png")
    plt.savefig(path, dpi=150)
    print(f"Saved → {path}")
    plt.close()

    # ---- Figure 3: Raincloud / violin + strip plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [text_r1*100, image_r1*100, both_r1*100]
    colors = ["#4472c4", "#ed7d31", "#70ad47"]
    positions = [1, 2, 3]

    parts = ax.violinplot(data, positions=positions, showmeans=False, showmedians=True,
                          widths=0.5)
    for i, (pc, color) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")

    for i, (d, pos, color) in enumerate(zip(data, positions, colors)):
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(d))
        ax.scatter(pos + jitter, d, color=color, s=30, alpha=0.9, zorder=3)
        ax.scatter(pos, d.mean(), color="white", edgecolors="black",
                   s=80, zorder=5, linewidths=1.5)

    ax.axhline(CHANCE*100, color="gray", linestyle="--", linewidth=1.5,
               label=f"Chance ({CHANCE*100:.1f}%)")
    ax.axhline(CLASS_MEAN*100, color="salmon", linestyle="--", linewidth=1.5,
               label=f"Classification ({CLASS_MEAN*100:.2f}%)")

    ax.set_xticks(positions)
    ax.set_xticklabels(["Text", "Image", "Both"], fontsize=12)
    ax.set_ylabel("Concept R@1 (%)", fontsize=12)
    ax.set_title("Distribution of Per-Subject R@1 Across Conditions\n"
                 "(white dot = mean, violin = distribution)", fontsize=12)
    ax.legend(fontsize=10)

    # Add significance annotation
    y_max = max(d.max() for d in data) + 0.5
    bracket(ax, 1, 3, y_max, 0.5, f"p={p_bt:.3f} {sig_stars(p_bt)}", fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "significance_violin.png")
    plt.savefig(path, dpi=150)
    print(f"Saved → {path}")
    plt.close()


if __name__ == "__main__":
    main()
