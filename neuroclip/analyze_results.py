"""
NeuroCLIP results analysis and report generation.

Reads the JSON result files from neuroclip/results/ and produces:
  1. Console summary table (markdown-ready)
  2. Comparison plot: NeuroCLIP vs classification baseline vs chance
  3. Per-subject variability plot
  4. Chunk attention analysis (if chunk models exist)

Run from EEG2Video/:
    python neuroclip/analyze_results.py
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR    = "neuroclip/results"
FIGURES_DIR    = "neuroclip/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Classification baseline from midterm (DE, 7-fold LOBO, 20 subjects)
CLASSIFICATION_BASELINE = {
    "top1": 0.0437, "top1_std": 0.0264,
    "top5": 0.1717, "top5_std": 0.0544,
}

CHANCE = {
    "concept_r1": 1/40, "concept_r5": 5/40, "concept_r10": 10/40,
    "clip_r1":    1/200, "clip_r5":  5/200,  "clip_r10":   10/200,
}


def load_result(fname):
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def print_table(results_dict):
    """Print a markdown-ready results table."""
    print("\n" + "="*80)
    print("NeuroCLIP Results Summary")
    print("="*80)
    print(f"{'Condition':<30} {'Concept R@1':>12} {'Concept R@5':>12} "
          f"{'Clip R@1':>10} {'Clip R@10':>10}")
    print("-"*80)

    chance_row = (f"{'Chance':<30} {CHANCE['concept_r1']*100:>11.2f}% "
                  f"{CHANCE['concept_r5']*100:>11.2f}% "
                  f"{CHANCE['clip_r1']*100:>9.2f}% "
                  f"{CHANCE['clip_r10']*100:>9.2f}%")
    print(chance_row)

    cls = CLASSIFICATION_BASELINE
    print(f"{'Classification (DE, midterm)':<30} {cls['top1']*100:>10.2f}% "
          f"{cls['top5']*100:>10.2f}%     —          —")

    for label, r in results_dict.items():
        if r is None:
            print(f"  {label:<28}  (not found)")
            continue
        r1  = r.get('mean_concept_r1',  0) * 100
        r5  = r.get('mean_concept_r5',  0) * 100
        cr1 = r.get('mean_clip_r1',     0) * 100
        cr10 = r.get('mean_clip_r10',   0) * 100
        s_r1  = r.get('std_concept_r1',  0) * 100
        s_r5  = r.get('std_concept_r5',  0) * 100
        print(f"  {label:<28} {r1:>8.2f}±{s_r1:.2f}% {r5:>8.2f}±{s_r5:.2f}% "
              f"{cr1:>8.2f}% {cr10:>8.2f}%")

    print("="*80)


def plot_comparison(results_dict):
    """Bar chart: NeuroCLIP concept R@1 across conditions vs baseline vs chance."""
    labels_plot = []
    vals  = []
    errs  = []
    colors = []

    # Chance
    labels_plot.append("Chance")
    vals.append(CHANCE["concept_r1"] * 100)
    errs.append(0)
    colors.append("#cccccc")

    # Classification baseline
    labels_plot.append("Classification\n(DE baseline)")
    vals.append(CLASSIFICATION_BASELINE["top1"] * 100)
    errs.append(CLASSIFICATION_BASELINE["top1_std"] * 100)
    colors.append("#ff9999")

    condition_colors = {"DE": "#4472c4", "Raw": "#ed7d31", "Raw+Chunks": "#70ad47"}
    for label, r in results_dict.items():
        if r is None:
            continue
        labels_plot.append(f"NeuroCLIP\n{label}")
        vals.append(r.get("mean_concept_r1", 0) * 100)
        errs.append(r.get("std_concept_r1", 0) * 100)
        col_key = "DE" if "de" in label.lower() else ("Raw+Chunks" if "chunk" in label.lower() else "Raw")
        colors.append(condition_colors.get(col_key, "#4472c4"))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels_plot))
    bars = ax.bar(x, vals, yerr=errs, capsize=5, color=colors, width=0.6,
                  error_kw={"elinewidth": 2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot, fontsize=11)
    ax.set_ylabel("Concept Retrieval R@1 (%)", fontsize=12)
    ax.set_title("NeuroCLIP Zero-Shot Concept Retrieval vs Classification Baseline\n"
                 "(SEED-DV, 40-class, mean ± std across 20 subjects)", fontsize=12)
    ax.axhline(CHANCE["concept_r1"] * 100, color="gray", linestyle="--", linewidth=1.5,
               label="Chance (2.5%)")
    ax.legend(fontsize=10)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "neuroclip_comparison.png")
    plt.savefig(path, dpi=150)
    print(f"Saved → {path}")
    plt.close()


def plot_per_subject(results_dict):
    """Per-subject R@1 distribution for best condition."""
    best_label = None
    best_mean  = -1
    best_r      = None
    for label, r in results_dict.items():
        if r is None:
            continue
        m = r.get("mean_concept_r1", 0)
        if m > best_mean:
            best_mean = m
            best_label = label
            best_r = r

    if best_r is None:
        return

    per_sub = best_r["per_subject"]["concept_r1"]
    n_subs  = len(per_sub)

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(n_subs)
    ax.bar(x, [v*100 for v in per_sub], color="#4472c4", alpha=0.8)
    ax.axhline(CHANCE["concept_r1"]*100, color="red",   linestyle="--", linewidth=1.5, label="Chance")
    ax.axhline(CLASSIFICATION_BASELINE["top1"]*100, color="orange",
               linestyle="--", linewidth=1.5, label="Classification baseline")
    ax.axhline(np.mean(per_sub)*100, color="#4472c4", linestyle="-", linewidth=1.5,
               label=f"NeuroCLIP mean ({np.mean(per_sub)*100:.1f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i+1}" for i in range(n_subs)], fontsize=9)
    ax.set_ylabel("Concept R@1 (%)", fontsize=11)
    ax.set_title(f"Per-Subject NeuroCLIP Concept R@1 — {best_label}\n"
                 f"(chance=2.5%, classification=4.37%)", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "neuroclip_per_subject.png")
    plt.savefig(path, dpi=150)
    print(f"Saved → {path}")
    plt.close()


def plot_retrieval_at_k(results_dict):
    """R@1/5/10 bar groups for concept and clip retrieval."""
    ks = [1, 5, 10]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.2
    offsets = np.linspace(-width*(len(results_dict)-1)/2, width*(len(results_dict)-1)/2,
                          len(results_dict))

    for ax_idx, (prefix, chance_prefix, title) in enumerate([
        ("concept", "concept", "Concept-Gallery Retrieval (40 concepts)"),
        ("clip",    "clip",    "Clip-Gallery Retrieval (200 clips)"),
    ]):
        ax = axes[ax_idx]
        x  = np.arange(len(ks))

        # Chance bars
        chance_vals = [CHANCE[f"{chance_prefix}_r{k}"] * 100 for k in ks]
        ax.bar(x - width*(len(results_dict)/2 + 0.5), chance_vals,
               width, label="Chance", color="#cccccc", alpha=0.9)

        colors_cycle = ["#4472c4", "#ed7d31", "#70ad47", "#ffc000"]
        for i, (label, r) in enumerate(results_dict.items()):
            if r is None:
                continue
            vals = [r.get(f"mean_{prefix}_r{k}", 0) * 100 for k in ks]
            errs = [r.get(f"std_{prefix}_r{k}",  0) * 100 for k in ks]
            ax.bar(x + offsets[i], vals, width, label=f"NeuroCLIP {label}",
                   color=colors_cycle[i % len(colors_cycle)], alpha=0.85,
                   yerr=errs, capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(["R@1", "R@5", "R@10"])
        ax.set_ylabel("%")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)

    plt.suptitle("NeuroCLIP Zero-Shot Retrieval (SEED-DV, 20 subjects)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "neuroclip_recall_at_k.png")
    plt.savefig(path, dpi=150)
    print(f"Saved → {path}")
    plt.close()


def main():
    results = {
        "Text (DE)":      load_result("results_de_k1.json"),
        "Image (DE)":     load_result("results_de_k1_image.json"),
        "Both (DE)":      load_result("results_de_k1_both.json"),
        "Raw+Chunks k=4": load_result("results_raw_k4.json"),
        "Raw k=1":        load_result("results_raw_k1.json"),
    }
    # Remove None entries from display
    results_found = {k: v for k, v in results.items() if v is not None}

    print_table(results_found)
    plot_comparison(results_found)
    plot_per_subject(results_found)
    plot_retrieval_at_k(results_found)
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
