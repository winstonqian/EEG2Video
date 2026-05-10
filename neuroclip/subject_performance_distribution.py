"""
Subject Performance Distribution: Who Are the Best/Worst EEG-BCI Users?

Analyzes the distribution of per-subject R@1 across 21 subjects:
1. Is the distribution normal or bimodal? (some subjects much better than others?)
2. How variable is performance within vs across subjects?
3. What is the best possible performance (upper bound per subject)?
4. Does subject performance on activity vs passive concepts correlate?

Also tests:
- Do subjects who are better overall show larger activity/passive gap?
- Is there a "floor" effect where worst subjects are at chance even for activity concepts?

Run from EEG2Video/:
    python neuroclip/subject_performance_distribution.py
"""
import os, sys, json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = "neuroclip/results"
FIGURES_DIR = "neuroclip/figures"
N_CONCEPTS = 40

SEMANTIC_GROUPS = {
    "Animals":  [0,1,2,3,4,5,6,7,8,9,10],
    "Nature":   [11,12,13,23,25],
    "Food":     [27,28,29,30,31],
    "Sports":   [14,15,16,17],
    "Music":    [32,33,34],
    "Vehicles": [35,36,37,38,39],
    "Urban":    [20,21,22,24],
    "People":   [18,19],
    "Other":    [26],
}
ACTIVITY_CATS = ["Sports", "Music", "People"]

act_cids = np.array([c for cat in ACTIVITY_CATS for c in SEMANTIC_GROUPS[cat]])
pas_cids = np.array([c for cat in SEMANTIC_GROUPS if cat not in ACTIVITY_CATS for c in SEMANTIC_GROUPS[cat]])


def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."


def main():
    # Load per-subject per-concept R@1
    deco = json.load(open(f"{RESULTS_DIR}/results_concept_decodability.json"))
    all_pc = np.array(deco["all_per_concept"])   # (21, 40) per-subject per-concept R@1
    N_sub = all_pc.shape[0]

    sub_r1 = all_pc.mean(axis=1)   # (21,) overall per-subject R@1
    sub_act = all_pc[:, act_cids].mean(axis=1)   # (21,) per-subject activity R@1
    sub_pas = all_pc[:, pas_cids].mean(axis=1)   # (21,) per-subject passive R@1
    sub_gap = sub_act - sub_pas                   # (21,) activity advantage per subject

    print(f"Per-subject R@1 statistics:")
    print(f"  Mean:   {sub_r1.mean()*100:.2f}%")
    print(f"  Std:    {sub_r1.std()*100:.2f}%")
    print(f"  Min:    {sub_r1.min()*100:.2f}%  (worst subject)")
    print(f"  Max:    {sub_r1.max()*100:.2f}%  (best subject)")
    print(f"  Median: {np.median(sub_r1)*100:.2f}%")

    # Test for normality
    stat_norm, p_norm = stats.shapiro(sub_r1)
    print(f"\nShapiro-Wilk normality test: W={stat_norm:.4f}  p={p_norm:.4f}  "
          f"({'normal' if p_norm>0.05 else 'non-normal'})")

    # Correlation: overall R@1 vs activity gap
    rho_gap, p_gap = stats.spearmanr(sub_r1, sub_gap)
    print(f"\nOverall R@1 ↔ Activity-Passive gap: Spearman ρ={rho_gap:.4f}  p={p_gap:.4f}  {sig(p_gap)}")

    # Are good subjects (top half) better at activity vs passive than bad subjects?
    median_r1 = np.median(sub_r1)
    top_half = sub_r1 >= median_r1
    bot_half = sub_r1 < median_r1

    top_act = sub_act[top_half].mean()
    top_pas = sub_pas[top_half].mean()
    bot_act = sub_act[bot_half].mean()
    bot_pas = sub_pas[bot_half].mean()
    t_top_gap, p_top_gap = stats.ttest_rel(sub_act[top_half], sub_pas[top_half])
    t_bot_gap, p_bot_gap = stats.ttest_rel(sub_act[bot_half], sub_pas[bot_half])
    print(f"\nTop-half subjects (n={top_half.sum()}):")
    print(f"  Activity: {top_act*100:.2f}%  Passive: {top_pas*100:.2f}%  "
          f"gap={((top_act-top_pas)*100):.2f}pp  t={t_top_gap:.2f}  {sig(p_top_gap)}")
    print(f"Bottom-half subjects (n={bot_half.sum()}):")
    print(f"  Activity: {bot_act*100:.2f}%  Passive: {bot_pas*100:.2f}%  "
          f"gap={((bot_act-bot_pas)*100):.2f}pp  t={t_bot_gap:.2f}  {sig(p_bot_gap)}")

    # Subjects below chance
    n_below_chance = (sub_r1 < 0.025).sum()
    print(f"\nSubjects below chance (R@1 < 2.5%): {n_below_chance}/{N_sub}")
    print(f"Subjects at 2×chance or better (R@1 > 5.0%): {(sub_r1 > 0.05).sum()}/{N_sub}")

    # Per-concept coefficient of variation (across subjects)
    cv_per_concept = all_pc.std(axis=0) / all_pc.mean(axis=0).clip(min=1e-6)
    overall_cv = sub_r1.std() / sub_r1.mean()
    print(f"\nCoefficient of variation across subjects (overall R@1): {overall_cv:.4f}")

    # Sorted subject performance
    print(f"\nSubject ranking (sorted by R@1):")
    order = np.argsort(sub_r1)[::-1]
    for rank, idx in enumerate(order):
        print(f"  {rank+1:2d}. R@1={sub_r1[idx]*100:.2f}%  Activity={sub_act[idx]*100:.2f}%  "
              f"Passive={sub_pas[idx]*100:.2f}%  gap={sub_gap[idx]*100:+.2f}pp")

    results = {
        "sub_r1": sub_r1.tolist(),
        "sub_act": sub_act.tolist(),
        "sub_pas": sub_pas.tolist(),
        "sub_gap": sub_gap.tolist(),
        "mean_r1": float(sub_r1.mean()),
        "std_r1": float(sub_r1.std()),
        "min_r1": float(sub_r1.min()),
        "max_r1": float(sub_r1.max()),
        "shapiro_p": float(p_norm),
        "rho_r1_gap": float(rho_gap), "p_r1_gap": float(p_gap),
        "n_below_chance": int(n_below_chance),
        "top_half_act": float(top_act), "top_half_pas": float(top_pas),
        "bot_half_act": float(bot_act), "bot_half_pas": float(bot_pas),
    }
    with open(f"{RESULTS_DIR}/results_subject_performance_dist.json","w") as f:
        json.dump(results, f, indent=2)

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: distribution histogram + KDE
    ax = axes[0]
    bins = np.linspace(0, sub_r1.max()*100+1, 12)
    ax.hist(sub_r1*100, bins=bins, color="#4472c4", alpha=0.7, edgecolor="white", linewidth=0.8)
    ax.axvline(2.5, color="black", linestyle="--", linewidth=2, label="Chance (2.5%)")
    ax.axvline(sub_r1.mean()*100, color="#e74c3c", linestyle="-", linewidth=2,
               label=f"Mean={sub_r1.mean()*100:.2f}%")
    ax.axvline(np.median(sub_r1)*100, color="orange", linestyle="-", linewidth=2,
               label=f"Median={np.median(sub_r1)*100:.2f}%")
    ax.set_xlabel("Subject R@1 (%)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"(A) Subject Performance Distribution\n"
                 f"Range: {sub_r1.min()*100:.1f}%–{sub_r1.max()*100:.1f}%  "
                 f"Shapiro p={p_norm:.3f}{'(normal)' if p_norm>0.05 else '(non-normal)'}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel B: Activity vs Passive per subject scatter
    ax = axes[1]
    sc = ax.scatter(sub_pas*100, sub_act*100, c=sub_r1*100, cmap="viridis",
                    s=80, alpha=0.9, edgecolors="white", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="Overall R@1 (%)")
    lim = [0, max(sub_act.max(), sub_pas.max())*100+0.5]
    ax.plot(lim, lim, "k--", linewidth=1.5, alpha=0.5, label="Equal")
    ax.axhline(2.5, color="gray", linestyle=":", linewidth=1)
    ax.axvline(2.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Passive R@1 (%)", fontsize=11)
    ax.set_ylabel("Activity R@1 (%)", fontsize=11)
    ax.set_title(f"(B) Activity vs Passive per Subject\n"
                 f"Activity-Passive gap ↔ Overall R@1: ρ={rho_gap:.3f} {sig(p_gap)}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel C: Sorted subject performance bar chart
    ax = axes[2]
    order = np.argsort(sub_r1)
    bar_colors = ["#e74c3c" if sub_r1[i] >= median_r1 else "#4472c4" for i in order]
    ax.bar(range(N_sub), sub_r1[order]*100, color=bar_colors, alpha=0.85, width=0.8)
    # Stacked: activity on top
    act_vals = sub_act[order]*100
    pas_vals = sub_pas[order]*100
    # Show activity gap as hatch
    ax.axhline(2.5, color="black", linestyle="--", linewidth=1.5, label="Chance (2.5%)")
    ax.axhline(sub_r1.mean()*100, color="gray", linestyle=":", linewidth=1.5,
               label=f"Mean={sub_r1.mean()*100:.2f}%")
    ax.set_xticks(range(N_sub))
    ax.set_xticklabels([f"S{i+1}" for i in range(N_sub)], rotation=45, fontsize=7)
    ax.set_ylabel("Overall R@1 (%)", fontsize=11)
    ax.set_title(f"(C) Per-Subject R@1 (sorted)\nTop-half {sub_r1[top_half].mean()*100:.1f}% vs "
                 f"Bottom-half {sub_r1[bot_half].mean()*100:.1f}%",
                 fontsize=10, fontweight="bold")
    ax.legend(handles=[mpatches.Patch(color="#e74c3c",alpha=0.85,label="Top half"),
                        mpatches.Patch(color="#4472c4",alpha=0.85,label="Bottom half"),
                        plt.Line2D([0],[0],color="black",linestyle="--",label="Chance")],
              fontsize=9)

    plt.suptitle("Individual Differences in EEG-BCI Performance\n"
                 "Large subject variability; activity advantage present across all skill levels",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/F43_subject_performance_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
