"""
Cross-Metric Correlation: Do All Our Per-Concept Metrics Agree?

Loads per-concept results from all completed analyses and computes a
metric × metric Spearman correlation matrix. Tests whether:
1. R@1 correlates with MRR (consistency of retrieval quality)
2. Inter-subject consistency predicts R@1
3. CLIP isolation is uncorrelated with R@1 (the key null result)
4. All metrics agree on activity vs passive concepts

Metrics used:
  - R@1, R@5, MRR (from topk_retrieval)
  - Inter-subject EEG consistency (from concept_intersubject_consistency)
  - CLIP isolation (from concept_decodability)
  - CV across subjects (from concept_stability)
  - Entropy (from retrieval_confidence)
  - Within-category EEG cluster compactness (silhouette from category_centroid)

Run from EEG2Video/:
    python neuroclip/cross_metric_correlation.py
"""
import os, sys, json
import numpy as np
from scipy import stats

RESULTS_DIR = "neuroclip/results"
FIGURES_DIR = "neuroclip/figures"
N_CONCEPTS = 40

CONCEPT_NAMES = [
    "cat","husky","elephant","horses","panda","rabbit","bird","fish","jellyfish","whale",
    "turtle","flowers","mushrooms","forest","boxing","dancing","running","skiing","computer","construction",
    "crowd","beach","city","mountain","road","waterfall","fireworks","banana","cheesecake","drink",
    "pizza","watermelon","drums","guitar","piano","motorcycle","car","balloon","airplane","boat"
]

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

is_activity = np.isin(np.arange(N_CONCEPTS), act_cids).astype(float)

def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."


def main():
    metrics = {}

    # R@1, R@5, MRR
    try:
        d = json.load(open(f"{RESULTS_DIR}/results_topk_retrieval.json"))
        metrics["R@1"]  = np.array(d["conc_rk1"])
        metrics["R@5"]  = np.array(d["conc_rk5"])
        metrics["MRR"]  = np.array(d["conc_mrr"])
    except: print("Missing: topk_retrieval")

    # CLIP isolation and inter-subject consistency
    try:
        d = json.load(open(f"{RESULTS_DIR}/results_concept_decodability.json"))
        metrics["CLIP_isolation"] = np.array(d["clip_isolation"])
        metrics["CLIP_consistency"] = np.array(d["clip_consistency"])
    except: print("Missing: concept_decodability")

    # Inter-subject EEG consistency
    try:
        d = json.load(open(f"{RESULTS_DIR}/results_concept_intersubject_consistency.json"))
        metrics["EEG_consistency"] = np.array(d["concept_consistency"])
    except: print("Missing: concept_intersubject_consistency")

    # CV across subjects
    try:
        d = json.load(open(f"{RESULTS_DIR}/results_concept_stability.json"))
        metrics["CV"] = np.array(d["cv_per_concept"])
    except: print("Missing: concept_stability")

    # Retrieval entropy
    try:
        d = json.load(open(f"{RESULTS_DIR}/results_retrieval_confidence.json"))
        metrics["Entropy"] = np.array(d["conc_entropy"])
    except: print("Missing: retrieval_confidence")

    # Activity flag
    metrics["IsActivity"] = is_activity

    metric_names = list(metrics.keys())
    N_metrics = len(metric_names)
    print(f"Loaded {N_metrics} metrics: {metric_names}")

    # Compute metric × metric Spearman correlation matrix
    rho_mat = np.zeros((N_metrics, N_metrics))
    p_mat   = np.zeros((N_metrics, N_metrics))
    for i, m1 in enumerate(metric_names):
        for j, m2 in enumerate(metric_names):
            rho, p = stats.spearmanr(metrics[m1], metrics[m2])
            rho_mat[i, j] = rho
            p_mat[i, j] = p

    print(f"\nSpearman correlation matrix:")
    header = f"{'':18s}" + "".join(f"{n:>16s}" for n in metric_names)
    print(header)
    for i, m1 in enumerate(metric_names):
        row = f"{m1:18s}"
        for j, m2 in enumerate(metric_names):
            if i == j:
                row += f"{'  1.00':>16s}"
            else:
                row += f"{rho_mat[i,j]:>14.3f}{sig(p_mat[i,j]):>2s}"
        print(row)

    # Key correlations with R@1
    if "R@1" in metrics:
        print("\nKey correlations with R@1:")
        r1_idx = metric_names.index("R@1")
        for j, m in enumerate(metric_names):
            if m == "R@1": continue
            print(f"  R@1 ↔ {m:25s}: ρ={rho_mat[r1_idx,j]:.4f}  p={p_mat[r1_idx,j]:.4f}  {sig(p_mat[r1_idx,j])}")

    # Per-concept summary: rank each concept by # of metrics where it's above median
    above_median = {}
    for m in metric_names:
        if m in ["CV", "Entropy", "IsActivity"]: continue  # lower is worse for CV/Entropy
        med = np.median(metrics[m])
        above_median[m] = (metrics[m] > med).astype(int)
    if "CV" in metrics: above_median["low_CV"] = (metrics["CV"] < np.median(metrics["CV"])).astype(int)

    combined_score = np.zeros(N_CONCEPTS)
    for v in above_median.values(): combined_score += v
    n_metrics_above = len(above_median)

    print(f"\nTop-10 concepts by combined metric score (above median in {n_metrics_above} metrics):")
    for idx in np.argsort(combined_score)[::-1][:10]:
        cat = next(c for c,ids in SEMANTIC_GROUPS.items() if idx in ids)
        act = "ACTION" if cat in ACTIVITY_CATS else "static"
        print(f"  {CONCEPT_NAMES[idx]:15s}: {combined_score[idx]:.0f}/{n_metrics_above}  [{cat}] {act}")

    results = {
        "metric_names": metric_names,
        "rho_matrix": rho_mat.tolist(),
        "p_matrix": p_mat.tolist(),
        "concept_scores": combined_score.tolist(),
        "concept_names": CONCEPT_NAMES,
    }
    with open(f"{RESULTS_DIR}/results_cross_metric_correlation.json","w") as f:
        json.dump(results, f, indent=2)

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: correlation matrix heatmap
    ax = axes[0]
    mask_diag = np.eye(N_metrics, dtype=bool)
    rho_plot = rho_mat.copy()
    rho_plot[mask_diag] = 1
    im = ax.imshow(rho_plot, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman ρ")
    ax.set_xticks(range(N_metrics)); ax.set_yticks(range(N_metrics))
    ax.set_xticklabels(metric_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(metric_names, fontsize=9)
    # Annotate with significance
    for i in range(N_metrics):
        for j in range(N_metrics):
            if i == j: continue
            p = p_mat[i, j]
            s = sig(p)
            if s != "n.s.":
                ax.text(j, i, s, ha="center", va="center", fontsize=6,
                        color="black" if abs(rho_mat[i,j]) < 0.7 else "white")
    ax.set_title("(A) Cross-Metric Spearman Correlation Matrix\n(annotations: sig. correlations only)",
                 fontsize=10, fontweight="bold")

    # Panel B: per-concept scores scatter vs R@1
    ax = axes[1]
    if "R@1" in metrics:
        r1 = metrics["R@1"]*100
        is_act = is_activity.astype(bool)
        sc = ax.scatter(r1[~is_act], combined_score[~is_act], c="#4472c4",
                        s=50, alpha=0.8, edgecolors="white", linewidths=0.5, label="Passive")
        sc2 = ax.scatter(r1[is_act], combined_score[is_act], c="#e74c3c",
                         s=70, marker="^", alpha=0.9, edgecolors="white", linewidths=0.5, label="Activity")
        for cid in np.argsort(r1)[::-1][:8]:
            ax.annotate(CONCEPT_NAMES[cid], (r1[cid], combined_score[cid]),
                        fontsize=6, xytext=(2,2), textcoords="offset points")
        rho_r1_score, p_r1_score = stats.spearmanr(r1, combined_score)
        ax.set_xlabel("Per-Concept R@1 (%)", fontsize=11)
        ax.set_ylabel(f"Combined Score (metrics above median, max={n_metrics_above})", fontsize=10)
        ax.set_title(f"(B) R@1 vs Combined Metric Score\nSpearman ρ={rho_r1_score:.3f} {sig(p_r1_score)}",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)

    plt.suptitle("Cross-Metric Consistency: Do All Per-Concept Metrics Agree?\n"
                 "Testing internal coherence of NeuroCLIP concept decodability findings",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/F45_cross_metric_correlation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
