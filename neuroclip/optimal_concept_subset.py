"""
Optimal Concept Subset: Which N concepts from SEED-DV maximize NeuroCLIP R@1?

Uses actual per-concept R@1 (not predicted via regression) to find the
optimal subset of concepts for a k-way retrieval task.

Two selection strategies:
1. Greedy by R@1: pick top-K by per-concept R@1
2. Greedy by R@1 + diversity: pick K concepts maximizing R@1 while maintaining
   minimum CLIP diversity (one per category)

Also shows: R@1 as a function of N (how does performance scale with fewer concepts)?

Implication for BCI design: a practitioner can pre-select the "best" concepts
from a known stimulus library to maximize retrieval accuracy.

Run from EEG2Video/:
    python neuroclip/optimal_concept_subset.py
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from dataset import GT_LABEL

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

def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."


def main():
    deco = json.load(open(f"{RESULTS_DIR}/results_concept_decodability.json"))
    r1s  = np.array(deco["per_concept_r1"])  # (40,) per-concept R@1

    # Load per-concept × per-fold R@1 (needed for subset evaluation)
    all_per_conc = np.array(deco.get("all_per_concept", []))
    # all_per_concept is shape (40,) — mean across subjects/folds

    # Load CLIP gallery for diversity metric
    conc = torch.load("neuroclip/clip_concept_both_embs_v2.pt", weights_only=True)
    g = torch.zeros(N_CONCEPTS, 512); c = torch.zeros(N_CONCEPTS)
    for s in range(7):
        for pos in range(N_CONCEPTS):
            cid = int(GT_LABEL[s, pos]); g[cid] += conc[s, pos]; c[cid] += 1
    gallery = F.normalize(g / c.clamp(min=1).unsqueeze(1), dim=-1).numpy()  # (40,512)
    clip_sim = gallery @ gallery.T  # (40,40)

    # Strategy 1: Top-K by R@1 — how does R@1 change as we vary K?
    sorted_cids = np.argsort(r1s)[::-1]  # descending R@1
    topk_r1_byN = []
    for k in range(1, N_CONCEPTS+1):
        subset = sorted_cids[:k]
        topk_r1_byN.append(r1s[subset].mean())

    print("Top-K R@1 curve (by descending R@1):")
    for k in [5, 10, 15, 20, 25, 30, 40]:
        print(f"  K={k:2d}: {topk_r1_byN[k-1]*100:.2f}%  concepts: {[CONCEPT_NAMES[c] for c in sorted_cids[:k]][:5]}...")

    # Strategy 2: Activity-biased selection
    activity_cids = [c for cat in ["Sports","Music","People"] for c in SEMANTIC_GROUPS[cat]]
    passive_cids  = [c for cat in ["Animals","Nature","Food","Vehicles","Urban","Other"]
                     for c in SEMANTIC_GROUPS[cat]]
    act_r1 = r1s[activity_cids].mean()
    pas_r1 = r1s[passive_cids].mean()

    # R@1 for activity-only (9 concepts) vs passive-only (31 concepts)
    print(f"\nActivity-only (9 concepts): mean R@1 = {act_r1*100:.2f}%")
    print(f"Passive-only (31 concepts): mean R@1 = {pas_r1*100:.2f}%")
    print(f"Full set (40 concepts):     mean R@1 = {r1s.mean()*100:.2f}%")

    # Optimal subset by category diversity (at least 1 from each category)
    # For each category, pick the best concept. Then fill remaining slots with top R@1.
    best_per_cat = {cat: max(SEMANTIC_GROUPS[cat], key=lambda c: r1s[c]) for cat in SEMANTIC_GROUPS}
    cat_diverse = sorted(set(best_per_cat.values()), key=lambda c: -r1s[c])
    remaining = sorted([c for c in range(N_CONCEPTS) if c not in cat_diverse], key=lambda c: -r1s[c])
    diverse_subset = cat_diverse + remaining  # 40 ordered by: best per cat + rest

    print("\nCategory-diverse optimal order (1 per category first):")
    for i, cid in enumerate(diverse_subset[:15]):
        cat = next(c for c,ids in SEMANTIC_GROUPS.items() if cid in ids)
        print(f"  {i+1:2d}. {CONCEPT_NAMES[cid]:15s} R@1={r1s[cid]*100:.1f}% [{cat}]")

    # Performance curve for diverse subset
    diverse_r1_byN = [r1s[diverse_subset[:k]].mean() for k in range(1, N_CONCEPTS+1)]

    # Extrapolated best-10: how much better than SEED-DV-10?
    seed_random_10 = r1s.mean()  # using all 40 as representative
    top10_r1 = topk_r1_byN[9]
    print(f"\nIf using TOP-10 concepts vs ALL-40: {top10_r1*100:.2f}% vs {r1s.mean()*100:.2f}%")
    print(f"Improvement from concept selection: +{(top10_r1-r1s.mean())*100:.2f} pp")

    results = {
        "sorted_by_r1": [CONCEPT_NAMES[c] for c in sorted_cids],
        "topk_r1": [float(v) for v in topk_r1_byN],
        "diverse_r1": [float(v) for v in diverse_r1_byN],
        "activity_mean": float(act_r1),
        "passive_mean": float(pas_r1),
        "full_mean": float(r1s.mean()),
        "top10_names": [CONCEPT_NAMES[c] for c in sorted_cids[:10]],
        "top10_r1": float(topk_r1_byN[9]),
    }
    with open(f"{RESULTS_DIR}/results_optimal_subset.json","w") as f:
        json.dump(results, f, indent=2)

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: R@1 curve vs N selected concepts
    ax = axes[0]
    Ns = list(range(1, N_CONCEPTS+1))
    ax.plot(Ns, [v*100 for v in topk_r1_byN], "o-", color="#e74c3c", linewidth=2.5,
            markersize=5, label="Top-K by R@1")
    ax.plot(Ns, [v*100 for v in diverse_r1_byN], "s--", color="#4472c4", linewidth=2.5,
            markersize=5, label="Category-diverse")
    ax.axhline(r1s.mean()*100, color="gray", linestyle=":", linewidth=2,
               label=f"All-40 mean ({r1s.mean()*100:.1f}%)")
    ax.axhline(2.5, color="black", linestyle="--", linewidth=1.5, label="Chance (2.5%)")
    ax.fill_between([1,10], [2.5,2.5], [topk_r1_byN[9]*100, topk_r1_byN[9]*100],
                    alpha=0.08, color="#e74c3c")
    ax.axvline(10, color="#e74c3c", linestyle=":", alpha=0.6)
    ax.set_xlabel("N selected concepts", fontsize=11)
    ax.set_ylabel("Mean R@1 for selected concepts (%)", fontsize=11)
    ax.set_title(f"(A) R@1 vs Concept Set Size\nTop-10: {topk_r1_byN[9]*100:.1f}% vs All-40: {r1s.mean()*100:.1f}%",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")

    # Panel B: Per-concept R@1 bar chart, colored by category type
    ax = axes[1]
    order = np.argsort(r1s)[::-1]
    cols_b = []
    cat_lookup = {}
    for cat, ids in SEMANTIC_GROUPS.items():
        for cid in ids: cat_lookup[cid] = cat
    for cid in order:
        cat = cat_lookup[cid]
        cols_b.append("#e74c3c" if cat in ["Sports","Music","People"] else "#4472c4")
    ax.bar(range(N_CONCEPTS), r1s[order]*100, color=cols_b, alpha=0.8, width=0.8)
    ax.axhline(2.5, color="black", linestyle="--", linewidth=1.5, label="Chance")
    ax.axhline(r1s.mean()*100, color="gray", linestyle=":", linewidth=1.5, label="Mean")
    ax.set_xticks(range(N_CONCEPTS))
    ax.set_xticklabels([CONCEPT_NAMES[c] for c in order], rotation=90, fontsize=6)
    ax.set_ylabel("Per-Concept R@1 (%)", fontsize=11)
    ax.set_title("(B) Per-Concept R@1 (sorted)\nRed = Activity, Blue = Passive",
                 fontsize=11, fontweight="bold")
    ax.legend(handles=[mpatches.Patch(color="#e74c3c",alpha=0.8,label="Activity"),
                       mpatches.Patch(color="#4472c4",alpha=0.8,label="Passive"),
                       plt.Line2D([0],[0],color="black",linestyle="--",label="Chance"),
                       plt.Line2D([0],[0],color="gray",linestyle=":",label="Mean")],
              fontsize=8, loc="upper right")

    # Panel C: Category-level R@1 vs N activity concepts in set
    ax = axes[2]
    activity_fracs = np.arange(0, 1.01, 0.1)
    expected_r1s = []
    for frac in activity_fracs:
        n_act = int(round(frac * N_CONCEPTS))
        n_pas = N_CONCEPTS - n_act
        act_sorted = sorted(activity_cids, key=lambda c: -r1s[c])[:n_act]
        pas_sorted = sorted(passive_cids,  key=lambda c: -r1s[c])[:n_pas]
        subset = act_sorted + pas_sorted
        if subset:
            expected_r1s.append(r1s[subset].mean()*100)
        else:
            expected_r1s.append(0)
    ax.plot(activity_fracs*100, expected_r1s, "o-", color="#e74c3c", linewidth=2.5, markersize=7)
    ax.axvline(len(activity_cids)/N_CONCEPTS*100, color="gray", linestyle="--",
               linewidth=1.5, label=f"Current SEED-DV ({len(activity_cids)/N_CONCEPTS*100:.0f}% activity)")
    ax.axhline(r1s.mean()*100, color="gray", linestyle=":", linewidth=1.5)
    ax.set_xlabel("% Activity Concepts in Set", fontsize=11)
    ax.set_ylabel("Expected Mean R@1 (%)", fontsize=11)
    ax.set_title("(C) Expected R@1 vs Activity Fraction\n"
                 "More activity concepts → higher R@1",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    plt.suptitle("Optimal Concept Selection for EEG-BCI\n"
                 "Activity concepts drive R@1 — choosing them wisely improves performance",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/F34_optimal_concept_subset.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
