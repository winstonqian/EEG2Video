"""
Top-K Retrieval: R@1, R@3, R@5, R@10 and MRR Analysis.

Extends the standard R@1 metric to the full retrieval curve:
- R@K = fraction of trials where correct concept is in top-K predictions
- MRR = Mean Reciprocal Rank (= 1/rank of correct concept, averaged over trials)
- MedRR = Median Reciprocal Rank

Tests whether activity vs passive concepts differ at higher K
(maybe passive concepts are "almost right" but ranked lower).

Run from EEG2Video/:
    python neuroclip/topk_retrieval_analysis.py
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from dataset import GT_LABEL
from models_neuroclip import EEGEncoder

DE_DATA_DIR = "data/DE_1per1s"
RESULTS_DIR = "neuroclip/results"
FIGURES_DIR = "neuroclip/figures"
N_CONCEPTS, N_CLIPS, N_SESSIONS = 40, 5, 7
TEST_SESS = 0

ALL_SUBS = sorted([f.replace(".npy","") for f in os.listdir(DE_DATA_DIR) if f.endswith(".npy")])

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

KS = [1, 2, 3, 5, 10, 20]
CHANCE_K = {k: k/N_CONCEPTS for k in KS}


def build_gallery(device):
    conc = torch.load("neuroclip/clip_concept_both_embs_v2.pt", weights_only=True)
    g = torch.zeros(N_CONCEPTS, 512); c = torch.zeros(N_CONCEPTS)
    for s in range(N_SESSIONS):
        for pos in range(N_CONCEPTS):
            cid = int(GT_LABEL[s, pos]); g[cid] += conc[s, pos]; c[cid] += 1
    return F.normalize(g / c.clamp(min=1).unsqueeze(1), dim=-1).to(device)


def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."


def main():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")
    gallery = build_gallery(device)

    # Per-subject: (n_subs, n_trials) ranks; also per-concept
    all_ranks  = []        # (n_subs, 200) rank of true concept (1-indexed)
    all_cids   = []        # (n_subs, 200) true concept ids

    valid_subs = []
    for sub in ALL_SUBS:
        ckpt = f"{RESULTS_DIR}/{sub}_fold0_de_k1_both.pt"
        if not os.path.exists(ckpt): continue
        raw = np.load(f"{DE_DATA_DIR}/{sub}.npy")
        n_s,n_c,n_cl,n_seg,n_ch,n_b = raw.shape
        eeg_all = raw.mean(axis=3).reshape(n_s, n_c*n_cl, n_ch, n_b)
        model = EEGEncoder(n_channels=n_ch, n_time=n_b, embed_dim=512).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()
        flat = eeg_all[TEST_SESS].reshape(N_CONCEPTS*N_CLIPS, -1)
        norm = StandardScaler().fit_transform(flat).reshape(N_CONCEPTS*N_CLIPS, n_ch, n_b)
        eeg_t = torch.tensor(norm, dtype=torch.float32).to(device)
        cids  = np.repeat(GT_LABEL[TEST_SESS], N_CLIPS).astype(int)
        with torch.no_grad(): embs = model(eeg_t)
        sims = (embs @ gallery.T).cpu().numpy()          # (200, 40)
        # Rank of true concept (1 = top-1)
        ranks = np.array([
            int((sims[i, :].argsort()[::-1]).tolist().index(cids[i])) + 1
            for i in range(len(cids))
        ])
        all_ranks.append(ranks)
        all_cids.append(cids)
        valid_subs.append(sub)
        print(f"  {sub}: R@1={( ranks==1).mean()*100:.2f}%  MRR={(1/ranks).mean():.4f}")

    all_ranks = np.array(all_ranks)   # (N_sub, 200)
    all_cids  = np.array(all_cids)    # (N_sub, 200)
    N_sub = len(valid_subs)

    print(f"\n{N_sub} subjects")
    print("\nRetrieval curve (mean ± sem across subjects):")
    rk_by_sub = {}
    for k in KS:
        hit = (all_ranks <= k)  # (N_sub, 200)
        sub_rk = hit.mean(axis=1)*100   # per-subject R@K
        rk_by_sub[k] = sub_rk
        chance = CHANCE_K[k]*100
        t, p = stats.ttest_1samp(sub_rk, chance)
        print(f"  R@{k:2d}: {sub_rk.mean():.2f}% ± {sub_rk.std()/np.sqrt(N_sub):.2f}%  "
              f"(chance={chance:.1f}%)  t={t:.2f}  {sig(p)}")

    # MRR
    mrr_per_sub = (1.0 / all_ranks).mean(axis=1)
    chance_mrr  = np.mean([1/r for r in range(1, N_CONCEPTS+1)])
    t_mrr, p_mrr = stats.ttest_1samp(mrr_per_sub, chance_mrr)
    print(f"\n  MRR:  {mrr_per_sub.mean():.4f} ± {mrr_per_sub.std()/np.sqrt(N_sub):.4f}  "
          f"(chance={chance_mrr:.4f})  t={t_mrr:.2f}  {sig(p_mrr)}")

    # Per-concept: compute R@K for each concept
    # Use pooled data across subjects
    ranks_flat = all_ranks.flatten()         # (N_sub*200,)
    cids_flat  = all_cids.flatten()          # (N_sub*200,)

    # Per-concept R@1 and MRR
    conc_mrr  = np.zeros(N_CONCEPTS)
    conc_rk1  = np.zeros(N_CONCEPTS)
    conc_rk5  = np.zeros(N_CONCEPTS)
    for cid in range(N_CONCEPTS):
        mask = (cids_flat == cid)
        if mask.sum() == 0: continue
        conc_mrr[cid]  = (1.0 / ranks_flat[mask]).mean()
        conc_rk1[cid]  = (ranks_flat[mask] == 1).mean()
        conc_rk5[cid]  = (ranks_flat[mask] <= 5).mean()

    # Activity vs passive R@5 and MRR
    t_mrr5, p_mrr5 = stats.ttest_ind(conc_mrr[act_cids], conc_mrr[pas_cids])
    t_rk5, p_rk5   = stats.ttest_ind(conc_rk5[act_cids], conc_rk5[pas_cids])
    print(f"\nActivity MRR: {conc_mrr[act_cids].mean():.4f}  Passive MRR: {conc_mrr[pas_cids].mean():.4f}  "
          f"t={t_mrr5:.2f}  {sig(p_mrr5)}")
    print(f"Activity R@5: {conc_rk5[act_cids].mean()*100:.2f}%  Passive R@5: {conc_rk5[pas_cids].mean()*100:.2f}%  "
          f"t={t_rk5:.2f}  {sig(p_rk5)}")

    results = {
        "rk_means": {str(k): float(rk_by_sub[k].mean()) for k in KS},
        "rk_sems":  {str(k): float(rk_by_sub[k].std()/np.sqrt(N_sub)) for k in KS},
        "chance":   {str(k): CHANCE_K[k] for k in KS},
        "mrr_mean": float(mrr_per_sub.mean()),
        "mrr_sem":  float(mrr_per_sub.std()/np.sqrt(N_sub)),
        "chance_mrr": float(chance_mrr),
        "t_mrr": float(t_mrr), "p_mrr": float(p_mrr),
        "conc_mrr": conc_mrr.tolist(),
        "conc_rk1": conc_rk1.tolist(),
        "conc_rk5": conc_rk5.tolist(),
        "activity_mrr": float(conc_mrr[act_cids].mean()),
        "passive_mrr":  float(conc_mrr[pas_cids].mean()),
        "t_mrr_act_pas": float(t_mrr5), "p_mrr_act_pas": float(p_mrr5),
        "concept_names": CONCEPT_NAMES,
    }
    with open(f"{RESULTS_DIR}/results_topk_retrieval.json","w") as f:
        json.dump(results, f, indent=2)

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Retrieval curve R@K vs chance
    ax = axes[0]
    Ks_plot = KS
    means_k = [rk_by_sub[k].mean() for k in Ks_plot]
    sems_k  = [rk_by_sub[k].std()/np.sqrt(N_sub) for k in Ks_plot]
    chance_k = [CHANCE_K[k]*100 for k in Ks_plot]
    ax.errorbar(Ks_plot, means_k, yerr=sems_k, fmt="o-", color="#4472c4",
                linewidth=2.5, markersize=7, capsize=5, label="NeuroCLIP R@K")
    ax.plot(Ks_plot, chance_k, "k--", linewidth=2, label="Chance", alpha=0.7)
    for k, m, c_k in zip(Ks_plot, means_k, chance_k):
        ax.annotate(f"{m:.1f}%\n({c_k:.0f}%)", (k, m), fontsize=7.5,
                    xytext=(0, 8), textcoords="offset points", ha="center")
    ax.set_xlabel("K", fontsize=11); ax.set_ylabel("R@K (%)", fontsize=11)
    ax.set_title(f"(A) Retrieval Curve\nMRR={mrr_per_sub.mean():.4f} (chance={chance_mrr:.4f}) {sig(p_mrr)}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel B: per-concept R@1 vs R@5 scatter
    ax = axes[1]
    is_act = np.isin(np.arange(N_CONCEPTS), act_cids)
    ax.scatter(conc_rk1[~is_act]*100, conc_rk5[~is_act]*100, c="#4472c4",
               s=50, alpha=0.8, edgecolors="white", linewidths=0.5, label="Passive")
    ax.scatter(conc_rk1[is_act]*100, conc_rk5[is_act]*100, c="#e74c3c",
               s=70, alpha=0.9, edgecolors="white", linewidths=0.5, marker="^", label="Activity")
    lims = [0, max(conc_rk5.max()*100+2, 30)]
    ax.plot(lims, lims, "k--", linewidth=1.5, alpha=0.4, label="R@1=R@5 line")
    ax.axhline(CHANCE_K[5]*100, color="gray", linestyle=":", linewidth=1.5)
    ax.axvline(CHANCE_K[1]*100, color="gray", linestyle=":", linewidth=1.5)
    # Label top concepts
    for cid in range(N_CONCEPTS):
        if conc_rk5[cid] > 0.3 or conc_rk1[cid] > 0.08:
            ax.annotate(CONCEPT_NAMES[cid], (conc_rk1[cid]*100, conc_rk5[cid]*100),
                        fontsize=6, xytext=(2,2), textcoords="offset points")
    ax.set_xlabel("Per-Concept R@1 (%)", fontsize=11)
    ax.set_ylabel("Per-Concept R@5 (%)", fontsize=11)
    ax.set_title("(B) R@1 vs R@5 per Concept\n(points above diagonal = easier at K=5)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    # Panel C: per-category MRR
    ax = axes[2]
    cat_mrr = {}
    for cat, cids in SEMANTIC_GROUPS.items():
        cat_mrr[cat] = conc_mrr[cids].mean()
    cats_sorted = sorted(cat_mrr, key=lambda c: -cat_mrr[c])
    cols_c = ["#e74c3c" if c in ACTIVITY_CATS else "#4472c4" for c in cats_sorted]
    vals_c = [cat_mrr[c] for c in cats_sorted]
    bars = ax.bar(range(len(cats_sorted)), vals_c, color=cols_c, alpha=0.85, width=0.65)
    ax.axhline(chance_mrr, color="gray", linestyle="--", linewidth=1.5, label=f"Chance MRR={chance_mrr:.3f}")
    ax.set_xticks(range(len(cats_sorted)))
    ax.set_xticklabels(cats_sorted, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean Reciprocal Rank (MRR)", fontsize=10)
    act_patch = mpatches.Patch(color="#e74c3c", alpha=0.85, label=f"Activity MRR={conc_mrr[act_cids].mean():.3f}")
    pas_patch = mpatches.Patch(color="#4472c4", alpha=0.85, label=f"Passive MRR={conc_mrr[pas_cids].mean():.3f}")
    ax.legend(handles=[act_patch, pas_patch,
                        plt.Line2D([0],[0],color="gray",linestyle="--",label=f"Chance={chance_mrr:.3f}")],
              fontsize=8)
    ax.set_title(f"(C) Per-Category MRR\nActivity={conc_mrr[act_cids].mean():.3f} vs Passive={conc_mrr[pas_cids].mean():.3f} {sig(p_mrr5)}",
                 fontsize=10, fontweight="bold")

    plt.suptitle("Top-K Retrieval Performance: NeuroCLIP Beyond R@1\n"
                 "Activity concepts achieve higher MRR — they rank higher even when not top-1",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/F38_topk_retrieval_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
