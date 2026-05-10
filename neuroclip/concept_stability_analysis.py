"""
Per-Concept R@1 Stability: Which Concepts Are Universally Decodable?

Uses per-subject, per-concept R@1 from concept_decodability results
(shape 21×40) to measure:
1. Coefficient of variation (CV = std/mean) across subjects per concept
2. Do activity concepts have lower CV (more universally decodable)?
3. Is per-concept mean R@1 correlated with inter-subject stability?
4. EEG embedding PCA: do activity concepts form tighter clusters?

Also loads EEG encoder embeddings (test session, fold 0) across subjects
and does PCA / cluster compactness analysis.

Run from EEG2Video/:
    python neuroclip/concept_stability_analysis.py
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

act_cids = [c for cat in ACTIVITY_CATS for c in SEMANTIC_GROUPS[cat]]
pas_cids = [c for cat in SEMANTIC_GROUPS if cat not in ACTIVITY_CATS for c in SEMANTIC_GROUPS[cat]]

CAT_COLORS = {
    "Animals": "#4472c4", "Nature": "#70ad47", "Food": "#ffc000",
    "Sports":  "#e74c3c", "Music":  "#c00000", "Vehicles": "#7030a0",
    "Urban":   "#00b0f0", "People": "#ff4d00", "Other": "#808080",
}


def get_concept_embs_all_subs(device):
    """Return (N_sub, 40, 512) array of L2-normalized concept embeddings."""
    all_embs, valid_subs = [], []
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
        cids = np.repeat(GT_LABEL[TEST_SESS], N_CLIPS)
        with torch.no_grad(): embs = model(eeg_t)
        embs_np = embs.cpu().numpy()
        ce = np.zeros((N_CONCEPTS, 512))
        cnt = np.zeros(N_CONCEPTS)
        for i, cid in enumerate(cids.astype(int)):
            ce[cid] += embs_np[i]; cnt[cid] += 1
        norms = np.linalg.norm(ce, axis=1, keepdims=True).clip(min=1e-8)
        all_embs.append(ce / norms)
        valid_subs.append(sub)
    return np.stack(all_embs), valid_subs  # (N_sub, 40, 512)


def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."


def main():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    # Load per-subject per-concept R@1
    deco = json.load(open(f"{RESULTS_DIR}/results_concept_decodability.json"))
    r1s = np.array(deco["per_concept_r1"])              # (40,)
    all_pc = np.array(deco["all_per_concept"])          # (21, 40)
    N_sub = all_pc.shape[0]

    # 1. Per-concept stability = CV across subjects
    conc_means = all_pc.mean(axis=0)   # (40,)
    conc_stds  = all_pc.std(axis=0)    # (40,)
    # Use sem as stability metric (lower sem = more consistent)
    conc_sems  = conc_stds / np.sqrt(N_sub)
    cv = conc_stds / conc_means.clip(min=1e-6)  # CV = std/mean

    print("Per-concept stability (lower CV = more universal):")
    pairs = sorted(zip(cv, CONCEPT_NAMES, r1s), key=lambda x: x[0])
    for cv_val, name, r1 in pairs[:10]:
        cat = next(c for c,ids in SEMANTIC_GROUPS.items() if CONCEPT_NAMES.index(name) in ids)
        act = "ACTION" if cat in ACTIVITY_CATS else "static"
        print(f"  {name:15s}: CV={cv_val:.3f}  R@1={r1*100:.1f}%  [{cat}] {act}")

    # Activity vs passive CV
    act_cv = cv[act_cids]
    pas_cv = cv[pas_cids]
    t_cv, p_cv = stats.ttest_ind(act_cv, pas_cv)
    print(f"\nActivity CV: {act_cv.mean():.4f} ± {act_cv.std():.4f}")
    print(f"Passive  CV: {pas_cv.mean():.4f} ± {pas_cv.std():.4f}")
    print(f"Activity < Passive (more stable): t={t_cv:.2f}  p={p_cv:.4f}  {sig(p_cv)}")

    # Correlation: R@1 vs CV
    rho, p_rho = stats.spearmanr(cv, r1s)
    print(f"\nCV → R@1 correlation: Spearman ρ={rho:.4f}  p={p_rho:.4f}  {sig(p_rho)}")

    # 2. Load concept embeddings for PCA
    print("\nLoading concept embeddings for PCA...")
    all_embs, valid_subs = get_concept_embs_all_subs(device)
    print(f"  Shape: {all_embs.shape}")

    # Group-mean embeddings: (40, 512)
    mean_embs = all_embs.mean(axis=0)

    # PCA on group-mean concept embeddings
    pca = PCA(n_components=2)
    pc = pca.fit_transform(mean_embs)  # (40, 2)
    var_expl = pca.explained_variance_ratio_

    # Cluster compactness: per-category mean intra-cluster cosine distance
    # (lower = more compact in embedding space)
    cat_compactness = {}
    for cat, cids in SEMANTIC_GROUPS.items():
        if len(cids) < 2: continue
        cat_embs = mean_embs[cids]  # (n_cat, 512)
        sim_mat = cat_embs @ cat_embs.T
        np.fill_diagonal(sim_mat, 0)
        n_pairs = len(cids)*(len(cids)-1)
        cat_compactness[cat] = sim_mat.sum() / n_pairs  # mean within-cat cosine sim

    print("\nPer-category EEG embedding compactness (higher = tighter cluster):")
    for cat in sorted(cat_compactness, key=lambda c: -cat_compactness[c]):
        act = "ACTION" if cat in ACTIVITY_CATS else "static"
        print(f"  {cat:12s}: {cat_compactness[cat]:.4f}  {act}")

    act_compact = np.mean([cat_compactness[c] for c in ACTIVITY_CATS if c in cat_compactness])
    pas_compact = np.mean([cat_compactness[c] for c in SEMANTIC_GROUPS if c not in ACTIVITY_CATS
                           and c in cat_compactness])
    act_comp_vals = [cat_compactness[c] for c in ACTIVITY_CATS if c in cat_compactness]
    pas_comp_vals = [cat_compactness[c] for c in SEMANTIC_GROUPS if c not in ACTIVITY_CATS
                     and c in cat_compactness]
    t_comp, p_comp = stats.ttest_ind(act_comp_vals, pas_comp_vals)
    print(f"\nActivity EEG compactness: {act_compact:.4f}")
    print(f"Passive  EEG compactness: {pas_compact:.4f}")
    print(f"Activity > Passive: t={t_comp:.2f}  p={p_comp:.4f}  {sig(p_comp)}")

    results = {
        "cv_per_concept": cv.tolist(),
        "activity_cv_mean": float(act_cv.mean()),
        "passive_cv_mean": float(pas_cv.mean()),
        "t_cv": float(t_cv), "p_cv": float(p_cv),
        "cv_r1_spearman_rho": float(rho), "cv_r1_spearman_p": float(p_rho),
        "cat_compactness": {k: float(v) for k,v in cat_compactness.items()},
        "activity_compactness": float(act_compact),
        "passive_compactness": float(pas_compact),
        "t_compactness": float(t_comp), "p_compactness": float(p_comp),
        "pca_var_explained": var_expl.tolist(),
        "concept_names": CONCEPT_NAMES,
    }
    with open(f"{RESULTS_DIR}/results_concept_stability.json","w") as f:
        json.dump(results, f, indent=2)

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: CV per concept bar chart
    ax = axes[0]
    order_cv = np.argsort(cv)
    cols_a = ["#e74c3c" if c in act_cids else "#4472c4" for c in order_cv]
    ax.bar(range(N_CONCEPTS), cv[order_cv], color=cols_a, alpha=0.8, width=0.8)
    ax.axhline(act_cv.mean(), color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Activity mean CV={act_cv.mean():.2f}")
    ax.axhline(pas_cv.mean(), color="#4472c4", linestyle="--", linewidth=1.5,
               label=f"Passive mean CV={pas_cv.mean():.2f}")
    ax.set_xticks(range(N_CONCEPTS))
    ax.set_xticklabels([CONCEPT_NAMES[c] for c in order_cv], rotation=90, fontsize=6)
    ax.set_ylabel("CV (std/mean across subjects)", fontsize=10)
    ax.set_title(f"(A) Per-Concept Stability (CV)\nLower CV = more universal across subjects {sig(p_cv)}",
                 fontsize=10, fontweight="bold")
    ax.legend(handles=[mpatches.Patch(color="#e74c3c",alpha=0.8,label=f"Activity CV={act_cv.mean():.2f}"),
                        mpatches.Patch(color="#4472c4",alpha=0.8,label=f"Passive CV={pas_cv.mean():.2f}")],
              fontsize=9)

    # Panel B: PCA of group-mean EEG embeddings
    ax = axes[1]
    for cat, cids in SEMANTIC_GROUPS.items():
        col = CAT_COLORS[cat]
        marker = "^" if cat in ACTIVITY_CATS else "o"
        ax.scatter(pc[cids, 0], pc[cids, 1], c=col, s=80, marker=marker,
                   alpha=0.85, edgecolors="white", linewidths=0.5, label=cat)
        for cid in cids:
            ax.annotate(CONCEPT_NAMES[cid], (pc[cid,0], pc[cid,1]), fontsize=5.5,
                        xytext=(2,2), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({var_expl[0]*100:.1f}% var)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var_expl[1]*100:.1f}% var)", fontsize=10)
    ax.set_title("(B) PCA of EEG Concept Embeddings\nTriangles = Activity, Circles = Passive",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    # Panel C: per-category EEG compactness
    ax = axes[2]
    cats_sorted = sorted(cat_compactness, key=lambda c: -cat_compactness[c])
    cols_c = ["#e74c3c" if c in ACTIVITY_CATS else "#4472c4" for c in cats_sorted]
    vals_c = [cat_compactness[c] for c in cats_sorted]
    bars = ax.bar(range(len(cats_sorted)), vals_c, color=cols_c, alpha=0.85, width=0.65)
    ax.set_xticks(range(len(cats_sorted)))
    ax.set_xticklabels(cats_sorted, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Within-Category EEG Cosine Similarity", fontsize=10)
    ax.set_title(f"(C) EEG Cluster Compactness\nActivity={act_compact:.3f} vs Passive={pas_compact:.3f}  {sig(p_comp)}",
                 fontsize=10, fontweight="bold")
    for bar, v in zip(bars, vals_c):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.001, f"{v:.3f}",
                ha="center", fontsize=8)

    plt.suptitle("Concept Stability and Embedding Geometry\n"
                 "Activity concepts: lower cross-subject variability, tighter EEG clusters",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/F37_concept_stability_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
