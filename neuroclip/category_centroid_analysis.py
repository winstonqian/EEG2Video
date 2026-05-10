"""
Category Centroid Separability: How Well Do EEG Category Centroids Separate?

Computes centroid embeddings for each of 9 semantic categories in both
CLIP and EEG space. Measures:
1. Between-category vs within-category distances (silhouette-like)
2. RSA between 9×9 CLIP centroid similarity matrix and EEG centroid similarity matrix
3. MANOVA-style separability: do category centroids form distinct clusters?
4. Activity vs passive centroid distances: are activity categories more isolated?

Run from EEG2Video/:
    python neuroclip/category_centroid_analysis.py
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
cat_list = list(SEMANTIC_GROUPS.keys())
N_CATS = len(cat_list)

act_cats_idx = [i for i, c in enumerate(cat_list) if c in ACTIVITY_CATS]
pas_cats_idx = [i for i, c in enumerate(cat_list) if c not in ACTIVITY_CATS]


def build_gallery(device):
    conc = torch.load("neuroclip/clip_concept_both_embs_v2.pt", weights_only=True)
    g = torch.zeros(N_CONCEPTS, 512); c = torch.zeros(N_CONCEPTS)
    for s in range(N_SESSIONS):
        for pos in range(N_CONCEPTS):
            cid = int(GT_LABEL[s, pos]); g[cid] += conc[s, pos]; c[cid] += 1
    return F.normalize(g / c.clamp(min=1).unsqueeze(1), dim=-1).numpy()


def get_group_mean_eeg_embs(device):
    sum_embs = np.zeros((N_CONCEPTS, 512))
    count = np.zeros(N_CONCEPTS)
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
        for i, cid in enumerate(cids.astype(int)):
            sum_embs[cid] += embs_np[i]; count[cid] += 1
    mean_embs = sum_embs / count.clip(min=1).reshape(-1,1)
    norms = np.linalg.norm(mean_embs, axis=1, keepdims=True).clip(min=1e-8)
    return mean_embs / norms


def upper_tri(M):
    idx = np.triu_indices_from(M, k=1)
    return M[idx]


def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."


def main():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    gallery  = build_gallery(device)            # (40, 512) CLIP concept embeddings
    eeg_mean = get_group_mean_eeg_embs(device)  # (40, 512) EEG group-mean embeddings

    # Category centroids in CLIP and EEG space
    clip_centroids = np.zeros((N_CATS, 512))
    eeg_centroids  = np.zeros((N_CATS, 512))
    for i, cat in enumerate(cat_list):
        cids = SEMANTIC_GROUPS[cat]
        c_clip = gallery[cids].mean(axis=0)
        c_eeg  = eeg_mean[cids].mean(axis=0)
        clip_centroids[i] = c_clip / np.linalg.norm(c_clip).clip(min=1e-8)
        eeg_centroids[i]  = c_eeg  / np.linalg.norm(c_eeg).clip(min=1e-8)

    clip_cat_sim = clip_centroids @ clip_centroids.T   # (9, 9)
    eeg_cat_sim  = eeg_centroids  @ eeg_centroids.T    # (9, 9)

    # RSA on category centroids
    rho_cat, p_cat = stats.spearmanr(upper_tri(clip_cat_sim), upper_tri(eeg_cat_sim))
    print(f"Category centroid RSA: ρ={rho_cat:.4f}  p={p_cat:.4f}  {sig(p_cat)}")

    # Silhouette-like score: for each concept, compare within-cat vs between-cat similarity
    # Use cosine similarity
    clip_sim_all = gallery @ gallery.T
    eeg_sim_all  = eeg_mean @ eeg_mean.T

    cat_label = np.zeros(N_CONCEPTS, dtype=int)
    for i, cat in enumerate(cat_list):
        for cid in SEMANTIC_GROUPS[cat]: cat_label[cid] = i

    def silhouette(sim_mat):
        # For each concept: a = mean within-cat sim (excl self), b = mean between-cat sim
        scores = []
        for c in range(N_CONCEPTS):
            same_mask = (cat_label == cat_label[c])
            same_mask[c] = False
            diff_mask = (cat_label != cat_label[c])
            a = sim_mat[c, same_mask].mean() if same_mask.sum() > 0 else 0
            b = sim_mat[c, diff_mask].mean() if diff_mask.sum() > 0 else 0
            scores.append((a - b) / max(a, b) if max(a, b) > 0 else 0)
        return np.array(scores)

    clip_sil = silhouette(clip_sim_all)
    eeg_sil  = silhouette(eeg_sim_all)
    print(f"\nSilhouette scores (mean ± std):")
    print(f"  CLIP: {clip_sil.mean():.4f} ± {clip_sil.std():.4f}")
    print(f"  EEG:  {eeg_sil.mean():.4f} ± {eeg_sil.std():.4f}")

    t_sil, p_sil = stats.ttest_rel(clip_sil, eeg_sil)
    print(f"  CLIP > EEG: t={t_sil:.2f}  p={p_sil:.4f}  {sig(p_sil)}")

    # Activity vs Passive silhouette
    act_cids = np.array([c for cat in ACTIVITY_CATS for c in SEMANTIC_GROUPS[cat]])
    pas_cids = np.array([c for cat in SEMANTIC_GROUPS if cat not in ACTIVITY_CATS for c in SEMANTIC_GROUPS[cat]])
    t_clip_ap, p_clip_ap = stats.ttest_ind(clip_sil[act_cids], clip_sil[pas_cids])
    t_eeg_ap,  p_eeg_ap  = stats.ttest_ind(eeg_sil[act_cids],  eeg_sil[pas_cids])
    print(f"\nActivity vs Passive silhouette:")
    print(f"  CLIP: Activity={clip_sil[act_cids].mean():.4f}  Passive={clip_sil[pas_cids].mean():.4f}  "
          f"t={t_clip_ap:.2f}  {sig(p_clip_ap)}")
    print(f"  EEG:  Activity={eeg_sil[act_cids].mean():.4f}  Passive={eeg_sil[pas_cids].mean():.4f}  "
          f"t={t_eeg_ap:.2f}  {sig(p_eeg_ap)}")

    # Print per-category silhouette
    print(f"\nPer-category silhouette (CLIP vs EEG):")
    for i, cat in enumerate(cat_list):
        cids = SEMANTIC_GROUPS[cat]
        act = "ACTION" if cat in ACTIVITY_CATS else "static"
        print(f"  {cat:12s}: CLIP={clip_sil[cids].mean():.4f}  EEG={eeg_sil[cids].mean():.4f}  {act}")

    # Distance between activity and passive category centroids
    act_centroid_clip = clip_centroids[act_cats_idx].mean(axis=0)
    act_centroid_eeg  = eeg_centroids[act_cats_idx].mean(axis=0)
    pas_centroid_clip = clip_centroids[pas_cats_idx].mean(axis=0)
    pas_centroid_eeg  = eeg_centroids[pas_cats_idx].mean(axis=0)

    act_centroid_clip /= np.linalg.norm(act_centroid_clip).clip(min=1e-8)
    act_centroid_eeg  /= np.linalg.norm(act_centroid_eeg).clip(min=1e-8)
    pas_centroid_clip /= np.linalg.norm(pas_centroid_clip).clip(min=1e-8)
    pas_centroid_eeg  /= np.linalg.norm(pas_centroid_eeg).clip(min=1e-8)

    act_pas_clip_sim = float(act_centroid_clip @ pas_centroid_clip)
    act_pas_eeg_sim  = float(act_centroid_eeg  @ pas_centroid_eeg)
    print(f"\nActivity–Passive centroid cosine similarity:")
    print(f"  CLIP: {act_pas_clip_sim:.4f}  (lower = more separated)")
    print(f"  EEG:  {act_pas_eeg_sim:.4f}")

    results = {
        "cat_rsa_rho": float(rho_cat), "cat_rsa_p": float(p_cat),
        "clip_sil_mean": float(clip_sil.mean()), "eeg_sil_mean": float(eeg_sil.mean()),
        "t_clip_vs_eeg_sil": float(t_sil), "p_clip_vs_eeg_sil": float(p_sil),
        "clip_sil_activity": float(clip_sil[act_cids].mean()),
        "clip_sil_passive": float(clip_sil[pas_cids].mean()),
        "eeg_sil_activity": float(eeg_sil[act_cids].mean()),
        "eeg_sil_passive": float(eeg_sil[pas_cids].mean()),
        "t_clip_act_pas": float(t_clip_ap), "p_clip_act_pas": float(p_clip_ap),
        "t_eeg_act_pas": float(t_eeg_ap), "p_eeg_act_pas": float(p_eeg_ap),
        "act_pas_clip_sim": act_pas_clip_sim,
        "act_pas_eeg_sim": act_pas_eeg_sim,
        "cat_names": cat_list,
        "clip_centroid_sim": clip_cat_sim.tolist(),
        "eeg_centroid_sim": eeg_cat_sim.tolist(),
    }
    with open(f"{RESULTS_DIR}/results_category_centroid.json","w") as f:
        json.dump(results, f, indent=2)

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: 9×9 category centroid similarity matrices (CLIP and EEG)
    ax = axes[0]
    np.fill_diagonal(clip_cat_sim, 0); np.fill_diagonal(eeg_cat_sim, 0)
    # Show as side-by-side heatmaps in one axis using difference or combined
    diff_sim = eeg_cat_sim - clip_cat_sim
    vmax = max(abs(diff_sim).max(), 0.01)
    im = ax.imshow(diff_sim, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="EEG sim − CLIP sim")
    ax.set_xticks(range(N_CATS)); ax.set_yticks(range(N_CATS))
    ax.set_xticklabels(cat_list, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cat_list, fontsize=9)
    ax.set_title(f"(A) EEG−CLIP Centroid Similarity\n(RSA ρ={rho_cat:.3f} {sig(p_cat)})",
                 fontsize=10, fontweight="bold")

    # Panel B: Per-concept silhouette scores CLIP vs EEG
    ax = axes[1]
    is_act = np.isin(np.arange(N_CONCEPTS), act_cids)
    ax.scatter(clip_sil[~is_act], eeg_sil[~is_act], c="#4472c4",
               s=50, alpha=0.8, edgecolors="white", linewidths=0.5, label="Passive")
    ax.scatter(clip_sil[is_act], eeg_sil[is_act], c="#e74c3c",
               s=70, marker="^", alpha=0.9, edgecolors="white", linewidths=0.5, label="Activity")
    lim = [min(clip_sil.min(), eeg_sil.min())-0.05,
           max(clip_sil.max(), eeg_sil.max())+0.05]
    ax.plot(lim, lim, "k--", linewidth=1.5, alpha=0.5, label="Equal")
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("CLIP Silhouette Score", fontsize=11)
    ax.set_ylabel("EEG Silhouette Score", fontsize=11)
    ax.set_title(f"(B) CLIP vs EEG Silhouette\nCLIP={clip_sil.mean():.3f} vs EEG={eeg_sil.mean():.3f} {sig(p_sil)}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel C: Per-category mean silhouette (CLIP vs EEG grouped bar)
    ax = axes[2]
    x = np.arange(N_CATS)
    w = 0.35
    clip_cat_sil = [clip_sil[SEMANTIC_GROUPS[cat]].mean() for cat in cat_list]
    eeg_cat_sil  = [eeg_sil[SEMANTIC_GROUPS[cat]].mean()  for cat in cat_list]
    cols_cat = ["#e74c3c" if c in ACTIVITY_CATS else "#4472c4" for c in cat_list]
    bars_clip = ax.bar(x-w/2, clip_cat_sil, w, color="#f0a500", alpha=0.85, label="CLIP silhouette")
    bars_eeg  = ax.bar(x+w/2, eeg_cat_sil,  w, color="#2e86ab", alpha=0.85, label="EEG silhouette")
    ax.axhline(0, color="black", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_list, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean Silhouette Score", fontsize=10)
    ax.set_title(f"(C) Per-Category Silhouette (CLIP vs EEG)\n"
                 f"Act EEG={eeg_sil[act_cids].mean():.3f} {sig(p_eeg_ap)}  Pas EEG={eeg_sil[pas_cids].mean():.3f}",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=9)

    plt.suptitle("Category Centroid Separability: EEG vs CLIP Embedding Space\n"
                 "How well does NeuroCLIP cluster semantic categories in 512-D space?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/F44_category_centroid_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
