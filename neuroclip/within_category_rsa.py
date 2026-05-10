"""
Within-Category RSA: Does CLIP geometry predict EEG geometry within each category?

The global RSA (ρ=0.039**) measures whole-matrix CLIP↔EEG alignment.
Here we ask: within each of the 9 semantic categories, do the within-group
pairwise concept similarities in CLIP space predict within-group EEG similarities?

This tests fine-grained structure: beyond coarse categorical clustering, does
NeuroCLIP encode the subtle similarity relationships within a category?

Run from EEG2Video/:
    python neuroclip/within_category_rsa.py
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


def build_gallery(device):
    conc = torch.load("neuroclip/clip_concept_both_embs_v2.pt", weights_only=True)
    g = torch.zeros(N_CONCEPTS, 512); c = torch.zeros(N_CONCEPTS)
    for s in range(N_SESSIONS):
        for pos in range(N_CONCEPTS):
            cid = int(GT_LABEL[s, pos]); g[cid] += conc[s, pos]; c[cid] += 1
    return F.normalize(g / c.clamp(min=1).unsqueeze(1), dim=-1)


def get_eeg_concept_embs(device):
    """Returns (N_sub, 40, 512) group-mean EEG concept embeddings (test session, fold 0)."""
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
        ce = np.zeros((N_CONCEPTS, 512)); cnt = np.zeros(N_CONCEPTS)
        for i, cid in enumerate(cids.astype(int)):
            ce[cid] += embs_np[i]; cnt[cid] += 1
        norms = np.linalg.norm(ce, axis=1, keepdims=True).clip(min=1e-8)
        all_embs.append(ce / norms)
        valid_subs.append(sub)
    return np.stack(all_embs), valid_subs


def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."


def upper_tri(M):
    """Return upper triangle (excluding diagonal) as 1D array."""
    idx = np.triu_indices_from(M, k=1)
    return M[idx]


def main():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    # Build CLIP gallery and similarity matrix
    gallery = build_gallery(device).numpy()         # (40, 512)
    clip_sim = gallery @ gallery.T                  # (40, 40)

    # Load EEG concept embeddings
    all_embs, valid_subs = get_eeg_concept_embs(device)   # (N_sub, 40, 512)
    N_sub = len(valid_subs)
    mean_embs = all_embs.mean(axis=0)                     # (40, 512) group-mean
    eeg_sim = mean_embs @ mean_embs.T                      # (40, 40)

    # GLOBAL RSA (sanity check)
    global_rho, global_p = stats.spearmanr(upper_tri(clip_sim), upper_tri(eeg_sim))
    print(f"Global RSA (sanity check): ρ={global_rho:.4f}  p={global_p:.4f}  {sig(global_p)}")

    # WITHIN-CATEGORY RSA per category
    print("\nWithin-category RSA:")
    cat_rho = {}
    cat_p = {}
    for cat, cids in SEMANTIC_GROUPS.items():
        if len(cids) < 3:
            print(f"  {cat:12s}: skipped (n={len(cids)} < 3)")
            continue
        clip_sub = clip_sim[np.ix_(cids, cids)]
        eeg_sub  = eeg_sim[np.ix_(cids, cids)]
        rho, p = stats.spearmanr(upper_tri(clip_sub), upper_tri(eeg_sub))
        cat_rho[cat] = float(rho)
        cat_p[cat] = float(p)
        act = "ACTION" if cat in ACTIVITY_CATS else "static"
        print(f"  {cat:12s}: ρ={rho:.4f}  p={p:.4f}  {sig(p)}  n={len(cids)}  {act}")

    # Activity vs Passive within-cat RSA
    act_rhos = [cat_rho[c] for c in ACTIVITY_CATS if c in cat_rho]
    pas_rhos = [cat_rho[c] for c in SEMANTIC_GROUPS if c not in ACTIVITY_CATS and c in cat_rho]
    if len(act_rhos) >= 2 and len(pas_rhos) >= 2:
        t_rho, p_rho = stats.ttest_ind(act_rhos, pas_rhos)
        print(f"\nActivity within-cat ρ: {np.mean(act_rhos):.4f}")
        print(f"Passive  within-cat ρ: {np.mean(pas_rhos):.4f}")
        print(f"Activity > Passive: t={t_rho:.2f}  p={p_rho:.4f}  {sig(p_rho)}")
    else:
        t_rho, p_rho = 0, 1

    # PER-SUBJECT global RSA to verify
    sub_rhos = []
    for sub_emb in all_embs:
        sub_sim = sub_emb @ sub_emb.T
        rho_s, _ = stats.spearmanr(upper_tri(clip_sim), upper_tri(sub_sim))
        sub_rhos.append(rho_s)
    sub_rhos = np.array(sub_rhos)
    t_sub, p_sub = stats.ttest_1samp(sub_rhos, 0)
    print(f"\nPer-subject global RSA: mean ρ={sub_rhos.mean():.4f} ± {sub_rhos.std():.4f}  "
          f"t={t_sub:.2f}  {sig(p_sub)}")

    results = {
        "global_rho": float(global_rho), "global_p": float(global_p),
        "cat_rho": cat_rho, "cat_p": cat_p,
        "activity_mean_rho": float(np.mean(act_rhos)) if act_rhos else None,
        "passive_mean_rho":  float(np.mean(pas_rhos)) if pas_rhos else None,
        "t_act_vs_pas": float(t_rho), "p_act_vs_pas": float(p_rho),
        "sub_mean_rho": float(sub_rhos.mean()), "sub_std_rho": float(sub_rhos.std()),
        "t_sub": float(t_sub), "p_sub": float(p_sub),
    }
    with open(f"{RESULTS_DIR}/results_within_category_rsa.json","w") as f:
        json.dump(results, f, indent=2)

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: within-category RSA bar chart
    ax = axes[0]
    cats_plot = [c for c in sorted(cat_rho, key=lambda x: -cat_rho[x])]
    cols_a = ["#e74c3c" if c in ACTIVITY_CATS else "#4472c4" for c in cats_plot]
    bars = ax.bar(range(len(cats_plot)), [cat_rho[c] for c in cats_plot],
                  color=cols_a, alpha=0.85, width=0.65)
    ax.axhline(0, color="black", linewidth=1.5)
    ax.axhline(global_rho, color="gray", linestyle="--", linewidth=1.5,
               label=f"Global RSA ρ={global_rho:.3f}")
    ax.set_xticks(range(len(cats_plot)))
    ax.set_xticklabels(cats_plot, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Within-Category Spearman ρ (CLIP↔EEG)", fontsize=10)
    ax.set_title("(A) Within-Category RSA\nDoes CLIP geometry predict EEG within categories?",
                 fontsize=10, fontweight="bold")
    for bar, c in zip(bars, cats_plot):
        p = cat_p.get(c, 1)
        if p < 0.05:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    sig(p), ha="center", fontsize=9)
    ax.legend(handles=[mpatches.Patch(color="#e74c3c",alpha=0.85,label="Activity"),
                        mpatches.Patch(color="#4472c4",alpha=0.85,label="Passive"),
                        plt.Line2D([0],[0],color="gray",linestyle="--",label=f"Global ρ={global_rho:.3f}")],
              fontsize=9)

    # Panel B: CLIP similarity matrix (ordered by category)
    ax = axes[1]
    order = []
    for cat in SEMANTIC_GROUPS: order.extend(SEMANTIC_GROUPS[cat])
    clip_ord = clip_sim[np.ix_(order, order)]
    im = ax.imshow(clip_ord, cmap="RdYlBu_r", aspect="auto", vmin=-0.5, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine similarity")
    boundary = 0
    for cat in SEMANTIC_GROUPS:
        boundary += len(SEMANTIC_GROUPS[cat])
        ax.axhline(boundary-0.5, color="white", linewidth=0.8)
        ax.axvline(boundary-0.5, color="white", linewidth=0.8)
    ax.set_title("(B) CLIP Similarity Matrix\n(ordered by semantic category)",
                 fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    # Panel C: EEG similarity matrix (ordered by category)
    ax = axes[2]
    eeg_ord = eeg_sim[np.ix_(order, order)]
    im2 = ax.imshow(eeg_ord, cmap="RdYlBu_r", aspect="auto", vmin=eeg_sim.min(), vmax=eeg_sim.max())
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label="Cosine similarity")
    boundary = 0
    for cat in SEMANTIC_GROUPS:
        boundary += len(SEMANTIC_GROUPS[cat])
        ax.axhline(boundary-0.5, color="white", linewidth=0.8)
        ax.axvline(boundary-0.5, color="white", linewidth=0.8)
    ax.set_title(f"(C) EEG Similarity Matrix\n(group-mean embeddings, fold 0, global ρ={global_rho:.3f})",
                 fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("Within-Category RSA: Fine-Grained CLIP↔EEG Geometry Alignment\n"
                 "Does NeuroCLIP encode subtle within-category concept structure?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/F39_within_category_rsa.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
