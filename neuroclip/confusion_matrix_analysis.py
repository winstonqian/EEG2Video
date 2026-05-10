"""
Confusion Matrix Analysis: What Does NeuroCLIP Actually Confuse?

For each subject's test session (fold 0), run inference and record
the predicted concept for every EEG trial. Build a 40×40 confusion
matrix (true concept vs predicted concept) pooled across subjects.

Tests:
1. Do errors cluster within semantic categories? (categorical confusion)
2. Are activity concepts more distinctly classified? (lower within-set confusion)
3. Does CLIP similarity predict confusion probability? (geometry → confusability)

Run from EEG2Video/:
    python neuroclip/confusion_matrix_analysis.py
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

cat_lookup = {}
for cat, ids in SEMANTIC_GROUPS.items():
    for cid in ids: cat_lookup[cid] = cat


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

    # (40, 40) pooled confusion matrix: C[true, pred] = count
    conf_matrix = np.zeros((N_CONCEPTS, N_CONCEPTS), dtype=float)

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
        cids = np.repeat(GT_LABEL[TEST_SESS], N_CLIPS)
        with torch.no_grad(): embs = model(eeg_t)
        preds = (embs @ gallery.T).argmax(1).cpu().numpy()
        for true, pred in zip(cids.astype(int), preds.astype(int)):
            conf_matrix[true, pred] += 1
        valid_subs.append(sub)
        print(f"  {sub}: done")

    print(f"\n{len(valid_subs)} subjects processed")
    # Normalize by rows (true concept total counts)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    conf_norm = conf_matrix / row_sums.clip(min=1)  # (40, 40) row-normalized

    # 1. Are within-category errors more common than expected?
    cat_labels = np.array([next(i for i,(c,ids) in enumerate(SEMANTIC_GROUPS.items()) if cid in ids)
                           for cid in range(N_CONCEPTS)])
    within_cat_conf = []
    between_cat_conf = []
    for true_c in range(N_CONCEPTS):
        for pred_c in range(N_CONCEPTS):
            if true_c == pred_c: continue
            val = conf_norm[true_c, pred_c]
            if cat_labels[true_c] == cat_labels[pred_c]:
                within_cat_conf.append(val)
            else:
                between_cat_conf.append(val)
    within_mean = np.mean(within_cat_conf)
    between_mean = np.mean(between_cat_conf)
    t_cat, p_cat = stats.ttest_ind(within_cat_conf, between_cat_conf)
    print(f"\nWithin-category confusion rate: {within_mean:.4f}")
    print(f"Between-category confusion rate: {between_mean:.4f}")
    print(f"Within > Between: t={t_cat:.2f}  p={p_cat:.6f}  {sig(p_cat)}")

    # 2. Activity vs passive: per-concept off-diagonal confusion (confusion = 1 - R@1)
    deco = json.load(open(f"{RESULTS_DIR}/results_concept_decodability.json"))
    r1s = np.array(deco["per_concept_r1"])
    act_cids = [c for cat in ACTIVITY_CATS for c in SEMANTIC_GROUPS[cat]]
    pas_cids = [c for cat in SEMANTIC_GROUPS if cat not in ACTIVITY_CATS for c in SEMANTIC_GROUPS[cat]]
    act_conf = 1 - r1s[act_cids]
    pas_conf = 1 - r1s[pas_cids]
    t2, p2 = stats.ttest_ind(act_conf, pas_conf)
    print(f"\nActivity confusion (1-R@1): {act_conf.mean():.4f}")
    print(f"Passive  confusion (1-R@1): {pas_conf.mean():.4f}")
    print(f"Passive > Activity: t={t2:.2f}  p={p2:.6f}  {sig(p2)}")

    # 3. Does CLIP gallery similarity predict confusion probability?
    gallery_np = gallery.cpu().numpy()
    clip_sim = gallery_np @ gallery_np.T  # (40, 40)
    # Vectorize: for each off-diagonal pair (true, pred), correlate conf_norm with clip_sim
    mask = ~np.eye(N_CONCEPTS, dtype=bool)
    conf_flat = conf_norm[mask]
    clip_flat  = clip_sim[mask]
    rho, p_rho = stats.spearmanr(clip_flat, conf_flat)
    r_p, p_r   = stats.pearsonr(clip_flat, conf_flat)
    print(f"\nCLIP similarity → confusion probability:")
    print(f"  Spearman ρ={rho:.4f}  p={p_rho:.6f}  {sig(p_rho)}")
    print(f"  Pearson  r={r_p:.4f}  p={p_r:.6f}  {sig(p_r)}")

    # Top confused pairs
    print("\nTop-10 most confused pairs (off-diagonal):")
    off_diag_idx = np.array([(i,j) for i in range(N_CONCEPTS) for j in range(N_CONCEPTS) if i!=j])
    off_diag_vals = np.array([conf_norm[i,j] for i,j in off_diag_idx])
    top_idx = np.argsort(off_diag_vals)[::-1][:10]
    for rank, idx in enumerate(top_idx):
        i, j = off_diag_idx[idx]
        print(f"  {rank+1:2d}. {CONCEPT_NAMES[i]:15s} → {CONCEPT_NAMES[j]:15s}  "
              f"rate={conf_norm[i,j]:.3f}  CLIP_sim={clip_sim[i,j]:.3f}  "
              f"[{cat_lookup[i]}→{cat_lookup[j]}]")

    results = {
        "within_cat_confusion": float(within_mean),
        "between_cat_confusion": float(between_mean),
        "t_within_vs_between": float(t_cat),
        "p_within_vs_between": float(p_cat),
        "clip_spearman_rho": float(rho),
        "clip_spearman_p": float(p_rho),
        "clip_pearson_r": float(r_p),
        "clip_pearson_p": float(p_r),
        "conf_matrix": conf_norm.tolist(),
        "concept_names": CONCEPT_NAMES,
    }
    with open(f"{RESULTS_DIR}/results_confusion_matrix.json","w") as f:
        json.dump(results, f, indent=2)

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: confusion matrix heatmap (log scale)
    ax = axes[0]
    order = []
    for cat in SEMANTIC_GROUPS:
        order.extend(SEMANTIC_GROUPS[cat])
    conf_ordered = conf_norm[np.ix_(order, order)]
    # Use linear scale but clip to [0, max_off_diag]
    max_val = conf_norm[~np.eye(N_CONCEPTS, dtype=bool)].max()
    im = ax.imshow(conf_ordered, cmap="hot_r", aspect="auto", vmin=0, vmax=max_val)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Confusion rate")
    # Category boundaries
    boundary = 0
    for cat in SEMANTIC_GROUPS:
        boundary += len(SEMANTIC_GROUPS[cat])
        ax.axhline(boundary-0.5, color="cyan", linewidth=0.8, alpha=0.7)
        ax.axvline(boundary-0.5, color="cyan", linewidth=0.8, alpha=0.7)
    ax.set_title(f"(A) Confusion Matrix\nWithin-cat={within_mean:.4f} vs Between-cat={between_mean:.4f} {sig(p_cat)}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Concept", fontsize=10)
    ax.set_ylabel("True Concept", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # Panel B: CLIP similarity vs confusion probability scatter
    ax = axes[1]
    ax.scatter(clip_flat, conf_flat, alpha=0.08, s=5, color="#4472c4")
    # Bin and show means
    bins = np.linspace(clip_flat.min(), clip_flat.max(), 20)
    bin_centers = (bins[:-1]+bins[1:])/2
    bin_means = [conf_flat[(clip_flat>=bins[i])&(clip_flat<bins[i+1])].mean()
                 for i in range(len(bins)-1)]
    ax.plot(bin_centers, bin_means, "o-", color="#e74c3c", linewidth=2, markersize=5, label="Binned mean")
    ax.set_xlabel("CLIP Gallery Cosine Similarity", fontsize=10)
    ax.set_ylabel("Confusion Rate", fontsize=10)
    ax.set_title(f"(B) CLIP Similarity → Confusion\nSpearman ρ={rho:.3f} {sig(p_rho)}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel C: per-concept accuracy (1-confusion) by activity vs passive
    ax = axes[2]
    act_r1s = r1s[act_cids]*100
    pas_r1s = r1s[pas_cids]*100
    data = [act_r1s, pas_r1s]
    vp = ax.violinplot(data, positions=[0,1], showmedians=True, showextrema=False)
    for pc, col in zip(vp["bodies"], ["#e74c3c","#4472c4"]):
        pc.set_facecolor(col); pc.set_alpha(0.6)
    ax.scatter(np.zeros(len(act_r1s))+np.random.normal(0,0.04,len(act_r1s)), act_r1s,
               c="#e74c3c", s=50, zorder=3, alpha=0.8, edgecolors="white", linewidths=0.5)
    ax.scatter(np.ones(len(pas_r1s))+np.random.normal(0,0.04,len(pas_r1s)), pas_r1s,
               c="#4472c4", s=50, zorder=3, alpha=0.8, edgecolors="white", linewidths=0.5)
    ax.axhline(2.5, color="gray", linestyle="--", linewidth=1.5)
    y_top = max(act_r1s.max(), pas_r1s.max()) + 0.3
    ax.plot([0,0,1,1],[y_top,y_top+0.3,y_top+0.3,y_top], lw=1.5, color="black")
    t_r1, p_r1 = stats.ttest_ind(act_r1s, pas_r1s)
    ax.text(0.5, y_top+0.4, f"t={t_r1:.1f} {sig(p_r1)}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks([0,1])
    ax.set_xticklabels(["Activity\n(Sports+Music+People)", "Passive\n(Others)"], fontsize=10)
    ax.set_ylabel("Per-Concept R@1 (%)", fontsize=10)
    ax.set_title(f"(C) Activity vs Passive R@1\n{act_r1s.mean():.2f}% vs {pas_r1s.mean():.2f}%", fontsize=10, fontweight="bold")

    plt.suptitle("EEG Confusion Structure: CLIP Geometry Predicts Errors\n"
                 "Within-category confusions dominate; activity concepts more distinctly classified",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/F35_confusion_matrix_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
