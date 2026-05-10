"""
CLIP Geometry → EEG Confusion Patterns.

Tests the central thesis: if CLIP geometry governs EEG-BCI alignment,
then concept pairs that are close in CLIP space should be confused more
often in EEG retrieval.

For each pair (i,j): compute CLIP cosine similarity and EEG confusion rate.
Spearman correlation across all 40×39 directed pairs.

General contribution: direct test that CLIP pairwise geometry predicts
error patterns in EEG decoding — not just decodability, but WHICH errors occur.

Run from EEG2Video/:
    python neuroclip/clip_confusion_correlation.py
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

CONCEPT_NAMES = [
    "cat","husky","elephant","horses","panda","rabbit","bird","fish","jellyfish","whale",
    "turtle","flowers","mushrooms","forest","boxing","dancing","running","skiing","computer","construction",
    "crowd","beach","city","mountain","road","waterfall","fireworks","banana","cheesecake","drink",
    "pizza","watermelon","drums","guitar","piano","motorcycle","car","balloon","airplane","boat"
]

ALL_SUBS = sorted([f.replace(".npy","") for f in os.listdir(DE_DATA_DIR) if f.endswith(".npy")])


def build_gallery(device):
    conc = torch.load("neuroclip/clip_concept_both_embs_v2.pt", weights_only=True)
    g = torch.zeros(N_CONCEPTS,512); c = torch.zeros(N_CONCEPTS)
    for s in range(N_SESSIONS):
        for pos in range(N_CONCEPTS):
            cid = int(GT_LABEL[s,pos]); g[cid]+=conc[s,pos]; c[cid]+=1
    gallery = F.normalize(g/c.clamp(min=1).unsqueeze(1), dim=-1)
    clip_sim = (gallery @ gallery.T).numpy()
    np.fill_diagonal(clip_sim, 0)
    return F.normalize(g/c.clamp(min=1).unsqueeze(1), dim=-1).to(device), clip_sim, gallery.numpy()


def main():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    gallery_t, clip_sim, gallery_np = build_gallery(device)

    # Accumulate confusion matrix: confusion[true_cid, pred_cid]
    confusion = np.zeros((N_CONCEPTS, N_CONCEPTS))

    for sub in ALL_SUBS:
        raw = np.load(f"{DE_DATA_DIR}/{sub}.npy")
        n_s,n_c,n_cl,n_seg,n_ch,n_b = raw.shape
        eeg_all = raw.mean(axis=3).reshape(n_s, n_c*n_cl, n_ch, n_b)
        cids_all = np.repeat(GT_LABEL, N_CLIPS, axis=1)

        for fold in range(N_SESSIONS):
            ckpt = f"{RESULTS_DIR}/{sub}_fold{fold}_de_k1_both.pt"
            if not os.path.exists(ckpt): continue
            model = EEGEncoder(n_channels=n_ch, n_time=n_b, embed_dim=512).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            model.eval()
            flat = eeg_all[fold].reshape(N_CONCEPTS*N_CLIPS,-1)
            norm = StandardScaler().fit_transform(flat).reshape(N_CONCEPTS*N_CLIPS, n_ch, n_b)
            eeg_t = torch.tensor(norm, dtype=torch.float32).to(device)
            with torch.no_grad(): embs = model(eeg_t)
            true_cids = cids_all[fold]
            preds = (embs @ gallery_t.T).argmax(1).cpu().numpy()
            for true, pred in zip(true_cids, preds):
                confusion[int(true), int(pred)] += 1
        print(f"  {sub}: done")

    # Normalize confusion to rates (excluding diagonal = correct)
    row_sums = confusion.sum(axis=1, keepdims=True)
    conf_rate = confusion / row_sums.clip(min=1)
    np.fill_diagonal(conf_rate, 0)  # exclude correct predictions

    # Extract upper triangle (directed pairs) for correlation
    rows, cols = np.triu_indices(N_CONCEPTS, k=1)
    clip_vals  = clip_sim[rows, cols]
    conf_vals  = conf_rate[rows, cols] + conf_rate[cols, rows]  # symmetric: total confusion

    rho_s, p_s = stats.spearmanr(clip_vals, conf_vals)
    r_p, p_p   = stats.pearsonr(clip_vals, conf_vals)
    print(f"\nCLIP similarity vs EEG confusion rate:")
    print(f"  Spearman ρ={rho_s:.4f}  p={p_s:.4f}")
    print(f"  Pearson  r={r_p:.4f}  p={p_p:.4f}")

    # Top confused pairs
    conf_sym = conf_rate + conf_rate.T
    np.fill_diagonal(conf_sym, 0)
    top_pairs = np.dstack(np.unravel_index(np.argsort(conf_sym.ravel())[::-1][:20],
                                            conf_sym.shape))[0]
    print("\nTop confused concept pairs:")
    seen = set()
    for i,j in top_pairs:
        if (j,i) in seen or i==j: continue
        seen.add((i,j))
        print(f"  {CONCEPT_NAMES[i]:12s} ↔ {CONCEPT_NAMES[j]:12s}  "
              f"conf={conf_sym[i,j]:.4f}  CLIP_sim={clip_sim[i,j]:.4f}")

    results = {
        "spearman_rho": float(rho_s), "spearman_p": float(p_s),
        "pearson_r": float(r_p), "pearson_p": float(p_p),
        "confusion_matrix": confusion.tolist(),
        "clip_sim_matrix": clip_sim.tolist(),
        "concept_names": CONCEPT_NAMES,
    }
    with open(f"{RESULTS_DIR}/results_clip_confusion.json","w") as f:
        json.dump(results, f, indent=2)

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: scatter CLIP sim vs confusion rate
    ax = axes[0]
    sc = ax.scatter(clip_vals, conf_vals*100, alpha=0.3, s=20, c=clip_vals, cmap="RdYlGn_r")
    m, b = np.polyfit(clip_vals, conf_vals*100, 1)
    x_line = np.linspace(clip_vals.min(), clip_vals.max(), 100)
    ax.plot(x_line, m*x_line+b, "k-", linewidth=2.5)
    ax.set_xlabel("CLIP Pairwise Similarity", fontsize=11)
    ax.set_ylabel("EEG Confusion Rate (%)", fontsize=11)
    ax.set_title(f"(A) CLIP Geometry → EEG Confusion\nSpearman ρ={rho_s:.3f} {sig(p_s)}", fontsize=11, fontweight="bold")
    plt.colorbar(sc, ax=ax, label="CLIP similarity")

    # Panel B: confusion matrix heatmap
    ax = axes[1]
    im = ax.imshow(conf_rate*100, cmap="Reds", aspect="auto")
    ax.set_xticks(range(0,40,5)); ax.set_yticks(range(0,40,5))
    ax.set_xticklabels([CONCEPT_NAMES[i] for i in range(0,40,5)], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([CONCEPT_NAMES[i] for i in range(0,40,5)], fontsize=7)
    ax.set_xlabel("Predicted Concept", fontsize=10); ax.set_ylabel("True Concept", fontsize=10)
    ax.set_title("(B) EEG Confusion Matrix\n(darker = more confused)", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Confusion Rate (%)")

    # Panel C: CLIP sim matrix heatmap
    ax = axes[2]
    im2 = ax.imshow(clip_sim, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.6)
    ax.set_xticks(range(0,40,5)); ax.set_yticks(range(0,40,5))
    ax.set_xticklabels([CONCEPT_NAMES[i] for i in range(0,40,5)], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([CONCEPT_NAMES[i] for i in range(0,40,5)], fontsize=7)
    ax.set_xlabel("Concept", fontsize=10); ax.set_ylabel("Concept", fontsize=10)
    ax.set_title("(C) CLIP Pairwise Similarity Matrix\n(green=similar, red=dissimilar)", fontsize=11, fontweight="bold")
    plt.colorbar(im2, ax=ax, label="CLIP cosine sim")

    plt.suptitle(f"CLIP Geometry Predicts EEG Confusion Patterns: ρ={rho_s:.3f} {sig(p_s)}\n"
                 "Concept pairs close in CLIP space are confused more often in EEG retrieval",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/F28_clip_confusion_correlation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {path}")

if __name__ == "__main__":
    main()
