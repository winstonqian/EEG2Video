"""
Concept Decodability vs CLIP Geometry: What makes a concept easy to decode from EEG?

Hypothesis: concepts that are more isolated (farther from others) in CLIP space
are easier to retrieve from EEG, because the decoder has a clearer target.

Measures:
  - Per-concept R@1: mean over subjects and folds
  - CLIP isolation: mean cosine distance to all other 39 concepts
  - CLIP self-similarity: std of 7-session CLIP embeddings (concept consistency)

General contribution: explains inter-concept variability in EEG decoding
via CLIP geometry — a general principle for any EEG-CLIP system.

Run from EEG2Video/:
    python neuroclip/concept_decodability.py
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

ALL_SUBS = sorted([f.replace(".npy", "") for f in os.listdir(DE_DATA_DIR) if f.endswith(".npy")])


def build_gallery(concept_embs, device):
    g = torch.zeros(N_CONCEPTS, 512)
    c = torch.zeros(N_CONCEPTS)
    for s in range(N_SESSIONS):
        for pos in range(N_CONCEPTS):
            cid = int(GT_LABEL[s, pos])
            g[cid] += concept_embs[s, pos]
            c[cid] += 1
    return F.normalize(g / c.clamp(min=1).unsqueeze(1), dim=-1).to(device)


def load_eeg(sub_name):
    raw = np.load(os.path.join(DE_DATA_DIR, f"{sub_name}.npy"))
    n_s, n_c, n_cl, n_seg, n_ch, n_b = raw.shape
    return raw.mean(axis=3).reshape(n_s, n_c * n_cl, n_ch, n_b)


def compute_per_concept_r1(sub, device, gallery):
    eeg_all = load_eeg(sub)
    n_ch, n_b = eeg_all.shape[2], eeg_all.shape[3]
    cids_all = np.repeat(GT_LABEL, N_CLIPS, axis=1)

    per_concept_hits = np.zeros(N_CONCEPTS)
    per_concept_count = np.zeros(N_CONCEPTS)

    for fold in range(N_SESSIONS):
        ckpt = os.path.join(RESULTS_DIR, f"{sub}_fold{fold}_de_k1_both.pt")
        if not os.path.exists(ckpt):
            continue
        model = EEGEncoder(n_channels=n_ch, n_time=n_b, embed_dim=512).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()

        flat = eeg_all[fold].reshape(N_CONCEPTS * N_CLIPS, -1)
        norm = StandardScaler().fit_transform(flat).reshape(N_CONCEPTS * N_CLIPS, n_ch, n_b)
        eeg_t = torch.tensor(norm, dtype=torch.float32).to(device)
        with torch.no_grad():
            embs = model(eeg_t)
        true_cids = cids_all[fold]
        preds = (embs @ gallery.T).argmax(1).cpu().numpy()

        for i, (pred, cid) in enumerate(zip(preds, true_cids)):
            per_concept_hits[int(cid)] += int(pred == cid)
            per_concept_count[int(cid)] += 1

    return per_concept_hits / per_concept_count.clip(min=1)


def compute_clip_isolation(concept_embs):
    gallery = torch.zeros(N_CONCEPTS, 512)
    cnt = torch.zeros(N_CONCEPTS)
    for s in range(N_SESSIONS):
        for pos in range(N_CONCEPTS):
            cid = int(GT_LABEL[s, pos])
            gallery[cid] += concept_embs[s, pos]
            cnt[cid] += 1
    gallery = F.normalize(gallery / cnt.clamp(min=1).unsqueeze(1), dim=-1)

    sim = (gallery @ gallery.T).numpy()  # (40, 40) pairwise cosine sims
    np.fill_diagonal(sim, 0)

    isolation = 1.0 - sim.mean(axis=1)  # higher = more isolated (farther from others)
    neighbor_sim = sim.mean(axis=1)     # lower = more isolated

    # Also compute intra-session consistency: std of per-session embeddings
    per_sess = []
    for s in range(N_SESSIONS):
        ce = F.normalize(concept_embs[s], dim=-1)  # (40, 512)
        per_sess.append(ce.numpy())
    per_sess = np.stack(per_sess)  # (7, 40, 512)
    # For each concept, compute mean cosine sim between all session pairs
    consistency = []
    for cid in range(N_CONCEPTS):
        vecs = per_sess[:, cid, :]  # (7, 512)
        sims = []
        for i in range(N_SESSIONS):
            for j in range(i+1, N_SESSIONS):
                sims.append(float(np.dot(vecs[i], vecs[j])))
        consistency.append(np.mean(sims))
    consistency = np.array(consistency)

    return isolation, neighbor_sim, consistency, gallery.numpy()


def main():
    device = (torch.device("mps")  if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    concept_embs = torch.load("neuroclip/clip_concept_both_embs_v2.pt", weights_only=True)
    isolation, neighbor_sim, consistency, gallery = compute_clip_isolation(concept_embs)

    gallery_t = torch.tensor(gallery, dtype=torch.float32).to(device)
    print("Computing per-concept R@1 for each subject...")

    all_per_concept = np.zeros((len(ALL_SUBS), N_CONCEPTS))
    for si, sub in enumerate(ALL_SUBS):
        pc_r1 = compute_per_concept_r1(sub, device, gallery_t)
        all_per_concept[si] = pc_r1
        print(f"  {sub}: mean={pc_r1.mean()*100:.2f}%  best={CONCEPT_NAMES[pc_r1.argmax()]} ({pc_r1.max()*100:.0f}%)  worst={CONCEPT_NAMES[pc_r1.argmin()]} ({pc_r1.min()*100:.0f}%)")

    mean_per_concept = all_per_concept.mean(axis=0)

    # Correlations
    r_iso, p_iso = stats.pearsonr(isolation, mean_per_concept)
    r_con, p_con = stats.pearsonr(consistency, mean_per_concept)
    r_nei, p_nei = stats.pearsonr(neighbor_sim, mean_per_concept)

    print(f"\nCorrelations with per-concept R@1:")
    print(f"  CLIP isolation (1-mean_sim):  r={r_iso:.3f}  p={p_iso:.4f}")
    print(f"  CLIP consistency (sess-sim):  r={r_con:.3f}  p={p_con:.4f}")
    print(f"  Neighbor similarity:          r={r_nei:.3f}  p={p_nei:.4f}")

    results = {
        "concept_names": CONCEPT_NAMES,
        "per_concept_r1": mean_per_concept.tolist(),
        "clip_isolation": isolation.tolist(),
        "clip_consistency": consistency.tolist(),
        "neighbor_sim": neighbor_sim.tolist(),
        "r_isolation": float(r_iso), "p_isolation": float(p_iso),
        "r_consistency": float(r_con), "p_consistency": float(p_con),
        "all_per_concept": all_per_concept.tolist()
    }
    with open(os.path.join(RESULTS_DIR, "results_concept_decodability.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def sig(p):
        return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."

    # Panel A: isolation vs R@1
    ax = axes[0]
    sc = ax.scatter(isolation, mean_per_concept*100, c=mean_per_concept*100,
                    cmap="RdYlGn", s=80, alpha=0.85, edgecolors="black", linewidths=0.5)
    m, b = np.polyfit(isolation, mean_per_concept*100, 1)
    x_line = np.linspace(isolation.min(), isolation.max(), 100)
    ax.plot(x_line, m*x_line + b, "k--", linewidth=2)
    for i, name in enumerate(CONCEPT_NAMES):
        if mean_per_concept[i] > np.percentile(mean_per_concept, 85) or \
           mean_per_concept[i] < np.percentile(mean_per_concept, 15):
            ax.annotate(name, (isolation[i], mean_per_concept[i]*100),
                       xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("CLIP Isolation (1 − mean sim to other concepts)", fontsize=10)
    ax.set_ylabel("Mean Per-Concept R@1 (%)", fontsize=10)
    ax.set_title(f"(A) CLIP Isolation vs Decodability\nr={r_iso:.3f} {sig(p_iso)}", fontsize=11, fontweight="bold")
    plt.colorbar(sc, ax=ax, label="R@1 (%)")

    # Panel B: consistency vs R@1
    ax = axes[1]
    sc2 = ax.scatter(consistency, mean_per_concept*100, c=isolation,
                     cmap="cool", s=80, alpha=0.85, edgecolors="black", linewidths=0.5)
    m2, b2 = np.polyfit(consistency, mean_per_concept*100, 1)
    x_line2 = np.linspace(consistency.min(), consistency.max(), 100)
    ax.plot(x_line2, m2*x_line2 + b2, "k--", linewidth=2)
    for i, name in enumerate(CONCEPT_NAMES):
        if mean_per_concept[i] > np.percentile(mean_per_concept, 85) or \
           mean_per_concept[i] < np.percentile(mean_per_concept, 15):
            ax.annotate(name, (consistency[i], mean_per_concept[i]*100),
                       xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("CLIP Consistency (mean inter-session cosine sim)", fontsize=10)
    ax.set_ylabel("Mean Per-Concept R@1 (%)", fontsize=10)
    ax.set_title(f"(B) CLIP Consistency vs Decodability\nr={r_con:.3f} {sig(p_con)}", fontsize=11, fontweight="bold")
    plt.colorbar(sc2, ax=ax, label="Isolation")

    # Panel C: bar chart of per-concept R@1 sorted
    ax = axes[2]
    order = np.argsort(mean_per_concept)[::-1]
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, N_CONCEPTS))[::-1]
    bars = ax.bar(range(N_CONCEPTS), mean_per_concept[order]*100,
                  color=colors, alpha=0.85, width=0.8)
    ax.axhline(1/40*100, color="gray", linestyle="--", linewidth=1.5, label="Chance (2.5%)")
    ax.axhline(mean_per_concept.mean()*100, color="black", linestyle="-",
               linewidth=1.5, label=f"Mean ({mean_per_concept.mean()*100:.1f}%)")
    ax.set_xticks(range(N_CONCEPTS))
    ax.set_xticklabels([CONCEPT_NAMES[i] for i in order], rotation=90, fontsize=7)
    ax.set_ylabel("Per-Concept R@1 (%)", fontsize=10)
    ax.set_title("(C) Per-Concept Decodability (sorted)\n(green=best, red=worst)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    plt.suptitle(
        "Concept Decodability: CLIP Geometry Predicts EEG Retrieval Performance\n"
        "NeuroCLIP-Both (DE) · 21 subjects · 40 concepts",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "F20_concept_decodability.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")

    print("\n=== Top 5 most decodable concepts ===")
    for i in order[:5]:
        print(f"  {CONCEPT_NAMES[i]:15s}: R@1={mean_per_concept[i]*100:.1f}%  isolation={isolation[i]:.4f}  consistency={consistency[i]:.4f}")
    print("\n=== Bottom 5 least decodable concepts ===")
    for i in order[-5:]:
        print(f"  {CONCEPT_NAMES[i]:15s}: R@1={mean_per_concept[i]*100:.1f}%  isolation={isolation[i]:.4f}  consistency={consistency[i]:.4f}")


if __name__ == "__main__":
    main()
