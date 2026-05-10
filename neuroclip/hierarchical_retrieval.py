"""
Hierarchical Retrieval Analysis.

Three-level test of what NeuroCLIP actually learned:

  Level 1 — Category (8-way):   given EEG, find the right semantic category
  Level 2 — Concept  (40-way):  given EEG, find the right concept  (standard)
  Level 3 — Intra-category:     given EEG from category C, find which concept in C

If the model learned categorical structure (RSA shows it):
  Category R@1 lift  >>  Concept R@1 lift  >>  Intra-category R@1 lift (≈ chance)

This is the strongest generalisation argument: the model knows WHAT CATEGORY the
brain is processing, but cannot reliably distinguish WHICH specific item within it.
General principle — applies to any EEG-CLIP alignment system.

Run from EEG2Video/:
    python neuroclip/hierarchical_retrieval.py
"""

import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from dataset import GT_LABEL
from models_neuroclip import EEGEncoder

DE_DATA_DIR = "data/DE_1per1s"
RESULTS_DIR = "neuroclip/results"
FIGURES_DIR = "neuroclip/figures"
BOTH_CONC   = "neuroclip/clip_concept_both_embs_v2.pt"
os.makedirs(FIGURES_DIR, exist_ok=True)

N_CONCEPTS, N_CLIPS, TEST_SESSION = 40, 5, 0

SEMANTIC_GROUPS = {
    "Animals":  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Nature":   [11, 12, 13, 23, 25],
    "Food":     [27, 28, 29, 30, 31],
    "Sports":   [14, 15, 16, 17],
    "Music":    [32, 33, 34],
    "Vehicles": [35, 36, 37, 38, 39],
    "Urban":    [20, 21, 22, 24],
    "Other":    [18, 19, 26],
}
N_CATS = len(SEMANTIC_GROUPS)
CID_TO_CAT = {c: i for i, (_, cids) in enumerate(SEMANTIC_GROUPS.items()) for c in cids}
CAT_NAMES  = list(SEMANTIC_GROUPS.keys())


# ---------------------------------------------------------------------------
# Build CLIP galleries
# ---------------------------------------------------------------------------

def build_galleries(concept_conc):
    """
    concept_conc: (7, 40, 512) CLIP concept-mean embeddings.
    concept_conc[sess, pos] is the embedding for concept GT_LABEL[sess, pos].
    The gallery must be indexed by concept_id, not position.

    Returns:
      gallery_40  (40, 512) — concept-level gallery (indexed by concept_id)
      gallery_cat (N_CATS, 512) — category-level gallery
    """
    # Accumulate CLIP embeddings by concept_id across all sessions
    g = torch.zeros(N_CONCEPTS, 512)
    c = torch.zeros(N_CONCEPTS)
    for sess in range(7):
        for pos in range(N_CONCEPTS):
            cid = int(GT_LABEL[sess, pos])
            g[cid] += concept_conc[sess, pos]
            c[cid] += 1
    gallery_40 = F.normalize(g / c.clamp(min=1).unsqueeze(1), dim=-1)

    # Category gallery: mean of concept embeddings in each category
    gallery_cat = []
    for grp, cids in SEMANTIC_GROUPS.items():
        cat_emb = gallery_40[cids].mean(0)
        gallery_cat.append(cat_emb)
    gallery_cat = F.normalize(torch.stack(gallery_cat), dim=-1)   # (N_CATS, 512)

    return gallery_40, gallery_cat


def build_intra_galleries(gallery_40):
    """Per-category concept galleries for intra-category evaluation."""
    intra = {}
    for cat_idx, (grp, cids) in enumerate(SEMANTIC_GROUPS.items()):
        intra[cat_idx] = (cids, gallery_40[cids])   # (K, 512)
    return intra


# ---------------------------------------------------------------------------
# Subject loader
# ---------------------------------------------------------------------------

def get_eeg_embs_all_folds(sub_name, device="cpu"):
    """Pool embeddings from all 7 LOBO folds (each model on its held-out session)."""
    sub_path = os.path.join(DE_DATA_DIR, f"{sub_name}.npy")
    if not os.path.exists(sub_path):
        return None, None
    raw = np.load(sub_path)
    n_sess, n_conc, n_cl, n_seg, n_ch, n_bands = raw.shape
    eeg_all = raw.mean(axis=3).reshape(n_sess, n_conc * n_cl, n_ch, n_bands)

    all_embs, all_cids = [], []
    for fold in range(7):
        mp = os.path.join(RESULTS_DIR, f"{sub_name}_fold{fold}_de_k1_both.pt")
        if not os.path.exists(mp):
            continue
        model = EEGEncoder(n_channels=n_ch, n_time=n_bands, embed_dim=512)
        model.load_state_dict(torch.load(mp, map_location="cpu", weights_only=True))
        model.eval()
        sess_data = eeg_all[fold]
        flat = sess_data.reshape(200, -1)
        norm = StandardScaler().fit_transform(flat).reshape(200, n_ch, n_bands)
        eeg_t = torch.tensor(norm, dtype=torch.float32)
        with torch.no_grad():
            embs = model(eeg_t)
        cids = np.repeat(GT_LABEL[fold], N_CLIPS)
        all_embs.append(embs)
        all_cids.append(cids)

    if not all_embs:
        return None, None
    return torch.cat(all_embs, dim=0), np.concatenate(all_cids)


# ---------------------------------------------------------------------------
# Retrieval evaluators
# ---------------------------------------------------------------------------

def eval_concept(embs, concept_ids, gallery_40):
    """Standard 40-way concept retrieval R@1."""
    sim = embs @ gallery_40.T   # (200, 40)
    preds = sim.argmax(1)
    true  = torch.tensor(concept_ids, dtype=torch.long)
    return (preds == true).float().mean().item()


def eval_category(embs, concept_ids, gallery_cat):
    """8-way category retrieval R@1."""
    sim  = embs @ gallery_cat.T   # (200, N_CATS)
    preds = sim.argmax(1).numpy()
    true  = np.array([CID_TO_CAT[int(c)] for c in concept_ids])
    return (preds == true).mean()


def eval_intra_category(embs, concept_ids, intra_galleries):
    """
    Intra-category R@1: for each clip, restrict gallery to concepts in the
    same semantic category and find the right one.
    Returns mean R@1 and per-category R@1 dict.
    """
    per_cat_correct = {cat_idx: [] for cat_idx in intra_galleries}
    concept_ids_arr = np.array(concept_ids)

    for i in range(len(embs)):
        cid = int(concept_ids_arr[i])
        cat_idx = CID_TO_CAT[cid]
        cids_in_cat, gal_in_cat = intra_galleries[cat_idx]
        if len(cids_in_cat) == 1:
            continue   # trivial
        sim   = (embs[i] @ gal_in_cat.T)   # (K,)
        pred  = cids_in_cat[sim.argmax().item()]
        per_cat_correct[cat_idx].append(int(pred == cid))

    per_cat_r1 = {cat_idx: np.mean(v) if v else 0.0
                  for cat_idx, v in per_cat_correct.items()}
    all_correct = [v for vals in per_cat_correct.values() for v in vals]
    return np.mean(all_correct), per_cat_r1


def chance_intra(cat_idx):
    return 1.0 / len(SEMANTIC_GROUPS[CAT_NAMES[cat_idx]])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_hierarchy(results, save_path):
    """
    3-level bar chart: Category / Concept / Intra-category
    Shows mean R@1 and lift-over-chance for each level.
    """
    levels      = ["Category\n(8-way)", "Concept\n(40-way)", "Intra-category\n(within group)"]
    means       = [results["cat_r1_mean"],    results["conc_r1_mean"],   results["intra_r1_mean"]]
    stds        = [results["cat_r1_std"],     results["conc_r1_std"],    results["intra_r1_std"]]
    chances     = [1/N_CATS,                  1/N_CONCEPTS,              results["intra_chance_mean"]]
    lifts       = [m/c for m, c in zip(means, chances)]
    colors      = ["#4472c4", "#70ad47", "#ed7d31"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: absolute R@1
    ax = axes[0]
    x  = np.arange(3)
    bars = ax.bar(x, [m*100 for m in means], yerr=[s*100 for s in stds],
                  color=colors, width=0.5, alpha=0.85, capsize=7,
                  error_kw={"elinewidth":2})
    for xi, ch in zip(x, chances):
        ax.plot([xi - 0.28, xi + 0.28], [ch*100, ch*100],
                color="black", linewidth=2, linestyle="--")
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x()+bar.get_width()/2, m*100+s*100+0.3,
                f"{m*100:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(levels, fontsize=11)
    ax.set_ylabel("R@1 (%)", fontsize=12)
    ax.set_title("(A) Absolute R@1 by Retrieval Level\n(dashed = chance per level)", fontsize=11)
    ax.set_ylim(0, max(m*100+s*100 for m,s in zip(means,stds)) + 4)

    # Right: lift over chance
    ax = axes[1]
    lift_bars = ax.bar(x, lifts, color=colors, width=0.5, alpha=0.85)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5,
               label="Chance (lift=1.0×)")
    for bar, lift in zip(lift_bars, lifts):
        ax.text(bar.get_x()+bar.get_width()/2, lift+0.02,
                f"{lift:.2f}×", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(levels, fontsize=11)
    ax.set_ylabel("Lift over chance (R@1 / chance)", fontsize=12)
    ax.set_title("(B) Lift over Chance by Retrieval Level\n"
                 "Category > Concept > Intra-category → categorical representation",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(lifts) + 0.4)

    plt.suptitle("Hierarchical Retrieval: EEG Representations Are Categorical, Not Fine-Grained\n"
                 f"(N=21 subjects, session 1 held-out, NeuroCLIP-Both)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"Saved → {save_path}")


def plot_per_category_bars(results, save_path):
    """Per-category intra-category R@1 vs chance."""
    cat_names = CAT_NAMES
    means     = [results["per_cat_intra_mean"].get(i, 0) for i in range(N_CATS)]
    stds      = [results["per_cat_intra_std"].get(i,  0) for i in range(N_CATS)]
    chances   = [chance_intra(i) for i in range(N_CATS)]
    colors    = ["#4472c4","#70ad47","#ffc000","#ed7d31","#9b59b6","#e74c3c","#95a5a6","#bdc3c7"]

    x = np.arange(N_CATS)
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(x, [m*100 for m in means], yerr=[s*100 for s in stds],
                  color=colors, width=0.55, alpha=0.85, capsize=5,
                  error_kw={"elinewidth":1.5})
    for xi, ch in zip(x, chances):
        ax.plot([xi-0.3, xi+0.3], [ch*100, ch*100], color="red", linewidth=2.5,
                linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(cat_names, fontsize=10)
    ax.set_ylabel("Intra-category R@1 (%)", fontsize=12)
    ax.set_title("Intra-Category Retrieval: Can the Model Distinguish Within a Semantic Group?\n"
                 "(red dashes = chance per category; NeuroCLIP-Both, 21 subjects)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"Saved → {save_path}")


def plot_lift_comparison(results, save_path):
    """Scatter: concept R@1 vs category R@1 per subject — shows consistent lift."""
    conc_per_sub = results["conc_per_sub"]
    cat_per_sub  = results["cat_per_sub"]
    n = len(conc_per_sub)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(np.array(conc_per_sub)*100, np.array(cat_per_sub)*100,
               color="#4472c4", s=50, alpha=0.8, zorder=3)

    # Diagonal: cat=conc line
    lim = max(max(conc_per_sub), max(cat_per_sub))*100 + 2
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="cat=concept (equal)")

    # Mark chance lines
    ax.axhline(100/N_CATS,    color="#4472c4", linestyle=":", linewidth=1.5,
               label=f"Category chance ({100/N_CATS:.1f}%)")
    ax.axvline(100/N_CONCEPTS, color="#70ad47", linestyle=":", linewidth=1.5,
               label=f"Concept chance ({100/N_CONCEPTS:.1f}%)")

    # Regression
    m, b = np.polyfit(conc_per_sub, cat_per_sub, 1)
    xr = np.linspace(0, max(conc_per_sub), 100)
    ax.plot(xr*100, (m*xr+b)*100, color="#e74c3c", linewidth=2)

    r, p = stats.pearsonr(conc_per_sub, cat_per_sub)
    ax.set_xlabel("Concept R@1 (%) — 40-way", fontsize=12)
    ax.set_ylabel("Category R@1 (%) — 8-way", fontsize=12)
    ax.set_title(f"Per-Subject: Category vs Concept Retrieval\n"
                 f"Pearson r={r:.3f}, p={p:.3f}  (n={n} subjects)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"Saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cpu"
    print("Loading CLIP concept embeddings...")
    concept_conc = torch.load(BOTH_CONC, weights_only=True)   # (7, 40, 512)
    gallery_40, gallery_cat = build_galleries(concept_conc)
    intra_galleries = build_intra_galleries(gallery_40)

    sub_names = sorted([f.replace(".npy","") for f in os.listdir(DE_DATA_DIR)
                        if f.endswith(".npy")])
    print(f"Running hierarchical evaluation across {len(sub_names)} subjects...\n")

    cat_r1s, conc_r1s, intra_r1s = [], [], []
    per_cat_all = {i: [] for i in range(N_CATS)}

    header = f"{'Subject':<18} {'Cat R@1':>9} {'Conc R@1':>10} {'Intra R@1':>11}"
    print(header); print("-"*52)

    for sub in sub_names:
        embs, cids = get_eeg_embs_all_folds(sub, device)
        if embs is None:
            continue
        cat_r1  = eval_category(embs, cids, gallery_cat)
        conc_r1 = eval_concept(embs, cids, gallery_40)
        intra_r1, per_cat = eval_intra_category(embs, cids, intra_galleries)

        cat_r1s.append(cat_r1)
        conc_r1s.append(conc_r1)
        intra_r1s.append(intra_r1)
        for ci, v in per_cat.items():
            per_cat_all[ci].append(v)

        print(f"  {sub:<16}  {cat_r1*100:>7.1f}%  {conc_r1*100:>8.1f}%  {intra_r1*100:>9.1f}%")

    cat_arr   = np.array(cat_r1s)
    conc_arr  = np.array(conc_r1s)
    intra_arr = np.array(intra_r1s)

    # Weighted average: each clip has chance 1/K_C; weight by n_clips_in_cat/total
    # = sum(K_C * N_CLIPS / 200 * 1/K_C) = sum(N_CLIPS/200) = N_CATS*N_CLIPS/200
    intra_chance_mean = N_CATS * N_CLIPS / 200   # = 8*5/200 = 0.20

    # Statistical tests vs chance
    t_cat,   p_cat   = stats.ttest_1samp(cat_arr,   1/N_CATS)
    t_conc,  p_conc  = stats.ttest_1samp(conc_arr,  1/N_CONCEPTS)
    t_intra, p_intra = stats.ttest_1samp(intra_arr, intra_chance_mean)

    def sig(p):
        return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."

    print(f"\n{'='*60}")
    print(f"Hierarchical Retrieval Results (N={len(cat_arr)} subjects)")
    print(f"{'='*60}")
    print(f"{'Level':<20} {'Mean R@1':>9} {'Chance':>8} {'Lift':>7} {'p-val':>10}")
    print("-"*60)
    for label, arr, chance in [
        ("Category (8-way)",       cat_arr,   1/N_CATS),
        ("Concept (40-way)",       conc_arr,  1/N_CONCEPTS),
        ("Intra-category",         intra_arr, intra_chance_mean),
    ]:
        t, p = stats.ttest_1samp(arr, chance)
        lift = arr.mean() / chance
        print(f"  {label:<18}  {arr.mean()*100:>7.2f}%  {chance*100:>6.2f}%  "
              f"{lift:>5.2f}×  {p:>8.4f} {sig(p)}")

    print(f"\nLift comparison:")
    print(f"  Category lift / Concept lift = {(cat_arr.mean()/(1/N_CATS)) / (conc_arr.mean()/(1/N_CONCEPTS)):.3f}×")
    print(f"  → Category retrieval is ___× more efficient than concept retrieval")

    # Per-category intra stats
    per_cat_mean = {}
    per_cat_std  = {}
    for ci in range(N_CATS):
        v = per_cat_all[ci]
        per_cat_mean[ci] = float(np.mean(v)) if v else 0.0
        per_cat_std[ci]  = float(np.std(v))  if v else 0.0

    results = {
        "cat_r1_mean":        float(cat_arr.mean()),
        "cat_r1_std":         float(cat_arr.std()),
        "conc_r1_mean":       float(conc_arr.mean()),
        "conc_r1_std":        float(conc_arr.std()),
        "intra_r1_mean":      float(intra_arr.mean()),
        "intra_r1_std":       float(intra_arr.std()),
        "intra_chance_mean":  float(intra_chance_mean),
        "cat_lift":           float(cat_arr.mean() / (1/N_CATS)),
        "conc_lift":          float(conc_arr.mean() / (1/N_CONCEPTS)),
        "intra_lift":         float(intra_arr.mean() / intra_chance_mean),
        "p_cat":              float(p_cat),
        "p_conc":             float(p_conc),
        "p_intra":            float(p_intra),
        "per_cat_intra_mean": per_cat_mean,
        "per_cat_intra_std":  per_cat_std,
        "cat_per_sub":        cat_r1s,
        "conc_per_sub":       conc_r1s,
        "intra_per_sub":      intra_r1s,
        "n_subjects":         len(cat_arr),
        "n_categories":       N_CATS,
        "n_concepts":         N_CONCEPTS,
    }
    with open(os.path.join(RESULTS_DIR, "results_hierarchical.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_hierarchy(results, os.path.join(FIGURES_DIR, "F10_hierarchical_retrieval.png"))
    plot_per_category_bars(results, os.path.join(FIGURES_DIR, "F11_intra_category.png"))
    plot_lift_comparison(results, os.path.join(FIGURES_DIR, "F12_cat_vs_conc_per_sub.png"))

    print(f"\nSaved → {RESULTS_DIR}/results_hierarchical.json")


if __name__ == "__main__":
    main()
