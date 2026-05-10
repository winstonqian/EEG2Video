"""
Representational Similarity Analysis (RSA) for NeuroCLIP.

Core question: Does the structure of EEG representations mirror the semantic
structure of CLIP space?

Method (Kriegeskorte et al., 2008):
  1. Compute 40×40 pairwise cosine similarity matrix for EEG concept-mean embeddings
  2. Compute 40×40 pairwise cosine similarity matrix for CLIP concept embeddings
  3. Spearman-correlate the upper-triangle vectors of the two matrices
  4. Permutation test (1000 label shuffles) for significance

Analyses:
  A. CLIP-EEG RSA with three CLIP conditions (text, image, both)
  B. Category-EEG RSA (binary same-vs-different semantic group structure)
  C. Heatmap visualizations with semantic group ordering
  D. Per-subject distributions and stats

Run from EEG2Video/:
    python neuroclip/rsa_analysis.py
"""

import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from dataset import GT_LABEL
from models_neuroclip import EEGEncoder

DE_DATA_DIR  = "data/DE_1per1s"
RESULTS_DIR  = "neuroclip/results"
FIGURES_DIR  = "neuroclip/figures"
BOTH_CONC    = "neuroclip/clip_concept_both_embs_v2.pt"
TEXT_CONC    = "neuroclip/clip_concept_text_embs_v2.pt"
IMAGE_CONC   = "neuroclip/clip_concept_image_embs_v2.pt"
os.makedirs(FIGURES_DIR, exist_ok=True)

N_CONCEPTS, N_CLIPS, N_SESSIONS = 40, 5, 7
TEST_SESSION = 0   # fold 0 holds out session 0
N_PERMS      = 2000

# ---------------------------------------------------------------------------
# Semantic group structure (40 SEED-DV concepts)
# ---------------------------------------------------------------------------

SEMANTIC_GROUPS = {
    "Animals":  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Nature":   [11, 12, 13, 23, 25],
    "Food":     [27, 28, 29, 30, 31],
    "Sports":   [14, 15, 16, 17],
    "Music":    [32, 33, 34],
    "Vehicles": [35, 36, 37, 38, 39],
    "Urban":    [20, 21, 22, 24],
    "People":   [18, 19],
    "Other":    [26],
}

CONCEPT_NAMES = {
    0:"cat", 1:"husky", 2:"elephant", 3:"horses", 4:"panda", 5:"rabbit",
    6:"bird", 7:"fish", 8:"jellyfish", 9:"whale", 10:"turtle",
    11:"flowers", 12:"mushrooms", 13:"forest", 14:"boxing", 15:"dancing",
    16:"running", 17:"skiing", 18:"computer", 19:"construction",
    20:"crowd", 21:"beach", 22:"city", 23:"mountain", 24:"road",
    25:"waterfall", 26:"fireworks", 27:"banana", 28:"cheesecake",
    29:"drink", 30:"pizza", 31:"watermelon", 32:"drums", 33:"guitar",
    34:"piano", 35:"motorcycle", 36:"car", 37:"balloon", 38:"airplane",
    39:"boat",
}

GROUP_COLORS = {
    "Animals": "#4472c4", "Nature": "#70ad47", "Food": "#ffc000",
    "Sports": "#ed7d31", "Music": "#9b59b6", "Vehicles": "#e74c3c",
    "Urban": "#95a5a6", "People": "#1abc9c", "Other": "#bdc3c7",
}

# Canonical ordering: group concepts by semantic category for heatmap display
GROUP_ORDER = []
GROUP_BOUNDARIES = []
for grp, cids in SEMANTIC_GROUPS.items():
    GROUP_BOUNDARIES.append((len(GROUP_ORDER), len(GROUP_ORDER) + len(cids), grp))
    GROUP_ORDER.extend(cids)
# Map: concept_id → position in GROUP_ORDER
CID_TO_POS = {cid: pos for pos, cid in enumerate(GROUP_ORDER)}


# ---------------------------------------------------------------------------
# Core RSA helpers
# ---------------------------------------------------------------------------

def sim_matrix(embs):
    """(N, D) L2-normed → (N, N) cosine similarity matrix."""
    embs = F.normalize(embs, dim=-1)
    return (embs @ embs.T).cpu().numpy()


def upper_tri(mat):
    """Return upper-triangle (excluding diagonal) as a 1-D vector."""
    n = mat.shape[0]
    idx = np.triu_indices(n, k=1)
    return mat[idx]


def rsa_spearman(eeg_sim, clip_sim):
    """Spearman correlation between upper triangles of two (N, N) sim matrices."""
    rho, pval = stats.spearmanr(upper_tri(eeg_sim), upper_tri(clip_sim))
    return float(rho), float(pval)


def permutation_test(eeg_sim, clip_sim, n_perms=N_PERMS, seed=42):
    """
    Permutation test: shuffle EEG concept labels, recompute RSA.
    Returns empirical p-value (one-tailed: how often perm_rho >= observed_rho).
    """
    rng  = np.random.default_rng(seed)
    obs_rho, _ = rsa_spearman(eeg_sim, clip_sim)
    n    = eeg_sim.shape[0]
    null = []
    for _ in range(n_perms):
        perm = rng.permutation(n)
        perm_sim = eeg_sim[np.ix_(perm, perm)]
        r, _ = rsa_spearman(perm_sim, clip_sim)
        null.append(r)
    p_emp = (np.sum(np.array(null) >= obs_rho) + 1) / (n_perms + 1)
    return obs_rho, p_emp, np.array(null)


def build_category_matrix():
    """40×40 binary matrix: 1 if same semantic group, 0 if different."""
    mat = np.zeros((N_CONCEPTS, N_CONCEPTS))
    cid_to_group = {}
    for grp, cids in SEMANTIC_GROUPS.items():
        for c in cids:
            cid_to_group[c] = grp
    for i in range(N_CONCEPTS):
        for j in range(N_CONCEPTS):
            if i != j and cid_to_group[i] == cid_to_group[j]:
                mat[i, j] = 1.0
    return mat


# ---------------------------------------------------------------------------
# Load EEG embeddings per subject
# ---------------------------------------------------------------------------

def get_eeg_concept_means(sub_name, device):
    """Return (40, 512) concept-mean EEG embeddings for one subject (fold 0)."""
    sub_path   = os.path.join(DE_DATA_DIR, f"{sub_name}.npy")
    model_path = os.path.join(RESULTS_DIR, f"{sub_name}_fold0_de_k1_both.pt")
    if not os.path.exists(sub_path) or not os.path.exists(model_path):
        return None

    raw = np.load(sub_path)
    n_sess, n_conc, n_cl, n_seg, n_ch, n_bands = raw.shape
    eeg_all = raw.mean(axis=3).reshape(n_sess, n_conc * n_cl, n_ch, n_bands)

    model = EEGEncoder(n_channels=n_ch, n_time=n_bands, embed_dim=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    sess_data = eeg_all[TEST_SESSION]
    flat = sess_data.reshape(200, -1)
    norm = StandardScaler().fit_transform(flat).reshape(200, n_ch, n_bands)
    eeg_t = torch.tensor(norm, dtype=torch.float32).to(device)

    with torch.no_grad():
        eeg_embs = model(eeg_t)   # (200, 512)

    concept_ids = np.repeat(GT_LABEL[TEST_SESSION], repeats=N_CLIPS)
    concept_means = torch.zeros(N_CONCEPTS, 512, device=device)
    counts = torch.zeros(N_CONCEPTS, device=device)
    for i, cid in enumerate(concept_ids):
        concept_means[int(cid)] += eeg_embs[i]
        counts[int(cid)] += 1
    return F.normalize(concept_means / counts.clamp(min=1).unsqueeze(1), dim=-1)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def reorder(mat, order):
    """Reorder rows and columns of matrix by semantic group order."""
    return mat[np.ix_(order, order)]


def plot_sim_matrices(clip_sim_sorted, mean_eeg_sim_sorted, save_path):
    """Side-by-side: CLIP similarity matrix vs mean EEG similarity matrix."""
    labels = [CONCEPT_NAMES[c] for c in GROUP_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    titles = ["CLIP Semantic Similarity", "EEG Representational Similarity (mean, 21 subjects)"]
    data   = [clip_sim_sorted, mean_eeg_sim_sorted]
    cmaps  = ["RdBu_r", "RdBu_r"]

    for ax, title, mat, cmap in zip(axes, titles, data, cmaps):
        im = ax.imshow(mat, aspect="auto", cmap=cmap,
                       vmin=mat.min(), vmax=mat.max())
        ax.set_xticks(range(N_CONCEPTS))
        ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
        ax.set_yticks(range(N_CONCEPTS))
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Cosine similarity")

        # Draw group boundaries
        for start, end, grp in GROUP_BOUNDARIES:
            for spine_val in [start - 0.5, end - 0.5]:
                ax.axhline(spine_val, color="black", linewidth=1.2)
                ax.axvline(spine_val, color="black", linewidth=1.2)
            mid = (start + end - 1) / 2
            ax.text(mid, -1.5, grp, ha="center", va="top",
                    fontsize=7, fontweight="bold", rotation=0,
                    transform=ax.transData)

    plt.suptitle("Representational Similarity Analysis (RSA): EEG vs CLIP\n"
                 "Concepts ordered by semantic category", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.close()


def plot_rsa_results(results, save_path):
    """Bar chart of RSA rho across CLIP conditions with significance markers."""
    labels   = list(results.keys())
    rhos     = [results[l]["rho"] for l in labels]
    p_emps   = [results[l]["p_emp"] for l in labels]
    null_stds = [results[l]["null_std"] for l in labels]

    def sig_str(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "n.s."

    colors = ["#4472c4", "#ed7d31", "#70ad47", "#ffc000", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(labels)), rhos, color=colors[:len(labels)],
                  width=0.55, alpha=0.85,
                  yerr=null_stds, capsize=6,
                  error_kw={"elinewidth": 2, "label": "Permutation null ±1 SD"})

    ax.axhline(0, color="black", linewidth=1)
    for i, (bar, rho, p) in enumerate(zip(bars, rhos, p_emps)):
        y = rho + null_stds[i] + 0.005
        ax.text(bar.get_x() + bar.get_width()/2, y,
                f"ρ={rho:.3f}\n{sig_str(p)}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("RSA Spearman ρ (EEG–CLIP)", fontsize=12)
    ax.set_title("RSA: EEG Representational Structure vs CLIP Semantic Structure\n"
                 "(error bars = permutation null ±1 SD; n=21 subjects)", fontsize=11)
    ax.set_ylim(min(min(rhos) - 0.05, -0.02), max(max(rhos) + 0.08, 0.15))
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved → {save_path}")
    plt.close()


def plot_per_subject_rsa(per_sub_rhos, save_path):
    """Violin + strip plot of per-subject RSA rho for each condition."""
    conditions = list(per_sub_rhos.keys())
    data = [per_sub_rhos[c] for c in conditions]
    colors = ["#4472c4", "#ed7d31", "#70ad47", "#ffc000", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(9, 5))
    parts = ax.violinplot(data, positions=range(len(conditions)),
                          showmeans=False, showmedians=True, widths=0.5)
    for i, (pc, col) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")

    rng = np.random.default_rng(0)
    for i, (d, col) in enumerate(zip(data, colors)):
        jitter = rng.uniform(-0.08, 0.08, len(d))
        ax.scatter(i + jitter, d, color=col, s=25, alpha=0.85, zorder=3)
        ax.scatter(i, np.mean(d), color="white", edgecolors="black",
                   s=70, zorder=5, linewidths=1.5)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.5,
               label="Zero (no alignment)")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_ylabel("RSA Spearman ρ per subject", fontsize=12)
    ax.set_title("Per-Subject RSA: EEG–CLIP Alignment (white dot = mean)",
                 fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved → {save_path}")
    plt.close()


def plot_null_distribution(null, obs_rho, label, save_path):
    """Permutation null distribution vs observed RSA rho."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null, bins=50, color="#aaaaaa", edgecolor="white", alpha=0.8,
            label="Permutation null")
    ax.axvline(obs_rho, color="#e74c3c", linewidth=2.5,
               label=f"Observed ρ = {obs_rho:.3f}")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    p_emp = (np.sum(null >= obs_rho) + 1) / (len(null) + 1)
    ax.set_xlabel("RSA Spearman ρ", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"RSA Permutation Test — {label}\n"
                 f"p_emp = {p_emp:.4f}  ({len(null)} permutations)", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved → {save_path}")
    plt.close()


def plot_scatter_eeg_vs_clip(eeg_sim, clip_sim, label, save_path):
    """Scatter: clip pairwise sims vs eeg pairwise sims (upper tri only)."""
    x = upper_tri(clip_sim)
    y = upper_tri(eeg_sim)
    rho, _ = stats.spearmanr(x, y)

    # Color points by whether same group
    cat_mat = build_category_matrix()
    same_grp = upper_tri(cat_mat).astype(bool)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x[~same_grp], y[~same_grp], s=8, alpha=0.35,
               color="#aaaaaa", label="Different category")
    ax.scatter(x[same_grp],  y[same_grp],  s=14, alpha=0.7,
               color="#4472c4", label="Same category")

    # Regression line
    m, b = np.polyfit(x, y, 1)
    xr = np.linspace(x.min(), x.max(), 100)
    ax.plot(xr, m*xr + b, color="#e74c3c", linewidth=2, label=f"ρ = {rho:.3f}")

    ax.set_xlabel(f"CLIP {label} pairwise similarity", fontsize=11)
    ax.set_ylabel("EEG pairwise similarity (concept means)", fontsize=11)
    ax.set_title(f"CLIP vs EEG Concept Similarity — {label}\n"
                 "(each point = one concept pair)", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved → {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cpu")
    print("Device: cpu")

    # Load CLIP concept embeddings (averaged across sessions)
    both_conc  = torch.load(BOTH_CONC,  weights_only=True).mean(0)  # (40,512)
    text_conc  = torch.load(TEXT_CONC,  weights_only=True).mean(0)
    image_conc = torch.load(IMAGE_CONC, weights_only=True).mean(0)
    both_conc  = F.normalize(both_conc,  dim=-1)
    text_conc  = F.normalize(text_conc,  dim=-1)
    image_conc = F.normalize(image_conc, dim=-1)

    clip_sims = {
        "CLIP-Both":  sim_matrix(both_conc),
        "CLIP-Text":  sim_matrix(text_conc),
        "CLIP-Image": sim_matrix(image_conc),
    }
    cat_mat = build_category_matrix()

    # Load all subjects
    sub_names = sorted([f.replace(".npy","") for f in os.listdir(DE_DATA_DIR)
                        if f.endswith(".npy")])
    print(f"Loading {len(sub_names)} subjects...")

    all_eeg_sims = []
    loaded_subs  = []
    for sub in sub_names:
        cm = get_eeg_concept_means(sub, device)
        if cm is not None:
            all_eeg_sims.append(sim_matrix(cm))
            loaded_subs.append(sub)
    print(f"  Loaded {len(loaded_subs)} subjects successfully.")

    mean_eeg_sim = np.mean(all_eeg_sims, axis=0)   # (40, 40)

    # --- RSA: CLIP conditions + category structure ---
    all_rsa_targets = {**clip_sims, "Category": cat_mat}
    per_sub_rhos = {k: [] for k in all_rsa_targets}
    results      = {}

    print(f"\n{'Condition':<16}  {'ρ (mean EEG)':>13}  {'p_emp':>8}  {'sig':>5}")
    print("-" * 50)

    for label, ref_mat in all_rsa_targets.items():
        # Per-subject RSA rho (using each subject's own EEG sim matrix)
        sub_rhos = [rsa_spearman(esim, ref_mat)[0] for esim in all_eeg_sims]
        per_sub_rhos[label] = sub_rhos

        # Permutation test on mean EEG sim matrix
        obs_rho, p_emp, null = permutation_test(mean_eeg_sim, ref_mat)

        # Also one-sample t-test of per-subject rhos vs 0
        t, p_t = stats.ttest_1samp(sub_rhos, 0)

        sig = ("***" if p_emp < 0.001 else "**" if p_emp < 0.01
               else "*" if p_emp < 0.05 else "n.s.")
        results[label] = {
            "rho":        obs_rho,
            "p_emp":      float(p_emp),
            "null_std":   float(null.std()),
            "sig":        sig,
            "per_sub_rhos": sub_rhos,
            "mean_sub_rho": float(np.mean(sub_rhos)),
            "std_sub_rho":  float(np.std(sub_rhos)),
            "t_vs_zero":  float(t),
            "p_vs_zero":  float(p_t),
        }
        print(f"  {label:<14}  ρ={obs_rho:>+.4f}  p_emp={p_emp:.4f}  {sig}")
        print(f"              per-sub: {np.mean(sub_rhos):+.4f}±{np.std(sub_rhos):.4f}  "
              f"t={t:.3f} p={p_t:.4f}")

        # Null distribution plot for CLIP-Both
        if label == "CLIP-Both":
            plot_null_distribution(
                null, obs_rho, label,
                os.path.join(FIGURES_DIR, "rsa_null_both.png"))

    # --- Figures ---
    clip_sorted = reorder(clip_sims["CLIP-Both"], GROUP_ORDER)
    eeg_sorted  = reorder(mean_eeg_sim, GROUP_ORDER)
    plot_sim_matrices(clip_sorted, eeg_sorted,
                      os.path.join(FIGURES_DIR, "rsa_sim_matrices.png"))

    # Bar chart: CLIP conditions only
    clip_results = {k: v for k, v in results.items() if k.startswith("CLIP")}
    plot_rsa_results(clip_results,
                     os.path.join(FIGURES_DIR, "rsa_bar.png"))

    # Per-subject violin
    plot_per_subject_rsa(per_sub_rhos,
                         os.path.join(FIGURES_DIR, "rsa_per_subject.png"))

    # Scatter CLIP-Both vs mean EEG
    plot_scatter_eeg_vs_clip(
        mean_eeg_sim, clip_sims["CLIP-Both"], "Both",
        os.path.join(FIGURES_DIR, "rsa_scatter_both.png"))

    # Save
    with open(os.path.join(RESULTS_DIR, "results_rsa.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {RESULTS_DIR}/results_rsa.json")

    # Summary
    print(f"\n=== RSA Summary ===")
    print(f"{'Condition':<16}  {'ρ':>7}  {'p_emp':>8}  {'Sub ρ mean':>12}")
    for label, r in results.items():
        print(f"  {label:<14}  {r['rho']:>+.4f}  {r['p_emp']:>7.4f} {r['sig']:>4}  "
              f"{r['mean_sub_rho']:>+.4f}±{r['std_sub_rho']:.4f}")


if __name__ == "__main__":
    main()
