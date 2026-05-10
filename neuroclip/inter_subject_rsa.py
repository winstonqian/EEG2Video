"""
Inter-Subject RSA: Do different brains represent concepts similarly?

For each pair of subjects, compute Spearman correlation between their
EEG pairwise similarity matrices (after contrastive training).

High inter-subject RSA → shared brain representations → cross-subject decoding possible
Low inter-subject RSA → idiosyncratic representations → explains flat scaling curve

This provides a mechanistic explanation for the subject scaling finding:
cross-subject R@1 doesn't improve with N because each brain encodes concepts differently.

General contribution: quantifies neural representation idiosyncrasy —
a fundamental barrier to cross-subject EEG-BCI transfer.

Run from EEG2Video/:
    python neuroclip/inter_subject_rsa.py
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
ALL_SUBS = sorted([f.replace(".npy","") for f in os.listdir(DE_DATA_DIR) if f.endswith(".npy")])


def build_clip_gallery():
    conc = torch.load("neuroclip/clip_concept_both_embs_v2.pt", weights_only=True)
    g = torch.zeros(N_CONCEPTS, 512); c = torch.zeros(N_CONCEPTS)
    for s in range(N_SESSIONS):
        for pos in range(N_CONCEPTS):
            cid = int(GT_LABEL[s, pos]); g[cid] += conc[s, pos]; c[cid] += 1
    gallery = F.normalize(g / c.clamp(min=1).unsqueeze(1), dim=-1)
    clip_sim = (gallery @ gallery.T).numpy()
    np.fill_diagonal(clip_sim, 0)
    return clip_sim


def get_eeg_sim_matrix(sub, device):
    raw = np.load(os.path.join(DE_DATA_DIR, f"{sub}.npy"))
    n_s,n_c,n_cl,n_seg,n_ch,n_b = raw.shape
    eeg_all = raw.mean(axis=3).reshape(n_s, n_c*n_cl, n_ch, n_b)

    # Average EEG embeddings per concept over training folds (all except fold 0)
    ckpt = os.path.join(RESULTS_DIR, f"{sub}_fold0_de_k1_both.pt")
    if not os.path.exists(ckpt):
        return None

    model = EEGEncoder(n_channels=n_ch, n_time=n_b, embed_dim=512).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    # Use TEST session (fold 0 held-out session 0) — consistent with rsa_analysis.py
    TEST_SESS = 0
    flat = eeg_all[TEST_SESS].reshape(N_CONCEPTS*N_CLIPS, -1)
    norm = StandardScaler().fit_transform(flat).reshape(N_CONCEPTS*N_CLIPS, n_ch, n_b)
    eeg_t = torch.tensor(norm, dtype=torch.float32).to(device)
    cids = np.repeat(GT_LABEL[TEST_SESS], N_CLIPS)
    concept_embs = torch.zeros(N_CONCEPTS, 512, device=device)
    counts = torch.zeros(N_CONCEPTS, device=device)
    with torch.no_grad():
        embs = model(eeg_t)
    for i, cid in enumerate(cids):
        concept_embs[int(cid)] += embs[i]; counts[int(cid)] += 1

    concept_embs = F.normalize(concept_embs / counts.clamp(min=1).unsqueeze(1), dim=-1)
    sim = (concept_embs @ concept_embs.T).cpu().numpy()
    np.fill_diagonal(sim, 0)
    return sim


def upper_tri(M):
    return M[np.triu_indices(40, k=1)]


def main():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    clip_sim = build_clip_gallery()
    clip_upper = upper_tri(clip_sim)

    print("Computing per-subject EEG similarity matrices...")
    eeg_sims = {}
    for sub in ALL_SUBS:
        sim = get_eeg_sim_matrix(sub, device)
        if sim is not None:
            eeg_sims[sub] = sim
            print(f"  {sub}: done")

    valid_subs = list(eeg_sims.keys())
    N = len(valid_subs)
    print(f"\n{N} valid subjects")

    # Pairwise inter-subject RSA
    pair_rhos = []
    pair_info = []
    for i in range(N):
        for j in range(i+1, N):
            rho, _ = stats.spearmanr(upper_tri(eeg_sims[valid_subs[i]]),
                                     upper_tri(eeg_sims[valid_subs[j]]))
            pair_rhos.append(float(rho))
            pair_info.append((valid_subs[i], valid_subs[j]))

    pair_rhos = np.array(pair_rhos)
    mean_inter_rsa = float(pair_rhos.mean())
    std_inter_rsa  = float(pair_rhos.std())
    t_vs_zero, p_vs_zero = stats.ttest_1samp(pair_rhos, 0)

    # Per-subject EEG vs CLIP similarity RSA
    sub_clip_rhos = []
    for sub in valid_subs:
        rho, _ = stats.spearmanr(upper_tri(eeg_sims[sub]), clip_upper)
        sub_clip_rhos.append(float(rho))
    sub_clip_rhos = np.array(sub_clip_rhos)
    t_clip, p_clip = stats.ttest_1samp(sub_clip_rhos, 0)

    print(f"\nInter-subject EEG RSA: {mean_inter_rsa:.4f} ± {std_inter_rsa:.4f}")
    print(f"  t={t_vs_zero:.3f}  p={p_vs_zero:.4f}")
    print(f"EEG-CLIP RSA per subject: {sub_clip_rhos.mean():.4f} ± {sub_clip_rhos.std():.4f}")
    print(f"  t={t_clip:.3f}  p={p_clip:.4f}")
    print(f"N pairs: {len(pair_rhos)}")

    results = {
        "valid_subjects": valid_subs,
        "mean_inter_rsa": mean_inter_rsa,
        "std_inter_rsa":  std_inter_rsa,
        "t_vs_zero": float(t_vs_zero), "p_vs_zero": float(p_vs_zero),
        "pair_rhos": pair_rhos.tolist(),
        "per_subject_clip_rsa": sub_clip_rhos.tolist(),
        "mean_clip_rsa": float(sub_clip_rhos.mean()),
        "t_clip": float(t_clip), "p_clip": float(p_clip),
        "n_pairs": int(len(pair_rhos))
    }
    with open(os.path.join(RESULTS_DIR, "results_inter_subject_rsa.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Distribution of pairwise inter-subject RSA
    ax = axes[0]
    ax.hist(pair_rhos, bins=25, color="#4472c4", alpha=0.75, edgecolor="white")
    ax.axvline(mean_inter_rsa, color="red", linewidth=2.5,
               label=f"Mean={mean_inter_rsa:.4f}")
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="Zero")
    ax.set_xlabel("Inter-Subject EEG RSA ρ", fontsize=11)
    ax.set_ylabel("Count (subject pairs)", fontsize=11)
    ax.set_title(f"(A) Pairwise Inter-Subject EEG RSA\n"
                 f"Mean={mean_inter_rsa:.4f}  {sig(p_vs_zero)}  (N={len(pair_rhos)} pairs)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel B: Inter-subject RSA matrix (N×N heatmap)
    ax = axes[1]
    rsa_matrix = np.eye(N)
    k = 0
    for i in range(N):
        for j in range(i+1, N):
            rsa_matrix[i, j] = pair_rhos[k]
            rsa_matrix[j, i] = pair_rhos[k]
            k += 1
    im = ax.imshow(rsa_matrix, cmap="RdYlGn", vmin=-0.1, vmax=0.3, aspect="auto")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    short_names = [s.replace("sub","S") for s in valid_subs]
    ax.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax.set_yticklabels(short_names, fontsize=7)
    plt.colorbar(im, ax=ax, label="RSA ρ")
    ax.set_title(f"(B) Inter-Subject RSA Matrix\n(green=similar, red=dissimilar representations)",
                 fontsize=11, fontweight="bold")

    # Panel C: Per-subject EEG-CLIP RSA vs inter-subject RSA (mean per subject)
    ax = axes[2]
    # Compute mean inter-subject RSA per subject
    mean_per_sub = []
    for i, sub in enumerate(valid_subs):
        row_rhos = [pair_rhos[k] for k, (s1,s2) in enumerate(pair_info) if s1==sub or s2==sub]
        mean_per_sub.append(np.mean(row_rhos))
    mean_per_sub = np.array(mean_per_sub)

    sc = ax.scatter(mean_per_sub, sub_clip_rhos,
                    c=mean_per_sub, cmap="coolwarm", s=80,
                    edgecolors="black", linewidths=0.5)
    for i, sub in enumerate(valid_subs):
        ax.annotate(sub.replace("sub","S"), (mean_per_sub[i], sub_clip_rhos[i]),
                   xytext=(3,2), textcoords="offset points", fontsize=7)
    r_scatter, p_scatter = stats.pearsonr(mean_per_sub, sub_clip_rhos)
    m, b = np.polyfit(mean_per_sub, sub_clip_rhos, 1)
    x_line = np.linspace(mean_per_sub.min(), mean_per_sub.max(), 100)
    ax.plot(x_line, m*x_line+b, "k--", linewidth=2)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Mean Inter-Subject RSA ρ (subject typicality)", fontsize=10)
    ax.set_ylabel("EEG-CLIP RSA ρ (per subject)", fontsize=10)
    ax.set_title(f"(C) Neural Typicality vs CLIP Alignment\n"
                 f"r={r_scatter:.3f} {sig(p_scatter)}",
                 fontsize=11, fontweight="bold")
    plt.colorbar(sc, ax=ax, label="Neural Typicality")

    plt.suptitle(
        "Inter-Subject RSA: Are EEG Representations Idiosyncratic Across Brains?\n"
        "NeuroCLIP-Both (DE) · Low inter-subject RSA → Individual variability bottleneck",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "F23_inter_subject_rsa.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")

    print(f"\n=== Key Finding ===")
    print(f"Mean inter-subject EEG RSA: {mean_inter_rsa:.4f} {sig(p_vs_zero)}")
    if p_vs_zero > 0.05:
        print("→ EEG representations are IDIOSYNCRATIC across subjects")
        print("  (explains flat scaling curve: more subjects don't help)")
    else:
        print(f"→ Some shared structure across subjects (ρ={mean_inter_rsa:.4f})")


if __name__ == "__main__":
    main()
