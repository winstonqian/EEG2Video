"""
RSA Noise Ceiling: What fraction of the theoretically achievable categorical
RSA does NeuroCLIP capture?

Method: Split-half reliability (Nili et al. 2014).
  - Split 21 subjects into halves A (10) and B (11)
  - Compute average EEG similarity matrix per half
  - Correlate half-A vs half-B average → noise ceiling
  - Spearman-Brown correct for full-N reliability

Our categorical RSA ρ=+0.107 as fraction of noise ceiling tells us
how much of the available categorical structure NeuroCLIP captures.

General contribution: contextualizes RSA effect size — are we near the
maximum achievable alignment, or is there substantial room to improve?

Run from EEG2Video/:
    python neuroclip/rsa_noise_ceiling.py
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

# Correct SEED-DV semantic groups (0=cat,1=husky,...,38=airplane,39=boat)
CAT_GROUPS = {
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
cat_label = np.zeros(40, dtype=int)
for gi, (_, cids) in enumerate(CAT_GROUPS.items()):
    for cid in cids: cat_label[cid] = gi
ref = (cat_label[:,None]==cat_label[None,:]).astype(float)
np.fill_diagonal(ref, 0)
ref_upper = ref[np.triu_indices(40,k=1)]

def upper_tri(M): return M[np.triu_indices(40,k=1)]


def get_eeg_sim_matrix(sub, device):
    raw = np.load(f"{DE_DATA_DIR}/{sub}.npy")
    n_s,n_c,n_cl,n_seg,n_ch,n_b = raw.shape
    eeg_all = raw.mean(axis=3).reshape(n_s, n_c*n_cl, n_ch, n_b)
    ckpt = f"{RESULTS_DIR}/{sub}_fold0_de_k1_both.pt"
    if not os.path.exists(ckpt): return None
    model = EEGEncoder(n_channels=n_ch, n_time=n_b, embed_dim=512).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    # Use TEST session (fold 0 held-out session 0) — consistent with rsa_analysis.py
    TEST_SESS = 0
    flat = eeg_all[TEST_SESS].reshape(N_CONCEPTS*N_CLIPS, -1)
    norm = StandardScaler().fit_transform(flat).reshape(N_CONCEPTS*N_CLIPS, n_ch, n_b)
    eeg_t = torch.tensor(norm, dtype=torch.float32).to(device)
    cids = np.repeat(GT_LABEL[TEST_SESS], N_CLIPS)
    ce = torch.zeros(N_CONCEPTS,512,device=device)
    cnt = torch.zeros(N_CONCEPTS,device=device)
    with torch.no_grad(): embs = model(eeg_t)
    for i,cid in enumerate(cids):
        ce[int(cid)]+=embs[i]; cnt[int(cid)]+=1
    ce = F.normalize(ce/cnt.clamp(min=1).unsqueeze(1), dim=-1)
    sim = (ce@ce.T).cpu().numpy(); np.fill_diagonal(sim,0)
    return sim


def main():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    print("Computing per-subject EEG similarity matrices...")
    sub_sims = {}
    for sub in ALL_SUBS:
        sim = get_eeg_sim_matrix(sub, device)
        if sim is not None:
            sub_sims[sub] = sim
            rho,_ = stats.spearmanr(upper_tri(sim), ref_upper)
            print(f"  {sub}: cat_RSA={rho:.4f}")

    valid_subs = list(sub_sims.keys())
    N = len(valid_subs)
    print(f"\n{N} valid subjects")

    # Our observed RSA (group average vs reference)
    avg_sim_all = np.mean([sub_sims[s] for s in valid_subs], axis=0)
    rho_obs, _ = stats.spearmanr(upper_tri(avg_sim_all), ref_upper)
    print(f"Group-average RSA ρ (all {N}): {rho_obs:.4f}")

    # Also load the permutation-test RSA from results_rsa.json
    rsa_loaded = json.load(open(f"{RESULTS_DIR}/results_rsa.json"))
    rho_single_sub = rsa_loaded["Category"]["rho"]
    print(f"Single-subject-avg RSA ρ (from results_rsa.json): {rho_single_sub:.4f}")

    # Split-half noise ceiling (20 random splits)
    np.random.seed(42)
    split_rhos = []
    for trial in range(50):
        idx = np.random.permutation(N)
        half_a = [valid_subs[i] for i in idx[:N//2]]
        half_b = [valid_subs[i] for i in idx[N//2:]]
        avg_a = np.mean([sub_sims[s] for s in half_a], axis=0)
        avg_b = np.mean([sub_sims[s] for s in half_b], axis=0)
        rho_ab, _ = stats.spearmanr(upper_tri(avg_a), upper_tri(avg_b))
        # Spearman-Brown correction: rho_full = 2*rho_half/(1+rho_half)
        rho_full = 2*rho_ab/(1+rho_ab) if rho_ab > -1 else 0
        split_rhos.append(float(rho_full))

    noise_ceiling = float(np.mean(split_rhos))
    nc_std = float(np.std(split_rhos))
    print(f"\nNoise ceiling (split-half, SB-corrected, 50 splits): {noise_ceiling:.4f} ± {nc_std:.4f}")
    print(f"Single-subject RSA as fraction of ceiling: {rho_single_sub/noise_ceiling*100:.1f}%")
    print(f"Group-average RSA as fraction of ceiling:  {rho_obs/noise_ceiling*100:.1f}%")

    # Per-subject RSA values
    per_sub_rsa = np.array([stats.spearmanr(upper_tri(sub_sims[s]), ref_upper)[0] for s in valid_subs])

    results = {
        "noise_ceiling": noise_ceiling, "nc_std": nc_std,
        "rho_single_sub": float(rho_single_sub),
        "rho_group_avg": float(rho_obs),
        "fraction_single": float(rho_single_sub/noise_ceiling) if noise_ceiling>0 else None,
        "fraction_group":  float(rho_obs/noise_ceiling) if noise_ceiling>0 else None,
        "per_subject_rsa": per_sub_rsa.tolist(),
        "split_rhos_corrected": split_rhos,
        "n_subjects": N,
    }
    with open(f"{RESULTS_DIR}/results_rsa_noise_ceiling.json","w") as f:
        json.dump(results, f, indent=2)

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."

    fig, axes = plt.subplots(1,2,figsize=(11,4))

    ax = axes[0]
    ax.hist(per_sub_rsa, bins=15, color="#4472c4", alpha=0.75, edgecolor="white", label="Per-subject RSA")
    ax.axvline(rho_single_sub, color="#4472c4", linewidth=2.5, linestyle="-",
               label=f"Group mean ρ={rho_single_sub:.4f}")
    ax.axvline(noise_ceiling, color="red", linewidth=2.5, linestyle="--",
               label=f"Noise ceiling ρ={noise_ceiling:.4f}")
    ax.axvline(0, color="gray", linewidth=1.5, linestyle=":")
    frac = rho_single_sub/noise_ceiling if noise_ceiling>0 else 0
    ax.fill_betweenx([0,6],[0,0],[noise_ceiling,noise_ceiling], alpha=0.08, color="red")
    ax.set_xlabel("Categorical RSA ρ", fontsize=11)
    ax.set_ylabel("Count (subjects)", fontsize=11)
    ax.set_title(f"(A) RSA vs Noise Ceiling\nNeuroCLIP captures {frac*100:.0f}% of available signal",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[1]
    labels = ["Noise\nCeiling\n(max possible)","NeuroCLIP\nGroup Avg","NeuroCLIP\nSingle-Sub Avg"]
    vals = [noise_ceiling, rho_obs, rho_single_sub]
    errs = [nc_std, 0, 0]
    cols = ["#e74c3c","#70ad47","#4472c4"]
    bars = ax.bar(range(3), vals, yerr=errs, color=cols, width=0.5, alpha=0.85, capsize=6)
    ax.axhline(0, color="black", linewidth=1)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2, max(v,0)+0.005,
                f"ρ={v:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Categorical RSA ρ", fontsize=11)
    ax.set_title(f"(B) Fraction of Noise Ceiling Captured\n({frac*100:.0f}% — {'near ceiling' if frac>0.6 else 'substantial room to improve'})",
                 fontsize=11, fontweight="bold")

    plt.suptitle("RSA Noise Ceiling: How Much Categorical Structure Does NeuroCLIP Capture?\n"
                 "Split-half reliability (Spearman-Brown corrected, 50 random splits)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/F29_rsa_noise_ceiling.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {path}")

if __name__ == "__main__":
    main()
