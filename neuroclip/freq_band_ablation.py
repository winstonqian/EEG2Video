"""
Frequency Band Ablation: Which EEG oscillations drive CLIP alignment?

For each of the 5 DE frequency bands (delta, theta, alpha, beta, gamma),
zero out that band in the test set and re-evaluate within-subject R@1.
Drop in performance = band's importance for semantic EEG-CLIP alignment.

Uses trained NeuroCLIP-Both (DE) models across all 21 subjects × 7 folds.

General contribution: identifies WHICH neural oscillations carry
semantic content alignable to vision-language embeddings.

Run from EEG2Video/:
    python neuroclip/freq_band_ablation.py
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
os.makedirs(FIGURES_DIR, exist_ok=True)

N_CONCEPTS, N_CLIPS, N_SESSIONS = 40, 5, 7
BAND_NAMES = ["Delta\n(1-3 Hz)", "Theta\n(4-7 Hz)", "Alpha\n(8-13 Hz)",
              "Beta\n(14-30 Hz)", "Gamma\n(31-50 Hz)"]
BAND_SHORT  = ["delta", "theta", "alpha", "beta", "gamma"]

ALL_SUBS = sorted([f.replace(".npy", "") for f in os.listdir(DE_DATA_DIR) if f.endswith(".npy")])


def build_gallery(device):
    conc = torch.load("neuroclip/clip_concept_both_embs_v2.pt", weights_only=True)
    g = torch.zeros(N_CONCEPTS, 512)
    c = torch.zeros(N_CONCEPTS)
    for s in range(N_SESSIONS):
        for pos in range(N_CONCEPTS):
            cid = int(GT_LABEL[s, pos])
            g[cid] += conc[s, pos]
            c[cid] += 1
    return F.normalize(g / c.clamp(min=1).unsqueeze(1), dim=-1).to(device)


def load_eeg(sub_name):
    p = os.path.join(DE_DATA_DIR, f"{sub_name}.npy")
    raw = np.load(p)
    n_s, n_c, n_cl, n_seg, n_ch, n_b = raw.shape
    return raw.mean(axis=3).reshape(n_s, n_c * n_cl, n_ch, n_b)


def eval_subject(sub, device, gallery, ablate_band=None):
    eeg_all = load_eeg(sub)
    n_ch, n_b = eeg_all.shape[2], eeg_all.shape[3]
    cids_all = np.repeat(GT_LABEL, N_CLIPS, axis=1)

    r1s = []
    for fold in range(N_SESSIONS):
        ckpt = os.path.join(RESULTS_DIR, f"{sub}_fold{fold}_de_k1_both.pt")
        if not os.path.exists(ckpt):
            continue
        model = EEGEncoder(n_channels=n_ch, n_time=n_b, embed_dim=512).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()

        flat = eeg_all[fold].reshape(N_CONCEPTS * N_CLIPS, -1)
        norm = StandardScaler().fit_transform(flat).reshape(N_CONCEPTS * N_CLIPS, n_ch, n_b)

        if ablate_band is not None:
            norm[:, :, ablate_band] = 0.0  # zero out one frequency band

        eeg_t = torch.tensor(norm, dtype=torch.float32).to(device)
        with torch.no_grad():
            embs = model(eeg_t)
        true_cids = torch.tensor(cids_all[fold], dtype=torch.long, device=device)
        preds = (embs @ gallery.T).argmax(1)
        r1s.append((preds == true_cids).float().mean().item())

    return float(np.mean(r1s)) if r1s else 0.0


def main():
    device = (torch.device("mps")  if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")
    gallery = build_gallery(device)

    print("Computing full-model baseline...")
    full_r1 = []
    for sub in ALL_SUBS:
        r = eval_subject(sub, device, gallery, ablate_band=None)
        full_r1.append(r)
        print(f"  {sub}: {r*100:.2f}%")
    full_r1 = np.array(full_r1)
    print(f"Full model: {full_r1.mean()*100:.2f}% ± {full_r1.std()*100:.2f}%\n")

    ablation_results = {}
    for bi, bname in enumerate(BAND_SHORT):
        print(f"Ablating {bname}...")
        abl_r1 = []
        for sub in ALL_SUBS:
            r = eval_subject(sub, device, gallery, ablate_band=bi)
            abl_r1.append(r)
        abl_r1 = np.array(abl_r1)
        drop = full_r1 - abl_r1
        t, p = stats.ttest_rel(full_r1, abl_r1)
        ablation_results[bname] = {
            "mean_r1": float(abl_r1.mean()),
            "std_r1":  float(abl_r1.std()),
            "mean_drop": float(drop.mean()),
            "std_drop":  float(drop.std()),
            "t": float(t), "p": float(p),
            "per_subject": abl_r1.tolist()
        }
        print(f"  {bname}: {abl_r1.mean()*100:.2f}%  drop={drop.mean()*100:.2f}%  t={t:.2f}  p={p:.4f}")

    ablation_results["full"] = {
        "mean_r1": float(full_r1.mean()),
        "std_r1":  float(full_r1.std()),
        "per_subject": full_r1.tolist()
    }

    with open(os.path.join(RESULTS_DIR, "results_freq_ablation.json"), "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\nSaved → {RESULTS_DIR}/results_freq_ablation.json")

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    def sig(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

    full_mean = full_r1.mean() * 100
    full_sem  = full_r1.std() / np.sqrt(len(full_r1)) * 100

    abl_means = [ablation_results[b]["mean_r1"] * 100 for b in BAND_SHORT]
    abl_sems  = [ablation_results[b]["std_r1"] / np.sqrt(len(ALL_SUBS)) * 100 for b in BAND_SHORT]
    drops     = [ablation_results[b]["mean_drop"] * 100 for b in BAND_SHORT]
    pvals     = [ablation_results[b]["p"] for b in BAND_SHORT]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: absolute R@1 per condition
    ax = axes[0]
    colors_band = ["#4472c4", "#70ad47", "#ed7d31", "#ffc000", "#9b59b6"]
    x = np.arange(len(BAND_SHORT))
    bars = ax.bar(x, abl_means, yerr=abl_sems, color=colors_band, width=0.55,
                  alpha=0.85, capsize=5, error_kw={"elinewidth": 1.8})
    ax.axhline(full_mean, color="black", linestyle="--", linewidth=2,
               label=f"Full model ({full_mean:.2f}%)")
    ax.axhline(1/40*100, color="gray", linestyle=":", linewidth=1.5, label="Chance (2.5%)")
    for bar, m, p in zip(bars, abl_means, pvals):
        ax.text(bar.get_x() + bar.get_width()/2, m + 0.15,
                sig(p), ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_NAMES, fontsize=9)
    ax.set_ylabel("Concept R@1 (%)", fontsize=11)
    ax.set_ylim(0, full_mean + 2.0)
    ax.set_title("(A) R@1 when each frequency band is ablated\n(* = sig. drop from full model)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # Right: performance drop (importance)
    ax = axes[1]
    drop_sems = [ablation_results[b]["std_drop"] / np.sqrt(len(ALL_SUBS)) * 100 for b in BAND_SHORT]
    bar_colors = [colors_band[i] if drops[i] > 0 else "#aaaaaa" for i in range(len(BAND_SHORT))]
    bars2 = ax.bar(x, drops, yerr=drop_sems, color=bar_colors, width=0.55,
                   alpha=0.85, capsize=5, error_kw={"elinewidth": 1.8})
    ax.axhline(0, color="black", linewidth=1)
    for bar, d, p in zip(bars2, drops, pvals):
        yoff = d + 0.05 if d >= 0 else d - 0.2
        ax.text(bar.get_x() + bar.get_width()/2, yoff + drop_sems[bars2.index(bar) if bar in list(bars2) else 0],
                sig(p), ha="center", fontsize=11, fontweight="bold")
    # Fix text placement
    for i, (bar, d, sem, p) in enumerate(zip(bars2, drops, drop_sems, pvals)):
        ax.texts[i].set_position((bar.get_x() + bar.get_width()/2, d + sem + 0.05))

    ax.set_xticks(x)
    ax.set_xticklabels(BAND_NAMES, fontsize=9)
    ax.set_ylabel("R@1 drop from full model (pp)", fontsize=11)
    ax.set_title("(B) Frequency band importance\n(drop = percentage-point decrease when ablated)", fontsize=11, fontweight="bold")

    plt.suptitle(
        "Frequency Band Ablation: Which EEG Oscillations Drive CLIP Alignment?\n"
        "NeuroCLIP-Both (DE) · 21 subjects · 7-fold CV · paired t-test vs full model",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "F19_freq_band_ablation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")

    print("\n=== Frequency Band Importance (sorted by drop) ===")
    sorted_bands = sorted(zip(BAND_SHORT, drops, pvals), key=lambda x: -x[1])
    for b, d, p in sorted_bands:
        print(f"  {b:6s}: drop={d:.3f}pp  {sig(p)}")


if __name__ == "__main__":
    main()
