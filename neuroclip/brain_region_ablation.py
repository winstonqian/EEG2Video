"""
Brain Region Ablation: Which scalp regions drive EEG-CLIP semantic alignment?

Uses SEED-IV/DV 62-channel layout (international 10-20 system).
Zero out one brain region at a time and measure R@1 drop.

Regions: Frontal | Fronto-Central | Central+Temporal | Parietal | Occipital

General contribution: spatial specificity of semantic EEG signals —
which scalp regions carry information alignable to CLIP visual semantics.

Run from EEG2Video/:
    python neuroclip/brain_region_ablation.py
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

# SEED 62-channel layout (SEED-IV standard, same as SEED-DV)
CHANNEL_NAMES = [
    'FP1','FPZ','FP2','AF3','AF4',                          # 0-4
    'F7','F5','F3','F1','FZ','F2','F4','F6','F8',            # 5-13
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8',   # 14-22
    'T7','C5','C3','C1','CZ','C2','C4','C6','T8',            # 23-31
    'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',   # 32-40
    'P7','P5','P3','P1','PZ','P2','P4','P6','P8',            # 41-49
    'PO7','PO5','PO3','POZ','PO4','PO6','PO8',               # 50-56
    'CB1','O1','OZ','O2','CB2'                               # 57-61
]

REGIONS = {
    "Frontal\n(FP,AF,F)":          list(range(0, 14)),   # 14 ch
    "Fronto-Central\n(FC,FT)":     list(range(14, 23)),  #  9 ch
    "Central+Temporal\n(C,T)":     list(range(23, 32)),  #  9 ch
    "Parietal\n(TP,CP,P)":         list(range(32, 50)),  # 18 ch
    "Occipital\n(PO,O,CB)":        list(range(50, 62)),  # 12 ch
}

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
    raw = np.load(os.path.join(DE_DATA_DIR, f"{sub_name}.npy"))
    n_s, n_c, n_cl, n_seg, n_ch, n_b = raw.shape
    return raw.mean(axis=3).reshape(n_s, n_c * n_cl, n_ch, n_b)


def eval_subject(sub, device, gallery, ablate_channels=None):
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

        if ablate_channels is not None:
            norm[:, ablate_channels, :] = 0.0  # zero out entire region

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
        r = eval_subject(sub, device, gallery, ablate_channels=None)
        full_r1.append(r)
    full_r1 = np.array(full_r1)
    print(f"Full model: {full_r1.mean()*100:.2f}% ± {full_r1.std()*100:.2f}%\n")

    region_results = {}
    region_names = list(REGIONS.keys())
    for rname, ch_ids in REGIONS.items():
        print(f"Ablating {rname.replace(chr(10),' ')} ({len(ch_ids)} channels)...")
        abl_r1 = []
        for sub in ALL_SUBS:
            r = eval_subject(sub, device, gallery, ablate_channels=ch_ids)
            abl_r1.append(r)
        abl_r1 = np.array(abl_r1)
        drop = full_r1 - abl_r1
        t, p = stats.ttest_rel(full_r1, abl_r1)
        region_results[rname] = {
            "n_channels": len(ch_ids),
            "mean_r1": float(abl_r1.mean()),
            "std_r1":  float(abl_r1.std()),
            "mean_drop": float(drop.mean()),
            "std_drop":  float(drop.std()),
            "t": float(t), "p": float(p),
            "per_subject": abl_r1.tolist()
        }
        def sig(p):
            return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."
        print(f"  → {abl_r1.mean()*100:.2f}%  drop={drop.mean()*100:.2f}pp  t={t:.2f}  p={p:.4f} {sig(p)}")

    region_results["full"] = {
        "mean_r1": float(full_r1.mean()),
        "std_r1":  float(full_r1.std()),
        "per_subject": full_r1.tolist()
    }

    with open(os.path.join(RESULTS_DIR, "results_brain_region_ablation.json"), "w") as f:
        json.dump(region_results, f, indent=2)
    print(f"\nSaved → {RESULTS_DIR}/results_brain_region_ablation.json")

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    def sig(p):
        return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."

    full_mean = full_r1.mean() * 100
    full_sem  = full_r1.std() / np.sqrt(len(full_r1)) * 100

    region_short = ["Frontal", "Fronto-\nCentral", "Central+\nTemporal", "Parietal", "Occipital"]
    abl_means = [region_results[k]["mean_r1"] * 100 for k in region_names]
    abl_sems  = [region_results[k]["std_r1"] / np.sqrt(len(ALL_SUBS)) * 100 for k in region_names]
    drops     = [region_results[k]["mean_drop"] * 100 for k in region_names]
    drop_sems = [region_results[k]["std_drop"] / np.sqrt(len(ALL_SUBS)) * 100 for k in region_names]
    pvals     = [region_results[k]["p"] for k in region_names]
    n_chs     = [region_results[k]["n_channels"] for k in region_names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Anterior→Posterior color gradient (frontal=blue, occipital=red)
    region_colors = ["#4472c4", "#70ad47", "#ffc000", "#ed7d31", "#e74c3c"]
    x = np.arange(len(region_names))

    # Left panel: R@1 when ablated
    ax = axes[0]
    bars = ax.bar(x, abl_means, yerr=abl_sems, color=region_colors, width=0.55,
                  alpha=0.85, capsize=5, error_kw={"elinewidth": 1.8})
    ax.axhline(full_mean, color="black", linestyle="--", linewidth=2,
               label=f"Full model ({full_mean:.2f}%)")
    ax.axhline(1/40*100, color="gray", linestyle=":", linewidth=1.5, label="Chance (2.5%)")
    for bar, m, p, n in zip(bars, abl_means, pvals, n_chs):
        ax.text(bar.get_x() + bar.get_width()/2, m + 0.1,
                sig(p), ha="center", fontsize=11, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width()/2, 0.15,
                f"n={n}", ha="center", fontsize=7, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(region_short, fontsize=9)
    ax.set_ylabel("Concept R@1 (%)", fontsize=11)
    ax.set_ylim(0, full_mean + 2.0)
    ax.set_title("(A) R@1 when each brain region is ablated\n(* = sig. drop; n = channels zeroed)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # Right panel: performance drop
    ax = axes[1]
    bars2 = ax.bar(x, drops, yerr=drop_sems, color=region_colors, width=0.55,
                   alpha=0.85, capsize=5, error_kw={"elinewidth": 1.8})
    ax.axhline(0, color="black", linewidth=1)
    for i, (bar, d, sem, p) in enumerate(zip(bars2, drops, drop_sems, pvals)):
        ypos = d + sem + 0.05 if d >= 0 else d - sem - 0.2
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                sig(p), ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(region_short, fontsize=9)
    ax.set_ylabel("R@1 drop from full model (pp)", fontsize=11)
    ax.set_title("(B) Region importance (pp drop when ablated)\n(anterior→posterior: blue→red)", fontsize=11, fontweight="bold")

    plt.suptitle(
        "Brain Region Ablation: Spatial Specificity of Semantic EEG-CLIP Alignment\n"
        "NeuroCLIP-Both (DE) · 21 subjects · SEED-DV 62-channel layout · paired t-test",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "F21_brain_region_ablation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")

    print("\n=== Brain Region Importance (sorted by drop) ===")
    sorted_regions = sorted(zip(region_short, drops, pvals, n_chs), key=lambda x: -x[1])
    for rn, d, p, n in sorted_regions:
        print(f"  {rn.replace(chr(10),' '):22s}: drop={d:.3f}pp  {sig(p)}  (n={n} channels)")


if __name__ == "__main__":
    main()
