"""
Temporal Dynamics: Early (0-1s) vs Late (1-2s) segment R@1.

Each 2-second clip has 2 EEG segments. Tests whether:
1. Early vs late segment is more informative (visual onset vs sustained)
2. Activity concepts show stronger temporal differentiation than static ones.

EEG data shape: (7, 40, 5, 2, 62, 5) — n_segs=2 axis at index 3.
Instead of mean(axis=3), take each segment separately.

Run from EEG2Video/:
    python neuroclip/temporal_dynamics.py
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
act_cids = [c for cat in ACTIVITY_CATS for c in SEMANTIC_GROUPS[cat]]
pas_cids = [c for cat in ["Animals","Nature","Food","Vehicles","Urban","Other"]
            for c in SEMANTIC_GROUPS[cat]]


def build_gallery(device):
    conc = torch.load("neuroclip/clip_concept_both_embs_v2.pt", weights_only=True)
    g = torch.zeros(N_CONCEPTS, 512); c = torch.zeros(N_CONCEPTS)
    for s in range(N_SESSIONS):
        for pos in range(N_CONCEPTS):
            cid = int(GT_LABEL[s, pos]); g[cid] += conc[s, pos]; c[cid] += 1
    return F.normalize(g / c.clamp(min=1).unsqueeze(1), dim=-1).to(device)


def eval_segment(sub, seg_idx, gallery, device):
    raw = np.load(f"{DE_DATA_DIR}/{sub}.npy")
    n_s, n_c, n_cl, n_seg, n_ch, n_b = raw.shape
    # Take only seg_idx segment (shape: n_s, n_c, n_cl, n_ch, n_b)
    seg_data = raw[:, :, :, seg_idx, :, :]  # (7, 40, 5, 62, 5)
    eeg_all = seg_data.reshape(n_s, n_c * n_cl, n_ch, n_b)

    r1s = []
    for fold in range(N_SESSIONS):
        ckpt = f"{RESULTS_DIR}/{sub}_fold{fold}_de_k1_both.pt"
        if not os.path.exists(ckpt): continue
        model = EEGEncoder(n_channels=n_ch, n_time=n_b, embed_dim=512).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()
        flat = eeg_all[fold].reshape(N_CONCEPTS * N_CLIPS, -1)
        norm = StandardScaler().fit_transform(flat).reshape(N_CONCEPTS * N_CLIPS, n_ch, n_b)
        eeg_t = torch.tensor(norm, dtype=torch.float32).to(device)
        cids_fold = np.repeat(GT_LABEL[fold], N_CLIPS)
        with torch.no_grad(): embs = model(eeg_t)
        preds = (embs @ gallery.T).argmax(1).cpu().numpy()
        r1s.append((preds == cids_fold).mean())
    return np.array(r1s) if r1s else None


def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."


def main():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")
    gallery = build_gallery(device)

    early_r1s, late_r1s = [], []
    for sub in ALL_SUBS:
        e = eval_segment(sub, 0, gallery, device)
        l = eval_segment(sub, 1, gallery, device)
        if e is not None and l is not None:
            early_r1s.append(e.mean())
            late_r1s.append(l.mean())
            print(f"  {sub}: early={e.mean()*100:.2f}%  late={l.mean()*100:.2f}%")

    early_r1s = np.array(early_r1s)
    late_r1s  = np.array(late_r1s)
    t, p = stats.ttest_rel(late_r1s, early_r1s)
    print(f"\nEarly (0-1s): {early_r1s.mean()*100:.2f}% ± {early_r1s.std()*100:.2f}%")
    print(f"Late  (1-2s): {late_r1s.mean()*100:.2f}% ± {late_r1s.std()*100:.2f}%")
    print(f"Late > Early: t={t:.2f}  p={p:.4f}  {sig(p)}")

    results = {
        "early_mean": float(early_r1s.mean()), "early_std": float(early_r1s.std()),
        "late_mean":  float(late_r1s.mean()),  "late_std":  float(late_r1s.std()),
        "t": float(t), "p": float(p),
        "early_per_sub": early_r1s.tolist(), "late_per_sub": late_r1s.tolist(),
    }
    with open(f"{RESULTS_DIR}/results_temporal_dynamics.json","w") as f:
        json.dump(results, f, indent=2)

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    vals = [early_r1s.mean()*100, late_r1s.mean()*100]
    sems = [early_r1s.std()*100/np.sqrt(len(early_r1s)), late_r1s.std()*100/np.sqrt(len(late_r1s))]
    cols = ["#4472c4","#70ad47"]
    bars = ax.bar([0,1], vals, yerr=sems, color=cols, width=0.5, alpha=0.85, capsize=8)
    ax.axhline(2.5, color="gray", linestyle="--", linewidth=1.5, label="Chance (2.5%)")
    y_top = max(v+e for v,e in zip(vals,sems))+0.2
    ax.plot([0,0,1,1],[y_top,y_top+0.1,y_top+0.1,y_top],lw=1.5,color="black")
    ax.text(0.5, y_top+0.12, f"t={t:.1f} {sig(p)}", ha="center", fontsize=11, fontweight="bold")
    for bar,v,e in zip(bars,vals,sems):
        ax.text(bar.get_x()+bar.get_width()/2, v+e+0.05, f"{v:.2f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks([0,1]); ax.set_xticklabels(["Early (0–1s)","Late (1–2s)"], fontsize=11)
    ax.set_ylabel("Mean R@1 (%)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_title(f"(A) Temporal Dynamics\nEarly vs Late Segment R@1", fontsize=11, fontweight="bold")

    ax = axes[1]
    ax.scatter(early_r1s*100, late_r1s*100, c="#4472c4", s=70, alpha=0.8,
               edgecolors="white", linewidths=0.5)
    lim = [min(early_r1s.min(), late_r1s.min())*100-0.3,
           max(early_r1s.max(), late_r1s.max())*100+0.3]
    ax.plot(lim, lim, "k--", linewidth=1.5, alpha=0.5, label="Equal")
    ax.set_xlabel("Early R@1 (%)", fontsize=11)
    ax.set_ylabel("Late R@1 (%)", fontsize=11)
    n_late_better = (late_r1s > early_r1s).sum()
    ax.set_title(f"(B) Per-Subject: Early vs Late\n{n_late_better}/{len(late_r1s)} subjects: Late > Early",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    plt.suptitle("Temporal Dynamics: When Is EEG Most Informative?\n"
                 "Early (0–1s) vs Late (1–2s) segment within each 2-second clip",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/F33_temporal_dynamics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
