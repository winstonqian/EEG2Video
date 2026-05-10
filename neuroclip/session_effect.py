"""
Session Effect Analysis: Does EEG-CLIP alignment vary across the 7 recording sessions?

SEED-DV has 7 sessions (blocks). LOBO fold k = test on session k.
Per-session R@1 tells us: are earlier or later sessions easier to decode?

This tests:
  - Practice/familiarity effects: subjects show cleaner neural responses later
  - Fatigue effects: neural signal degrades over sessions
  - Habituation: reduced novelty response over repeated exposure

General contribution: quantifies temporal stability of EEG-CLIP alignment,
relevant for any multi-session BCI deployment.

Run from EEG2Video/:
    python neuroclip/session_effect.py
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


def eval_fold(sub, fold, device, gallery):
    raw = np.load(os.path.join(DE_DATA_DIR, f"{sub}.npy"))
    n_s, n_c, n_cl, n_seg, n_ch, n_b = raw.shape
    eeg_all = raw.mean(axis=3).reshape(n_s, n_c * n_cl, n_ch, n_b)
    cids_all = np.repeat(GT_LABEL, N_CLIPS, axis=1)

    ckpt = os.path.join(RESULTS_DIR, f"{sub}_fold{fold}_de_k1_both.pt")
    if not os.path.exists(ckpt):
        return None
    model = EEGEncoder(n_channels=n_ch, n_time=n_b, embed_dim=512).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    flat = eeg_all[fold].reshape(N_CONCEPTS * N_CLIPS, -1)
    norm = StandardScaler().fit_transform(flat).reshape(N_CONCEPTS * N_CLIPS, n_ch, n_b)
    eeg_t = torch.tensor(norm, dtype=torch.float32).to(device)
    with torch.no_grad():
        embs = model(eeg_t)
    true_cids = torch.tensor(cids_all[fold], dtype=torch.long, device=device)
    preds = (embs @ gallery.T).argmax(1)
    return (preds == true_cids).float().mean().item()


def main():
    device = (torch.device("mps")  if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")
    gallery = build_gallery(device)

    # (n_subjects, n_sessions) per-fold R@1
    per_fold_r1 = np.full((len(ALL_SUBS), N_SESSIONS), np.nan)
    for si, sub in enumerate(ALL_SUBS):
        for fold in range(N_SESSIONS):
            r = eval_fold(sub, fold, device, gallery)
            if r is not None:
                per_fold_r1[si, fold] = r
        print(f"  {sub}: {[f'{v*100:.1f}%' for v in per_fold_r1[si]]}")

    session_means = np.nanmean(per_fold_r1, axis=0)
    session_sems  = np.nanstd(per_fold_r1, axis=0) / np.sqrt(len(ALL_SUBS))

    print(f"\nPer-session R@1:")
    for s in range(N_SESSIONS):
        print(f"  Session {s+1}: {session_means[s]*100:.2f}% ± {session_sems[s]*100:.2f}%")

    # Test linear trend across sessions
    x = np.arange(N_SESSIONS)
    r_trend, p_trend = stats.pearsonr(x, session_means)
    print(f"\nLinear trend: r={r_trend:.3f}  p={p_trend:.4f}")

    # Test first vs last session
    first_r1 = per_fold_r1[:, 0][~np.isnan(per_fold_r1[:, 0])]
    last_r1  = per_fold_r1[:, -1][~np.isnan(per_fold_r1[:, -1])]
    t_fl, p_fl = stats.ttest_rel(first_r1, last_r1)
    print(f"Session 1 vs 7: t={t_fl:.2f}  p={p_fl:.4f}")

    results = {
        "per_fold_r1": per_fold_r1.tolist(),
        "session_means": session_means.tolist(),
        "session_sems": session_sems.tolist(),
        "r_trend": float(r_trend), "p_trend": float(p_trend),
        "t_first_last": float(t_fl), "p_first_last": float(p_fl),
        "subjects": ALL_SUBS
    }
    with open(os.path.join(RESULTS_DIR, "results_session_effect.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    def sig(p):
        return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: mean R@1 per session with error bars
    ax = axes[0]
    xs = np.arange(1, N_SESSIONS + 1)
    ax.errorbar(xs, session_means*100, yerr=session_sems*100, fmt="o-",
                color="#4472c4", linewidth=2.5, markersize=9, capsize=6,
                label="Mean R@1 ± SEM")
    ax.fill_between(xs,
                    (session_means - session_sems)*100,
                    (session_means + session_sems)*100,
                    color="#4472c4", alpha=0.15)
    ax.axhline(1/40*100, color="gray", linestyle="--", linewidth=1.5, label="Chance (2.5%)")
    m_fit, b_fit = np.polyfit(xs, session_means*100, 1)
    ax.plot(xs, m_fit*xs + b_fit, "r--", linewidth=1.5,
            label=f"Trend: r={r_trend:.2f} {sig(p_trend)}")
    ax.set_xlabel("Session Number (1-7)", fontsize=11)
    ax.set_ylabel("Concept R@1 (%)", fontsize=11)
    ax.set_xticks(xs)
    ax.set_ylim(0, session_means.max()*100 + 1.5)
    ax.set_title(f"(A) R@1 across Sessions\n(linear trend: r={r_trend:.3f}, {sig(p_trend)})", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # Right: per-subject heatmap (subjects × sessions)
    ax = axes[1]
    valid = per_fold_r1[~np.any(np.isnan(per_fold_r1), axis=1)]
    im = ax.imshow(valid * 100, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=15, interpolation="nearest")
    ax.set_xlabel("Session Number", fontsize=11)
    ax.set_ylabel("Subject", fontsize=11)
    ax.set_xticks(range(N_SESSIONS))
    ax.set_xticklabels([f"S{i+1}" for i in range(N_SESSIONS)], fontsize=9)
    ax.set_yticks(range(len(valid)))
    ax.set_yticklabels([ALL_SUBS[i] for i in range(len(valid))], fontsize=7)
    plt.colorbar(im, ax=ax, label="R@1 (%)")
    ax.set_title("(B) Per-Subject × Session R@1 Heatmap\n(green=high, red=low)", fontsize=11, fontweight="bold")

    plt.suptitle(
        f"Session Effect: EEG-CLIP Alignment Stability Across 7 Recording Sessions\n"
        f"NeuroCLIP-Both (DE) · {len(ALL_SUBS)} subjects · Chance = 2.5%",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "F22_session_effect.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
