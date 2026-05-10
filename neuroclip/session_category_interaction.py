"""
Session × Category Interaction: Is Activity Advantage Consistent Across Sessions?

For each of 7 LOBO sessions, computes per-category R@1 by loading
the checkpoint trained with that session held out (fold k) and
evaluating on session k's data.

Tests:
1. Does the Activity > Passive advantage hold in each individual session?
2. Is there a session × category type interaction?
3. Does any session show reversed (Passive > Activity) patterns?

Run from EEG2Video/:
    python neuroclip/session_category_interaction.py
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
act_cids = np.array([c for cat in ACTIVITY_CATS for c in SEMANTIC_GROUPS[cat]])
pas_cids = np.array([c for cat in SEMANTIC_GROUPS if cat not in ACTIVITY_CATS for c in SEMANTIC_GROUPS[cat]])


def build_gallery(device):
    conc = torch.load("neuroclip/clip_concept_both_embs_v2.pt", weights_only=True)
    g = torch.zeros(N_CONCEPTS, 512); c = torch.zeros(N_CONCEPTS)
    for s in range(N_SESSIONS):
        for pos in range(N_CONCEPTS):
            cid = int(GT_LABEL[s, pos]); g[cid] += conc[s, pos]; c[cid] += 1
    return F.normalize(g / c.clamp(min=1).unsqueeze(1), dim=-1).to(device)


def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."


def main():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")
    gallery = build_gallery(device)

    # For each session (fold), collect activity R@1 and passive R@1 across subjects
    sess_act_r1 = {fold: [] for fold in range(N_SESSIONS)}
    sess_pas_r1 = {fold: [] for fold in range(N_SESSIONS)}
    sess_r1     = {fold: [] for fold in range(N_SESSIONS)}

    for sub in ALL_SUBS:
        raw = np.load(f"{DE_DATA_DIR}/{sub}.npy")
        n_s,n_c,n_cl,n_seg,n_ch,n_b = raw.shape
        eeg_all = raw.mean(axis=3).reshape(n_s, n_c*n_cl, n_ch, n_b)

        for fold in range(N_SESSIONS):
            ckpt = f"{RESULTS_DIR}/{sub}_fold{fold}_de_k1_both.pt"
            if not os.path.exists(ckpt): continue
            model = EEGEncoder(n_channels=n_ch, n_time=n_b, embed_dim=512).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            model.eval()
            flat = eeg_all[fold].reshape(N_CONCEPTS*N_CLIPS, -1)
            norm = StandardScaler().fit_transform(flat).reshape(N_CONCEPTS*N_CLIPS, n_ch, n_b)
            eeg_t = torch.tensor(norm, dtype=torch.float32).to(device)
            cids  = np.repeat(GT_LABEL[fold], N_CLIPS).astype(int)
            with torch.no_grad(): embs = model(eeg_t)
            preds = (embs @ gallery.T).argmax(1).cpu().numpy()
            correct = (preds == cids)

            # Per-category R@1
            act_mask = np.isin(cids, act_cids)
            pas_mask = np.isin(cids, pas_cids)
            sess_act_r1[fold].append(correct[act_mask].mean() if act_mask.sum() > 0 else np.nan)
            sess_pas_r1[fold].append(correct[pas_mask].mean() if pas_mask.sum() > 0 else np.nan)
            sess_r1[fold].append(correct.mean())

        print(f"  {sub}: done all folds")

    # Compute stats per session
    print("\nPer-session Activity vs Passive R@1:")
    sess_gaps = []
    for fold in range(N_SESSIONS):
        act_arr = np.array([v for v in sess_act_r1[fold] if not np.isnan(v)])
        pas_arr = np.array([v for v in sess_pas_r1[fold] if not np.isnan(v)])
        r1_arr  = np.array([v for v in sess_r1[fold]    if not np.isnan(v)])
        if len(act_arr) == 0: continue
        t, p = stats.ttest_rel(act_arr, pas_arr)
        gap = act_arr.mean() - pas_arr.mean()
        sess_gaps.append(gap)
        print(f"  Session {fold+1}: R@1={r1_arr.mean()*100:.2f}%  "
              f"Activity={act_arr.mean()*100:.2f}%  Passive={pas_arr.mean()*100:.2f}%  "
              f"gap={gap*100:+.2f}pp  t={t:.2f}  {sig(p)}")

    # Test consistency of gap across sessions
    sess_gaps = np.array(sess_gaps)
    t_gap, p_gap = stats.ttest_1samp(sess_gaps, 0)
    print(f"\nMean activity gap across sessions: {sess_gaps.mean()*100:+.2f}pp  "
          f"t={t_gap:.2f}  p={p_gap:.4f}  {sig(p_gap)}")
    print(f"Sessions with positive gap: {(sess_gaps > 0).sum()}/{len(sess_gaps)}")

    # 2×7 ANOVA-style: session × category type
    # For each subject × session, we have act_r1 and pas_r1
    # Test interaction: does session effect differ for activity vs passive?
    all_act = np.array([[v for v in sess_act_r1[f]] for f in range(N_SESSIONS)]).T   # (n_sub, 7)
    all_pas = np.array([[v for v in sess_pas_r1[f]] for f in range(N_SESSIONS)]).T   # (n_sub, 7)
    gap_by_sess = all_act - all_pas   # (n_sub, 7) activity advantage by session
    # Test if gap varies across sessions (within-subject)
    # F-test: is variance of gap across sessions > 0?
    f_stat = np.nanvar(gap_by_sess, axis=1).mean() / np.nanvar(gap_by_sess, ddof=1)
    rho_sess, p_sess = stats.spearmanr(np.arange(N_SESSIONS), np.nanmean(gap_by_sess, axis=0))
    print(f"\nActivity gap trend across sessions: ρ={rho_sess:.4f}  p={p_sess:.4f}  {sig(p_sess)}")

    results = {
        "sess_act_mean": [float(np.nanmean(sess_act_r1[f])) for f in range(N_SESSIONS)],
        "sess_pas_mean": [float(np.nanmean(sess_pas_r1[f])) for f in range(N_SESSIONS)],
        "sess_r1_mean":  [float(np.nanmean(sess_r1[f]))     for f in range(N_SESSIONS)],
        "sess_gaps": sess_gaps.tolist(),
        "t_gap": float(t_gap), "p_gap": float(p_gap),
        "n_positive_gap": int((sess_gaps > 0).sum()),
        "rho_sess": float(rho_sess), "p_sess": float(p_sess),
    }
    with open(f"{RESULTS_DIR}/results_session_category_interaction.json","w") as f:
        json.dump(results, f, indent=2)

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sess_labels = [f"S{i+1}" for i in range(N_SESSIONS)]
    act_means = [np.nanmean(sess_act_r1[f])*100 for f in range(N_SESSIONS)]
    pas_means = [np.nanmean(sess_pas_r1[f])*100 for f in range(N_SESSIONS)]
    r1_means  = [np.nanmean(sess_r1[f])*100     for f in range(N_SESSIONS)]

    # Panel A: activity vs passive per session
    ax = axes[0]
    x = np.arange(N_SESSIONS)
    w = 0.35
    ax.bar(x-w/2, act_means, w, color="#e74c3c", alpha=0.85, label="Activity")
    ax.bar(x+w/2, pas_means, w, color="#4472c4", alpha=0.85, label="Passive")
    ax.axhline(2.5, color="black", linestyle="--", linewidth=1.5, label="Chance (2.5%)")
    for i, (a, p) in enumerate(zip(act_means, pas_means)):
        gap = a - p
        col = "green" if gap > 0 else "red"
        ax.text(i, max(a, p)+0.1, f"{gap:+.1f}pp", ha="center", fontsize=7.5,
                fontweight="bold", color=col)
    ax.set_xticks(x); ax.set_xticklabels(sess_labels, fontsize=10)
    ax.set_ylabel("Mean R@1 (%)", fontsize=11)
    ax.set_title(f"(A) Activity vs Passive R@1 per Session\n"
                 f"Gap consistent: {(sess_gaps > 0).sum()}/{len(sess_gaps)} sessions positive",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel B: session gap over time
    ax = axes[1]
    ax.plot(range(1, N_SESSIONS+1), sess_gaps*100, "o-", color="#4472c4",
            linewidth=2.5, markersize=8)
    ax.fill_between(range(1, N_SESSIONS+1), sess_gaps*100, 0,
                    alpha=0.2, color="#4472c4")
    ax.axhline(0, color="black", linewidth=1.5)
    ax.axhline(sess_gaps.mean()*100, color="#e74c3c", linestyle="--", linewidth=2,
               label=f"Mean gap={sess_gaps.mean()*100:+.2f}pp {sig(p_gap)}")
    ax.set_xlabel("Session (1–7)", fontsize=11)
    ax.set_ylabel("Activity − Passive R@1 (pp)", fontsize=11)
    ax.set_xticks(range(1, N_SESSIONS+1))
    ax.set_title(f"(B) Activity Advantage Across Sessions\nρ={rho_sess:.3f} {sig(p_sess)} (trend over time)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)

    plt.suptitle("Session × Category Interaction\n"
                 "Is the Activity > Passive Decodability Advantage Robust Across All Sessions?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/F46_session_category_interaction.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
