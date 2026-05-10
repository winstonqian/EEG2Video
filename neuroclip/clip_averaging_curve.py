"""
Clip Averaging Curve: How Does R@1 Scale with EEG Repetitions?

Each SEED-DV concept has 5 video clips shown per session.
Instead of evaluating each clip independently (standard R@1),
we can average N clips before retrieval — reducing EEG noise.

Tests how R@1 scales from N=1 (single clip) to N=5 (all clips averaged).
This mimics real BCI scenarios: more repetitions = better but slower.

Hypothesis: averaging reduces neural noise → monotonically increasing R@1.
The slope tells us about the neural signal-to-noise ratio.

Run from EEG2Video/:
    python neuroclip/clip_averaging_curve.py
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from scipy import stats
import itertools

sys.path.insert(0, os.path.dirname(__file__))
from dataset import GT_LABEL
from models_neuroclip import EEGEncoder

DE_DATA_DIR = "data/DE_1per1s"
RESULTS_DIR = "neuroclip/results"
FIGURES_DIR = "neuroclip/figures"
N_CONCEPTS, N_CLIPS, N_SESSIONS = 40, 5, 7
TEST_SESS = 0

ALL_SUBS = sorted([f.replace(".npy","") for f in os.listdir(DE_DATA_DIR) if f.endswith(".npy")])

CONCEPT_NAMES = [
    "cat","husky","elephant","horses","panda","rabbit","bird","fish","jellyfish","whale",
    "turtle","flowers","mushrooms","forest","boxing","dancing","running","skiing","computer","construction",
    "crowd","beach","city","mountain","road","waterfall","fireworks","banana","cheesecake","drink",
    "pizza","watermelon","drums","guitar","piano","motorcycle","car","balloon","airplane","boat"
]

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


def eval_avg_clips(sub, n_avg, gallery, device, n_reps=20):
    """
    For each concept, randomly sample n_avg clips, average their EEG embeddings,
    and compute R@1. Repeat n_reps times and return mean R@1.
    """
    ckpt = f"{RESULTS_DIR}/{sub}_fold0_de_k1_both.pt"
    if not os.path.exists(ckpt): return None
    raw = np.load(f"{DE_DATA_DIR}/{sub}.npy")
    n_s,n_c,n_cl,n_seg,n_ch,n_b = raw.shape
    eeg_all = raw.mean(axis=3).reshape(n_s, n_c*n_cl, n_ch, n_b)
    model = EEGEncoder(n_channels=n_ch, n_time=n_b, embed_dim=512).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    # Normalize all clips
    flat = eeg_all[TEST_SESS].reshape(N_CONCEPTS*N_CLIPS, -1)
    norm = StandardScaler().fit_transform(flat).reshape(N_CONCEPTS*N_CLIPS, n_ch, n_b)
    eeg_t = torch.tensor(norm, dtype=torch.float32).to(device)
    cids = np.repeat(GT_LABEL[TEST_SESS], N_CLIPS).astype(int)
    with torch.no_grad(): embs = model(eeg_t)
    embs_np = embs.cpu().numpy()  # (200, 512)

    # Group by concept: embs_by_concept[cid] = list of clip embeddings
    embs_by_concept = {}
    for i, cid in enumerate(cids):
        if cid not in embs_by_concept: embs_by_concept[cid] = []
        embs_by_concept[cid].append(embs_np[i])

    rng = np.random.default_rng(42)
    r1_trials = []
    for _ in range(n_reps):
        avg_embs = np.zeros((N_CONCEPTS, 512))
        for cid in range(N_CONCEPTS):
            clips = embs_by_concept.get(cid, [])
            if len(clips) < n_avg: continue
            selected = rng.choice(len(clips), n_avg, replace=False)
            avg_embs[cid] = np.stack([clips[i] for i in selected]).mean(axis=0)
        # L2-normalize averaged embeddings
        norms = np.linalg.norm(avg_embs, axis=1, keepdims=True).clip(min=1e-8)
        avg_embs = avg_embs / norms
        avg_t = torch.tensor(avg_embs, dtype=torch.float32).to(device)
        preds = (avg_t @ gallery.T).argmax(1).cpu().numpy()
        r1_trials.append((preds == np.arange(N_CONCEPTS)).mean())
    return np.mean(r1_trials)


def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."


def main():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")
    gallery = build_gallery(device)

    n_avgs = [1, 2, 3, 4, 5]
    results_by_n = {n: [] for n in n_avgs}

    for sub in ALL_SUBS:
        ckpt = f"{RESULTS_DIR}/{sub}_fold0_de_k1_both.pt"
        if not os.path.exists(ckpt): continue
        sub_vals = []
        for n in n_avgs:
            r1 = eval_avg_clips(sub, n, gallery, device, n_reps=30)
            if r1 is not None:
                results_by_n[n].append(r1)
                sub_vals.append(r1)
        if sub_vals:
            print(f"  {sub}: " + "  ".join(f"N={n}:{sub_vals[i]*100:.2f}%" for i,n in enumerate(n_avgs)))

    print("\nClip Averaging Curve:")
    means = [np.mean(results_by_n[n])*100 for n in n_avgs]
    sems  = [np.std(results_by_n[n])*100/np.sqrt(len(results_by_n[n])) for n in n_avgs]
    for n, m, s in zip(n_avgs, means, sems):
        t, p = stats.ttest_1samp(np.array(results_by_n[n])*100, 2.5)
        print(f"  N={n}: {m:.2f}% ± {s:.2f}%  vs chance t={t:.2f}  {sig(p)}")

    # Linear trend: does R@1 increase with N?
    flat_n = np.repeat(n_avgs, [len(results_by_n[n]) for n in n_avgs])
    flat_r1 = np.concatenate([results_by_n[n] for n in n_avgs])
    slope, intercept, r, p_trend, se = stats.linregress(flat_n, flat_r1)
    print(f"\nLinear trend: slope={slope*100:.3f} pp/clip  r={r:.4f}  p={p_trend:.4f}  {sig(p_trend)}")

    # Improvement from N=1 to N=5
    n1_arr = np.array(results_by_n[1])
    n5_arr = np.array(results_by_n[5])
    t_gain, p_gain = stats.ttest_rel(n5_arr, n1_arr)
    gain = (n5_arr.mean() - n1_arr.mean())*100
    print(f"\nGain from N=1 to N=5: +{gain:.2f} pp  t={t_gain:.2f}  p={p_gain:.4f}  {sig(p_gain)}")

    results = {
        "n_avgs": n_avgs,
        "means": means,
        "sems":  sems,
        "slope_pp_per_clip": float(slope*100),
        "r": float(r), "p_trend": float(p_trend),
        "n1_mean": float(n1_arr.mean()*100), "n5_mean": float(n5_arr.mean()*100),
        "gain_pp": float(gain),
        "t_gain": float(t_gain), "p_gain": float(p_gain),
        "per_n_r1": {str(n): [float(v) for v in results_by_n[n]] for n in n_avgs},
    }
    with open(f"{RESULTS_DIR}/results_clip_averaging_curve.json","w") as f:
        json.dump(results, f, indent=2)

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.errorbar(n_avgs, means, yerr=sems, fmt="o-", color="#4472c4",
                linewidth=2.5, markersize=8, capsize=6, label="NeuroCLIP")
    ax.axhline(2.5, color="gray", linestyle="--", linewidth=1.5, label="Chance (2.5%)")
    # Fit and plot trend line
    x_fit = np.linspace(1, 5, 100)
    ax.plot(x_fit, (slope*x_fit + intercept)*100, "k-", linewidth=1.5, alpha=0.5,
            label=f"Linear fit: +{slope*100:.2f}pp/clip")
    for n, m, s in zip(n_avgs, means, sems):
        ax.text(n, m+s+0.15, f"{m:.2f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("N clips averaged per concept", fontsize=11)
    ax.set_ylabel("R@1 (%)", fontsize=11)
    ax.set_xticks(n_avgs)
    ax.set_title(f"(A) EEG Averaging Curve\n+{gain:.2f}pp gain from N=1 to N=5 {sig(p_gain)}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[1]
    # Box plots for each N
    data = [np.array(results_by_n[n])*100 for n in n_avgs]
    bp = ax.boxplot(data, positions=n_avgs, widths=0.5, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2))
    for patch in bp["boxes"]:
        patch.set_facecolor("#4472c4"); patch.set_alpha(0.7)
    ax.axhline(2.5, color="gray", linestyle="--", linewidth=1.5, label="Chance (2.5%)")
    ax.set_xlabel("N clips averaged per concept", fontsize=11)
    ax.set_ylabel("R@1 (%)", fontsize=11)
    ax.set_xticks(n_avgs)
    ax.set_title(f"(B) Subject Distribution by N\n(r={r:.3f}, slope=+{slope*100:.3f}pp/clip {sig(p_trend)})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    plt.suptitle("EEG Signal Averaging: More Repetitions → Better Retrieval?\n"
                 "Trade-off between acquisition time and retrieval accuracy in BCI",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/F42_clip_averaging_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
