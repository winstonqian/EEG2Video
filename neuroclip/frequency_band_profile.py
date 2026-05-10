"""
Frequency Band Profile per Category: What DE Pattern Characterizes Activity Concepts?

Computes the mean Differential Entropy (DE) value per frequency band
(delta, theta, alpha, beta, gamma) for each of 9 semantic categories.

Tests whether activity concepts (Sports, Music, People) have a distinctive
frequency band profile — e.g., higher gamma/theta and lower alpha — which
would explain why they're more decodable.

The DE features are the raw input to NeuroCLIP. Understanding which bands
drive activity vs passive concept processing gives mechanistic insight.

Run from EEG2Video/:
    python neuroclip/frequency_band_profile.py
"""
import os, sys, json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from dataset import GT_LABEL

DE_DATA_DIR = "data/DE_1per1s"
RESULTS_DIR = "neuroclip/results"
FIGURES_DIR = "neuroclip/figures"
N_CONCEPTS, N_CLIPS, N_SESSIONS = 40, 5, 7

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

BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
N_BANDS = 5

cat_lookup = {}
for cat, ids in SEMANTIC_GROUPS.items():
    for cid in ids: cat_lookup[cid] = cat


def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."


def main():
    # For each concept, accumulate mean DE per band (averaged over channels)
    conc_de = np.zeros((N_CONCEPTS, N_BANDS))   # (40, 5) mean DE
    conc_cnt = np.zeros(N_CONCEPTS)

    for sub in ALL_SUBS:
        path = f"{DE_DATA_DIR}/{sub}.npy"
        if not os.path.exists(path): continue
        raw = np.load(path)  # (n_sess, n_concepts, n_clips, n_segs, n_ch, n_bands)
        n_s,n_c,n_cl,n_seg,n_ch,n_b = raw.shape

        for sess in range(n_s):
            for pos in range(n_c):
                cid = int(GT_LABEL[sess, pos])
                # Mean over clips, segments, channels
                de = raw[sess, pos, :, :, :, :].mean(axis=(0,1,2))  # (5,) mean over clips,segs,ch
                conc_de[cid] += de
                conc_cnt[cid] += 1

    conc_de /= conc_cnt.clip(min=1).reshape(-1,1)  # (40, 5) mean DE per concept per band

    print("Per-concept DE band profile (delta, theta, alpha, beta, gamma):")
    for cid in range(N_CONCEPTS):
        cat = cat_lookup[cid]
        act = "A" if cat in ACTIVITY_CATS else "P"
        de_str = "  ".join(f"{conc_de[cid,b]:.3f}" for b in range(N_BANDS))
        print(f"  [{act}] {CONCEPT_NAMES[cid]:15s}: {de_str}  [{cat}]")

    # Per-band: activity vs passive t-test
    print(f"\nPer-band Activity vs Passive DE comparison:")
    for b, band in enumerate(BAND_NAMES):
        act_de = conc_de[act_cids, b]
        pas_de = conc_de[pas_cids, b]
        t, p = stats.ttest_ind(act_de, pas_de)
        print(f"  {band:7s}: Activity={act_de.mean():.4f} ± {act_de.std():.4f}  "
              f"Passive={pas_de.mean():.4f} ± {pas_de.std():.4f}  "
              f"t={t:.2f}  p={p:.4f}  {sig(p)}")

    # Per-category mean DE profile (normalized by overall mean)
    overall_de = conc_de.mean(axis=0)  # (5,) overall mean
    cat_de = {}
    for cat, cids in SEMANTIC_GROUPS.items():
        cat_de[cat] = conc_de[cids].mean(axis=0)   # (5,) mean DE for this category

    print(f"\nOverall mean DE by band: {' '.join(f'{v:.3f}' for v in overall_de)}")

    # ANOVA per band across categories
    for b, band in enumerate(BAND_NAMES):
        cat_vals = [conc_de[SEMANTIC_GROUPS[cat], b] for cat in SEMANTIC_GROUPS]
        f, p = stats.f_oneway(*cat_vals)
        print(f"  {band:7s} ANOVA across 9 categories: F={f:.2f}  p={p:.4f}  {sig(p)}")

    # Correlation: per-concept gamma DE vs R@1
    deco = json.load(open(f"{RESULTS_DIR}/results_concept_decodability.json"))
    r1s = np.array(deco["per_concept_r1"])
    print(f"\nCorrelation of each band's DE with R@1:")
    for b, band in enumerate(BAND_NAMES):
        rho, p = stats.spearmanr(conc_de[:, b], r1s)
        print(f"  {band:7s}: ρ={rho:.4f}  p={p:.4f}  {sig(p)}")

    results = {
        "band_names": BAND_NAMES,
        "concept_de": conc_de.tolist(),
        "concept_names": CONCEPT_NAMES,
        "cat_de": {cat: cat_de[cat].tolist() for cat in SEMANTIC_GROUPS},
        "overall_de": overall_de.tolist(),
    }
    with open(f"{RESULTS_DIR}/results_frequency_band_profile.json","w") as f:
        json.dump(results, f, indent=2)

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel A: per-category DE band profile (radar/polar)
    ax = axes[0]
    cat_order = sorted(SEMANTIC_GROUPS, key=lambda c: -np.mean(cat_de[c]))
    cols_a = {cat: ("#e74c3c" if cat in ACTIVITY_CATS else "#4472c4") for cat in cat_order}
    x = np.arange(N_BANDS)
    for i, cat in enumerate(cat_order[:6]):  # top-6 most distinct categories
        vals = (cat_de[cat] - overall_de) / (overall_de.clip(min=1e-8))
        linestyle = "-" if cat in ACTIVITY_CATS else "--"
        ax.plot(x, vals*100, linestyle, color=cols_a[cat], linewidth=2,
                marker="o", markersize=5, label=cat)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1.5)
    ax.set_xticks(x); ax.set_xticklabels(BAND_NAMES, fontsize=10)
    ax.set_ylabel("% deviation from overall mean DE", fontsize=10)
    ax.set_title("(A) Category DE Band Profile\n(deviation from overall mean)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

    # Panel B: per-band activity vs passive DE
    ax = axes[1]
    act_de_by_band = conc_de[act_cids].mean(axis=0)
    pas_de_by_band = conc_de[pas_cids].mean(axis=0)
    act_sem = conc_de[act_cids].std(axis=0) / np.sqrt(len(act_cids))
    pas_sem = conc_de[pas_cids].std(axis=0) / np.sqrt(len(pas_cids))
    x = np.arange(N_BANDS)
    w = 0.35
    ax.bar(x-w/2, act_de_by_band, w, yerr=act_sem, color="#e74c3c", alpha=0.85, capsize=5, label="Activity")
    ax.bar(x+w/2, pas_de_by_band, w, yerr=pas_sem, color="#4472c4", alpha=0.85, capsize=5, label="Passive")
    for b in range(N_BANDS):
        t, p = stats.ttest_ind(conc_de[act_cids, b], conc_de[pas_cids, b])
        if p < 0.05:
            y = max(act_de_by_band[b]+act_sem[b], pas_de_by_band[b]+pas_sem[b])
            ax.text(b, y+0.005, sig(p), ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(BAND_NAMES, fontsize=10)
    ax.set_ylabel("Mean DE", fontsize=11)
    ax.set_title("(B) Activity vs Passive DE per Band\n(error bars = SEM across concepts)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel C: per-concept gamma DE vs R@1
    ax = axes[2]
    gamma_de = conc_de[:, 4]  # gamma band
    theta_de = conc_de[:, 1]  # theta band
    is_act = np.isin(np.arange(N_CONCEPTS), act_cids)
    ax.scatter(theta_de[~is_act], r1s[~is_act]*100, c="#4472c4",
               s=50, alpha=0.8, edgecolors="white", linewidths=0.5, label="Passive")
    ax.scatter(theta_de[is_act], r1s[is_act]*100, c="#e74c3c",
               s=70, marker="^", alpha=0.9, edgecolors="white", linewidths=0.5, label="Activity")
    m, b_fit = np.polyfit(theta_de, r1s*100, 1)
    xl = np.linspace(theta_de.min(), theta_de.max(), 100)
    rho_theta, p_theta = stats.spearmanr(theta_de, r1s)
    ax.plot(xl, m*xl+b_fit, "k-", linewidth=1.5, alpha=0.7,
            label=f"Fit: ρ={rho_theta:.3f} {sig(p_theta)}")
    ax.set_xlabel("Theta Band Mean DE", fontsize=11)
    ax.set_ylabel("Concept R@1 (%)", fontsize=11)
    ax.set_title("(C) Theta DE → R@1\nDoes theta power predict concept decodability?",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)

    plt.suptitle("Frequency Band DE Profiles: Neural Signature of Semantic Categories\n"
                 "Do activity concepts have distinctive frequency-domain EEG patterns?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/F47_frequency_band_profile.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
