"""
Retrieval Confidence Analysis: How Certain Is NeuroCLIP?

For each EEG trial, compute the softmax probability distribution over
40 gallery concepts and measure retrieval confidence:
  - Entropy H = -sum(p * log(p)): lower = more confident
  - Max probability: higher = more confident
  - Margin = top-1 prob - top-2 prob: higher = more confident

Tests:
1. Do correctly retrieved trials have lower entropy? (calibration)
2. Are activity concepts retrieved with higher confidence?
3. Does per-concept mean confidence correlate with R@1?

Temperature scaling is NOT applied — raw logit similarities are softmax'd.

Run from EEG2Video/:
    python neuroclip/retrieval_confidence_analysis.py
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


def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."


def main():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")
    gallery = build_gallery(device)

    # Collect per-trial (entropy, max_prob, margin, correct, concept_id)
    all_entropy, all_maxprob, all_margin, all_correct, all_cids = [], [], [], [], []

    valid_subs = []
    for sub in ALL_SUBS:
        ckpt = f"{RESULTS_DIR}/{sub}_fold0_de_k1_both.pt"
        if not os.path.exists(ckpt): continue
        raw = np.load(f"{DE_DATA_DIR}/{sub}.npy")
        n_s,n_c,n_cl,n_seg,n_ch,n_b = raw.shape
        eeg_all = raw.mean(axis=3).reshape(n_s, n_c*n_cl, n_ch, n_b)
        model = EEGEncoder(n_channels=n_ch, n_time=n_b, embed_dim=512).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()
        flat = eeg_all[TEST_SESS].reshape(N_CONCEPTS*N_CLIPS, -1)
        norm = StandardScaler().fit_transform(flat).reshape(N_CONCEPTS*N_CLIPS, n_ch, n_b)
        eeg_t = torch.tensor(norm, dtype=torch.float32).to(device)
        cids  = np.repeat(GT_LABEL[TEST_SESS], N_CLIPS).astype(int)
        with torch.no_grad():
            embs = model(eeg_t)
            logits = embs @ gallery.T     # (200, 40)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()  # (200, 40)

        preds = probs.argmax(axis=1)
        correct = (preds == cids).astype(float)
        probs_sorted = np.sort(probs, axis=1)[:, ::-1]
        entropy  = -np.sum(probs * np.log(probs.clip(min=1e-12)), axis=1)  # (200,)
        max_prob = probs.max(axis=1)
        margin   = probs_sorted[:, 0] - probs_sorted[:, 1]

        all_entropy.append(entropy)
        all_maxprob.append(max_prob)
        all_margin.append(margin)
        all_correct.append(correct)
        all_cids.append(cids)
        valid_subs.append(sub)

    all_entropy = np.concatenate(all_entropy)  # (N_sub*200,)
    all_maxprob = np.concatenate(all_maxprob)
    all_margin  = np.concatenate(all_margin)
    all_correct = np.concatenate(all_correct)
    all_cids    = np.concatenate(all_cids)
    print(f"\n{len(valid_subs)} subjects, {len(all_entropy)} trials")

    # 1. Calibration: correct vs incorrect trial entropy
    H_correct   = all_entropy[all_correct == 1]
    H_incorrect = all_entropy[all_correct == 0]
    t_calib, p_calib = stats.ttest_ind(H_correct, H_incorrect)
    print(f"\nEntropy — Correct: {H_correct.mean():.4f} ± {H_correct.std():.4f}")
    print(f"Entropy — Incorrect: {H_incorrect.mean():.4f} ± {H_incorrect.std():.4f}")
    print(f"Correct < Incorrect (lower entropy = more confident): t={t_calib:.2f}  p={p_calib:.6f}  {sig(p_calib)}")

    # Max-prob calibration
    mp_correct   = all_maxprob[all_correct == 1]
    mp_incorrect = all_maxprob[all_correct == 0]
    t_mp, p_mp = stats.ttest_ind(mp_correct, mp_incorrect)
    print(f"\nMax-Prob — Correct: {mp_correct.mean():.4f}  Incorrect: {mp_incorrect.mean():.4f}")
    print(f"Correct > Incorrect: t={t_mp:.2f}  p={p_mp:.6f}  {sig(p_mp)}")

    # 2. Per-concept mean entropy and confidence
    conc_entropy = np.zeros(N_CONCEPTS)
    conc_maxprob = np.zeros(N_CONCEPTS)
    conc_r1      = np.zeros(N_CONCEPTS)
    for cid in range(N_CONCEPTS):
        mask = (all_cids == cid)
        if mask.sum() == 0: continue
        conc_entropy[cid] = all_entropy[mask].mean()
        conc_maxprob[cid] = all_maxprob[mask].mean()
        conc_r1[cid]      = all_correct[mask].mean()

    # Correlation: entropy vs R@1
    rho_H, p_H = stats.spearmanr(conc_entropy, conc_r1)
    print(f"\nPer-concept entropy → R@1: Spearman ρ={rho_H:.4f}  p={p_H:.4f}  {sig(p_H)}")

    # 3. Activity vs passive confidence
    act_H = conc_entropy[act_cids]
    pas_H = conc_entropy[pas_cids]
    t_H, p_H2 = stats.ttest_ind(act_H, pas_H)
    print(f"\nActivity entropy: {act_H.mean():.4f} ± {act_H.std():.4f}")
    print(f"Passive  entropy: {pas_H.mean():.4f} ± {pas_H.std():.4f}")
    print(f"Activity < Passive (more confident): t={t_H:.2f}  p={p_H2:.4f}  {sig(p_H2)}")

    # Chance entropy: uniform over 40
    chance_H = np.log(N_CONCEPTS)  # max entropy for 40-way
    print(f"\nChance entropy (uniform): {chance_H:.4f}")
    print(f"Observed mean entropy: {all_entropy.mean():.4f}  "
          f"({(1-all_entropy.mean()/chance_H)*100:.1f}% below chance)")

    results = {
        "mean_entropy": float(all_entropy.mean()), "chance_entropy": float(chance_H),
        "correct_entropy": float(H_correct.mean()), "incorrect_entropy": float(H_incorrect.mean()),
        "t_calibration": float(t_calib), "p_calibration": float(p_calib),
        "activity_entropy": float(act_H.mean()), "passive_entropy": float(pas_H.mean()),
        "t_entropy_act_vs_pas": float(t_H), "p_entropy_act_vs_pas": float(p_H2),
        "rho_entropy_r1": float(rho_H), "p_entropy_r1": float(p_H),
        "conc_entropy": conc_entropy.tolist(),
        "conc_maxprob": conc_maxprob.tolist(),
        "conc_r1": conc_r1.tolist(),
        "concept_names": CONCEPT_NAMES,
    }
    with open(f"{RESULTS_DIR}/results_retrieval_confidence.json","w") as f:
        json.dump(results, f, indent=2)

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Correct vs Incorrect entropy distributions
    ax = axes[0]
    ax.hist(H_incorrect, bins=30, alpha=0.6, color="#4472c4", density=True, label="Incorrect retrieval")
    ax.hist(H_correct,   bins=30, alpha=0.8, color="#e74c3c", density=True, label="Correct retrieval")
    ax.axvline(chance_H, color="black", linestyle="--", linewidth=2, label=f"Chance H={chance_H:.2f}")
    ax.axvline(H_correct.mean(), color="#e74c3c", linestyle="-", linewidth=2, alpha=0.8)
    ax.axvline(H_incorrect.mean(), color="#4472c4", linestyle="-", linewidth=2, alpha=0.8)
    ax.set_xlabel("Retrieval Entropy H (nats)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"(A) Retrieval Calibration\nCorrect H={H_correct.mean():.3f} vs Incorrect H={H_incorrect.mean():.3f} {sig(p_calib)}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel B: per-concept entropy vs R@1
    ax = axes[1]
    is_act = np.isin(np.arange(N_CONCEPTS), act_cids)
    ax.scatter(conc_entropy[~is_act], conc_r1[~is_act]*100, c="#4472c4",
               s=50, alpha=0.8, edgecolors="white", linewidths=0.5, label="Passive")
    ax.scatter(conc_entropy[is_act], conc_r1[is_act]*100, c="#e74c3c",
               s=70, alpha=0.9, marker="^", edgecolors="white", linewidths=0.5, label="Activity")
    m, b = np.polyfit(conc_entropy, conc_r1*100, 1)
    xl = np.linspace(conc_entropy.min(), conc_entropy.max(), 100)
    ax.plot(xl, m*xl+b, "k-", linewidth=2, alpha=0.7)
    for cid in range(N_CONCEPTS):
        if conc_r1[cid] > 0.07 or conc_entropy[cid] < conc_entropy.mean()-0.05:
            ax.annotate(CONCEPT_NAMES[cid], (conc_entropy[cid], conc_r1[cid]*100),
                        fontsize=6, xytext=(2,2), textcoords="offset points")
    ax.set_xlabel("Per-Concept Mean Entropy (nats)", fontsize=11)
    ax.set_ylabel("Per-Concept R@1 (%)", fontsize=11)
    ax.set_title(f"(B) Entropy → R@1\nSpearman ρ={rho_H:.3f} {sig(p_H)}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel C: per-concept entropy bar, sorted, colored by activity
    ax = axes[2]
    order_H = np.argsort(conc_entropy)
    cols_c = ["#e74c3c" if c in act_cids else "#4472c4" for c in order_H]
    ax.bar(range(N_CONCEPTS), conc_entropy[order_H], color=cols_c, alpha=0.8, width=0.8)
    ax.axhline(act_H.mean(), color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Activity mean={act_H.mean():.3f}")
    ax.axhline(pas_H.mean(), color="#4472c4", linestyle="--", linewidth=1.5,
               label=f"Passive mean={pas_H.mean():.3f}")
    ax.axhline(chance_H, color="black", linestyle=":", linewidth=1.5, label=f"Chance={chance_H:.2f}")
    ax.set_xticks(range(N_CONCEPTS))
    ax.set_xticklabels([CONCEPT_NAMES[c] for c in order_H], rotation=90, fontsize=6)
    ax.set_ylabel("Mean Retrieval Entropy (nats)", fontsize=10)
    ax.set_title(f"(C) Per-Concept Entropy\nActivity={act_H.mean():.3f} vs Passive={pas_H.mean():.3f} {sig(p_H2)}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)

    plt.suptitle("EEG Retrieval Confidence: Entropy of Softmax over 40-Way Gallery\n"
                 "Calibrated uncertainty: correct retrievals are more confident",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/F40_retrieval_confidence_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
