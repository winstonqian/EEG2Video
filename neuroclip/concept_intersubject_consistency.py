"""
Per-concept inter-subject EEG consistency vs R@1.

For each of 40 concepts, compute how consistently subjects encode that concept
in EEG space: mean pairwise cosine similarity of concept embeddings across all
21×20/2=210 subject pairs.

Tests whether activity concepts (Sports, Music, People) have more universally
consistent EEG representations — explaining their higher decodability.

Run from EEG2Video/:
    python neuroclip/concept_intersubject_consistency.py
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
PASSIVE_CATS  = ["Animals", "Nature", "Food", "Vehicles", "Urban", "Other"]


def get_concept_embs(sub, device):
    raw = np.load(f"{DE_DATA_DIR}/{sub}.npy")
    n_s,n_c,n_cl,n_seg,n_ch,n_b = raw.shape
    eeg_all = raw.mean(axis=3).reshape(n_s, n_c*n_cl, n_ch, n_b)
    ckpt = f"{RESULTS_DIR}/{sub}_fold0_de_k1_both.pt"
    if not os.path.exists(ckpt): return None
    model = EEGEncoder(n_channels=n_ch, n_time=n_b, embed_dim=512).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    flat = eeg_all[TEST_SESS].reshape(N_CONCEPTS*N_CLIPS, -1)
    norm = StandardScaler().fit_transform(flat).reshape(N_CONCEPTS*N_CLIPS, n_ch, n_b)
    eeg_t = torch.tensor(norm, dtype=torch.float32).to(device)
    cids = np.repeat(GT_LABEL[TEST_SESS], N_CLIPS)
    with torch.no_grad(): embs = model(eeg_t)
    ce = torch.zeros(N_CONCEPTS, 512, device=device)
    cnt = torch.zeros(N_CONCEPTS, device=device)
    for i, cid in enumerate(cids):
        ce[int(cid)] += embs[i]; cnt[int(cid)] += 1
    return F.normalize(ce / cnt.clamp(min=1).unsqueeze(1), dim=-1).cpu().numpy()


def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."


def main():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    # Build per-subject concept embeddings
    all_embs = []
    valid_subs = []
    for sub in ALL_SUBS:
        embs = get_concept_embs(sub, device)
        if embs is not None:
            all_embs.append(embs)
            valid_subs.append(sub)
        print(f"  {sub}: done")

    N = len(valid_subs)
    all_embs = np.stack(all_embs)  # (N, 40, 512)
    print(f"\n{N} valid subjects")

    # Per-concept inter-subject consistency:
    # For each concept, compute mean pairwise cosine similarity across subject pairs
    concept_consistency = np.zeros(N_CONCEPTS)
    for c in range(N_CONCEPTS):
        concept_embs = all_embs[:, c, :]  # (N, 512) — each subject's embedding for concept c
        # Pairwise cosine similarity (already L2-normalized)
        sim_matrix = concept_embs @ concept_embs.T
        np.fill_diagonal(sim_matrix, 0)
        n_pairs = N * (N-1)
        concept_consistency[c] = sim_matrix.sum() / n_pairs

    # Load per-concept R@1
    deco = json.load(open(f"{RESULTS_DIR}/results_concept_decodability.json"))
    r1s = np.array(deco["per_concept_r1"])

    # Correlate consistency with R@1
    rho_s, p_s = stats.spearmanr(concept_consistency, r1s)
    r_p, p_p   = stats.pearsonr(concept_consistency, r1s)
    print(f"\nConsistency → R@1: Spearman ρ={rho_s:.4f} p={p_s:.4f} {sig(p_s)}")
    print(f"                   Pearson  r={r_p:.4f} p={p_p:.4f} {sig(p_p)}")

    # Activity vs Passive consistency
    act_cids = [c for cat in ACTIVITY_CATS for c in SEMANTIC_GROUPS[cat]]
    pas_cids = [c for cat in PASSIVE_CATS  for c in SEMANTIC_GROUPS[cat]]
    t, p = stats.ttest_ind(concept_consistency[act_cids], concept_consistency[pas_cids])
    print(f"\nActivity consistency: {concept_consistency[act_cids].mean():.4f}")
    print(f"Passive  consistency: {concept_consistency[pas_cids].mean():.4f}")
    print(f"t={t:.2f}  p={p:.4f}  {sig(p)}")

    # Print per-concept table
    print("\nPer-concept inter-subject consistency:")
    pairs = sorted(zip(concept_consistency, CONCEPT_NAMES, r1s), reverse=True)
    for cons, name, r1 in pairs:
        cat = next(c for c, ids in SEMANTIC_GROUPS.items() if CONCEPT_NAMES.index(name) in ids)
        act = "ACTION" if cat in ACTIVITY_CATS else "static"
        print(f"  {name:15s}: cons={cons:.4f}  R@1={r1*100:.1f}%  [{cat}] {act}")

    results = {
        "concept_consistency": concept_consistency.tolist(),
        "concept_names": CONCEPT_NAMES,
        "spearman_rho": float(rho_s), "spearman_p": float(p_s),
        "pearson_r": float(r_p), "pearson_p": float(p_p),
        "activity_consistency": float(concept_consistency[act_cids].mean()),
        "passive_consistency":  float(concept_consistency[pas_cids].mean()),
        "t_act_vs_pas": float(t), "p_act_vs_pas": float(p),
    }
    with open(f"{RESULTS_DIR}/results_concept_intersubject_consistency.json","w") as f:
        json.dump(results, f, indent=2)

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: scatter consistency vs R@1
    ax = axes[0]
    colors = ["#e74c3c" if CONCEPT_NAMES[i] in [CONCEPT_NAMES[c] for cat in ACTIVITY_CATS for c in SEMANTIC_GROUPS[cat]] else "#4472c4"
              for i in range(N_CONCEPTS)]
    ax.scatter(concept_consistency, r1s*100, c=colors, alpha=0.8, s=60, edgecolors="white", linewidths=0.5)
    m, b = np.polyfit(concept_consistency, r1s*100, 1)
    xl = np.linspace(concept_consistency.min(), concept_consistency.max(), 100)
    ax.plot(xl, m*xl+b, "k-", linewidth=2)
    ax.set_xlabel("Inter-subject Consistency", fontsize=11)
    ax.set_ylabel("Concept R@1 (%)", fontsize=11)
    ax.set_title(f"(A) Consistency → R@1\nSpearman ρ={rho_s:.3f} {sig(p_s)}", fontsize=11, fontweight="bold")
    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color="#e74c3c",alpha=0.8,label="Activity"),
                       mpatches.Patch(color="#4472c4",alpha=0.8,label="Passive")], fontsize=9)
    # Label top concepts
    for cons, name, r1 in sorted(zip(concept_consistency, CONCEPT_NAMES, r1s), reverse=True)[:5]:
        ax.annotate(name, (cons, r1*100), fontsize=7, xytext=(3,3), textcoords="offset points")

    # Panel B: per-category consistency
    ax = axes[1]
    cat_order = sorted(SEMANTIC_GROUPS.keys(),
                       key=lambda c: -np.mean(concept_consistency[SEMANTIC_GROUPS[c]]))
    cat_cols = ["#e74c3c" if c in ACTIVITY_CATS else "#4472c4" for c in cat_order]
    cat_vals = [np.mean(concept_consistency[SEMANTIC_GROUPS[c]]) for c in cat_order]
    cat_errs = [np.std(concept_consistency[SEMANTIC_GROUPS[c]])/np.sqrt(len(SEMANTIC_GROUPS[c]))
                for c in cat_order]
    bars = ax.bar(range(len(cat_order)), cat_vals, yerr=cat_errs, color=cat_cols,
                  alpha=0.85, capsize=5, width=0.65)
    ax.set_xticks(range(len(cat_order)))
    ax.set_xticklabels(cat_order, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean Inter-subject Consistency", fontsize=11)
    ax.set_title(f"(B) Per-Category Consistency\nActivity: {concept_consistency[act_cids].mean():.3f}  Passive: {concept_consistency[pas_cids].mean():.3f}",
                 fontsize=11, fontweight="bold")

    # Panel C: activity vs passive violin
    ax = axes[2]
    data = [concept_consistency[act_cids], concept_consistency[pas_cids]]
    vp = ax.violinplot(data, positions=[0,1], showmedians=True, showextrema=False)
    for pc, col in zip(vp["bodies"], ["#e74c3c","#4472c4"]):
        pc.set_facecolor(col); pc.set_alpha(0.6)
    ax.scatter(np.zeros(len(act_cids))+np.random.normal(0,0.04,len(act_cids)), concept_consistency[act_cids],
               c="#e74c3c",s=60,zorder=3,alpha=0.8,edgecolors="white",linewidths=0.5)
    ax.scatter(np.ones(len(pas_cids))+np.random.normal(0,0.04,len(pas_cids)), concept_consistency[pas_cids],
               c="#4472c4",s=60,zorder=3,alpha=0.8,edgecolors="white",linewidths=0.5)
    y_top = max(concept_consistency.max(), 0)+0.01
    ax.plot([0,0,1,1],[y_top,y_top+0.005,y_top+0.005,y_top], lw=1.5, color="black")
    ax.text(0.5, y_top+0.006, f"t={t:.1f} {sig(p)}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks([0,1])
    ax.set_xticklabels(["Activity\n(Sports+Music+People)","Passive\n(Others)"], fontsize=10)
    ax.set_ylabel("Inter-subject Consistency", fontsize=11)
    ax.set_title(f"(C) Activity vs Passive\nUniversality of neural representation",
                 fontsize=11, fontweight="bold")

    plt.suptitle("Action Concepts Have More Universal EEG Representations\n"
                 "Inter-subject consistency predicts concept decodability",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/F32_concept_intersubject_consistency.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
