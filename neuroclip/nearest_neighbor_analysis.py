"""
Nearest-Neighbor Analysis: Does EEG Pick the Same Nearest Neighbor as CLIP?

For each of 40 concepts, find its:
  - CLIP nearest neighbor (most similar in CLIP gallery space)
  - EEG nearest neighbor (most similar in EEG embedding space)

Questions:
1. How often do EEG and CLIP agree on the nearest neighbor?
2. When they disagree, is the EEG nearest neighbor semantically related?
3. Are activity concepts more likely to have matching nearest neighbors?

Also computes the rank of the EEG nearest neighbor in CLIP space:
if EEG and CLIP disagree on nearest neighbor, how close to the CLIP
nearest neighbor is the EEG choice?

Run from EEG2Video/:
    python neuroclip/nearest_neighbor_analysis.py
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

cat_lookup = {}
for cat, ids in SEMANTIC_GROUPS.items():
    for cid in ids: cat_lookup[cid] = cat


def build_gallery(device):
    conc = torch.load("neuroclip/clip_concept_both_embs_v2.pt", weights_only=True)
    g = torch.zeros(N_CONCEPTS, 512); c = torch.zeros(N_CONCEPTS)
    for s in range(N_SESSIONS):
        for pos in range(N_CONCEPTS):
            cid = int(GT_LABEL[s, pos]); g[cid] += conc[s, pos]; c[cid] += 1
    return F.normalize(g / c.clamp(min=1).unsqueeze(1), dim=-1).numpy()


def get_group_mean_eeg_embs(device):
    """Returns (40, 512) group-mean EEG concept embeddings."""
    sum_embs = np.zeros((N_CONCEPTS, 512))
    count = np.zeros(N_CONCEPTS)
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
        cids = np.repeat(GT_LABEL[TEST_SESS], N_CLIPS)
        with torch.no_grad(): embs = model(eeg_t)
        embs_np = embs.cpu().numpy()
        for i, cid in enumerate(cids.astype(int)):
            sum_embs[cid] += embs_np[i]; count[cid] += 1
    mean_embs = sum_embs / count.clip(min=1).reshape(-1,1)
    norms = np.linalg.norm(mean_embs, axis=1, keepdims=True).clip(min=1e-8)
    return mean_embs / norms


def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."


def main():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    gallery = build_gallery(device)                    # (40, 512) CLIP
    eeg_mean = get_group_mean_eeg_embs(device)         # (40, 512) EEG

    clip_sim = gallery @ gallery.T                     # (40, 40)
    eeg_sim  = eeg_mean @ eeg_mean.T                   # (40, 40)

    # Nearest neighbor analysis (excluding self, so zero diagonal then argmax)
    np.fill_diagonal(clip_sim, -1)
    np.fill_diagonal(eeg_sim, -1)

    clip_nn = clip_sim.argmax(axis=1)   # (40,) CLIP nearest neighbor per concept
    eeg_nn  = eeg_sim.argmax(axis=1)    # (40,) EEG nearest neighbor per concept

    # Agreement rate
    nn_agree = (clip_nn == eeg_nn)
    print(f"Nearest-neighbor agreement: {nn_agree.mean()*100:.1f}%  ({nn_agree.sum()}/{N_CONCEPTS})")

    # Same-category agreement
    same_cat_clip = np.array([cat_lookup[clip_nn[c]] == cat_lookup[c] for c in range(N_CONCEPTS)])
    same_cat_eeg  = np.array([cat_lookup[eeg_nn[c]]  == cat_lookup[c] for c in range(N_CONCEPTS)])
    print(f"\nCLIP NN in same category: {same_cat_clip.mean()*100:.1f}%")
    print(f"EEG  NN in same category: {same_cat_eeg.mean()*100:.1f}%")

    # Chance same-category rate
    cat_sizes = {cat: len(ids) for cat,ids in SEMANTIC_GROUPS.items()}
    chance_same_cat = np.mean([cat_sizes[cat_lookup[c]]/(N_CONCEPTS-1) for c in range(N_CONCEPTS)])
    print(f"Chance same-category rate: {chance_same_cat*100:.1f}%")

    t_clip, p_clip = stats.binomtest(same_cat_clip.sum(), N_CONCEPTS, chance_same_cat).pvalue, 0
    t_clip, p_clip = 0, stats.binomtest(same_cat_clip.sum(), N_CONCEPTS, chance_same_cat).pvalue
    t_eeg,  p_eeg  = 0, stats.binomtest(same_cat_eeg.sum(),  N_CONCEPTS, chance_same_cat).pvalue
    print(f"\nCLIP NN same-cat > chance: p={p_clip:.4f}  {sig(p_clip)}")
    print(f"EEG  NN same-cat > chance: p={p_eeg:.4f}  {sig(p_eeg)}")

    # Print all NN pairs
    print("\nPer-concept nearest neighbors (CLIP vs EEG):")
    for c in range(N_CONCEPTS):
        agree_str = "✓" if nn_agree[c] else "✗"
        print(f"  {agree_str} {CONCEPT_NAMES[c]:15s} | CLIP NN: {CONCEPT_NAMES[clip_nn[c]]:15s} "
              f"[{cat_lookup[clip_nn[c]]}]  |  EEG NN: {CONCEPT_NAMES[eeg_nn[c]]:15s} "
              f"[{cat_lookup[eeg_nn[c]]}]")

    # Rank of EEG-chosen NN in CLIP space (where does the EEG nearest neighbor rank in CLIP?)
    # For each concept c, EEG says NN = eeg_nn[c].
    # What rank is eeg_nn[c] in CLIP similarity to c?
    np.fill_diagonal(clip_sim, -1)  # already done
    eeg_nn_clip_rank = []
    for c in range(N_CONCEPTS):
        eeg_choice = eeg_nn[c]
        # Rank of eeg_choice in CLIP sorted by similarity to c (1 = most similar)
        rank = (clip_sim[c, :] > clip_sim[c, eeg_choice]).sum() + 1
        eeg_nn_clip_rank.append(rank)
    eeg_nn_clip_rank = np.array(eeg_nn_clip_rank)
    print(f"\nEEG nearest neighbor is CLIP rank {eeg_nn_clip_rank.mean():.2f} on average "
          f"(1 = perfect match, {N_CONCEPTS-1} = worst)")

    # Activity vs passive NN agreement
    act_agree = nn_agree[act_cids].mean()
    pas_agree = nn_agree[pas_cids].mean()
    t_agree, p_agree = stats.ttest_ind(nn_agree[act_cids].astype(float),
                                        nn_agree[pas_cids].astype(float))
    print(f"\nActivity NN agreement: {act_agree*100:.1f}%")
    print(f"Passive  NN agreement: {pas_agree*100:.1f}%")
    print(f"t={t_agree:.2f}  p={p_agree:.4f}  {sig(p_agree)}")

    results = {
        "nn_agreement_rate": float(nn_agree.mean()),
        "same_cat_clip": float(same_cat_clip.mean()),
        "same_cat_eeg":  float(same_cat_eeg.mean()),
        "chance_same_cat": float(chance_same_cat),
        "p_clip_same_cat": float(p_clip),
        "p_eeg_same_cat":  float(p_eeg),
        "eeg_nn_clip_rank_mean": float(eeg_nn_clip_rank.mean()),
        "activity_agree": float(act_agree),
        "passive_agree":  float(pas_agree),
        "t_agree": float(t_agree), "p_agree": float(p_agree),
        "clip_nn": [int(c) for c in clip_nn],
        "eeg_nn":  [int(c) for c in eeg_nn],
        "concept_names": CONCEPT_NAMES,
    }
    with open(f"{RESULTS_DIR}/results_nearest_neighbor.json","w") as f:
        json.dump(results, f, indent=2)

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel A: CLIP vs EEG nearest neighbor agreement heatmap
    ax = axes[0]
    # Show agreement matrix: rows=concepts, cols=[CLIP NN, EEG NN] agreement
    # Use a visual summary instead: scatter of CLIP similarity to CLIP-NN vs to EEG-NN
    clip_to_clipnn = np.array([clip_sim[c, clip_nn[c]] for c in range(N_CONCEPTS)])
    clip_to_eegnn  = np.array([clip_sim[c, eeg_nn[c]]  for c in range(N_CONCEPTS)])
    is_act = np.isin(np.arange(N_CONCEPTS), act_cids)
    ax.scatter(clip_to_clipnn[~is_act], clip_to_eegnn[~is_act], c="#4472c4",
               s=50, alpha=0.8, edgecolors="white", linewidths=0.5, label="Passive")
    ax.scatter(clip_to_clipnn[is_act], clip_to_eegnn[is_act], c="#e74c3c",
               s=70, marker="^", alpha=0.9, edgecolors="white", linewidths=0.5, label="Activity")
    lim = [min(clip_to_clipnn.min(), clip_to_eegnn.min())-0.01,
           max(clip_to_clipnn.max(), clip_to_eegnn.max())+0.01]
    ax.plot(lim, lim, "k--", linewidth=1.5, alpha=0.5, label="Equal (perfect agreement)")
    ax.set_xlabel("CLIP similarity to CLIP nearest neighbor", fontsize=10)
    ax.set_ylabel("CLIP similarity to EEG nearest neighbor", fontsize=10)
    ax.set_title(f"(A) NN Agreement\n{nn_agree.sum()}/{N_CONCEPTS} concepts: EEG NN = CLIP NN",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    for c in range(N_CONCEPTS):
        if nn_agree[c]:
            ax.annotate(CONCEPT_NAMES[c], (clip_to_clipnn[c], clip_to_eegnn[c]),
                        fontsize=5, xytext=(2,2), textcoords="offset points", color="green")

    # Panel B: EEG NN rank in CLIP space
    ax = axes[1]
    cols_b = ["#e74c3c" if c in act_cids else "#4472c4" for c in np.argsort(eeg_nn_clip_rank)]
    ax.bar(range(N_CONCEPTS), np.sort(eeg_nn_clip_rank), color=cols_b, alpha=0.8, width=0.8)
    ax.axhline(1, color="green", linestyle="--", linewidth=2, label="Perfect (rank 1)")
    ax.axhline(eeg_nn_clip_rank.mean(), color="gray", linestyle=":", linewidth=1.5,
               label=f"Mean rank = {eeg_nn_clip_rank.mean():.2f}")
    ax.set_xlabel("Concept (sorted by EEG NN CLIP rank)", fontsize=10)
    ax.set_ylabel("CLIP rank of EEG nearest neighbor", fontsize=10)
    ax.set_title(f"(B) CLIP Rank of EEG Nearest Neighbor\nMean rank = {eeg_nn_clip_rank.mean():.2f}/{N_CONCEPTS-1}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel C: Same-category NN rates
    ax = axes[2]
    cats_with_nn = [c for c in SEMANTIC_GROUPS if len(SEMANTIC_GROUPS[c]) >= 2]
    cat_clip_samecat = [np.mean([cat_lookup[clip_nn[c]] == cat for c in SEMANTIC_GROUPS[cat]])
                         for cat in cats_with_nn]
    cat_eeg_samecat  = [np.mean([cat_lookup[eeg_nn[c]]  == cat for c in SEMANTIC_GROUPS[cat]])
                         for cat in cats_with_nn]
    x = np.arange(len(cats_with_nn))
    width = 0.35
    cols_clip = ["#f0a500" for _ in cats_with_nn]
    cols_eeg  = ["#2e86ab" for _ in cats_with_nn]
    ax.bar(x-width/2, [v*100 for v in cat_clip_samecat], width, color="#f0a500", alpha=0.85, label="CLIP NN")
    ax.bar(x+width/2, [v*100 for v in cat_eeg_samecat],  width, color="#2e86ab", alpha=0.85, label="EEG NN")
    ax.axhline(chance_same_cat*100, color="gray", linestyle="--", linewidth=1.5, label=f"Chance ({chance_same_cat*100:.1f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(cats_with_nn, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("% Nearest Neighbor in Same Category", fontsize=10)
    ax.set_title(f"(C) Same-Category NN Rate\nCLIP={same_cat_clip.mean()*100:.0f}% vs EEG={same_cat_eeg.mean()*100:.0f}%",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)

    plt.suptitle("Nearest-Neighbor Alignment: Does EEG Match CLIP's Concept Proximity?\n"
                 f"Agreement rate: {nn_agree.sum()}/{N_CONCEPTS} concepts share the same nearest neighbor",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/F41_nearest_neighbor_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
