"""
CLIP Modality Comparison: Image vs Text vs Combined Gallery for Retrieval.

Uses the same trained EEG encoder but evaluates R@1 against three different
CLIP galleries:
  - Image-only gallery (averaged per-concept image CLIP embeddings)
  - Text-only gallery (averaged per-concept text CLIP embeddings)
  - Both-combined gallery (used during training)

Tests whether image or text CLIP is a better EEG retrieval target, and
whether the advantage differs for activity vs passive concepts.

Hypothesis: Text CLIP may be less suitable for activity concepts (action
verbs are harder to text-embed than concrete objects), suggesting image
CLIP alignment is the primary driver of decodability.

Run from EEG2Video/:
    python neuroclip/clip_modality_comparison.py
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

act_cids = [c for cat in ACTIVITY_CATS for c in SEMANTIC_GROUPS[cat]]
pas_cids = [c for cat in SEMANTIC_GROUPS if cat not in ACTIVITY_CATS for c in SEMANTIC_GROUPS[cat]]


def build_gallery_from(pt_file, device):
    conc = torch.load(pt_file, weights_only=True)
    g = torch.zeros(N_CONCEPTS, 512); c = torch.zeros(N_CONCEPTS)
    for s in range(N_SESSIONS):
        for pos in range(N_CONCEPTS):
            cid = int(GT_LABEL[s, pos]); g[cid] += conc[s, pos]; c[cid] += 1
    return F.normalize(g / c.clamp(min=1).unsqueeze(1), dim=-1).to(device)


def eval_with_gallery(gallery, device):
    """Returns per-subject R@1, and per-subject per-concept R@1."""
    sub_r1s = []
    sub_concept_r1s = []
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
        preds = (embs @ gallery.T).argmax(1).cpu().numpy()
        hit = (preds == cids.astype(int))
        sub_r1s.append(hit.mean())
        # Per-concept R@1 for this subject
        conc_r1 = np.zeros(N_CONCEPTS)
        for cid in range(N_CONCEPTS):
            mask = (cids == cid)
            if mask.sum() > 0:
                conc_r1[cid] = hit[mask].mean()
        sub_concept_r1s.append(conc_r1)
    return np.array(sub_r1s), np.array(sub_concept_r1s)  # (N_sub,), (N_sub, 40)


def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."


def main():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    galleries = {
        "Image": build_gallery_from("neuroclip/clip_concept_image_embs_v2.pt", device),
        "Text":  build_gallery_from("neuroclip/clip_concept_text_embs_v2.pt", device),
        "Both":  build_gallery_from("neuroclip/clip_concept_both_embs_v2.pt", device),
    }

    results_by_modality = {}
    for mod, gallery in galleries.items():
        print(f"\nEvaluating with {mod} gallery...")
        sub_r1s, sub_conc_r1s = eval_with_gallery(gallery, device)
        results_by_modality[mod] = {"sub_r1s": sub_r1s, "sub_conc_r1s": sub_conc_r1s}
        print(f"  Overall R@1: {sub_r1s.mean()*100:.2f}% ± {sub_r1s.std()*100:.2f}%")
        conc_means = sub_conc_r1s.mean(axis=0)
        act_r1 = conc_means[act_cids].mean()
        pas_r1 = conc_means[pas_cids].mean()
        print(f"  Activity: {act_r1*100:.2f}%  Passive: {pas_r1*100:.2f}%")

    # Statistical comparisons: Image vs Text R@1 (paired t-test across subjects)
    img_r1s = results_by_modality["Image"]["sub_r1s"]
    txt_r1s = results_by_modality["Text"]["sub_r1s"]
    bot_r1s = results_by_modality["Both"]["sub_r1s"]
    t_it, p_it = stats.ttest_rel(img_r1s, txt_r1s)
    t_bi, p_bi = stats.ttest_rel(bot_r1s, img_r1s)
    t_bt, p_bt = stats.ttest_rel(bot_r1s, txt_r1s)
    print(f"\nImage vs Text: t={t_it:.2f}  p={p_it:.4f}  {sig(p_it)}")
    print(f"Both vs Image: t={t_bi:.2f}  p={p_bi:.4f}  {sig(p_bi)}")
    print(f"Both vs Text:  t={t_bt:.2f}  p={p_bt:.4f}  {sig(p_bt)}")

    # Per-concept comparison: does image or text advantage vary by concept type?
    img_conc = results_by_modality["Image"]["sub_conc_r1s"].mean(0)
    txt_conc = results_by_modality["Text"]["sub_conc_r1s"].mean(0)
    bot_conc = results_by_modality["Both"]["sub_conc_r1s"].mean(0)
    img_adv = img_conc - txt_conc  # positive = image better per concept

    act_adv = img_adv[act_cids].mean()
    pas_adv = img_adv[pas_cids].mean()
    t_adv, p_adv = stats.ttest_ind(img_adv[act_cids], img_adv[pas_cids])
    print(f"\nImage-over-Text advantage:")
    print(f"  Activity concepts: {act_adv*100:+.2f} pp")
    print(f"  Passive  concepts: {pas_adv*100:+.2f} pp")
    print(f"  Activity vs Passive advantage: t={t_adv:.2f}  p={p_adv:.4f}  {sig(p_adv)}")

    # CLIP inter-modality similarity: how similar are image vs text galleries?
    img_g = galleries["Image"].cpu().numpy()
    txt_g = galleries["Text"].cpu().numpy()
    cross_sim = (img_g * txt_g).sum(axis=1)  # per-concept cosine similarity image vs text
    print(f"\nImage-Text gallery cosine similarity per concept:")
    print(f"  Overall: {cross_sim.mean():.4f} ± {cross_sim.std():.4f}")
    print(f"  Activity: {cross_sim[act_cids].mean():.4f}  Passive: {cross_sim[pas_cids].mean():.4f}")
    t_cs, p_cs = stats.ttest_ind(cross_sim[act_cids], cross_sim[pas_cids])
    print(f"  t={t_cs:.2f}  p={p_cs:.4f}  {sig(p_cs)}")

    results = {
        "image_mean": float(img_r1s.mean()), "image_std": float(img_r1s.std()),
        "text_mean":  float(txt_r1s.mean()), "text_std":  float(txt_r1s.std()),
        "both_mean":  float(bot_r1s.mean()), "both_std":  float(bot_r1s.std()),
        "t_image_vs_text": float(t_it), "p_image_vs_text": float(p_it),
        "t_both_vs_image": float(t_bi), "p_both_vs_image": float(p_bi),
        "t_both_vs_text":  float(t_bt), "p_both_vs_text":  float(p_bt),
        "image_advantage_activity": float(act_adv), "image_advantage_passive": float(pas_adv),
        "t_advantage_act_vs_pas": float(t_adv), "p_advantage_act_vs_pas": float(p_adv),
        "cross_modal_sim_activity": float(cross_sim[act_cids].mean()),
        "cross_modal_sim_passive":  float(cross_sim[pas_cids].mean()),
        "image_conc_r1": img_conc.tolist(),
        "text_conc_r1":  txt_conc.tolist(),
        "both_conc_r1":  bot_conc.tolist(),
        "concept_names": CONCEPT_NAMES,
    }
    with open(f"{RESULTS_DIR}/results_clip_modality_comparison.json","w") as f:
        json.dump(results, f, indent=2)

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Overall R@1 per gallery type
    ax = axes[0]
    mods = ["Image", "Text", "Both"]
    means = [img_r1s.mean()*100, txt_r1s.mean()*100, bot_r1s.mean()*100]
    sems  = [img_r1s.std()*100/np.sqrt(len(img_r1s)),
             txt_r1s.std()*100/np.sqrt(len(txt_r1s)),
             bot_r1s.std()*100/np.sqrt(len(bot_r1s))]
    cols  = ["#f0a500","#2e86ab","#e74c3c"]
    bars = ax.bar([0,1,2], means, yerr=sems, color=cols, width=0.5, alpha=0.85, capsize=8)
    ax.axhline(2.5, color="gray", linestyle="--", linewidth=1.5, label="Chance (2.5%)")
    for bar, m, e in zip(bars, means, sems):
        ax.text(bar.get_x()+bar.get_width()/2, m+e+0.05, f"{m:.2f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks([0,1,2]); ax.set_xticklabels(mods, fontsize=11)
    ax.set_ylabel("Mean R@1 (%)", fontsize=11)
    ax.set_title(f"(A) Gallery Modality Comparison\nImage vs Text gallery", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel B: Per-concept R@1 for Image vs Text, colored by category type
    ax = axes[1]
    order = np.argsort(bot_conc)[::-1]
    x = np.arange(N_CONCEPTS)
    ax.plot(x, img_conc[order]*100, "o-", color="#f0a500", linewidth=1.5, markersize=4, label="Image", alpha=0.8)
    ax.plot(x, txt_conc[order]*100, "s--", color="#2e86ab", linewidth=1.5, markersize=4, label="Text",  alpha=0.8)
    ax.plot(x, bot_conc[order]*100, "^:", color="#e74c3c",  linewidth=1.5, markersize=4, label="Both",  alpha=0.8)
    ax.axhline(2.5, color="gray", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Concept (sorted by Both R@1)", fontsize=10)
    ax.set_ylabel("Per-Concept R@1 (%)", fontsize=10)
    ax.set_title("(B) Per-Concept R@1 by Modality\n(sorted by combined gallery R@1)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # Panel C: Image-over-Text advantage per concept, activity vs passive
    ax = axes[2]
    is_act = np.array([any(cid in SEMANTIC_GROUPS[cat] for cat in ACTIVITY_CATS) for cid in range(N_CONCEPTS)])
    cols_c = ["#e74c3c" if is_act[c] else "#4472c4" for c in np.argsort(img_adv)[::-1]]
    ax.bar(range(N_CONCEPTS), np.sort(img_adv)[::-1]*100, color=cols_c, alpha=0.8, width=0.8)
    ax.axhline(0, color="black", linewidth=1.5)
    ax.axhline(act_adv*100, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Activity mean: {act_adv*100:+.2f}pp")
    ax.axhline(pas_adv*100, color="#4472c4", linestyle="--", linewidth=1.5,
               label=f"Passive mean: {pas_adv*100:+.2f}pp")
    ax.set_xlabel("Concept (sorted by Image advantage)", fontsize=10)
    ax.set_ylabel("Image − Text R@1 (pp)", fontsize=10)
    act_patch = mpatches.Patch(color="#e74c3c", alpha=0.8, label="Activity")
    pas_patch = mpatches.Patch(color="#4472c4", alpha=0.8, label="Passive")
    ax.legend(handles=[act_patch, pas_patch], fontsize=9)
    ax.set_title(f"(C) Image-over-Text Advantage\nActivity={act_adv*100:+.2f}pp  Passive={pas_adv*100:+.2f}pp  {sig(p_adv)}",
                 fontsize=10, fontweight="bold")

    plt.suptitle("CLIP Gallery Modality: Image vs Text vs Combined\n"
                 "Does the neural EEG representation align better with visual or linguistic CLIP?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/F36_clip_modality_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
