"""
Fusion analysis — understanding why text+image supervision outperforms either alone.

Figures generated:
  Fig F1: Per-subject R@1 — Text vs Image vs Both (grouped bar)
  Fig F2: Per-subject scatter — Both vs Text, Both vs Image
  Fig F3: Late fusion vs Early fusion comparison
  Fig F4: Text-image CLIP agreement per concept vs fusion benefit
  Fig F5: Per-concept R@1 — Text vs Image vs Both (sorted by fusion gain)
  Fig F6: Fusion weight sweep (inference-time α from 0→1)
  Fig F7: Where does fusion help? Subjects ranked by (Both - max(Text, Image))

Run from EEG2Video/:
    python neuroclip/fusion_analysis.py
"""

import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_subject, NeuroCLIPDataset, GT_LABEL
from models_neuroclip import EEGEncoder

RESULTS_DIR  = "neuroclip/results"
FIGURES_DIR  = "neuroclip/figures"
DE_DATA_DIR  = "data/DE_1per1s"
TEXT_EMB     = "neuroclip/clip_text_embs_v2.pt"
IMAGE_EMB    = "neuroclip/clip_image_embs_v2.pt"
BOTH_EMB     = "neuroclip/clip_both_embs_v2.pt"
CONCEPT_TEXT = "neuroclip/clip_concept_text_embs_v2.pt"
CONCEPT_IMG  = "neuroclip/clip_concept_image_embs_v2.pt"
CONCEPT_BOTH = "neuroclip/clip_concept_both_embs_v2.pt"
CAPTION_FILE = "data/Video/BLIP-caption/1st_10min.txt"
os.makedirs(FIGURES_DIR, exist_ok=True)

CHANCE   = 1/40
CLS_MEAN = 0.0437

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")

def load_concept_names():
    with open(CAPTION_FILE) as f:
        lines = [l.strip() for l in f.readlines()]
    return {int(GT_LABEL[0, p]): " ".join(lines[p*5].split()[:3]) for p in range(40)}

def make_gallery(concept_embs_tensor, sess):
    """(40,512) gallery indexed by concept_id for session sess."""
    g = torch.zeros(40, 512)
    c = torch.zeros(40)
    for pos in range(40):
        cid = int(GT_LABEL[sess, pos])
        g[cid] += concept_embs_tensor[sess, pos]
        c[cid] += 1
    return F.normalize(g / c.clamp(min=1).unsqueeze(1), dim=-1)

def recall_at_1(embs, gallery, true_cids):
    sim   = embs @ gallery.T
    top1  = sim.argmax(dim=1)
    return (top1.cpu() == torch.tensor(true_cids)).float().mean().item()

def run_inference_one_sub(sub_name, tag, eeg_data, text_embs, concept_embs,
                          concept_ids, device):
    """Run all 7 folds for one subject/tag. Returns per-fold dicts."""
    n_ch, n_time = eeg_data.shape[2], eeg_data.shape[3]
    folds_out = []
    for fold in range(7):
        model_path = os.path.join(RESULTS_DIR,
            f"{sub_name.replace('.npy','')}_fold{fold}_de_k1_{tag}.pt")
        if not os.path.exists(model_path):
            return None
        model = EEGEncoder(n_channels=n_ch, n_time=n_time, embed_dim=512)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.to(device).eval()

        ds = NeuroCLIPDataset(eeg_data, text_embs, concept_embs, concept_ids, [fold])
        loader = torch.utils.data.DataLoader(ds, batch_size=200, shuffle=False)

        embs, cids = [], []
        with torch.no_grad():
            for eeg_b, _, _, cid_b, _ in loader:
                embs.append(model(eeg_b.to(device)).cpu())
                cids.extend(cid_b.numpy())
        folds_out.append({"embs": torch.cat(embs), "cids": np.array(cids), "fold": fold})
    return folds_out


# ---------------------------------------------------------------------------
# SECTION A: Load per-subject JSON results
# ---------------------------------------------------------------------------

def load_results():
    tags = {"text": "results_de_k1.json",
            "image": "results_de_k1_image.json",
            "both":  "results_de_k1_both.json"}
    out = {}
    for tag, fname in tags.items():
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            d = json.load(f)
        out[tag] = np.array(d["per_subject"]["concept_r1"]) * 100
    return out

# ---------------------------------------------------------------------------
# Fig F1: Per-subject grouped bar
# ---------------------------------------------------------------------------

def fig_f1_per_subject_bar(results):
    n = len(results["text"])
    x = np.arange(n)
    w = 0.25
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.bar(x - w, results["text"],  w, label="Text",  color="#4472c4", alpha=0.85)
    ax.bar(x,     results["image"], w, label="Image", color="#ed7d31", alpha=0.85)
    ax.bar(x + w, results["both"],  w, label="Both",  color="#70ad47", alpha=0.85)
    ax.axhline(CHANCE*100,   color="red",    linestyle="--", linewidth=1.2, label="Chance (2.5%)")
    ax.axhline(CLS_MEAN*100, color="purple", linestyle="--", linewidth=1.2, label="Supervised (4.37%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i+1}" for i in range(n)], fontsize=8)
    ax.set_ylabel("Concept R@1 (%)")
    ax.set_title("Per-Subject NeuroCLIP Concept R@1 by Supervision Modality\n"
                 "(Text, Image, Both — DE features, 150ep, 7-fold CV)", fontsize=11)
    ax.legend(fontsize=9, ncol=5)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fusF1_per_subject_bar.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved → {path}")

# ---------------------------------------------------------------------------
# Fig F2: Scatter Both vs Text and Both vs Image
# ---------------------------------------------------------------------------

def fig_f2_scatter(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (tag, color, label) in zip(axes, [
        ("text",  "#4472c4", "NeuroCLIP-Text R@1 (%)"),
        ("image", "#ed7d31", "NeuroCLIP-Image R@1 (%)"),
    ]):
        x, y = results[tag], results["both"]
        ax.scatter(x, y, color=color, s=60, alpha=0.85, zorder=3)
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.annotate(f"S{i+1}", (xi, yi), fontsize=7,
                        textcoords="offset points", xytext=(4, 2))
        lo = min(x.min(), y.min()) - 0.3
        hi = max(x.max(), y.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="Equal performance")
        above = (y > x).sum()
        ax.set_xlabel(label)
        ax.set_ylabel("NeuroCLIP-Both R@1 (%)")
        ax.set_title(f"Both vs {tag.capitalize()} per Subject\n"
                     f"(Both better in {above}/{len(x)} subjects)", fontsize=11)
        ax.legend(fontsize=9)
    plt.suptitle("Multimodal Fusion Benefit per Subject", fontsize=12)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fusF2_scatter_both_vs.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved → {path}")

# ---------------------------------------------------------------------------
# Fig F3: Late fusion vs Early fusion
# Late fusion = average EEG embeddings from text model + image model at inference
# Early fusion = NeuroCLIP-Both (trained on averaged embeddings)
# ---------------------------------------------------------------------------

def fig_f3_late_vs_early_fusion(device):
    """
    Late fusion: average EEG embeddings from original Text model + Image model at inference.
    Early fusion: NeuroCLIP-Both (trained jointly on text+image supervision).
    Text model = sub1_fold*_de_k1.pt (original, unprojected text, 4.23% overall).
    Image model = sub1_fold*_de_k1_image.pt (projected image, 4.26% overall).
    """
    print("Fig F3: Late vs Early fusion (running inference on sub1)...")

    # Original text model uses original (unprojected) embeddings
    TEXT_EMB_ORIG    = "neuroclip/clip_text_embeddings.pt"
    CONCEPT_EMB_ORIG = "neuroclip/clip_concept_embeddings.pt"

    eeg_data, t_embs_orig, c_embs_orig, concept_ids = load_subject(
        os.path.join(DE_DATA_DIR, "sub1.npy"),
        TEXT_EMB_ORIG, CONCEPT_EMB_ORIG, feature="de"
    )
    _, img_embs_v2, concept_img_embs_v2, _ = load_subject(
        os.path.join(DE_DATA_DIR, "sub1.npy"),
        IMAGE_EMB, CONCEPT_IMG, feature="de"
    )
    _, both_embs_v2, concept_both_embs_v2, _ = load_subject(
        os.path.join(DE_DATA_DIR, "sub1.npy"),
        BOTH_EMB, CONCEPT_BOTH, feature="de"
    )
    n_ch, n_time = eeg_data.shape[2], eeg_data.shape[3]

    late_r1s, early_r1s, text_r1s, image_r1s = [], [], [], []

    for fold in range(7):
        # Original text model (no suffix), image model, both model
        text_path  = os.path.join(RESULTS_DIR, f"sub1_fold{fold}_de_k1.pt")
        image_path = os.path.join(RESULTS_DIR, f"sub1_fold{fold}_de_k1_image.pt")
        both_path  = os.path.join(RESULTS_DIR, f"sub1_fold{fold}_de_k1_both.pt")
        if not all(os.path.exists(p) for p in [text_path, image_path, both_path]):
            continue

        def get_embs(model_path, eeg_d, t_embs, c_embs, c_ids):
            model = EEGEncoder(n_channels=n_ch, n_time=n_time, embed_dim=512)
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            model.to(device).eval()
            ds = NeuroCLIPDataset(eeg_d, t_embs, c_embs, c_ids, [fold])
            loader = torch.utils.data.DataLoader(ds, batch_size=200, shuffle=False)
            embs, cids = [], []
            with torch.no_grad():
                for eeg_b, _, _, cid_b, _ in loader:
                    embs.append(model(eeg_b.to(device)).cpu())
                    cids.extend(cid_b.numpy())
            return F.normalize(torch.cat(embs), dim=-1), np.array(cids)

        text_embs_out,  cids = get_embs(text_path,  eeg_data, t_embs_orig,  c_embs_orig,          concept_ids)
        image_embs_out, _    = get_embs(image_path, eeg_data, img_embs_v2,  concept_img_embs_v2,  concept_ids)
        both_embs_out,  _    = get_embs(both_path,  eeg_data, both_embs_v2, concept_both_embs_v2, concept_ids)

        # Late fusion: average EEG embeddings from text + image models, then renormalize
        late_embs = F.normalize(text_embs_out + image_embs_out, dim=-1)

        # Evaluate all against both gallery (common reference)
        gallery = make_gallery(concept_both_embs_v2, fold).to(device)
        late_r1s.append(recall_at_1(late_embs.to(device),      gallery, cids))
        early_r1s.append(recall_at_1(both_embs_out.to(device), gallery, cids))

        # Individual models on their own galleries
        t_gal = make_gallery(c_embs_orig,         fold).to(device)
        i_gal = make_gallery(concept_img_embs_v2, fold).to(device)
        text_r1s.append(recall_at_1(text_embs_out.to(device),   t_gal, cids))
        image_r1s.append(recall_at_1(image_embs_out.to(device), i_gal, cids))

    conditions = ["Text only", "Image only", "Late Fusion\n(avg embeddings)", "Early Fusion\n(NeuroCLIP-Both)"]
    means = [np.mean(text_r1s)*100, np.mean(image_r1s)*100,
             np.mean(late_r1s)*100, np.mean(early_r1s)*100]
    colors = ["#4472c4", "#ed7d31", "#9dc3e6", "#70ad47"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(4), means, color=colors, width=0.55, alpha=0.9)
    ax.axhline(CHANCE*100,   color="red",    linestyle="--", linewidth=1.5, label="Chance (2.5%)")
    ax.axhline(CLS_MEAN*100, color="purple", linestyle="--", linewidth=1.5, label="Supervised (4.37%)")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_ylabel("Concept R@1 (%)")
    ax.set_title("Late Fusion vs Early Fusion (sub1, all 7 folds)\n"
                 "Late = average EEG embeddings at inference; Early = train on mixed target", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fusF3_late_vs_early.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved → {path}  late={np.mean(late_r1s)*100:.2f}%  early={np.mean(early_r1s)*100:.2f}%")

# ---------------------------------------------------------------------------
# Fig F4: CLIP text-image agreement vs fusion benefit per concept
# ---------------------------------------------------------------------------

def fig_f4_agreement_vs_benefit(device):
    print("Fig F4: Text-image agreement vs fusion benefit...")
    concept_names = load_concept_names()

    text_embs_all = torch.load(CONCEPT_TEXT, weights_only=True)   # (7,40,512) by concept_pos
    img_embs_all  = torch.load(CONCEPT_IMG,  weights_only=True)   # (7,40,512) by concept_pos

    # Text-image cosine similarity per concept_id, averaged across sessions
    ti_sim = torch.zeros(40)
    for s in range(7):
        for pos in range(40):
            cid = int(GT_LABEL[s, pos])
            ti_sim[cid] += (text_embs_all[s, pos] * img_embs_all[s, pos]).sum()
    ti_sim /= 7
    ti_sim = ti_sim.numpy()

    # Per-concept R@1 for text and both (from sub1 inference)
    eeg_data, t_embs, c_embs, c_ids = load_subject(
        os.path.join(DE_DATA_DIR, "sub1.npy"), TEXT_EMB, CONCEPT_TEXT, feature="de")
    _, bt_embs, bc_embs, _ = load_subject(
        os.path.join(DE_DATA_DIR, "sub1.npy"), BOTH_EMB, CONCEPT_BOTH, feature="de")
    n_ch, n_time = eeg_data.shape[2], eeg_data.shape[3]

    concept_r1_text = np.zeros(40)
    concept_r1_both = np.zeros(40)
    concept_counts  = np.zeros(40)

    for fold in range(7):
        for tag, mpath, t_e, c_e, r1_arr in [
            ("text", f"sub1_fold{fold}_de_k1_text.pt", t_embs, c_embs, concept_r1_text),
            ("both", f"sub1_fold{fold}_de_k1_both.pt", bt_embs, bc_embs, concept_r1_both),
        ]:
            mpath_full = os.path.join(RESULTS_DIR, mpath)
            if not os.path.exists(mpath_full): continue
            model = EEGEncoder(n_channels=n_ch, n_time=n_time, embed_dim=512)
            model.load_state_dict(torch.load(mpath_full, map_location="cpu", weights_only=True))
            model.to(device).eval()
            ds = NeuroCLIPDataset(eeg_data, t_e, c_e, c_ids, [fold])
            loader = torch.utils.data.DataLoader(ds, batch_size=200, shuffle=False)
            embs, true_cids, clip_idxs = [], [], []
            with torch.no_grad():
                for eeg_b, _, _, cid_b, cidx_b in loader:
                    embs.append(model(eeg_b.to(device)).cpu())
                    true_cids.extend(cid_b.numpy())
                    clip_idxs.extend(cidx_b.numpy())
            embs = torch.cat(embs)
            gallery = make_gallery(c_e, fold).to(device)
            sim = embs.to(device) @ gallery.T
            preds = sim.argmax(dim=1).cpu().numpy()
            for i, cid in enumerate(true_cids):
                r1_arr[cid] += int(preds[i] == cid)
                if tag == "text":
                    concept_counts[cid] += 1

    concept_counts = concept_counts.clip(min=1)
    text_r1 = concept_r1_text / concept_counts * 100
    both_r1 = concept_r1_both / concept_counts * 100
    gain    = both_r1 - text_r1

    corr = np.corrcoef(ti_sim, gain)[0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: scatter agreement vs gain
    ax = axes[0]
    sc = ax.scatter(ti_sim, gain, c=gain, cmap="RdYlGn", s=60, alpha=0.85, vmin=-5, vmax=5)
    plt.colorbar(sc, ax=ax, label="Fusion gain (Both - Text) %")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.2)
    for i in range(40):
        if abs(gain[i]) > 3:
            ax.annotate(concept_names.get(i, str(i)), (ti_sim[i], gain[i]),
                        fontsize=6, textcoords="offset points", xytext=(3, 2))
    # Trend line
    m, b = np.polyfit(ti_sim, gain, 1)
    xs = np.linspace(ti_sim.min(), ti_sim.max(), 100)
    ax.plot(xs, m*xs+b, color="red", linewidth=2, linestyle="--",
            label=f"Linear fit (r={corr:.2f})")
    ax.set_xlabel("CLIP Text-Image Cosine Similarity (per concept)")
    ax.set_ylabel("Fusion Gain: Both − Text R@1 (%)")
    ax.set_title("Concepts with High Text-Image Agreement:\nDoes Fusion Help More?", fontsize=11)
    ax.legend(fontsize=9)

    # Right: bar of top-10 fusion gainers and losers
    ax2 = axes[1]
    sorted_idx = np.argsort(gain)
    top5_lose  = sorted_idx[:5]
    top5_gain  = sorted_idx[-5:][::-1]
    show_idx   = list(top5_gain) + list(top5_lose)
    show_vals  = [gain[i] for i in show_idx]
    show_names = [concept_names.get(i, str(i)) for i in show_idx]
    colors     = ["#70ad47" if v >= 0 else "#ff6b6b" for v in show_vals]
    ax2.barh(range(10), show_vals, color=colors, alpha=0.85)
    ax2.set_yticks(range(10))
    ax2.set_yticklabels(show_names, fontsize=8)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Fusion Gain (%)")
    ax2.set_title("Top 5 Concepts Gained & Lost\nfrom Multimodal Fusion", fontsize=11)

    plt.suptitle(f"Text-Image CLIP Agreement vs NeuroCLIP Fusion Benefit (sub1, r={corr:.2f})", fontsize=12)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fusF4_agreement_vs_benefit.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved → {path}  (r={corr:.2f})")

# ---------------------------------------------------------------------------
# Fig F5: Fusion weight sweep at inference time
# Use Both EEG model, vary the gallery: α*text + (1-α)*image
# ---------------------------------------------------------------------------

def fig_f5_weight_sweep(device):
    print("Fig F5: Fusion weight sweep...")
    eeg_data, t_embs, c_embs_t, c_ids = load_subject(
        os.path.join(DE_DATA_DIR, "sub1.npy"), TEXT_EMB, CONCEPT_TEXT, feature="de")
    _, i_embs, c_embs_i, _ = load_subject(
        os.path.join(DE_DATA_DIR, "sub1.npy"), IMAGE_EMB, CONCEPT_IMG, feature="de")
    n_ch, n_time = eeg_data.shape[2], eeg_data.shape[3]

    alphas = np.linspace(0, 1, 11)   # 0=image only, 1=text only
    alpha_r1 = []

    for alpha in alphas:
        fold_r1s = []
        for fold in range(7):
            both_path = os.path.join(RESULTS_DIR, f"sub1_fold{fold}_de_k1_both.pt")
            if not os.path.exists(both_path): continue
            model = EEGEncoder(n_channels=n_ch, n_time=n_time, embed_dim=512)
            model.load_state_dict(torch.load(both_path, map_location="cpu", weights_only=True))
            model.to(device).eval()

            # Blend concept galleries
            t_gal = make_gallery(c_embs_t, fold)
            i_gal = make_gallery(c_embs_i, fold)
            mixed_gal = F.normalize(alpha * t_gal + (1-alpha) * i_gal, dim=-1).to(device)

            ds = NeuroCLIPDataset(eeg_data, t_embs, c_embs_t, c_ids, [fold])
            loader = torch.utils.data.DataLoader(ds, batch_size=200, shuffle=False)
            embs, cids = [], []
            with torch.no_grad():
                for eeg_b, _, _, cid_b, _ in loader:
                    embs.append(model(eeg_b.to(device)).cpu())
                    cids.extend(cid_b.numpy())
            embs = F.normalize(torch.cat(embs), dim=-1)
            fold_r1s.append(recall_at_1(embs.to(device), mixed_gal, np.array(cids)))
        alpha_r1.append(np.mean(fold_r1s) * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, alpha_r1, marker="o", linewidth=2.5, markersize=7, color="steelblue")
    ax.axvline(0.5, color="green",  linestyle="--", linewidth=1.5, label="α=0.5 (trained setting)")
    ax.axhline(CHANCE*100,   color="red",    linestyle=":",  linewidth=1.5, label="Chance (2.5%)")
    ax.axhline(CLS_MEAN*100, color="purple", linestyle=":",  linewidth=1.5, label="Supervised (4.37%)")
    ax.set_xlabel("Gallery mixing weight α  (0 = image only, 1 = text only)")
    ax.set_ylabel("Concept R@1 (%)")
    ax.set_title("Fusion Weight Sensitivity — NeuroCLIP-Both Encoder (sub1)\n"
                 "Gallery blended as α·text + (1-α)·image at inference", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fusF5_weight_sweep.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved → {path}  best_alpha={alphas[np.argmax(alpha_r1)]:.1f}  best_r1={max(alpha_r1):.2f}%")

# ---------------------------------------------------------------------------
# Fig F6: Subject-level fusion gain ranking
# ---------------------------------------------------------------------------

def fig_f6_gain_ranking(results):
    gain = results["both"] - np.maximum(results["text"], results["image"])
    sorted_idx = np.argsort(gain)[::-1]

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#70ad47" if g >= 0 else "#ff6b6b" for g in gain[sorted_idx]]
    ax.bar(range(len(gain)), gain[sorted_idx], color=colors, alpha=0.85, width=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(gain)))
    ax.set_xticklabels([f"S{sorted_idx[i]+1}" for i in range(len(gain))], fontsize=9)
    ax.set_ylabel("Fusion Gain: Both − max(Text, Image) (%)")
    ax.set_title("Per-Subject Fusion Gain from Multimodal Supervision\n"
                 "(sorted descending; green = fusion helps, red = fusion hurts)", fontsize=11)

    above = (gain >= 0).sum()
    ax.text(0.98, 0.95, f"Fusion helps: {above}/{len(gain)} subjects",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(facecolor="white", alpha=0.8))
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fusF6_gain_ranking.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved → {path}  (helps {above}/{len(gain)} subjects)")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = get_device()
    print(f"Device: {device}")
    results = load_results()

    print(f"\nMeans — Text: {results['text'].mean():.2f}%  "
          f"Image: {results['image'].mean():.2f}%  "
          f"Both: {results['both'].mean():.2f}%")

    fig_f1_per_subject_bar(results)
    fig_f2_scatter(results)
    fig_f4_agreement_vs_benefit(device)
    fig_f5_weight_sweep(device)
    fig_f6_gain_ranking(results)

    print(f"\nAll fusion figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
