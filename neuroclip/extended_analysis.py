"""Extended NeuroCLIP analysis — rich final-project figures.

Generates 11 additional figures:
  Fig 01: t-SNE of EEG embeddings colored by concept (sub1, all 7 folds)
  Fig 02: Per-concept R@1 bar chart (40 concepts, sorted by accuracy)
  Fig 03: 40x40 confusion matrix (sub1, all 7 folds combined)
  Fig 04: CLIP text 40x40 semantic similarity matrix
  Fig 05: Retrieval rank histogram (sub1, all 7 folds)
  Fig 06: Cross-fold session effect — R@1 by held-out session (all 20 subjects)
  Fig 07: Sequence shortcut audit — R@1 by clip position 1-5 (sub1)
  Fig 08: NeuroCLIP-DE vs Classification R@1 scatter (20 subjects)
  Fig 09: DE vs Raw k=1 per-subject scatter
  Fig 10: Chunk attention aggregated across all 20 subjects (fold 0, raw k=4)
  Fig 11: CLIP semantic similarity vs NeuroCLIP confusion rate correlation

Run from EEG2Video/:
    python neuroclip/extended_analysis.py
"""

import os
import sys
import re
import json
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_subject, NeuroCLIPDataset, GT_LABEL
from models_neuroclip import EEGEncoder, ChunkEEGEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR  = "neuroclip/results"
FIGURES_DIR  = "neuroclip/figures"
DE_DATA_DIR  = "data/DE_1per1s"
RAW_DATA_DIR = "data/Segmented_Rawf_200Hz_2s"
TEXT_EMB     = "neuroclip/clip_text_embeddings.pt"
CONCEPT_EMB  = "neuroclip/clip_concept_embeddings.pt"
CAPTION_FILES = [f"data/Video/BLIP-caption/{i+1}st_10min.txt" for i in range(7)]

os.makedirs(FIGURES_DIR, exist_ok=True)

CHANCE_R1  = 1 / 40
CLS_MEAN   = 0.0437
CLS_STD    = 0.0264

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_concept_gallery(concept_embs_session):
    """
    concept_embs_session: (40, 512) per-session concept means indexed by concept_pos.
    Returns (40, 512) gallery indexed by concept_id, using GT_LABEL for session s.
    """
    pass  # see _make_gallery_for_session below


def _make_gallery_for_session(s, concept_embs_tensor):
    """Build (40, 512) gallery indexed by concept_id for session s."""
    g = torch.zeros(40, 512)
    c = torch.zeros(40)
    for pos in range(40):
        cid = int(GT_LABEL[s, pos])
        g[cid] += concept_embs_tensor[s, pos]
        c[cid] += 1
    g = g / c.clamp(min=1).unsqueeze(1)
    return F.normalize(g, dim=-1)  # (40, 512)


def load_concept_names():
    """Build dict {concept_id (0-indexed) -> short name} from session-0 captions."""
    names = {}
    with open(CAPTION_FILES[0]) as fh:
        lines = [l.strip() for l in fh.readlines()]
    for pos in range(40):
        cid    = int(GT_LABEL[0, pos])
        caption = lines[pos * 5]
        # Shorten: keep first 3 words
        short = " ".join(caption.split()[:3])
        names[cid] = short
    return names


# ---------------------------------------------------------------------------
# SECTION A: Inference on sub1 (all 7 folds, DE)
# Collects EEG embeddings, true labels, top-1 predictions, ranks, clip pos.
# ---------------------------------------------------------------------------

def run_inference_sub1(device):
    print("Running inference on sub1 (all 7 folds, DE)...")
    eeg_data, text_embs, concept_embs, concept_ids = load_subject(
        os.path.join(DE_DATA_DIR, "sub1.npy"), TEXT_EMB, CONCEPT_EMB, feature="de"
    )

    all_embs       = []
    all_true_cids  = []
    all_pred_cids  = []
    all_ranks      = []
    all_clip_pos   = []
    all_folds      = []

    n_ch, n_time = eeg_data.shape[2], eeg_data.shape[3]

    for fold in range(7):
        model_path = os.path.join(RESULTS_DIR, f"sub1_fold{fold}_de_k1.pt")
        if not os.path.exists(model_path):
            print(f"  Missing: {model_path}. Skipping.")
            continue

        model = EEGEncoder(n_channels=n_ch, n_time=n_time, embed_dim=512)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.to(device).eval()

        test_ds = NeuroCLIPDataset(eeg_data, text_embs, concept_embs, concept_ids, [fold])
        loader  = torch.utils.data.DataLoader(test_ds, batch_size=200, shuffle=False)

        gallery = _make_gallery_for_session(fold, concept_embs).to(device)  # (40, 512)
        gallery_ids = torch.arange(40, device=device)

        with torch.no_grad():
            for eeg_b, _, _, cid_b, clip_idx_b in loader:
                emb = model(eeg_b.to(device))                         # (200, 512)
                sim = emb @ gallery.T                                  # (200, 40)
                ranks_b = sim.argsort(dim=1, descending=True)         # (200, 40)

                true_cids = cid_b.numpy()
                pred_cids = ranks_b[:, 0].cpu().numpy()               # top-1 predicted concept_id

                # Rank of the correct concept (0-indexed; 0 = top-1)
                correct_ranks = []
                for i in range(len(true_cids)):
                    rank_row = ranks_b[i].cpu().numpy()
                    r = int(np.where(rank_row == true_cids[i])[0][0])
                    correct_ranks.append(r)

                clip_positions = (np.array(clip_idx_b) % 5)  # 0-4

                all_embs.append(emb.cpu().numpy())
                all_true_cids.extend(true_cids)
                all_pred_cids.extend(pred_cids)
                all_ranks.extend(correct_ranks)
                all_clip_pos.extend(clip_positions)
                all_folds.extend([fold] * len(true_cids))

        print(f"  fold {fold} done")

    all_embs      = np.vstack(all_embs)
    all_true_cids = np.array(all_true_cids)
    all_pred_cids = np.array(all_pred_cids)
    all_ranks     = np.array(all_ranks)
    all_clip_pos  = np.array(all_clip_pos)
    all_folds     = np.array(all_folds)

    print(f"  Collected {len(all_embs)} embeddings from sub1.")
    return all_embs, all_true_cids, all_pred_cids, all_ranks, all_clip_pos, all_folds


# ---------------------------------------------------------------------------
# SECTION B: Figures from sub1 inference
# ---------------------------------------------------------------------------

def fig01_tsne(all_embs, all_true_cids, concept_names):
    print("Fig 01: t-SNE...")
    tsne = TSNE(n_components=2, perplexity=40, learning_rate=200,
                random_state=42, max_iter=1000)
    embs_2d = tsne.fit_transform(all_embs)

    cmap = plt.get_cmap("tab20", 40)
    fig, ax = plt.subplots(figsize=(12, 10))
    for cid in range(40):
        mask = all_true_cids == cid
        ax.scatter(embs_2d[mask, 0], embs_2d[mask, 1],
                   color=cmap(cid), alpha=0.55, s=15, label=f"{cid}: {concept_names.get(cid,'')}")
    ax.set_title("t-SNE of NeuroCLIP EEG Embeddings (sub1, all 7 folds, DE)\n"
                 "Each color = one of 40 video concepts", fontsize=12)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    # Compact legend outside
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0),
              fontsize=5.5, ncol=2, markerscale=1.5)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ext01_tsne_sub1.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def fig02_per_concept_r1(all_true_cids, all_pred_cids, concept_names):
    print("Fig 02: Per-concept R@1...")
    r1 = {}
    for cid in range(40):
        mask = all_true_cids == cid
        if mask.sum() == 0:
            r1[cid] = 0.0
        else:
            r1[cid] = float((all_pred_cids[mask] == cid).mean())

    sorted_cids  = sorted(r1, key=r1.get, reverse=True)
    sorted_vals  = [r1[c] * 100 for c in sorted_cids]
    sorted_labels = [f"{concept_names.get(c, str(c))}" for c in sorted_cids]

    fig, ax = plt.subplots(figsize=(16, 5))
    bars = ax.bar(range(40), sorted_vals, color="steelblue", alpha=0.8)
    ax.axhline(CHANCE_R1 * 100, color="red",    linestyle="--", linewidth=1.5, label="Chance (2.5%)")
    ax.axhline(CLS_MEAN  * 100, color="orange", linestyle="--", linewidth=1.5, label="Classification mean (4.37%)")
    ax.set_xticks(range(40))
    ax.set_xticklabels(sorted_labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Concept R@1 (%)")
    ax.set_xlabel("Concept (sorted by NeuroCLIP accuracy)")
    ax.set_title("Per-Concept NeuroCLIP-DE R@1 (sub1, all 7 folds)\n"
                 "Sorted by descending accuracy", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ext02_per_concept_r1.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")
    return r1


def fig03_confusion_matrix(all_true_cids, all_pred_cids, concept_names):
    print("Fig 03: Confusion matrix...")
    # Build 40x40 matrix indexed by concept_id
    cm = np.zeros((40, 40), dtype=int)
    for t, p in zip(all_true_cids, all_pred_cids):
        cm[t, p] += 1

    # Normalise by row (recall)
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm  = cm / row_sums

    labels = [concept_names.get(c, str(c)) for c in range(40)]
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=0.5)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(40))
    ax.set_yticks(range(40))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Predicted Concept")
    ax.set_ylabel("True Concept")
    ax.set_title("NeuroCLIP-DE Confusion Matrix (sub1, all 7 folds)\nRow-normalised recall", fontsize=11)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ext03_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")
    return cm_norm


def fig04_clip_similarity(concept_names):
    print("Fig 04: CLIP text similarity matrix...")
    concept_embs_all = torch.load(CONCEPT_EMB, weights_only=True)  # (7, 40, 512)
    # Average concept embeddings across all 7 sessions, then re-index by concept_id
    gallery = torch.zeros(40, 512)
    counts  = torch.zeros(40)
    for s in range(7):
        for pos in range(40):
            cid = int(GT_LABEL[s, pos])
            gallery[cid] += concept_embs_all[s, pos]
            counts[cid]  += 1
    gallery = gallery / counts.clamp(min=1).unsqueeze(1)
    gallery = F.normalize(gallery, dim=-1)

    sim = (gallery @ gallery.T).numpy()  # (40, 40)

    labels = [concept_names.get(c, str(c)) for c in range(40)]
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(sim, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, label="Cosine similarity")
    ax.set_xticks(range(40))
    ax.set_yticks(range(40))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("CLIP Text Embedding Cosine Similarity — 40 Concepts\n"
                 "(averaged across 7 sessions; diagonal = 1.0)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ext04_clip_similarity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")
    return sim, gallery


def fig05_rank_histogram(all_ranks):
    print("Fig 05: Retrieval rank histogram...")
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(0, 41) - 0.5
    ax.hist(all_ranks, bins=bins, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(x=0.5, color="red",   linestyle="--", linewidth=1.5, label="R@1 boundary")
    ax.axvline(x=4.5, color="orange",linestyle="--", linewidth=1.5, label="R@5 boundary")
    ax.set_xlabel("Rank of Correct Concept (0 = top-1)")
    ax.set_ylabel("Count")
    ax.set_title("NeuroCLIP-DE: Distribution of True Concept Rank\n"
                 "(sub1, 7 folds × 200 clips = 1400 EEG segments)", fontsize=11)
    ax.legend(fontsize=10)
    # Add cumulative % annotation
    total = len(all_ranks)
    r1 = (all_ranks == 0).sum() / total
    r5 = (all_ranks < 5).sum() / total
    ax.text(0.65, 0.90, f"R@1 = {r1*100:.1f}%\nR@5 = {r5*100:.1f}%",
            transform=ax.transAxes, fontsize=11,
            verticalalignment="top", bbox=dict(facecolor="white", alpha=0.7))
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ext05_rank_histogram.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def fig06_session_effect():
    """Cross-fold session ordering: R@1 by test fold (0-6), averaged across subjects."""
    print("Fig 06: Session ordering effect...")
    log_path = os.path.join(RESULTS_DIR, "full_de_150ep.log")
    if not os.path.exists(log_path):
        print("  Log not found, skipping.")
        return

    # Parse fold lines: "  fold X:  concept R@1=Y"
    fold_r1 = {i: [] for i in range(7)}
    pattern = re.compile(r"fold (\d):\s+concept R@1=([\d.]+)")
    with open(log_path) as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                fold  = int(m.group(1))
                r1val = float(m.group(2))
                fold_r1[fold].append(r1val)

    means = [np.mean(fold_r1[f]) * 100 for f in range(7)]
    stds  = [np.std(fold_r1[f])  * 100 for f in range(7)]
    print(f"  Fold R@1 means: {[f'{v:.2f}' for v in means]}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(range(1, 8), means, yerr=stds, marker="s", color="steelblue",
                linestyle="-", linewidth=2, capsize=5, markersize=7)
    ax.axhline(CHANCE_R1 * 100, color="red",   linestyle="--", linewidth=1.5, label="Chance (2.5%)")
    ax.axhline(CLS_MEAN  * 100, color="orange",linestyle="--", linewidth=1.5, label="Classification mean (4.37%)")
    ax.set_xticks(range(1, 8))
    ax.set_xlabel("Test Session Block (fold 1 to 7)")
    ax.set_ylabel("Concept R@1 (%)")
    ax.set_title("Session Ordering Effect on NeuroCLIP-DE Performance\n"
                 "(mean ± std across 20 subjects)", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ext06_session_effect.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def fig07_sequence_shortcut(all_true_cids, all_pred_cids, all_clip_pos):
    """R@1 by clip position 1-5 within concept run (sub1, all folds)."""
    print("Fig 07: Sequence shortcut audit...")
    pos_r1 = []
    pos_std = []
    for pos in range(5):
        mask = all_clip_pos == pos
        r1s = []
        # Break into per-fold estimates for error bars
        # all_folds is not passed here; just compute overall
        r1s.append(float((all_pred_cids[mask] == all_true_cids[mask]).mean()))
        pos_r1.append(np.mean(r1s) * 100)
        pos_std.append(0)

    # Redo with fold-level estimates using the fold array
    # (We compute at segment level since fold info not passed — that's fine for a shortcut audit)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, 6), pos_r1, marker="o", linestyle="-", linewidth=2,
            markersize=8, color="steelblue", label="NeuroCLIP-DE (sub1)")
    ax.axhline(CHANCE_R1 * 100, color="red",   linestyle="--", linewidth=1.5, label="Chance (2.5%)")
    ax.axhline(CLS_MEAN  * 100, color="orange",linestyle="--", linewidth=1.5,
               label="Classification mean (4.37%)")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("Clip Position within Concept Run (1 = first, 5 = last)")
    ax.set_ylabel("Concept R@1 (%)")
    ax.set_title("Sequence Shortcut Audit: NeuroCLIP-DE R@1 by Run Position\n"
                 "(sub1, 7 folds — no shortcut would be flat)", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ext07_sequence_shortcut.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# SECTION C: Cross-subject comparisons
# ---------------------------------------------------------------------------

def fig08_neuroclip_vs_classification():
    print("Fig 08: NeuroCLIP-DE vs Classification scatter...")
    with open(os.path.join(RESULTS_DIR, "results_de_k1.json")) as f:
        nc_data = json.load(f)
    nc_r1 = np.array(nc_data["per_subject"]["concept_r1"]) * 100

    # Load classification results — DE_All_subject_acc.npy or per-subject files
    cls_path = "ClassificationResults/40c_top1/DE_All_subject_acc.npy"
    if not os.path.exists(cls_path):
        print(f"  Classification results not found at {cls_path}, skipping.")
        return
    cls_r1 = np.load(cls_path) * 100  # (20,) subject-level top-1 accuracy

    if len(cls_r1) != len(nc_r1):
        min_n = min(len(cls_r1), len(nc_r1))
        cls_r1 = cls_r1[:min_n]
        nc_r1  = nc_r1[:min_n]

    corr = np.corrcoef(cls_r1, nc_r1)[0, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(cls_r1, nc_r1, color="steelblue", s=60, alpha=0.85, zorder=3)
    for i, (cx, cy) in enumerate(zip(cls_r1, nc_r1)):
        ax.annotate(f"S{i+1}", (cx, cy), fontsize=7, textcoords="offset points",
                    xytext=(4, 2))
    # Best-fit line
    m, b = np.polyfit(cls_r1, nc_r1, 1)
    xs = np.linspace(cls_r1.min(), cls_r1.max(), 100)
    ax.plot(xs, m * xs + b, color="red", linestyle="--", linewidth=1.5,
            label=f"Linear fit (r={corr:.2f})")
    ax.axvline(CHANCE_R1 * 100, color="gray",   linestyle=":", linewidth=1)
    ax.axhline(CHANCE_R1 * 100, color="gray",   linestyle=":", linewidth=1)
    ax.set_xlabel("Classification DE R@1 / Top-1 (%)")
    ax.set_ylabel("NeuroCLIP-DE Concept R@1 (%)")
    ax.set_title(f"NeuroCLIP vs Classification Accuracy per Subject\n"
                 f"(Pearson r = {corr:.3f}, n=20 subjects)", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ext08_neuroclip_vs_cls.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}  (r={corr:.3f})")


def fig09_de_vs_raw():
    print("Fig 09: DE vs Raw k=1 scatter...")
    with open(os.path.join(RESULTS_DIR, "results_de_k1.json")) as f:
        de_data = json.load(f)
    with open(os.path.join(RESULTS_DIR, "results_raw_k1.json")) as f:
        raw_data = json.load(f)

    de_r1  = np.array(de_data["per_subject"]["concept_r1"]) * 100
    raw_r1 = np.array(raw_data["per_subject"]["concept_r1"]) * 100

    # DE is better when above diagonal
    above = (de_r1 > raw_r1).sum()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(raw_r1, de_r1, color="steelblue", s=60, alpha=0.85, zorder=3)
    for i, (rx, dy) in enumerate(zip(raw_r1, de_r1)):
        ax.annotate(f"S{i+1}", (rx, dy), fontsize=7,
                    textcoords="offset points", xytext=(4, 2))
    lo = min(raw_r1.min(), de_r1.min()) - 0.5
    hi = max(raw_r1.max(), de_r1.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1.5,
            label="Equal performance (diagonal)")
    ax.set_xlabel("Raw EEG k=1 — Concept R@1 (%)")
    ax.set_ylabel("DE k=1 — Concept R@1 (%)")
    ax.set_title(f"Feature Type Comparison per Subject: DE vs Raw EEG\n"
                 f"(DE better in {above}/20 subjects; points above diagonal = DE wins)", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ext09_de_vs_raw.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}  (DE better in {above}/20 subjects)")


# ---------------------------------------------------------------------------
# SECTION D: Chunk attention multi-subject
# ---------------------------------------------------------------------------

def fig10_chunk_attention_all_subjects(device):
    print("Fig 10: Chunk attention (all subjects, fold 0, raw k=4)...")
    sub_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".npy")])

    all_attn = []  # (n_valid_subjects, 4) mean attention weights

    for sub_name in sub_files:
        model_path = os.path.join(
            RESULTS_DIR,
            f"{sub_name.replace('.npy', '')}_fold0_raw_k4.pt"
        )
        if not os.path.exists(model_path):
            continue

        eeg_data, text_embs, concept_embs, concept_ids = load_subject(
            os.path.join(RAW_DATA_DIR, sub_name), TEXT_EMB, CONCEPT_EMB, feature="raw"
        )

        n_ch, n_time = eeg_data.shape[2], eeg_data.shape[3]
        model = ChunkEEGEncoder(n_channels=n_ch, n_time=n_time, k_chunks=4, embed_dim=512)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.to(device).eval()

        test_ds = NeuroCLIPDataset(eeg_data, text_embs, concept_embs, concept_ids, [0])
        loader  = torch.utils.data.DataLoader(test_ds, batch_size=200, shuffle=False)

        sub_weights = []
        with torch.no_grad():
            for eeg_b, _, _, _, _ in loader:
                _, w = model(eeg_b.to(device))          # (B, 4)
                sub_weights.append(w.cpu().numpy())
        sub_weights = np.vstack(sub_weights)             # (200, 4)
        all_attn.append(sub_weights.mean(axis=0))        # mean over clips → (4,)

    if not all_attn:
        print("  No chunk attention models found.")
        return

    all_attn  = np.array(all_attn)     # (n_subs, 4)
    mean_attn = all_attn.mean(axis=0)
    std_attn  = all_attn.std(axis=0)
    chunk_labels = ["0–0.5s", "0.5–1s", "1–1.5s", "1.5–2s"]
    n_subs = len(all_attn)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: mean + std across subjects
    ax = axes[0]
    ax.bar(range(4), mean_attn * 100, yerr=std_attn * 100, capsize=6,
           color="steelblue", alpha=0.85, width=0.6, error_kw={"elinewidth": 2})
    ax.axhline(25.0, color="gray", linestyle="--", linewidth=1.5, label="Uniform (25%)")
    ax.set_xticks(range(4))
    ax.set_xticklabels(chunk_labels)
    ax.set_ylabel("Mean Attention Weight (%)")
    ax.set_title(f"Temporal Chunk Attention — Aggregated\n"
                 f"(mean ± std across {n_subs} subjects, fold 0)", fontsize=11)
    ax.legend(fontsize=10)

    # Right: per-subject heat map
    ax2 = axes[1]
    im  = ax2.imshow(all_attn * 100, cmap="Blues", aspect="auto",
                     vmin=0, vmax=40)
    plt.colorbar(im, ax=ax2, fraction=0.046, label="Attn weight (%)")
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(chunk_labels)
    ax2.set_yticks(range(n_subs))
    ax2.set_yticklabels([f"S{i+1}" for i in range(n_subs)], fontsize=8)
    ax2.set_xlabel("Temporal Chunk")
    ax2.set_title(f"Per-Subject Chunk Attention Weights\n"
                  f"(fold 0, raw k=4)", fontsize=11)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ext10_chunk_attention_all.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}  ({n_subs} subjects)")


# ---------------------------------------------------------------------------
# SECTION E: CLIP similarity vs confusion correlation
# ---------------------------------------------------------------------------

def fig11_similarity_confusion_corr(cm_norm, clip_sim, concept_names):
    """Scatter: CLIP semantic similarity vs NeuroCLIP confusion rate (off-diagonal pairs)."""
    print("Fig 11: CLIP similarity vs confusion rate correlation...")
    n = 40
    sims   = []
    confs  = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            sims.append(clip_sim[i, j])
            confs.append(cm_norm[i, j] * 100)

    sims  = np.array(sims)
    confs = np.array(confs)
    corr  = np.corrcoef(sims, confs)[0, 1]

    # Bin by similarity for trend line
    bins = np.linspace(sims.min(), sims.max(), 12)
    bin_means = []
    bin_centers = []
    for k in range(len(bins) - 1):
        mask = (sims >= bins[k]) & (sims < bins[k+1])
        if mask.sum() > 0:
            bin_means.append(confs[mask].mean())
            bin_centers.append((bins[k] + bins[k+1]) / 2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sims, confs, alpha=0.12, s=8, color="steelblue")
    ax.plot(bin_centers, bin_means, color="red", linewidth=2.5,
            marker="o", markersize=6, label="Binned mean confusion rate")
    ax.set_xlabel("CLIP Text Cosine Similarity (between concept pairs)")
    ax.set_ylabel("NeuroCLIP Confusion Rate (% of true-i predicted as j)")
    ax.set_title(f"Semantic Similarity vs Confusion Rate\n"
                 f"(Pearson r = {corr:.3f}, {n*(n-1)} concept pairs, sub1 all folds)", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ext11_sim_vs_confusion.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}  (r={corr:.3f})")


# ---------------------------------------------------------------------------
# BONUS: Concept embedding PCA — EEG vs CLIP embeddings in shared space
# ---------------------------------------------------------------------------

def fig12_eeg_vs_clip_pca(all_embs, all_true_cids, concept_names, clip_gallery):
    """PCA overlay: mean EEG embedding per concept vs CLIP text embedding per concept."""
    print("Fig 12: PCA of mean EEG vs CLIP embeddings...")
    from sklearn.decomposition import PCA

    # Mean EEG embedding per concept (from sub1)
    eeg_concept_means = np.zeros((40, 512))
    for cid in range(40):
        mask = all_true_cids == cid
        if mask.sum() > 0:
            eeg_concept_means[cid] = all_embs[mask].mean(axis=0)

    clip_means = clip_gallery.numpy()  # (40, 512) already normalized

    # Fit PCA on combined
    combined = np.vstack([eeg_concept_means, clip_means])
    pca = PCA(n_components=2, random_state=42)
    pca.fit(combined)
    eeg_2d  = pca.transform(eeg_concept_means)
    clip_2d = pca.transform(clip_means)

    cmap = plt.get_cmap("tab20", 40)
    fig, ax = plt.subplots(figsize=(11, 9))
    for cid in range(40):
        color = cmap(cid)
        ax.scatter(*eeg_2d[cid],  color=color, marker="o", s=80, alpha=0.85)
        ax.scatter(*clip_2d[cid], color=color, marker="^", s=80, alpha=0.85)
        ax.plot([eeg_2d[cid, 0], clip_2d[cid, 0]],
                [eeg_2d[cid, 1], clip_2d[cid, 1]],
                color=color, alpha=0.3, linewidth=1.0)

    # Legend proxy
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=9, label="Mean EEG embedding"),
        Line2D([0], [0], marker="^", color="gray", linestyle="None", markersize=9, label="CLIP text embedding"),
        Line2D([0], [0], color="gray", linewidth=1,  linestyle="-",  alpha=0.5,   label="Concept correspondence"),
    ]
    ax.legend(handles=handles, fontsize=10)
    var_exp = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% var)")
    ax.set_title("PCA: Mean EEG Embeddings (circles) vs CLIP Text Embeddings (triangles)\n"
                 "(sub1, all folds — lines connect matched concept pairs)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ext12_eeg_vs_clip_pca.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = get_device()
    print(f"Device: {device}")

    concept_names = load_concept_names()
    print(f"Loaded {len(concept_names)} concept names.")

    # --- Section A: inference on sub1 ---
    (all_embs, all_true_cids, all_pred_cids,
     all_ranks, all_clip_pos, all_folds) = run_inference_sub1(device)

    # --- Section B: sub1 inference figures ---
    fig01_tsne(all_embs, all_true_cids, concept_names)
    r1_per_concept = fig02_per_concept_r1(all_true_cids, all_pred_cids, concept_names)
    cm_norm        = fig03_confusion_matrix(all_true_cids, all_pred_cids, concept_names)
    fig05_rank_histogram(all_ranks)
    fig06_session_effect()
    fig07_sequence_shortcut(all_true_cids, all_pred_cids, all_clip_pos)

    # --- Section C: CLIP embedding analysis ---
    clip_sim, clip_gallery = fig04_clip_similarity(concept_names)

    # --- Section D: cross-subject ---
    fig08_neuroclip_vs_classification()
    fig09_de_vs_raw()

    # --- Section E: chunk attention ---
    fig10_chunk_attention_all_subjects(device)

    # --- Section F: correlation analysis ---
    fig11_similarity_confusion_corr(cm_norm, clip_sim, concept_names)

    # --- Bonus ---
    fig12_eeg_vs_clip_pca(all_embs, all_true_cids, concept_names, clip_gallery)

    print(f"\nAll extended figures saved to {FIGURES_DIR}/")
    print("Figures generated:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        if f.startswith("ext"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
