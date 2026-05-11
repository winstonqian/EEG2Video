"""
Train LATA on all 20 SEED-DV subjects and aggregate results.
=============================================================
Video features are shared across subjects (same video for all).
Per-subject delay distributions are saved and aggregated into a
single results figure: lata_all_subjects_results.png

Run
---
    python train_all_subjects.py
"""

import os, sys, io, zipfile, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from lata import LATA, lata_infonce_loss
from train_lata_seeddv import SEEDDVEEGDataset, EEGChunkEncoder, VideoProjector

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader

# ── Config ──────────────────────────────────────────────────────────────────
ZIP_PATH       = os.path.join(os.path.dirname(__file__), "..", "eeg2video_colab.zip")
VID_FEAT_PATH  = os.path.join(os.path.dirname(__file__), "video_features_clip_visual.npy")
OUT_DIR        = os.path.dirname(__file__)

SUBJECTS       = list(range(1, 21))   # sub1 … sub20
N_SESSIONS     = 7
N_TRAIN        = 6    # sessions used for training
EPOCHS         = 100  # converges well by 100
BATCH_SIZE     = 64
D_MODEL        = 128
N_HEADS        = 4
MAX_DELAY      = 3    # δ ∈ {0,1,2,3}
LR             = 3e-4
TEMPERATURE    = 0.07
K              = 4


def train_subject(subject_id: int, eeg_data: np.ndarray,
                  vid_all: np.ndarray, device: str) -> dict:
    """Train LATA on one subject, return results dict."""
    eeg_train = eeg_data[:N_TRAIN]
    eeg_val   = eeg_data[N_TRAIN:N_TRAIN + 1]
    video_train = vid_all[:N_TRAIN]
    video_val   = vid_all[N_TRAIN:N_TRAIN + 1]

    train_ds = SEEDDVEEGDataset(eeg_train, K=K, video_feats=video_train)
    val_ds   = SEEDDVEEGDataset(eeg_val,   K=K, video_feats=video_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    C, T_chunk  = 62, 100
    d_video     = 512
    eeg_encoder   = EEGChunkEncoder(C, T_chunk, D_MODEL).to(device)
    vid_projector = VideoProjector(d_video, D_MODEL).to(device)
    lata          = LATA(d_model=D_MODEL, n_heads=N_HEADS, max_delay=MAX_DELAY).to(device)

    params    = (list(eeg_encoder.parameters()) +
                 list(vid_projector.parameters()) +
                 list(lata.parameters()))
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    best_val   = float("inf")
    train_hist = []
    val_hist   = []

    for epoch in range(1, EPOCHS + 1):
        eeg_encoder.train(); vid_projector.train(); lata.train()
        tloss = 0.0
        for eeg_b, vid_b in train_loader:
            eeg_b, vid_b = eeg_b.to(device), vid_b.to(device)
            loss = lata_infonce_loss(eeg_encoder(eeg_b),
                                     lata,
                                     vid_projector(vid_b),
                                     TEMPERATURE)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            tloss += loss.item()
        scheduler.step()
        tloss /= len(train_loader)

        eeg_encoder.eval(); vid_projector.eval(); lata.eval()
        vloss = 0.0
        with torch.no_grad():
            for eeg_b, vid_b in val_loader:
                eeg_b, vid_b = eeg_b.to(device), vid_b.to(device)
                vloss += lata_infonce_loss(eeg_encoder(eeg_b),
                                           lata,
                                           vid_projector(vid_b),
                                           TEMPERATURE).item()
        vloss /= len(val_loader)
        train_hist.append(tloss)
        val_hist.append(vloss)

        if vloss < best_val:
            best_val = vloss

    return {
        "subject":      subject_id,
        "delay_dist":   lata.learned_delay.numpy().copy(),   # (MAX_DELAY+1,)
        "expected_delay": lata.expected_delay,
        "peak_delay":   int(lata.learned_delay.numpy().argmax()),
        "train_loss_final": train_hist[-1],
        "train_loss_init":  train_hist[0],
        "best_val_loss":    best_val,
        "train_hist":   train_hist,
        "val_hist":     val_hist,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load shared video features
    print(f"Loading video features: {VID_FEAT_PATH}")
    vid_all = np.load(VID_FEAT_PATH)   # (7, 40, 5, 4, 512)

    all_results = []

    with zipfile.ZipFile(ZIP_PATH) as z:
        for sub_id in SUBJECTS:
            key = f"data/Segmented_Rawf_200Hz_2s/sub{sub_id}.npy"
            t0  = time.time()
            print(f"\n── Subject {sub_id:2d}/{len(SUBJECTS)} ──────────────────────────────")
            with z.open(key) as f:
                eeg_data = np.load(io.BytesIO(f.read()))

            res = train_subject(sub_id, eeg_data, vid_all, device)
            elapsed = time.time() - t0

            dist_str = "  ".join(f"{v:.3f}" for v in res["delay_dist"])
            print(f"  done in {elapsed:.0f}s | train {res['train_loss_init']:.2f}→{res['train_loss_final']:.2f} "
                  f"| E[δ]={res['expected_delay']:.2f} | peak={res['peak_delay']} | w=[{dist_str}]")
            all_results.append(res)

    # ── Aggregate ──────────────────────────────────────────────────────────
    delay_dists = np.stack([r["delay_dist"] for r in all_results])   # (20, 4)
    mean_dist   = delay_dists.mean(axis=0)
    std_dist    = delay_dists.std(axis=0)
    expected    = np.array([r["expected_delay"] for r in all_results])
    peaks       = np.array([r["peak_delay"]     for r in all_results])
    peak_counts = np.bincount(peaks, minlength=MAX_DELAY + 1)

    print("\n" + "=" * 65)
    print(f"{'Sub':>4}  {'E[δ]':>6}  {'Peak':>5}  {'w[0]':>6}  {'w[1]':>6}  {'w[2]':>6}  {'w[3]':>6}")
    print("-" * 65)
    for r in all_results:
        w = r["delay_dist"]
        print(f"  {r['subject']:2d}   {r['expected_delay']:5.2f}    {r['peak_delay']}   "
              + "   ".join(f"{v:.3f}" for v in w))
    print("-" * 65)
    print(f"mean   {expected.mean():.2f}   peak mode={peaks.max()}   "
          + "   ".join(f"{v:.3f}" for v in mean_dist))
    print(f"std    {expected.std():.2f}         "
          + "   ".join(f"{v:.3f}" for v in std_dist))
    print(f"\nPeak δ distribution across subjects: {dict(enumerate(peak_counts))}")
    print("=" * 65)

    # Save raw results
    np.save(os.path.join(OUT_DIR, "lata_all_subjects_raw.npy"),
            np.array([r["delay_dist"] for r in all_results]))

    # ── Plot ──────────────────────────────────────────────────────────────
    C_BLUE  = "#005b96"
    C_LIGHT = "#b3cde0"
    C_RED   = "#c0392b"
    C_GRAY  = "#888888"
    C_GREEN = "#2ecc71"

    fig = plt.figure(figsize=(15, 5))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(1, 3, wspace=0.38)

    delay_ticks = list(range(MAX_DELAY + 1))

    # Panel 1: Mean delay distribution with error bars
    ax1 = fig.add_subplot(gs[0])
    peak_idx = int(mean_dist.argmax())
    colors = [C_BLUE if i == peak_idx else C_LIGHT for i in delay_ticks]
    bars = ax1.bar(delay_ticks, mean_dist, color=colors, edgecolor="white",
                   linewidth=0.8, zorder=3, width=0.55)
    ax1.errorbar(delay_ticks, mean_dist, yerr=std_dist,
                 fmt="none", color="#333", capsize=4, linewidth=1.5, zorder=4)
    for bar, m, s in zip(bars, mean_dist, std_dist):
        ax1.text(bar.get_x() + bar.get_width()/2, m + s + 0.006,
                 f"{m:.3f}", ha="center", va="bottom", fontsize=7.5, color="#333")
    ax1.set_title("Mean Learned Delay Distribution\n(N=20 subjects, error bars = ±1 SD)",
                  fontsize=10.5, fontweight="bold", color="#011f4b")
    ax1.set_xlabel("Delay δ (chunks)", fontsize=9)
    ax1.set_ylabel("P(δ)", fontsize=9)
    ax1.set_xticks(delay_ticks)
    ax1.set_xticklabels([f"δ={d}\n({d*500}ms)" for d in delay_ticks], fontsize=8)
    ax1.set_ylim(0, 0.48)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.spines[["top","right"]].set_visible(False)
    ax1.text(0.97, 0.97,
             f"E[δ] = {expected.mean():.2f} ± {expected.std():.2f}\n≈ {expected.mean()*500:.0f} ms",
             transform=ax1.transAxes, ha="right", va="top", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))

    # Panel 2: Per-subject expected delay
    ax2 = fig.add_subplot(gs[1])
    sub_ids = [r["subject"] for r in all_results]
    colors2 = [C_BLUE if p == peak_idx else C_LIGHT for p in peaks]
    ax2.bar(sub_ids, expected, color=colors2, edgecolor="white", linewidth=0.5, zorder=3)
    ax2.axhline(expected.mean(), color=C_RED, linestyle="--", linewidth=1.5,
                label=f"Mean = {expected.mean():.2f}", zorder=4)
    ax2.axhspan(expected.mean() - expected.std(),
                expected.mean() + expected.std(),
                alpha=0.12, color=C_RED, zorder=2)
    ax2.set_title("Expected Delay E[δ] per Subject\n(chunks × 0.5 s = ms/500)",
                  fontsize=10.5, fontweight="bold", color="#011f4b")
    ax2.set_xlabel("Subject", fontsize=9)
    ax2.set_ylabel("E[δ] (chunks)", fontsize=9)
    ax2.set_xticks(sub_ids[::2])
    ax2.set_xticklabels([f"S{s}" for s in sub_ids[::2]], fontsize=7.5)
    ax2.legend(fontsize=8.5, framealpha=0.8)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.spines[["top","right"]].set_visible(False)

    # Panel 3: Peak δ histogram across subjects
    ax3 = fig.add_subplot(gs[2])
    ax3.bar(delay_ticks, peak_counts, color=[C_BLUE if i == peak_idx else C_LIGHT
                                              for i in delay_ticks],
            edgecolor="white", linewidth=0.8, zorder=3, width=0.55)
    for i, cnt in enumerate(peak_counts):
        if cnt > 0:
            ax3.text(i, cnt + 0.15, str(int(cnt)), ha="center", va="bottom",
                     fontsize=9, fontweight="bold", color="#333")
    ax3.set_title(f"Peak δ Count Across {len(SUBJECTS)} Subjects\n(how many subjects prefer each delay)",
                  fontsize=10.5, fontweight="bold", color="#011f4b")
    ax3.set_xlabel("Peak delay δ (chunks)", fontsize=9)
    ax3.set_ylabel("# subjects", fontsize=9)
    ax3.set_xticks(delay_ticks)
    ax3.set_xticklabels([f"δ={d}\n({d*500}ms)" for d in delay_ticks], fontsize=8)
    ax3.set_ylim(0, max(peak_counts) + 2)
    ax3.grid(axis="y", alpha=0.3, linestyle="--")
    ax3.spines[["top","right"]].set_visible(False)

    fig.suptitle(
        f"LATA on SEED-DV: Learned EEG–Video Delay Across All {len(SUBJECTS)} Subjects",
        fontsize=13, fontweight="bold", color="#011f4b", y=1.0
    )
    plt.subplots_adjust(top=0.88)

    out_path = os.path.join(OUT_DIR, "lata_all_subjects_results.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"\nSaved figure: {out_path}")


if __name__ == "__main__":
    main()
