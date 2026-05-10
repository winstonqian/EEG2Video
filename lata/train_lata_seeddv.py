"""
LATA on SEED-DV: Real EEG–Video Alignment
==========================================
Applies the LATA module to real data from the SEED-DV dataset.

Data expected
-------------
eeg2video_colab.zip  →  data/Segmented_Rawf_200Hz_2s/subN.npy
    shape: (7, 40, 5, 62, 400)
           sessions × concepts × clips × channels × timepoints

Video features: one of
    (a) CLIP frame embeddings  — extract with extract_clip_features.py (TODO)
    (b) VideoMAE chunk embeds  — heavier, Colab recommended

Run
---
    python train_lata_seeddv.py --subject 1 --sessions 7 --epochs 200
"""

import argparse
import os
import sys
import zipfile
import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from lata import LATA, lata_infonce_loss


# ─────────────────────────────────────────────────────────────────────────────
# EEG Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SEEDDVEEGDataset(Dataset):
    """
    Loads segmented raw EEG for one subject and chunks each 2-second clip
    into K equal temporal chunks for LATA training.

    EEG data shape: (7, 40, 5, 62, 400)
    After chunking: (N_clips, K, C, T_chunk)
                     N_clips = 7*40*5 = 1400
                     K = 4 chunks × 100 timepoints each

    Parameters
    ----------
    eeg_array  : np.ndarray (7, 40, 5, 62, 400)
    K          : int — number of chunks per clip (default 4)
    video_feats: np.ndarray (N_clips, K, d_video) or None
                 If None, returns random placeholder features (for testing).
    """

    def __init__(self, eeg_array: np.ndarray, K: int = 4, video_feats=None):
        n_sess, n_cls, n_clips, C, T = eeg_array.shape
        assert T % K == 0, f"T={T} must be divisible by K={K}"

        T_chunk = T // K
        # Reshape to (N, K, C, T_chunk)
        eeg = eeg_array.reshape(n_sess * n_cls * n_clips, C, T)
        # Split time axis into K chunks: (N, K, C, T_chunk)
        eeg_chunks = eeg.reshape(len(eeg), C, K, T_chunk).transpose(0, 2, 1, 3)
        # → (N, K, C, T_chunk)

        self.eeg    = torch.tensor(eeg_chunks, dtype=torch.float32)  # (N, K, C, T_chunk)
        self.N      = len(self.eeg)
        self.K      = K
        self.C      = C
        self.T_chunk = T_chunk

        if video_feats is not None:
            # Accept either (N, K, d) already flat, or (sess, conc, clips, K, d)
            if video_feats.ndim == 5:
                # (sess, 40, 5, K, d) → (N, K, d)
                s, c, cl, k, d = video_feats.shape
                video_feats = video_feats.reshape(s * c * cl, k, d)
            self.video = torch.tensor(video_feats, dtype=torch.float32)
        else:
            print("WARNING: using random video features — run extract_video_features.py first!")
            self.video = torch.randn(self.N, K, 512)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.eeg[idx], self.video[idx]
        # shapes: (K, C, T_chunk), (K, d_video)


# ─────────────────────────────────────────────────────────────────────────────
# EEG Chunk Encoder
# ─────────────────────────────────────────────────────────────────────────────

class EEGChunkEncoder(nn.Module):
    """
    Lightweight temporal CNN that encodes one EEG chunk (C, T_chunk) → d_model.

    Applied independently to each of the K chunks per clip.
    Based on ShallowNet from models.py, adapted for variable chunk length.
    """

    def __init__(self, C: int, T_chunk: int, d_model: int):
        super().__init__()
        # Temporal + spatial convolution (ShallowNet-style)
        self.temporal = nn.Conv2d(1, 32, kernel_size=(1, 25), padding=(0, 12))
        self.spatial  = nn.Conv2d(32, 32, kernel_size=(C, 1))
        self.bn       = nn.BatchNorm2d(32)
        self.act      = nn.ELU()
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.proj     = nn.Linear(32, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, K, C, T_chunk)
        returns: (B, K, d_model)
        """
        B, K, C, T = x.shape
        # Merge batch and chunk dims for convolution
        x = x.view(B * K, 1, C, T)            # (B*K, 1, C, T)
        x = self.temporal(x)                   # (B*K, 32, C, T)
        x = self.act(self.bn(self.spatial(x))) # (B*K, 32, 1, T')
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B*K, 32)
        x = self.proj(x)                       # (B*K, d_model)
        return x.view(B, K, -1)                # (B, K, d_model)


class VideoProjector(nn.Module):
    """Projects raw video features (e.g. CLIP 512-d) to d_model."""
    def __init__(self, d_video: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_video, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load EEG ────────────────────────────────────────────────────────────
    zip_path = os.path.join(
        os.path.dirname(__file__), "..", "eeg2video_colab.zip"
    )
    sub_key = f"data/Segmented_Rawf_200Hz_2s/sub{args.subject}.npy"
    print(f"Loading EEG for subject {args.subject} ...")

    with zipfile.ZipFile(zip_path) as z:
        with z.open(sub_key) as f:
            eeg_data = np.load(io.BytesIO(f.read()))  # (7, 40, 5, 62, 400)
    print(f"  EEG shape: {eeg_data.shape}")

    # Use first `args.sessions` sessions for train, last for val
    n_train = args.sessions - 1
    eeg_train = eeg_data[:n_train]
    eeg_val   = eeg_data[n_train:n_train + 1]

    # ── Video features ───────────────────────────────────────────────────────
    # Shape on disk: (7, 40, 5, 4, 512) — sessions × concepts × clips × chunks × d
    vid_feat_path = os.path.join(os.path.dirname(__file__), "video_features_clip_text.npy")
    K       = 4
    d_video = 512

    if os.path.exists(vid_feat_path):
        print(f"Loading video features from {vid_feat_path} ...")
        vid_all = np.load(vid_feat_path)           # (7, 40, 5, 4, 512)
        print(f"  Video features shape: {vid_all.shape}")
        video_feats_train = vid_all[:n_train]      # (n_train, 40, 5, 4, 512)
        video_feats_val   = vid_all[n_train:n_train + 1]
    else:
        print("WARNING: video_features_clip_text.npy not found — using random placeholders.")
        print("Run extract_video_features.py first.")
        video_feats_train = None
        video_feats_val   = None

    train_ds = SEEDDVEEGDataset(eeg_train, K=K, video_feats=video_feats_train)
    val_ds   = SEEDDVEEGDataset(eeg_val,   K=K, video_feats=video_feats_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── Models ───────────────────────────────────────────────────────────────
    C, T_chunk = 62, 100       # 62 channels, 100 timepoints per chunk (0.5 s)
    d_model    = args.d_model

    eeg_encoder   = EEGChunkEncoder(C, T_chunk, d_model).to(device)
    vid_projector = VideoProjector(d_video, d_model).to(device)
    lata          = LATA(d_model=d_model, n_heads=args.n_heads,
                         max_delay=args.max_delay).to(device)

    params = (list(eeg_encoder.parameters()) +
              list(vid_projector.parameters()) +
              list(lata.parameters()))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs
    )

    print(f"\nLATA config: d_model={d_model}, max_delay={args.max_delay}")
    print(f"Training for {args.epochs} epochs ...\n")

    best_val  = float("inf")
    delay_log = []

    for epoch in range(1, args.epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        eeg_encoder.train(); vid_projector.train(); lata.train()
        train_loss = 0.0
        for eeg_batch, vid_batch in train_loader:
            eeg_batch = eeg_batch.to(device)   # (B, K, C, T_chunk)
            vid_batch = vid_batch.to(device)   # (B, K, d_video)

            eeg_enc = eeg_encoder(eeg_batch)   # (B, K, d_model)
            vid_enc = vid_projector(vid_batch)  # (B, K, d_model)

            loss = lata_infonce_loss(eeg_enc, lata, vid_enc,
                                     temperature=args.temperature)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # ── Validate ─────────────────────────────────────────────────────────
        eeg_encoder.eval(); vid_projector.eval(); lata.eval()
        val_loss = 0.0
        with torch.no_grad():
            for eeg_batch, vid_batch in val_loader:
                eeg_batch = eeg_batch.to(device)
                vid_batch = vid_batch.to(device)
                eeg_enc = eeg_encoder(eeg_batch)
                vid_enc = vid_projector(vid_batch)
                val_loss += lata_infonce_loss(eeg_enc, lata, vid_enc,
                                              temperature=args.temperature).item()
        val_loss /= len(val_loader)

        delay_log.append(lata.learned_delay.numpy().copy())

        if epoch % 20 == 0 or epoch == 1:
            dist_str = "  ".join(f"{v:.3f}" for v in lata.learned_delay.tolist())
            print(f"Epoch {epoch:4d} | train={train_loss:.4f} | val={val_loss:.4f} | "
                  f"E[δ]={lata.expected_delay:.2f} | w=[{dist_str}]")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "eeg_encoder":   eeg_encoder.state_dict(),
                "vid_projector": vid_projector.state_dict(),
                "lata":          lata.state_dict(),
                "delay_dist":    lata.learned_delay.numpy(),
                "epoch":         epoch,
            }, f"lata_sub{args.subject}_best.pt")

    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Learned delay distribution: {lata.learned_delay.numpy().round(3)}")
    print(f"Expected delay: {lata.expected_delay:.3f} chunks "
          f"(≈ {lata.expected_delay * 500:.0f} ms at 0.5 s/chunk)")
    print(f"\nNote: if E[δ] ≈ 0.2–0.6 chunks ≈ 100–300 ms, this matches "
          f"known P100/P300 visual ERP latencies ✓")

    return lata, delay_log


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train LATA on SEED-DV EEG data")
    p.add_argument("--subject",     type=int,   default=1)
    p.add_argument("--sessions",    type=int,   default=7)
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--d-model",     type=int,   default=128)
    p.add_argument("--n-heads",     type=int,   default=4)
    p.add_argument("--max-delay",   type=int,   default=3)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    args = p.parse_args()

    train(args)
