"""
Post-hoc visualization of chunk-level temporal attention weights.

After training NeuroCLIP with --chunks 4, this script:
  1. Loads a saved ChunkEEGEncoder for one subject/fold
  2. Computes attention weights for every test clip
  3. Plots mean weight per chunk (0..K-1) averaged over all test clips
     and broken down by concept

This reveals which 500ms window of the 2-second EEG carries the most
information — an interpretability diagnostic analogous to Section 7.1
of the midterm.

Run from EEG2Video/:
    python neuroclip/visualize_chunks.py --sub sub1 --fold 0 --chunks 4
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_subject, NeuroCLIPDataset, GT_LABEL
from models_neuroclip import ChunkEEGEncoder

CHUNK_LABELS = ["0–0.5s", "0.5–1s", "1–1.5s", "1.5–2s"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sub",        default="sub1")
    p.add_argument("--fold",       type=int, default=0)
    p.add_argument("--chunks",     type=int, default=4)
    p.add_argument("--feature",    default="raw", choices=["raw", "de", "psd"])
    p.add_argument("--output_dir", default="neuroclip/results")
    p.add_argument("--text_emb",   default="neuroclip/clip_text_embeddings.pt")
    return p.parse_args()


def main():
    args = parse_args()

    device = (
        torch.device("mps")  if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    if args.feature == "raw":
        data_dir = "data/Segmented_Rawf_200Hz_2s"
    elif args.feature == "de":
        data_dir = "data/DE_1per1s"
    else:
        data_dir = "data/PSD_1per1s"

    sub_file = f"{args.sub}.npy"
    eeg_data, text_embs, concept_embs, concept_ids = load_subject(
        os.path.join(data_dir, sub_file),
        args.text_emb,
        feature=args.feature,
    )

    n_time = eeg_data.shape[-1]
    model  = ChunkEEGEncoder(n_channels=62, n_time=n_time, k_chunks=args.chunks, embed_dim=512)

    model_path = os.path.join(
        args.output_dir,
        f"{args.sub}_fold{args.fold}_{args.feature}_k{args.chunks}.pt",
    )
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Train first with: python neuroclip/train_neuroclip.py --chunks 4 --sub sub1.npy")
        return

    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.to(device).eval()

    test_session = args.fold
    test_ds = NeuroCLIPDataset(eeg_data, text_embs, concept_embs, concept_ids, [test_session])
    loader  = torch.utils.data.DataLoader(test_ds, batch_size=200, shuffle=False)

    all_weights = []  # (200, K)
    with torch.no_grad():
        for eeg, _, _, _, _ in loader:
            _, weights = model(eeg.to(device))
            all_weights.append(weights.cpu())
    all_weights = torch.cat(all_weights, dim=0).numpy()  # (200, K)

    # ---- Plot 1: mean attention weight per chunk ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    mean_w = all_weights.mean(axis=0)
    axes[0].bar(range(args.chunks), mean_w, color="steelblue")
    axes[0].set_xticks(range(args.chunks))
    axes[0].set_xticklabels(CHUNK_LABELS[:args.chunks])
    axes[0].set_ylabel("Mean attention weight")
    axes[0].set_title(f"Temporal attention — {args.sub} fold {args.fold}")
    axes[0].axhline(1 / args.chunks, color="gray", linestyle="--", label="Uniform")
    axes[0].legend()

    # ---- Plot 2: attention by clip position within concept run ----
    # Reshape to (40 concepts, 5 clips, K)
    weights_by_pos = all_weights.reshape(40, 5, args.chunks)
    mean_by_pos = weights_by_pos.mean(axis=0)  # (5, K)

    for k in range(args.chunks):
        axes[1].plot(range(1, 6), mean_by_pos[:, k],
                     marker="o", label=CHUNK_LABELS[k])
    axes[1].set_xlabel("Clip position within concept run")
    axes[1].set_ylabel("Mean attention weight")
    axes[1].set_title("Attention shift across run positions")
    axes[1].legend()
    axes[1].set_xticks(range(1, 6))

    plt.tight_layout()
    save_path = os.path.join(args.output_dir, f"chunk_attention_{args.sub}_fold{args.fold}.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved → {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
