"""
Extract CLIP text features from BLIP captions for LATA training.
================================================================
Reads the pre-extracted BLIP captions from Video.zip (200 captions per
session = 40 concepts × 5 clips), encodes each with CLIP's text encoder,
and saves chunk-level video features ready to plug into train_lata_seeddv.py.

Output shape: (n_sessions, 40, 5, K, 512)
  → same index layout as EEG data (7, 40, 5, 62, 400)
  → K=4 chunks per clip (same caption repeated — clip-level approximation)

Run
---
    python extract_video_features.py

Outputs
-------
    video_features_clip_text.npy   shape (7, 40, 5, 4, 512)
"""

import os
import sys
import zipfile
import io
import numpy as np
import torch
from transformers import CLIPTokenizer, CLIPTextModel

# ── Config ─────────────────────────────────────────────────────────────────
VIDEO_ZIP   = os.path.join(os.path.dirname(__file__), "..", "Video.zip")
OUT_PATH    = os.path.join(os.path.dirname(__file__), "video_features_clip_text.npy")
CLIP_MODEL  = "openai/clip-vit-base-patch32"
N_SESSIONS  = 7
N_CONCEPTS  = 40
N_CLIPS     = 5      # clips per concept
K           = 4      # temporal chunks per clip (same feat repeated)
D_CLIP      = 512    # CLIP text embedding dim
BATCH_SIZE  = 64


def load_captions(zip_path: str) -> list[list[str]]:
    """
    Load BLIP captions from Video.zip.

    Returns: list of 7 session lists, each containing 200 captions
             (40 concepts × 5 clips, in order).
    """
    session_names = [f"{i}st_10min" if i == 1
                     else f"{i}nd_10min" if i == 2
                     else f"{i}rd_10min" if i == 3
                     else f"{i}th_10min"
                     for i in range(1, N_SESSIONS + 1)]

    all_captions = []
    with zipfile.ZipFile(zip_path) as z:
        for name in session_names:
            key = f"Video/BLIP-caption/{name}.txt"
            with z.open(key) as f:
                lines = f.read().decode("utf-8").strip().splitlines()
                lines = [l.strip() for l in lines]
                assert len(lines) == N_CONCEPTS * N_CLIPS, \
                    f"{key}: expected {N_CONCEPTS * N_CLIPS} lines, got {len(lines)}"
                all_captions.append(lines)

    print(f"Loaded captions: {N_SESSIONS} sessions × {N_CONCEPTS * N_CLIPS} clips")
    print(f"  Example: '{all_captions[0][0]}'")
    return all_captions


def encode_captions(captions_flat: list[str], tokenizer, model, device: str) -> np.ndarray:
    """
    Encode a flat list of captions with CLIP text encoder.
    Returns: (N, 512) float32 numpy array
    """
    all_embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(captions_flat), BATCH_SIZE):
            batch = captions_flat[i : i + BATCH_SIZE]
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)
            out = model(**tokens)
            # Use the pooled output (EOS token embedding = CLIP text embed)
            emb = out.pooler_output  # (B, 512)
            all_embs.append(emb.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)  # (N, 512)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading CLIP model: {CLIP_MODEL} ...")

    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    model     = CLIPTextModel.from_pretrained(CLIP_MODEL).to(device)

    print("Loading BLIP captions ...")
    all_captions = load_captions(VIDEO_ZIP)

    # Result array: (7, 40, 5, K, 512)
    features = np.zeros((N_SESSIONS, N_CONCEPTS, N_CLIPS, K, D_CLIP), dtype=np.float32)

    for sess_idx, session_captions in enumerate(all_captions):
        print(f"  Encoding session {sess_idx + 1}/{N_SESSIONS} ...")

        # session_captions: 200 strings, ordered as concept0_clip0, c0_c1, ..., c39_c4
        embs = encode_captions(session_captions, tokenizer, model, device)
        # embs: (200, 512)

        # Reshape to (40, 5, 512)
        embs = embs.reshape(N_CONCEPTS, N_CLIPS, D_CLIP)

        # Repeat across K chunks (same caption for all chunks of a clip)
        # (40, 5, K, 512)
        embs_chunked = np.repeat(embs[:, :, np.newaxis, :], K, axis=2)

        features[sess_idx] = embs_chunked

    # L2-normalise across the embedding dimension (standard for CLIP)
    norm = np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8
    features = features / norm

    np.save(OUT_PATH, features)
    print(f"\nSaved: {OUT_PATH}")
    print(f"Shape: {features.shape}  (sessions, concepts, clips, chunks, d_clip)")
    print(f"  → to use: load and reshape to (N_clips_total, K, 512) for DataLoader")
    print(f"  → N_clips_total per session = {N_CONCEPTS * N_CLIPS} = 200")

    # Quick sanity check
    print(f"\nSanity check:")
    print(f"  Embedding norm (should be ~1.0): {np.linalg.norm(features[0, 0, 0, 0]):.4f}")
    print(f"  Sim(clip0_chunk0, clip0_chunk1): "
          f"{(features[0,0,0,0] @ features[0,0,0,1]):.4f}  (should be 1.0 — same caption)")
    print(f"  Sim(clip0, clip1): "
          f"{(features[0,0,0,0] @ features[0,1,0,0]):.4f}  (should be < 1.0 — diff captions)")


if __name__ == "__main__":
    main()
