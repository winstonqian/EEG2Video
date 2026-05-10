"""
Extract CLIP *visual* features from SEED-DV video clips for LATA training.
==========================================================================
Uses PyAV to decode the session MP4s from Video.zip and CLIPModel to encode
one frame per 0.5-second chunk.

Timing derived from EEG_preprocessing/segment_raw_signals_200Hz.py:
  Each concept block = 3s hint + 5 × 2s clips = 13s
  Clip (c, v) starts at: c * 13 + 3 + v * 2   (seconds, 0-indexed)
  Chunk k center time:   clip_start + k * 0.5 + 0.25

Output
------
  video_features_clip_visual.npy   shape (7, 40, 5, 4, 512)

Run
---
  python extract_visual_features_pyav.py
"""

import os, io, sys, zipfile
import numpy as np
import torch
from PIL import Image
import av
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────
VIDEO_ZIP   = os.path.join(os.path.dirname(__file__), "..", "Video.zip")
OUT_PATH    = os.path.join(os.path.dirname(__file__), "video_features_clip_visual.npy")
CLIP_MODEL  = "openai/clip-vit-base-patch32"
N_SESSIONS  = 7
N_CONCEPTS  = 40
N_CLIPS     = 5       # per concept
K           = 4       # temporal chunks per clip
FPS         = 24.0    # nominal; actual ~23.976

# Timing constants (from segment_raw_signals_200Hz.py)
HINT_DUR    = 3.0     # seconds of hint/rest before each concept
CLIP_DUR    = 2.0     # seconds per clip
CONCEPT_DUR = HINT_DUR + N_CLIPS * CLIP_DUR   # = 13.0 s
CHUNK_DUR   = CLIP_DUR / K                    # = 0.5 s

BATCH_SIZE  = 32

SESSION_KEYS = [f"{i}st_10min" if i == 1
                else f"{i}nd_10min" if i == 2
                else f"{i}rd_10min" if i == 3
                else f"{i}th_10min"
                for i in range(1, N_SESSIONS + 1)]


def clip_chunk_timestamps() -> np.ndarray:
    """
    Return array of target timestamps, shape (40, 5, 4).
    ts[c, v, k] = center time (seconds) of chunk k in clip v of concept c.
    """
    ts = np.zeros((N_CONCEPTS, N_CLIPS, K), dtype=np.float32)
    for c in range(N_CONCEPTS):
        for v in range(N_CLIPS):
            clip_start = c * CONCEPT_DUR + HINT_DUR + v * CLIP_DUR
            for k in range(K):
                ts[c, v, k] = clip_start + k * CHUNK_DUR + CHUNK_DUR / 2
    return ts  # shape (40, 5, 4)


def extract_frames_linear(video_bytes: bytes, target_times: np.ndarray) -> list[Image.Image]:
    """
    Stream through video once and collect one frame per target time.

    Parameters
    ----------
    video_bytes   : raw bytes of the MP4
    target_times  : sorted 1-D array of target timestamps (seconds)

    Returns
    -------
    frames : list of PIL Images, same length as target_times
    """
    container = av.open(io.BytesIO(video_bytes))
    stream    = container.streams.video[0]
    stream.thread_type = "AUTO"

    # Use the stream's time_base for pts calculations
    tb = stream.time_base  # fractions.Fraction

    frames  = [None] * len(target_times)
    targets = list(enumerate(target_times))  # (original_idx, t)
    t_idx   = 0                              # index into targets

    prev_frame = None

    for packet in container.demux(stream):
        if t_idx >= len(targets):
            break
        for frame in packet.decode():
            if t_idx >= len(targets):
                break
            frame_t = float(frame.pts * tb)

            # Advance through all targets that this frame covers
            while t_idx < len(targets) and targets[t_idx][1] <= frame_t + CHUNK_DUR / 2:
                orig_idx, tgt = targets[t_idx]
                # Use this frame (closest decoded frame at or after target)
                use_frame = prev_frame if prev_frame is not None else frame
                img = use_frame.to_image()
                frames[orig_idx] = img
                t_idx += 1
            prev_frame = frame

    # Fill any remaining (shouldn't happen with valid timestamps)
    for i, img in enumerate(frames):
        if img is None:
            frames[i] = prev_frame.to_image() if prev_frame else Image.new("RGB", (224, 224))

    container.close()
    return frames


def encode_frames(frames: list[Image.Image], processor, model, device: str) -> np.ndarray:
    """
    Encode a list of PIL images with CLIP vision encoder.
    Returns (N, 512) float32 numpy array.
    """
    all_embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(frames), BATCH_SIZE):
            batch = frames[i : i + BATCH_SIZE]
            inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
            embs   = model.get_image_features(**inputs)  # (B, 512)
            all_embs.append(embs.cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)  # (N, 512)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading CLIP model: {CLIP_MODEL} ...")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model     = CLIPModel.from_pretrained(CLIP_MODEL).to(device)

    # Pre-compute target timestamps (same for every session)
    ts = clip_chunk_timestamps()          # (40, 5, 4)
    ts_flat = ts.reshape(-1)              # (40*5*4,) = 800 per session
    sort_idx  = np.argsort(ts_flat)       # extract in temporal order
    unsort_idx = np.argsort(sort_idx)     # undo sort after extraction

    print(f"\nTarget timestamps: {len(ts_flat)} per session")
    print(f"  First 4: {ts_flat[sort_idx[:4]].round(2)}")
    print(f"  Last 4:  {ts_flat[sort_idx[-4:]].round(2)}")

    features = np.zeros((N_SESSIONS, N_CONCEPTS, N_CLIPS, K, 512), dtype=np.float32)

    with zipfile.ZipFile(VIDEO_ZIP) as z:
        for sess_idx, name in enumerate(SESSION_KEYS):
            key = f"Video/{name}.mp4"
            print(f"\nSession {sess_idx + 1}/{N_SESSIONS}: loading {key} ...")
            with z.open(key) as f:
                video_bytes = f.read()
            print(f"  {len(video_bytes) / 1e6:.0f} MB loaded, extracting {len(ts_flat)} frames ...")

            frames_sorted = extract_frames_linear(
                video_bytes, ts_flat[sort_idx]
            )
            frames_orig = [frames_sorted[unsort_idx[i]] for i in range(len(ts_flat))]

            print(f"  Encoding with CLIP ...")
            embs = encode_frames(frames_orig, processor, model, device)  # (800, 512)

            # Reshape: (800,) → (40, 5, 4, 512) then store
            embs_grid = embs.reshape(N_CONCEPTS, N_CLIPS, K, 512)
            features[sess_idx] = embs_grid

    # L2-normalise
    norm = np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8
    features = features / norm

    np.save(OUT_PATH, features)
    print(f"\nSaved: {OUT_PATH}  shape={features.shape}")

    # Sanity checks
    print("\nSanity checks:")
    print(f"  Norm (should be ~1.0):              {np.linalg.norm(features[0,0,0,0]):.4f}")
    sim_same = features[0,0,0,0] @ features[0,0,0,1]
    sim_diff = features[0,0,0,0] @ features[0,0,0,2]
    sim_cross = features[0,0,0,0] @ features[0,1,0,0]
    print(f"  Sim(chunk0, chunk1 same clip):      {sim_same:.4f}  (want < 1.0 — diff frames)")
    print(f"  Sim(chunk0, chunk2 same clip):      {sim_diff:.4f}")
    print(f"  Sim(clip0 chunk0, clip1 chunk0):    {sim_cross:.4f}")
    print(f"\nDone! Use video_features_clip_visual.npy with train_lata_seeddv.py:")
    print(f"  vid_feat_path in train_lata_seeddv.py → 'video_features_clip_visual.npy'")


if __name__ == "__main__":
    main()
