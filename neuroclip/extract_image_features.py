"""
Extract CLIP image embeddings from SEED-DV video clips.

Timing from EEG preprocessing:
  Each session = 40 concepts × (3s hint + 5 × 2s clips) = 520s
  Clip (concept_pos=c, clip_pos=p) starts at: c*13 + 3 + p*2 seconds
  We sample the middle frame at: c*13 + 3 + p*2 + 1 seconds

Outputs (saved to neuroclip/):
  clip_image_embeddings.pt   — (7, 200, 512) per-clip CLIP image embeddings
  concept_image_embeddings.pt — (7, 40, 512) per-concept mean image embeddings

Run from EEG2Video/:
    python neuroclip/extract_image_features.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

sys.path.insert(0, os.path.dirname(__file__))
from dataset import GT_LABEL

VIDEO_DIR  = "data/Video"
OUT_DIR    = "neuroclip"
MODEL_NAME = "openai/clip-vit-base-patch32"

VIDEO_FILES = [
    "1st_10min.mp4", "2nd_10min.mp4", "3rd_10min.mp4", "4th_10min.mp4",
    "5th_10min.mp4", "6th_10min.mp4", "7th_10min.mp4",
]

# Timing constants (from EEG preprocessing script)
HINT_SECS  = 3      # hint screen before each concept
CLIP_SECS  = 2      # each video clip duration
N_CONCEPTS = 40
N_CLIPS    = 5

def concept_clip_time(concept_pos, clip_pos):
    """Return the time (seconds) of the middle frame of a clip."""
    start = concept_pos * (HINT_SECS + N_CLIPS * CLIP_SECS) + HINT_SECS + clip_pos * CLIP_SECS
    return start + 1.0   # middle of 2-second clip


def extract_frame(cap, time_sec):
    """Seek to time_sec and return a PIL Image."""
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    ret, frame = cap.read()
    if not ret:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def main():
    device = (
        torch.device("mps")  if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    print(f"Loading CLIP model: {MODEL_NAME}")
    model     = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    all_clip_embs    = torch.zeros(7, 200, 512)   # per-clip
    all_concept_embs = torch.zeros(7, 40,  512)   # per-concept mean

    for sess in range(7):
        video_path = os.path.join(VIDEO_DIR, VIDEO_FILES[sess])
        print(f"\nSession {sess+1}: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  ERROR: cannot open {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = total_frames / fps
        print(f"  FPS={fps:.1f}  duration={duration:.1f}s  frames={int(total_frames)}")

        sess_embs = torch.zeros(200, 512)

        for concept_pos in range(N_CONCEPTS):
            frames_batch = []
            clip_indices = []

            for clip_pos in range(N_CLIPS):
                t = concept_clip_time(concept_pos, clip_pos)
                frame = extract_frame(cap, t)
                if frame is None:
                    print(f"  WARNING: failed to extract frame at t={t:.1f}s (concept={concept_pos}, clip={clip_pos})")
                    frame = Image.new("RGB", (224, 224), color=(128, 128, 128))
                frames_batch.append(frame)
                clip_indices.append(concept_pos * N_CLIPS + clip_pos)

            # Encode batch of 5 frames
            inputs = processor(images=frames_batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                img_feats = model.get_image_features(**inputs)   # (5, 512)
                img_feats = F.normalize(img_feats, dim=-1)

            for i, clip_idx in enumerate(clip_indices):
                sess_embs[clip_idx] = img_feats[i].cpu()

            if (concept_pos + 1) % 10 == 0:
                print(f"  concept {concept_pos+1}/40 done")

        cap.release()
        all_clip_embs[sess] = sess_embs

        # Concept-mean embeddings indexed by concept_id (0-39)
        concept_embs_sess = torch.zeros(40, 512)
        counts = torch.zeros(40)
        for concept_pos in range(N_CONCEPTS):
            cid = int(GT_LABEL[sess, concept_pos])
            for clip_pos in range(N_CLIPS):
                clip_idx = concept_pos * N_CLIPS + clip_pos
                concept_embs_sess[cid] += sess_embs[clip_idx]
                counts[cid] += 1
        concept_embs_sess = concept_embs_sess / counts.clamp(min=1).unsqueeze(1)
        concept_embs_sess = F.normalize(concept_embs_sess, dim=-1)
        all_concept_embs[sess] = concept_embs_sess

        print(f"  Session {sess+1} done. Emb mean norm: {sess_embs.norm(dim=-1).mean():.3f}")

    # Save
    clip_out    = os.path.join(OUT_DIR, "clip_image_embeddings.pt")
    concept_out = os.path.join(OUT_DIR, "clip_concept_image_embeddings.pt")
    torch.save(all_clip_embs,    clip_out)
    torch.save(all_concept_embs, concept_out)
    print(f"\nSaved → {clip_out}  shape={tuple(all_clip_embs.shape)}")
    print(f"Saved → {concept_out}  shape={tuple(all_concept_embs.shape)}")

    # Quick sanity: text vs image similarity
    text_embs = torch.load("neuroclip/clip_text_embeddings.pt", weights_only=True)
    sim = (all_clip_embs * text_embs).sum(dim=-1)   # (7, 200) cosine sim
    print(f"\nSanity: mean cosine sim(image, text) = {sim.mean():.3f}  "
          f"(expected ~0.2-0.4 for matching pairs)")


if __name__ == "__main__":
    main()
