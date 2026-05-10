"""
Re-extract both text and image CLIP features using CLIPModel's projection heads,
so both live in the same 512-D CLIP embedding space.

Outputs (neuroclip/):
  clip_text_embs_v2.pt          (7, 200, 512) — projected text features
  clip_concept_text_embs_v2.pt  (7,  40, 512) — concept-mean projected text
  clip_image_embs_v2.pt         (7, 200, 512) — projected image features
  clip_concept_image_embs_v2.pt (7,  40, 512) — concept-mean projected image
  clip_both_embs_v2.pt          (7, 200, 512) — (text + image) / 2, re-normalised
  clip_concept_both_embs_v2.pt  (7,  40, 512) — concept-mean of both

Run from EEG2Video/:
    python neuroclip/extract_clip_features_v2.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from dataset import GT_LABEL

CAPTION_DIR = "data/Video/BLIP-caption"
VIDEO_DIR   = "data/Video"
OUT_DIR     = "neuroclip"
MODEL_NAME  = "openai/clip-vit-base-patch32"

SESSION_FILES_TXT = [f"{i+1}st_10min.txt" if i == 0
                     else f"{i+1}nd_10min.txt" if i == 1
                     else f"{i+1}rd_10min.txt" if i == 2
                     else f"{i+1}th_10min.txt"
                     for i in range(7)]
SESSION_FILES_TXT = [
    "1st_10min.txt","2nd_10min.txt","3rd_10min.txt","4th_10min.txt",
    "5th_10min.txt","6th_10min.txt","7th_10min.txt",
]
SESSION_FILES_VID = [
    "1st_10min.mp4","2nd_10min.mp4","3rd_10min.mp4","4th_10min.mp4",
    "5th_10min.mp4","6th_10min.mp4","7th_10min.mp4",
]

HINT_SECS  = 3
CLIP_SECS  = 2
N_CONCEPTS = 40
N_CLIPS    = 5
BATCH_SIZE = 32


def clip_middle_time(concept_pos, clip_pos):
    start = concept_pos * (HINT_SECS + N_CLIPS * CLIP_SECS) + HINT_SECS + clip_pos * CLIP_SECS
    return start + 1.0


def extract_frame(cap, time_sec):
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    ret, frame = cap.read()
    if not ret:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def make_concept_embs(clip_embs_sess):
    """Average 5 clips per concept_pos → (40, 512) indexed by concept_pos."""
    return F.normalize(clip_embs_sess.reshape(N_CONCEPTS, N_CLIPS, -1).mean(dim=1), dim=-1)


def main():
    device = (
        torch.device("mps")  if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")
    print(f"Loading CLIP: {MODEL_NAME}")
    model     = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
    model.eval()

    text_all    = torch.zeros(7, 200, 512)
    image_all   = torch.zeros(7, 200, 512)

    for sess in range(7):
        print(f"\n=== Session {sess+1} ===")

        # ---- TEXT ----
        with open(os.path.join(CAPTION_DIR, SESSION_FILES_TXT[sess])) as fh:
            captions = [l.strip() for l in fh.readlines()]
        assert len(captions) == 200

        sess_text = []
        for start in range(0, 200, BATCH_SIZE):
            batch = captions[start:start+BATCH_SIZE]
            tokens = tokenizer(batch, padding=True, truncation=True,
                               max_length=77, return_tensors="pt").to(device)
            with torch.no_grad():
                feats = model.get_text_features(**tokens)   # (B, 512) projected
                feats = F.normalize(feats, dim=-1)
            sess_text.append(feats.cpu())
        sess_text = torch.cat(sess_text, dim=0)   # (200, 512)
        text_all[sess] = sess_text
        print(f"  Text done. Mean norm={sess_text.norm(dim=-1).mean():.3f}")

        # ---- IMAGE ----
        cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, SESSION_FILES_VID[sess]))
        sess_img = torch.zeros(200, 512)

        for concept_pos in range(N_CONCEPTS):
            frames = []
            for clip_pos in range(N_CLIPS):
                t = clip_middle_time(concept_pos, clip_pos)
                frame = extract_frame(cap, t)
                if frame is None:
                    frame = Image.new("RGB", (224, 224), (128, 128, 128))
                frames.append(frame)

            inputs = processor(images=frames, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = model.get_image_features(**inputs)   # (5, 512) projected
                feats = F.normalize(feats, dim=-1)
            for clip_pos in range(N_CLIPS):
                sess_img[concept_pos * N_CLIPS + clip_pos] = feats[clip_pos].cpu()

        cap.release()
        image_all[sess] = sess_img
        print(f"  Image done. Mean norm={sess_img.norm(dim=-1).mean():.3f}")

        # Sanity: cosine sim between matching text/image pairs
        sim = (sess_text * sess_img).sum(dim=-1)
        print(f"  Text-Image cosine sim: mean={sim.mean():.3f}  max={sim.max():.3f}  min={sim.min():.3f}")

    # ---- BOTH = normalised average ----
    both_all = F.normalize(text_all + image_all, dim=-1)

    # ---- Concept-mean embeddings ----
    text_concept_all  = torch.stack([make_concept_embs(text_all[s])  for s in range(7)])
    image_concept_all = torch.stack([make_concept_embs(image_all[s]) for s in range(7)])
    both_concept_all  = torch.stack([make_concept_embs(both_all[s])  for s in range(7)])

    # ---- Save ----
    paths = {
        "clip_text_embs_v2.pt":          text_all,
        "clip_concept_text_embs_v2.pt":  text_concept_all,
        "clip_image_embs_v2.pt":         image_all,
        "clip_concept_image_embs_v2.pt": image_concept_all,
        "clip_both_embs_v2.pt":          both_all,
        "clip_concept_both_embs_v2.pt":  both_concept_all,
    }
    for fname, tensor in paths.items():
        path = os.path.join(OUT_DIR, fname)
        torch.save(tensor, path)
        print(f"Saved {path}  {tuple(tensor.shape)}")

    # Final sanity
    sim_all = (text_all * image_all).sum(dim=-1)
    print(f"\nOverall text-image cosine sim: mean={sim_all.mean():.3f}  std={sim_all.std():.3f}")
    print("Expected: ~0.25-0.35 for matching CLIP pairs")


if __name__ == "__main__":
    main()
