"""
Step 1: Pre-extract CLIP text embeddings from BLIP captions.

Loads the 7 BLIP caption files (200 captions each, one per 2-second video clip),
runs CLIP's text encoder on every caption, and saves the resulting embeddings to
neuroclip/clip_text_embeddings.pt

Shape saved: (7, 200, 512)  [sessions x clips_per_session x embed_dim]

Run once from the EEG2Video/ directory:
    python neuroclip/extract_text_features.py
"""

import os
import torch
from transformers import CLIPTokenizer, CLIPTextModel

CAPTION_DIR = "data/Video/BLIP-caption"
SAVE_PATH   = "neuroclip/clip_text_embeddings.pt"
SESSION_FILES = [
    "1st_10min.txt",
    "2nd_10min.txt",
    "3rd_10min.txt",
    "4th_10min.txt",
    "5th_10min.txt",
    "6th_10min.txt",
    "7th_10min.txt",
]
CLIPS_PER_SESSION = 200
BATCH_SIZE = 64

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
text_model.eval()

all_embeddings = []  # will be shape (7, 200, 512)

for session_idx, fname in enumerate(SESSION_FILES):
    fpath = os.path.join(CAPTION_DIR, fname)
    with open(fpath, "r") as f:
        captions = [line.strip() for line in f.readlines()]

    assert len(captions) == CLIPS_PER_SESSION, (
        f"Expected {CLIPS_PER_SESSION} captions in {fname}, got {len(captions)}"
    )

    session_embs = []
    for start in range(0, CLIPS_PER_SESSION, BATCH_SIZE):
        batch_caps = captions[start : start + BATCH_SIZE]
        tokens = tokenizer(
            batch_caps,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = text_model(**tokens)
            # Use the [EOS] token embedding (pooler_output) as the clip embedding
            emb = out.pooler_output  # (batch, 512)
            emb = torch.nn.functional.normalize(emb, dim=-1)
        session_embs.append(emb.cpu())

    session_tensor = torch.cat(session_embs, dim=0)  # (200, 512)
    all_embeddings.append(session_tensor)
    print(f"Session {session_idx+1}: {session_tensor.shape}")

embeddings = torch.stack(all_embeddings, dim=0)  # (7, 200, 512)
torch.save(embeddings, SAVE_PATH)
print(f"\nSaved CLIP text embeddings: {embeddings.shape} → {SAVE_PATH}")

# Also save concept-mean embeddings: average the 5 clips per concept
# Shape: (7, 40, 512)
concept_embs = embeddings.reshape(7, 40, 5, 512).mean(dim=2)
concept_path = SAVE_PATH.replace("clip_text_embeddings", "clip_concept_embeddings")
torch.save(concept_embs, concept_path)
print(f"Saved concept-mean embeddings: {concept_embs.shape} → {concept_path}")
