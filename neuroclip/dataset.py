"""
NeuroCLIP dataset utilities.

Supports raw EEG (62 x 400) and DE/PSD features (62 x 5, averaged over
the two 1-second sub-segments in DE_1per1s).

Index convention
----------------
clip_idx  (0..199) = concept_pos * 5 + clip_within_concept  (within a session)
concept_idx (0..39) = concept_pos within the session
concept_id  (0..39) = GT_LABEL[session, concept_pos]  (globally consistent label)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# Concept presentation order per session (0-indexed).
# GT_LABEL[session, concept_pos] = concept_id
GT_LABEL = np.array([
    [23,22,9,6,18,14,5,36,25,19,28,35,3,16,24,40,15,27,38,33,34,4,39,17,1,26,20,29,13,32,37,2,11,12,30,31,8,21,7,10],
    [27,33,22,28,31,12,38,4,18,17,35,39,40,5,24,32,15,13,2,16,34,25,19,30,23,3,8,29,7,20,11,14,37,6,21,1,10,36,26,9],
    [15,36,31,1,34,3,37,12,4,5,21,24,14,16,39,20,28,29,18,32,2,27,8,19,13,10,30,40,17,26,11,9,33,25,35,7,38,22,23,6],
    [16,28,23,1,39,10,35,14,19,27,37,31,5,18,11,25,29,13,20,24,7,34,26,4,40,12,8,22,21,30,17,2,38,9,3,36,33,6,32,15],
    [18,29,7,35,22,19,12,36,8,15,28,1,34,23,20,13,37,9,16,30,2,33,27,21,14,38,10,17,31,3,24,39,11,32,4,25,40,5,26,6],
    [29,16,1,22,34,39,24,10,8,35,27,31,23,17,2,15,25,40,3,36,26,6,14,37,9,12,19,30,5,28,32,4,13,18,21,20,7,11,33,38],
    [38,34,40,10,28,7,1,37,22,9,16,5,12,36,20,30,6,15,35,2,31,26,18,24,8,3,23,19,14,13,21,4,25,11,32,17,39,29,33,27],
], dtype=np.int64) - 1  # 0-indexed: values now 0..39


def load_subject(
    sub_path: str,
    text_emb_path: str  = "neuroclip/clip_text_embeddings.pt",
    concept_emb_path: str = "neuroclip/clip_concept_embeddings.pt",
    feature: str = "raw",
):
    """
    Load EEG + pre-computed CLIP embeddings for one subject.

    Returns
    -------
    eeg          : np.ndarray  (7, 200, C, T)
    text_embs    : torch.Tensor (7, 200, 512)  — per-clip text embeddings
    concept_embs : torch.Tensor (7, 40, 512)   — per-concept mean embeddings
    concept_ids  : np.ndarray   (7, 200)        — concept_id per clip
    """
    raw = np.load(sub_path)

    if feature == "raw":
        # raw: (7, 40, 5, 62, 400)
        n_sessions, n_concepts, n_clips, n_ch, n_time = raw.shape
        eeg = raw.reshape(n_sessions, n_concepts * n_clips, n_ch, n_time)
    else:
        # DE/PSD: (7, 40, 5, 2, 62, 5) — average 2 sub-segments
        n_sessions, n_concepts, n_clips, n_segs, n_ch, n_bands = raw.shape
        eeg = raw.mean(axis=3)                                    # (7, 40, 5, 62, 5)
        eeg = eeg.reshape(n_sessions, n_concepts * n_clips, n_ch, n_bands)

    # Concept ID for each of the 200 clips in every session: (7, 200)
    concept_ids = np.repeat(GT_LABEL, repeats=5, axis=1)  # repeat each concept_id 5 times

    text_embs    = torch.load(text_emb_path,    weights_only=True)  # (7, 200, 512)
    concept_embs = torch.load(concept_emb_path, weights_only=True)  # (7, 40, 512)

    return eeg, text_embs, concept_embs, concept_ids


class NeuroCLIPDataset(Dataset):
    """
    Returns (eeg, concept_emb, concept_id, clip_text_emb) for contrastive training.

    eeg            : (C, T) float32 — normalised within session
    concept_emb    : (512,) float32 — mean CLIP embedding of concept's 5 clips
    concept_id     : int (0-39)
    clip_text_emb  : (512,) float32 — individual clip's CLIP text embedding
    """

    def __init__(
        self,
        eeg_data: np.ndarray,            # (n_sessions, 200, C, T)
        text_embs: torch.Tensor,         # (7, 200, 512)
        concept_embs: torch.Tensor,      # (7, 40, 512)
        concept_ids: np.ndarray,         # (7, 200) int, 0-indexed concept labels
        session_ids: list,
        normalize: bool = True,
    ):
        self.samples = []

        for sess in session_ids:
            raw = eeg_data[sess]              # (200, C, T)
            c_embs = concept_embs[sess]       # (40, 512)
            t_embs = text_embs[sess]          # (200, 512)
            cids   = concept_ids[sess]        # (200,) int

            if normalize:
                flat = raw.reshape(200, -1)
                scaler = StandardScaler()
                scaler.fit(flat)
                raw = scaler.transform(flat).reshape(raw.shape)

            for clip_idx in range(200):
                cid = int(cids[clip_idx])
                # concept position within session = cid in GT_LABEL[sess]
                # The concept_embs are indexed by position, not by concept_id.
                # We need the session-specific position index for concept_embs.
                # GT_LABEL[sess] maps position -> concept_id.
                # We need position index where GT_LABEL[sess, pos] == cid.
                # Precompute this mapping.
                self.samples.append({
                    "eeg":          torch.tensor(raw[clip_idx], dtype=torch.float32),
                    "concept_emb":  c_embs[clip_idx // 5],  # clips 0-4 belong to concept_pos 0, etc.
                    "clip_text_emb": t_embs[clip_idx],
                    "concept_id":   cid,
                    "clip_idx":     clip_idx,
                    "session":      sess,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s["eeg"],
            s["concept_emb"],
            s["clip_text_emb"],
            torch.tensor(s["concept_id"], dtype=torch.long),
            s["clip_idx"],
        )
