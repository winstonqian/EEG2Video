"""
NeuroCLIP model definitions.

EEGEncoder        — full-clip lightweight temporal CNN → 512-D embedding
ChunkEEGEncoder   — same but processes K temporal chunks + learned attention pooling
infonce_loss      — symmetric CLIP-style InfoNCE
concept_infonce   — concept-aware InfoNCE that masks within-concept false negatives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# EEG Encoders
# ---------------------------------------------------------------------------

class EEGEncoder(nn.Module):
    """
    Input:  (B, C, T)  — raw EEG (C=62, T=400) or DE/PSD (C=62, T=5)
    Output: (B, embed_dim) — L2 normalised
    """

    def __init__(self, n_channels: int = 62, n_time: int = 400, embed_dim: int = 512):
        super().__init__()
        self.n_channels = n_channels
        self.n_time     = n_time

        # Spatial filter: learn linear electrode combinations
        self.spatial = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        if n_time >= 100:  # raw EEG
            self.temporal = nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=25, stride=5, padding=12),  # T→80
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Conv1d(128, 256, kernel_size=10, stride=2, padding=4),  # T→40
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=2),   # T→20
                nn.BatchNorm1d(512),
                nn.GELU(),
            )
        else:  # DE/PSD (T=5)
            self.temporal = nn.Sequential(
                nn.Conv1d(64, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(512),
                nn.GELU(),
            )

        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.2)
        self.proj    = nn.Linear(512, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial(x)
        x = self.temporal(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.proj(x)
        return F.normalize(x, dim=-1)


class ChunkEEGEncoder(nn.Module):
    """
    Splits raw EEG (C, T) into K equal temporal chunks, encodes each with a
    shared EEGEncoder, then applies learned attention pooling over chunks.

    Input:  (B, C, T)
    Output:
        emb:     (B, embed_dim) — L2 normalised aggregated embedding
        weights: (B, K)         — softmax attention weights per chunk
    """

    def __init__(
        self,
        n_channels: int = 62,
        n_time:     int = 400,
        k_chunks:   int = 4,
        embed_dim:  int = 512,
    ):
        super().__init__()
        assert n_time % k_chunks == 0, "n_time must be divisible by k_chunks"
        self.k_chunks   = k_chunks
        self.chunk_len  = n_time // k_chunks
        self.chunk_enc  = EEGEncoder(n_channels, self.chunk_len, embed_dim)
        self.attn_query = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor):
        B, C, T = x.shape
        # (B, K, C, chunk_len)
        chunks = x.reshape(B, C, self.k_chunks, self.chunk_len).permute(0, 2, 1, 3)
        chunks_flat = chunks.reshape(B * self.k_chunks, C, self.chunk_len)

        chunk_embs = self.chunk_enc(chunks_flat)                    # (B*K, D)
        chunk_embs = chunk_embs.reshape(B, self.k_chunks, -1)       # (B, K, D)

        scores  = self.attn_query(chunk_embs).squeeze(-1)           # (B, K)
        weights = torch.softmax(scores, dim=-1)                     # (B, K)
        emb     = (weights.unsqueeze(-1) * chunk_embs).sum(dim=1)   # (B, D)
        return F.normalize(emb, dim=-1), weights


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def infonce_loss(
    eeg_emb: torch.Tensor,
    vid_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE. Both inputs must be L2-normalised."""
    logits    = eeg_emb @ vid_emb.T / temperature   # (B, B)
    labels    = torch.arange(len(eeg_emb), device=eeg_emb.device)
    loss_e2v  = F.cross_entropy(logits,   labels)
    loss_v2e  = F.cross_entropy(logits.T, labels)
    return (loss_e2v + loss_v2e) / 2


def concept_infonce(
    eeg_emb:     torch.Tensor,   # (B, D) L2-normed
    concept_emb: torch.Tensor,   # (B, D) L2-normed concept-mean target
    concept_ids: torch.Tensor,   # (B,) int, 0-indexed concept label
    gallery_emb: torch.Tensor,   # (N_concepts, D) full concept gallery, L2-normed
    gallery_ids: torch.Tensor,   # (N_concepts,) int
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE where the gallery is the full set of concept-mean embeddings.

    For each EEG sample, the positive is its concept-mean embedding and all
    other concept embeddings are negatives.  This eliminates within-concept
    false negatives entirely.

    eeg_emb   @ gallery_emb.T → (B, N_concepts) logits
    label[i]  = index in gallery where gallery_ids == concept_ids[i]
    """
    logits = eeg_emb @ gallery_emb.T / temperature  # (B, N_concepts)

    # Map each sample's concept_id to its index in the gallery
    # gallery_ids is a sorted 0..39 tensor
    labels = concept_ids  # direct index since gallery_ids = arange(40)

    return F.cross_entropy(logits, labels)
