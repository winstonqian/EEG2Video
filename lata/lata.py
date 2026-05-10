"""
Latency-Aware Temporal Alignment (LATA)
========================================
A plug-and-play PyTorch module that replaces standard cross-attention
with a biologically-motivated version.

Standard cross-attention assumes time t in modality A corresponds to
time t in modality B. This is physically wrong for brain signals: neural
responses always LAG the stimulus by some biological transit delay δ
(e.g. visual P100 at ~100 ms, P300 at ~300 ms).

LATA learns a soft distribution over candidate delays
    δ ∈ {0, 1, ..., max_delay}
and aligns the two temporal streams accordingly — with no manual tuning.

Applicable to any paired neural–sensory time series:
    EEG  ↔  Video        (visual decoding, this project)
    EEG  ↔  Audio        (auditory BCI)
    fMRI ↔  Text         (hemodynamic delay ~6 s)
    sEEG ↔  Robotics     (motor BCI)

Usage
-----
    lata = LATA(d_model=256, n_heads=4, max_delay=3)
    out  = lata(eeg_chunks, video_chunks)   # (B, K, 256)
    print(lata.learned_delay)               # e.g. [0.03, 0.91, 0.05, 0.01]
    print(lata.expected_delay)              # e.g. 1.04  (≈ 1 chunk = 125 ms)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LATA(nn.Module):
    """
    Latency-Aware Temporal Alignment module.

    Parameters
    ----------
    d_model   : int   — embedding dimension (both streams must match)
    n_heads   : int   — number of attention heads
    max_delay : int   — maximum candidate delay in chunk steps (inclusive)
    dropout   : float — attention dropout rate

    Inputs
    ------
    neural_seq   : (B, K, d_model)  — neural chunks, e.g. EEG
    stimulus_seq : (B, K, d_model)  — stimulus chunks, e.g. video frames

    Output
    ------
    (B, K, d_model) — latency-aligned attended neural representation

    Learnable parameters
    --------------------
    delay_logits : (max_delay+1,)  — unnormalised delay scores; softmax
                                     gives the delay distribution
    q_proj, k_proj, v_proj, out_proj : standard cross-attention weights
    """

    def __init__(
        self,
        d_model:   int,
        n_heads:   int   = 4,
        max_delay: int   = 3,
        dropout:   float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model   = d_model
        self.n_heads   = n_heads
        self.max_delay = max_delay
        self.d_head    = d_model // n_heads
        self.scale     = math.sqrt(self.d_head)

        # ── Learnable delay distribution ───────────────────────────────────
        # One logit per candidate delay δ ∈ {0, …, max_delay}.
        # softmax(delay_logits) is the learned probability mass over delays.
        # Initialised to uniform (all zeros → softmax = 1/max_delay+1).
        self.delay_logits = nn.Parameter(torch.zeros(max_delay + 1))

        # ── Cross-attention projections ─────────────────────────────────────
        self.q_proj    = nn.Linear(d_model, d_model, bias=False)
        self.k_proj    = nn.Linear(d_model, d_model, bias=False)
        self.v_proj    = nn.Linear(d_model, d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _lag(x: torch.Tensor, delta: int) -> torch.Tensor:
        """
        Lag x by `delta` steps along the K (chunk-time) dimension.

            lag(x, δ)[b, k, :] = x[b, k-δ, :]   for k >= δ
                                = 0               for k <  δ

        Interpretation: at neural chunk k the brain is processing the
        stimulus content that was presented δ steps earlier.
        """
        if delta == 0:
            return x
        B, K, d = x.shape
        pad = x.new_zeros(B, delta, d)
        return torch.cat([pad, x[:, : K - delta, :]], dim=1)

    def _latency_correct(self, stimulus_seq: torch.Tensor) -> torch.Tensor:
        """
        Soft latency-corrected stimulus:

            LC[b, k, :] = Σ_{δ=0}^{max_delay}  w[δ] · stimulus[b, k-δ, :]

        When the learned w is peaked at δ*, LC[k] ≈ stimulus[k-δ*],
        i.e. the stimulus content that the neural chunk at position k is
        actually responding to.
        """
        w = F.softmax(self.delay_logits, dim=0)          # (max_delay+1,)
        return sum(
            w[d] * self._lag(stimulus_seq, d)
            for d in range(self.max_delay + 1)
        )                                                 # (B, K, d_model)

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        neural_seq:   torch.Tensor,
        stimulus_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        neural_seq   : (B, K, d_model)
        stimulus_seq : (B, K, d_model)

        Returns
        -------
        (B, K, d_model)  latency-aligned neural representation
        """
        B, K, _ = neural_seq.shape

        # Step 1 — build latency-corrected keys/values from the stimulus
        lc = self._latency_correct(stimulus_seq)          # (B, K, d_model)

        # Step 2 — multi-head cross-attention
        #   Q  : from neural  (what the brain is doing *now*)
        #   K,V: from latency-corrected stimulus (what it is *responding to*)
        Q  = self.q_proj(neural_seq)                      # (B, K, d_model)
        Kp = self.k_proj(lc)
        V  = self.v_proj(lc)

        def to_heads(t: torch.Tensor) -> torch.Tensor:
            # (B, K, d_model) → (B, n_heads, K, d_head)
            return t.view(B, K, self.n_heads, self.d_head).transpose(1, 2)

        Q, Kp, V = to_heads(Q), to_heads(Kp), to_heads(V)

        attn = (Q @ Kp.transpose(-2, -1)) / self.scale    # (B, h, K, K)
        attn = self.attn_drop(attn.softmax(dim=-1))
        out  = attn @ V                                    # (B, h, K, d_head)

        out = out.transpose(1, 2).contiguous().view(B, K, self.d_model)
        return self.out_proj(out)

    # ── Diagnostic properties ────────────────────────────────────────────────

    @property
    def learned_delay(self) -> torch.Tensor:
        """Learned soft delay distribution over {0, …, max_delay} (CPU, detached)."""
        return F.softmax(self.delay_logits, dim=0).detach().cpu()

    @property
    def expected_delay(self) -> float:
        """Expected delay in chunk steps: E[δ] = Σ δ · w[δ]."""
        w = self.learned_delay
        return (w * torch.arange(self.max_delay + 1, dtype=w.dtype)).sum().item()

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"max_delay={self.max_delay}"
        )


# ── Convenience: latency-aware InfoNCE loss ──────────────────────────────────

def lata_infonce_loss(
    neural_enc:   torch.Tensor,
    lata:         LATA,
    stimulus_seq: torch.Tensor,
    temperature:  float = 0.07,
) -> torch.Tensor:
    """
    Latency-aware InfoNCE loss.

    Drives LATA to learn the correct delay by maximising the similarity
    between each neural chunk and the latency-corrected stimulus chunk
    at the SAME temporal position, while repelling all other positions.

    How it works
    ------------
    If neural[k] = f(stimulus[k - δ_true]) + noise, then when LATA has
    learned δ ≈ δ_true, latency_corrected[k] ≈ stimulus[k - δ_true] ≈
    neural[k].  The InfoNCE loss then has a clear positive pair (k, k) and
    pushes the delay weights toward δ_true via the gradient through
    softmax(delay_logits).

    Parameters
    ----------
    neural_enc   : (B, K, d) — encoded neural chunks (after EEG encoder)
    lata         : LATA      — the module (delay_logits are trained here)
    stimulus_seq : (B, K, d) — stimulus chunk features (e.g. video)
    temperature  : float

    Returns
    -------
    scalar loss
    """
    B, K, d = neural_enc.shape

    # Latency-corrected stimulus (gradient flows through delay_logits here)
    lc = lata._latency_correct(stimulus_seq)              # (B, K, d)

    # Flatten to (B*K, d) and L2-normalise
    n_flat  = F.normalize(neural_enc.reshape(B * K, d), dim=-1)
    lc_flat = F.normalize(lc.reshape(B * K, d), dim=-1)

    # Pairwise cosine similarity: (B*K) × (B*K)
    logits = n_flat @ lc_flat.T / temperature

    # Diagonal = positive pairs (chunk k matched to same chunk k after correction)
    labels = torch.arange(B * K, device=neural_enc.device)
    return F.cross_entropy(logits, labels)
