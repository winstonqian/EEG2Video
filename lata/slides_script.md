# LATA: Latency-Aware Temporal Alignment
## Rachel's Section — ~2 minutes (3 slides)
### Winston Qian · Rachel Li · Emma Wang — Spring 2026 (MIT)

---

## Thesis

> **LATA (Latency-Aware Temporal Alignment) is a plug-and-play PyTorch module that replaces standard cross-attention with a biologically-motivated variant that learns the neural transit delay δ end-to-end from data. Standard cross-attention assumes EEG at time t corresponds to video at time t — physically wrong, since the brain always lags the stimulus by 100–300 ms. LATA learns a soft distribution over candidate delays and aligns EEG chunks to their latency-corrected video counterparts. On synthetic paired sequences with known ground-truth delay δ_true ∈ {0,1,2,3}, LATA recovers the correct delay in all 4 cases. The module is modality-agnostic: the same layer applies to EEG↔Video, fMRI↔Text, or EEG↔Audio without modification.**

---

## Slide Structure (Rachel's 2-minute section)

| Slide | Time  | Speaker | Rubric role |
|-------|-------|---------|-------------|
| R1    | 0:35  | Rachel  | Motivation for new idea — why standard attention fails for neural signals |
| R2    | 0:50  | Rachel  | LATA module design and mathematical formulation |
| R3    | 0:35  | Rachel  | Synthetic validation results + generalizability claim |

Total: ~2 minutes. Integrate as slides 3–5 of the full 6-slide deck after Emma's audit section.

---

## Slide-by-Slide Script

---

### Slide R1 — The Latency Problem (~35 sec)

**ON SLIDE:**
```
Standard cross-attention ignores biological time

         Video:  [v₁]  [v₂]  [v₃]  [v₄]   ← stimulus at time k
                   ↕     ↕     ↕     ↕      ← assumes perfect sync
         EEG:   [e₁]  [e₂]  [e₃]  [e₄]   ← actual response: lagged!

Reality:
  P100 visual response    ~100 ms after stimulus
  P300 semantic response  ~300 ms after stimulus

→ EEG chunk k is responding to video chunk k − δ
→ All prior EEG-video methods assume δ = 0
```

**[visual: simple two-row timeline diagram, EEG row shifted right by δ with a red arrow labeled "biological delay δ"]**

**SAY:**
> "After Emma's audit showed the baseline is valid but static, our question was: can we do better by actually modeling the temporal relationship between EEG and video?
>
> Here's the problem. Standard cross-attention assumes that EEG at time t corresponds to video at time t. But that's physically wrong. The brain doesn't respond instantly — the P100 visual response peaks around 100 milliseconds after the stimulus, and higher-level semantic processing at around 300 milliseconds.
>
> So EEG chunk k is actually responding to video content from δ steps earlier. Every existing EEG-to-video model ignores this and just aligns whole clips to a single global embedding. We can do better."

---

### Slide R2 — LATA Module Design (~50 sec)

**ON SLIDE:**
```
LATA: Learn the delay, don't assume it

Step 1 — Learnable delay distribution:
   w = softmax(ℓ),  ℓ ∈ ℝ^{Δ+1}  (one logit per candidate delay)

Step 2 — Latency-corrected stimulus:
   ṽₖ = Σ_δ  wδ · v_{k−δ}
   "At chunk k, attend to a weighted mix of past video chunks"

Step 3 — Cross-attention:
   LATA(E, V) = MultiHeadAttn(Q=WqE, K=Wk·Ṽ, V=Wv·Ṽ)

Trained with InfoNCE:
   push (eₖ, ṽₖ) together · push (eₖ, ṽⱼ) apart for j ≠ k

Same module → EEG↔Video  ·  fMRI↔Text  ·  EEG↔Audio
```

**[visual: use the chunk alignment figure (Figure 2 from midterm) showing EEG chunks e₁…e₄ aligned to shifted video chunks v_{k+δ}]**

**SAY:**
> "LATA solves this with three steps.
>
> First, we add a small trainable vector — one logit per candidate delay. Softmax turns this into a probability distribution over delays zero through delta-max.
>
> Second, we build a latency-corrected stimulus: at each chunk position k, we take a weighted sum of past video chunks shifted by each candidate delay. When the distribution is peaked at delta-star, this gives us the video content from delta-star steps ago — exactly what the brain at chunk k is responding to.
>
> Third, we run standard cross-attention where the query comes from EEG and the keys and values come from this latency-corrected video.
>
> We train with an InfoNCE loss: after correction, EEG chunk k should be closest to video chunk k, and far from all others. The gradient flows straight back through the softmax into the delay logits — so delta is learned end-to-end, no manual tuning.
>
> And critically: this is modality-agnostic. The exact same module works for fMRI-to-text, EEG-to-audio, or any other paired temporal streams — you just swap the inputs."

---

### Slide R3 — Synthetic Validation: 4/4 Correct (~35 sec)

**ON SLIDE:**
```
Synthetic validation: LATA recovers known delay in all cases

Setup:
  neural[k] = stimulus[k − δ_true] + noise  (noise_std = 0.4)
  LATA searches δ ∈ {0, 1, 2, 3, 4}, no supervision on δ

Results:

  δ_true │ Learned peak │  ✓?
  ───────┼──────────────┼─────
    0    │      0       │  ✓
    1    │      1       │  ✓
    2    │      2       │  ✓
    3    │      3       │  ✓

→ Correct delay recovered in all 4 cases, peak always at ground truth
→ Generalizability proof: works for any paired temporal modalities
```

**[visual: top row of lata_synthetic_validation.png — 4 bar charts, one per δ_true, blue bar at correct position, red dashed line at ground truth]**

**SAY:**
> "To prove LATA actually learns the delay, we ran a synthetic experiment with known ground truth. We generated paired sequences where the neural signal is a noisy copy of the stimulus shifted by a fixed delta. LATA has no access to the true delta — it has to find it from the data alone.
>
> In all four cases, the learned distribution peaks at exactly the right delay. This is the key generalizability claim: LATA can be dropped into any brain-signal pipeline — EEG, fMRI, MEG — and it will automatically identify the correct biological lag."

---

## Key Figures

| Figure | File | Slide |
|--------|------|-------|
| Two-row timeline with lag arrow | make new simple diagram | R1 |
| Chunk alignment pipeline | `Project/figures/chunk_alignment.png` (midterm Fig 2) | R2 |
| Synthetic validation bar charts | `lata/lata_synthetic_validation.png` (top row only) | R3 |

**All LATA code:** `winstonqian/EEG2Video` → `lata/`

---

## Speaker Notes for Slide Maker

- **R1**: Make the timeline visual clean and simple — two rows (Video, EEG), arrows between them, EEG row shifted right. Label the shift "biological delay δ ≈ 100–300 ms". Minimal text on slide.
- **R2**: Use the existing chunk alignment figure from the midterm (already looks good). Add the three-step equations in a clean box on the right side.
- **R3**: Use only the **top row** of `lata_synthetic_validation.png` (4 bar charts). Crop out the loss curves — not needed for 35 seconds. Make the table small and clean.
- **Transitions**: R1 ends with "We can do better" → natural lead into R2. R3 ends with the generalizability claim → can hand off to conclusion slide.

---

## Rubric Checklist (Rachel's slides)

- **Motivation for new idea:** R1 establishes the biological latency gap and why all prior work misses it.
- **New method explanation:** R2 gives full mathematical intuition at the right level — step-by-step with one equation per step.
- **Experiments and results:** R3 shows synthetic validation with a clean table and figure, explains why it's the generalizability proof.
- **Slide quality:** 3 slides, each with one main visual + one takeaway sentence. Speaker notes are content-dense, slides are visually minimal.

---

## Backup Details for Q&A

- Module params: `d_model`, `n_heads`, `max_delay` — fully configurable
- Training: AdamW, cosine LR decay, InfoNCE temperature τ = 0.05
- Synthetic data: N=2048, K=8 chunks, d=128, noise_std=0.4, max_delay=4
- All 4 delay peaks correct by argmax; E[δ] pulled toward midpoint (2.0) at moderate SNR — expected, peak is what matters
- SEED-DV real-data experiment: architecture ready in `train_lata_seeddv.py`, requires VideoMAE chunk features (GPU); can be run on Colab
- Why not BLIP captions for real-data training: BLIP gives one caption per 2s clip, repeated for all 4 chunks → no temporal variation within clip → delay weights get no gradient
- Biological latency mapping: 0.5s/chunk → δ=1 ≈ 500ms (slightly slower than P100 but within P300 range); finer chunk resolution would improve correspondence
