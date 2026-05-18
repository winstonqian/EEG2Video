# EEG2Video Reproduction + NeuroCLIP/LATA Extensions

This repository contains our MAS.S60 / 6.S985 final project code for studying
EEG-to-video decoding on SEED-DV.

The original EEG2Video benchmark asks whether 62-channel EEG can decode the
visual concept, color, and motion of short natural video clips. Our project uses
that benchmark as a starting point, but focuses on two diagnostic questions:

1. **Which visual concepts are most reliably decodable from EEG?**
2. **When does the neural evidence for a video event arrive relative to the
   stimulus?**

The final report contribution is **NeuroCLIP + LATA: Action Semantics and
Latency-Aware Alignment for EEG-to-Video Decoding**.

---

## What Is In This Repo

### Baseline reproduction and validity audit

We reproduced EEG2Video-style 40-way within-subject concept classification and
audited the baseline for common EEG benchmark issues:

- train/test normalization leakage,
- run-position shortcuts across the five same-concept clips in each block,
- fold-level variance under DE and PSD features,
- raw 200 Hz EEG compatibility.

These checks live mainly in:

```text
EEG-VP/
EEG_preprocessing/
analysis/
output_dir/
```

### NeuroCLIP

NeuroCLIP maps EEG features into a frozen CLIP concept space and evaluates
40-way concept retrieval. This turns CLIP from a downstream generative prior
into an error-analysis tool for asking which SEED-DV concepts are decodable.

Main finding: **CLIP geometry does not explain EEG decodability.** Concepts that
are isolated in CLIP space are not systematically easier to decode. Activity-rich
concepts such as sports, music, and people are substantially more decodable than
passive scenes.

Important files:

```text
neuroclip/
├── train_neuroclip.py
├── models_neuroclip.py
├── dataset.py
├── concept_decodability.py
├── category_r1_analysis.py
├── frequency_band_profile.py
├── clip_confusion_correlation.py
├── session_category_interaction.py
├── optimal_concept_subset.py
└── figures/
```

### LATA

LATA (Latency-Aware Temporal Alignment) learns a soft distribution over
candidate EEG-video delays during contrastive chunk alignment. It tests whether
EEG-video alignment should assume zero lag or account for biological response
latency.

Main finding: **all 20 SEED-DV subjects prefer a nonzero delay.** Fourteen
subjects peak at a two-chunk delay, six peak at a one-chunk delay, and no
subject peaks at zero lag.

Important files:

```text
lata/
├── lata.py
├── synthetic_validation.py
├── train_lata_seeddv.py
├── train_all_subjects.py
├── plot_seeddv_results.py
├── lata_synthetic_validation.png
└── lata_all_subjects_results.png
```

---

## Key Results

| Result | Value |
|---|---:|
| Supervised DE baseline Top-1 | 4.37% ± 2.64% |
| Supervised DE baseline Top-5 | 17.17% ± 5.44% |
| NeuroCLIP text+image Recall@1 | **4.60% ± 2.70%** |
| NeuroCLIP text+image Recall@5 | **17.47% ± 5.14%** |
| Chance Recall@1 / Recall@5 | 2.50% / 12.50% |
| Activity-rich concept Recall@1 | **6.79%** |
| Passive concept Recall@1 | 3.83% |
| Activity vs. passive test | t = 6.72, p = 5.95e-8 |
| CLIP isolation vs. EEG Recall@1 | r = 0.036, p = 0.827 |
| LATA peak delay counts | 14/20 at δ=2, 6/20 at δ=1 |
| LATA expected delay | 1.58 ± 0.05 chunks ≈ 790 ms |

---

## Installation

Create a Python environment and install dependencies:

```bash
conda create -n eegvideo python=3.12
conda activate eegvideo
pip install -r requirements.txt
```

The SEED-DV data and extracted features are not committed to this repository.
Place the dataset/features in the expected local directories before running the
training scripts.

---

## Common Workflows

### Run EEG2Video-style classification

```bash
python EEG-VP/EEG_VP_train_test.py
```

The training script can be configured for DE/PSD features or raw 200 Hz EEG,
depending on the local feature files and script variables.

### Generate baseline audit summaries

```bash
python analysis/analyze_all.py
python analysis/generate_summary_tables.py
```

### Train NeuroCLIP

```bash
python neuroclip/train_neuroclip.py
```

Useful analysis scripts after training:

```bash
python neuroclip/concept_decodability.py
python neuroclip/category_r1_analysis.py
python neuroclip/frequency_band_profile.py
python neuroclip/clip_confusion_correlation.py
```

### Run LATA synthetic validation and SEED-DV training

```bash
python lata/synthetic_validation.py
python lata/train_all_subjects.py
python lata/plot_seeddv_results.py
```

---

## Repository Map

```text
.
├── EEG-VP/                  # EEG2Video-style classification training
├── EEG2Video/               # Original video generation / diffusion code
├── EEG_preprocessing/       # Raw EEG segmentation and DE/PSD extraction
├── analysis/                # Baseline audit scripts and summary tables
├── assets/                  # Original project assets
├── dataset/                 # Dataset metadata and channel layouts
├── lata/                    # Latency-aware temporal alignment experiments
├── neuroclip/               # CLIP-aligned EEG retrieval and analyses
├── output_dir/              # Local checkpoints/results
└── requirements.txt
```

---

## Final Takeaway

The project does not claim that EEG-to-video decoding is solved. Instead, it
shows that SEED-DV contains weak but meaningful semantic signal, and that this
signal is both **selective** and **delayed**:

- selective, because activity-rich concepts decode better than passive scenes;
- delayed, because learned EEG-video alignment consistently avoids zero lag.

Future EEG-to-video benchmarks should report concept-level decodability,
protocol audits, and latency-aware alignment diagnostics alongside aggregate
accuracy.
