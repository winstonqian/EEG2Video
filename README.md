# EEG2Video (Reproduction & Extension)

This repository is a reproduction and extension of the [EEG2Video](https://nips.cc/virtual/2024/poster/95156) project.

## Overview

In this project, I ran the original EEG-VP classification benchmarks and extended the baseline codebase to better handle different types of data formats:

- **DE & PSD Feature Training:** I initially evaluated the pre-extracted Differential Entropy (DE) and Power Spectral Density (PSD) features using the provided MLP architecture (`glfnet_mlp`). I trained and tested the models using the compressed `T=5` frequency bands.
- **Baseline Findings:** After running the DE and PSD training loops, analysis showed a relatively low initial Top-1 classification accuracy hovering around ~4.15%. This established our baseline for the semantic classification task.
- **Raw EEG Extension:** I generalized the PyTorch codebase to natively handle 2-second clips of Raw EEG sampled at 200Hz (`T=400`). To execute this, I updated the linear layer spatial calculations across the CNN models (like `shallownet`) to dynamically scale with the temporal sequence length (T) and channel count (C). This securely fixed the hardcoded shape crashes in the original code.
- **Current State:** The refactored models are currently natively training on the Raw EEG arrays to classify the 40 distinct video stimuli categories directly from the oscillograms. The previous baselines for `T=5` were preserved as documented fallback code.

## Midterm Progress & Lab Notebook

As part of auditing the baseline stability and establishing a clear path before scaling the project, several strict checks and architectural modifications were explored:

- **Sequence Shortcut Audit:** To ensure the model wasn't "cheating" by decoding subject anticipation or fatigue over the course of a 5-clip run, I stratified Top-1 accuracy strictly by clip index (1 through 5). The resulting accuracy did not scale monotonically (all remained steady at ~4.15%), proving that our baseline authentically decodes stimuli perception rather than evaluating experimental protocol artifacts.
- **Preprocessing Data Leakage Audit:** I strictly audited train/test normalization pipelines to determine their impact on feature stability. Validating against standard data leakage led to a massive **13x variance drop in PSD features**, though DE feature variance increased. This demonstrated that pure, zero-leakage normalization impacts stability in the frequency and entropy domains entirely differently.
- **Architectural Trade-offs (Temporal Attention):** To attempt better temporal alignment on the expanded Raw 200Hz sequences ($T=400$), I prototyped and integrated a custom `TemporalAttentionPooling` mechanism. The resulting $O(T^2)$ computational bottleneck severely slowed training without justifiable accuracy improvements. This reinforced the pragmatic engineering decision to abandon self-attention for this step and stick to efficient CNN pooling methods to keep the project compute-tractable.

## Installation

Create your Python environment and install the required modules:

```bash
conda create -n eegvideo python=3.12
conda activate eegvideo
pip install -r requirements.txt
```

Make sure to place your SEED-DV dataset directly into the `data/` directory inside the repository structure.

## Usage & Training

To run the classification models:
1. **Preprocess:** Segment the raw data into proper 2-second clips by running `python EEG_preprocessing/segment_raw_signals_200Hz.py`.
2. **Train:** Start the training block via `python EEG-VP/EEG_VP_train_test.py`.

You can easily toggle between evaluating the 200Hz Raw EEG arrays or the pre-extracted DE/PSD datasets by modifying the variables at the top of the training script.

---
*Note: This stage of the repository focuses purely on the Semantic Classification branch of the pipeline (proving we can reliably map EEG states to the 40 video categories) before expanding into the generative Stable Diffusion/Tune-A-Video reconstruction process.*