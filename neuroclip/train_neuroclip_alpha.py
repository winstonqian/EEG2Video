"""
Fusion weight ablation: train NeuroCLIP with different text/image mixing weights.

Creates on-the-fly blended embeddings:  α*text + (1-α)*image
Tests α ∈ {0.0, 0.25, 0.5, 0.75, 1.0} (0.5 = Both, 0.0 = Image-only, 1.0 = Text-only)

Run from EEG2Video/:
    python neuroclip/train_neuroclip_alpha.py
"""

import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from dataset import GT_LABEL
from models_neuroclip import EEGEncoder

DE_DATA_DIR  = "data/DE_1per1s"
TEXT_EMB     = "neuroclip/clip_text_embs_v2.pt"
IMAGE_EMB    = "neuroclip/clip_image_embs_v2.pt"
TEXT_CONC    = "neuroclip/clip_concept_text_embs_v2.pt"
IMAGE_CONC   = "neuroclip/clip_concept_image_embs_v2.pt"
RESULTS_DIR  = "neuroclip/results"
N_CONCEPTS, N_CLIPS, N_SESSIONS = 40, 5, 7
ALPHAS       = [0.0, 0.25, 0.5, 0.75, 1.0]
EPOCHS       = 150
BATCH_SIZE   = 64
LR           = 3e-4
TEMPERATURE  = 0.07


class AlphaDataset(Dataset):
    def __init__(self, sub_path, sess_test, blended_embs, concept_embs, concept_ids):
        """Leave-one-block-out: exclude sess_test."""
        raw = np.load(sub_path)
        n_sess, n_conc, n_cl, n_seg, n_ch, n_bands = raw.shape
        eeg = raw.mean(axis=3).reshape(n_sess, n_conc * n_cl, n_ch, n_bands)
        self.samples = []
        for sess in range(N_SESSIONS):
            if sess == sess_test:
                continue
            flat = eeg[sess].reshape(200, -1)
            norm = StandardScaler().fit_transform(flat).reshape(200, n_ch, n_bands)
            for clip_idx in range(200):
                cid = int(concept_ids[sess, clip_idx])
                self.samples.append({
                    "eeg":        torch.tensor(norm[clip_idx], dtype=torch.float32),
                    "concept_id": cid,
                })

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["eeg"], torch.tensor(s["concept_id"], dtype=torch.long)


def make_gallery(concept_embs, device):
    """Build 40-concept gallery averaged across all sessions."""
    g = torch.zeros(40, 512)
    c = torch.zeros(40)
    for sess in range(N_SESSIONS):
        for pos in range(N_CONCEPTS):
            cid = int(GT_LABEL[sess, pos])
            g[cid] += concept_embs[sess, pos]
            c[cid] += 1
    return F.normalize(g / c.clamp(min=1).unsqueeze(1), dim=-1).to(device)


def evaluate(model, sub_path, concept_embs, concept_ids, device):
    raw = np.load(sub_path)
    n_sess, n_conc, n_cl, n_seg, n_ch, n_bands = raw.shape
    eeg = raw.mean(axis=3).reshape(n_sess, n_conc * n_cl, n_ch, n_bands)
    gallery = make_gallery(concept_embs, device)
    model.eval()
    sess_r1s = []
    for sess in range(N_SESSIONS):
        flat = eeg[sess].reshape(200, -1)
        norm = StandardScaler().fit_transform(flat).reshape(200, n_ch, n_bands)
        eeg_t = torch.tensor(norm, dtype=torch.float32).to(device)
        with torch.no_grad():
            embs = model(eeg_t)
        true_cids = torch.tensor(concept_ids[sess], dtype=torch.long, device=device)
        preds = (embs @ gallery.T).argmax(dim=1)
        sess_r1s.append((preds == true_cids).float().mean().item())
    return np.mean(sess_r1s)


def blend_concept_embs(text_c, image_c, alpha):
    """α*text + (1-α)*image, renormalised."""
    blended = alpha * text_c + (1 - alpha) * image_c
    return F.normalize(blended, dim=-1)


def train_alpha(alpha, text_embs, image_embs, text_conc, image_conc,
                sub_files, concept_ids, device):
    print(f"\n{'='*60}")
    print(f"alpha={alpha:.2f}  (text={alpha:.0%}, image={1-alpha:.0%})")

    concept_embs = blend_concept_embs(text_conc, image_conc, alpha)
    gallery_all  = make_gallery(concept_embs, device)

    per_sub_r1 = []

    for sub_path in sub_files:
        raw = np.load(sub_path)
        n_sess, n_conc, n_cl, n_seg, n_ch, n_bands = raw.shape
        eeg_full = raw.mean(axis=3).reshape(n_sess, n_conc * n_cl, n_ch, n_bands)

        fold_r1s = []
        for sess_test in range(N_SESSIONS):
            # Build train set (all sessions except sess_test)
            train_samples = []
            for sess in range(N_SESSIONS):
                if sess == sess_test:
                    continue
                flat = eeg_full[sess].reshape(200, -1)
                norm = StandardScaler().fit_transform(flat).reshape(200, n_ch, n_bands)
                for clip_idx in range(200):
                    cid = int(concept_ids[sess, clip_idx])
                    train_samples.append((
                        torch.tensor(norm[clip_idx], dtype=torch.float32),
                        torch.tensor(cid, dtype=torch.long)
                    ))

            loader = DataLoader(train_samples, batch_size=BATCH_SIZE,
                                shuffle=True, drop_last=True)

            sample_eeg = train_samples[0][0]
            n_ch_m, n_time = sample_eeg.shape
            model = EEGEncoder(n_channels=n_ch_m, n_time=n_time, embed_dim=512).to(device)
            opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

            for epoch in range(EPOCHS):
                model.train()
                for eeg_b, cid_b in loader:
                    eeg_b = eeg_b.to(device)
                    cid_b = cid_b.to(device)
                    opt.zero_grad()
                    emb    = model(eeg_b)
                    logits = emb @ gallery_all.T / TEMPERATURE
                    loss   = F.cross_entropy(logits, cid_b)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()

            # Evaluate on held-out session
            model.eval()
            flat_test = eeg_full[sess_test].reshape(200, -1)
            norm_test  = StandardScaler().fit_transform(flat_test).reshape(200, n_ch_m, n_bands)
            eeg_t      = torch.tensor(norm_test, dtype=torch.float32).to(device)
            true_cids  = torch.tensor(concept_ids[sess_test], dtype=torch.long, device=device)
            with torch.no_grad():
                embs  = model(eeg_t)
            preds = (embs @ gallery_all.T).argmax(dim=1)
            fold_r1s.append((preds == true_cids).float().mean().item())

        sub_r1 = np.mean(fold_r1s)
        per_sub_r1.append(sub_r1)
        print(f"  {os.path.basename(sub_path)}: R@1={sub_r1:.3f}")

    mean_r1 = np.mean(per_sub_r1)
    std_r1  = np.std(per_sub_r1)
    print(f"  alpha={alpha:.2f}  Mean R@1={mean_r1*100:.3f}% ± {std_r1*100:.3f}%")
    return {"alpha": alpha, "mean_r1": mean_r1, "std_r1": std_r1,
            "per_sub_r1": per_sub_r1}


def main():
    device = (torch.device("mps")  if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    text_embs  = torch.load(TEXT_EMB,   weights_only=True)   # (7,200,512)
    image_embs = torch.load(IMAGE_EMB,  weights_only=True)   # (7,200,512)
    text_conc  = torch.load(TEXT_CONC,  weights_only=True)   # (7,40,512)
    image_conc = torch.load(IMAGE_CONC, weights_only=True)   # (7,40,512)
    concept_ids = np.repeat(GT_LABEL, repeats=N_CLIPS, axis=1)  # (7,200)

    sub_files = sorted([os.path.join(DE_DATA_DIR, f)
                        for f in os.listdir(DE_DATA_DIR) if f.endswith(".npy")])
    print(f"Subjects: {len(sub_files)}")

    all_results = []
    for alpha in ALPHAS:
        res = train_alpha(alpha, text_embs, image_embs, text_conc, image_conc,
                          sub_files, concept_ids, device)
        all_results.append(res)

    # Save
    out = {f"alpha_{r['alpha']:.2f}": r for r in all_results}
    path = os.path.join(RESULTS_DIR, "results_alpha_ablation.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {path}")

    print("\n=== Alpha Ablation Summary ===")
    for r in all_results:
        print(f"  α={r['alpha']:.2f}  R@1={r['mean_r1']*100:.3f}% ± {r['std_r1']*100:.3f}%")


if __name__ == "__main__":
    main()
