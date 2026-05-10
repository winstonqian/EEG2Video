"""
Subject Scaling Curve: Cross-Subject R@1 vs Number of Training Subjects.

Trains NeuroCLIP with N={2,4,6,8,10,15,20} training subjects,
evaluates on a fixed held-out test subject. 3 random seeds per N.

General question: How many brains do you need to decode a new brain?
This scaling law applies to any EEG-CLIP alignment system.

Run from EEG2Video/:
    python neuroclip/subject_scaling.py
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from dataset import GT_LABEL
from models_neuroclip import EEGEncoder

DE_DATA_DIR  = "data/DE_1per1s"
BOTH_CONC    = "neuroclip/clip_concept_both_embs_v2.pt"
RESULTS_DIR  = "neuroclip/results"
FIGURES_DIR  = "neuroclip/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

N_CONCEPTS, N_CLIPS, N_SESSIONS = 40, 5, 7
N_TRAINING   = [2, 4, 6, 8, 10, 15, 20]
N_SEEDS      = 3
EPOCHS       = 100     # slightly fewer for speed; same LR schedule
BATCH_SIZE   = 128
LR           = 3e-4
TEMPERATURE  = 0.07

# Fixed test subject — use the one with median within-subject performance
TEST_SUB     = "sub10"


def build_gallery(concept_embs, device):
    g = torch.zeros(40, 512)
    c = torch.zeros(40)
    for s in range(N_SESSIONS):
        for pos in range(N_CONCEPTS):
            cid = int(GT_LABEL[s, pos])
            g[cid] += concept_embs[s, pos]
            c[cid] += 1
    return F.normalize(g / c.clamp(min=1).unsqueeze(1), dim=-1).to(device)


def load_eeg(sub_name):
    p = os.path.join(DE_DATA_DIR, f"{sub_name}.npy")
    raw = np.load(p)
    n_s,n_c,n_cl,n_seg,n_ch,n_b = raw.shape
    return raw.mean(axis=3).reshape(n_s, n_c*n_cl, n_ch, n_b), n_ch, n_b


def make_train_samples(sub_names, concept_ids):
    samples = []
    for sub in sub_names:
        eeg_all, n_ch, n_b = load_eeg(sub)
        for sess in range(N_SESSIONS):
            flat = eeg_all[sess].reshape(200,-1)
            norm = StandardScaler().fit_transform(flat).reshape(200, n_ch, n_b)
            for ci in range(200):
                cid = int(concept_ids[sess, ci])
                samples.append((torch.tensor(norm[ci],dtype=torch.float32),
                                torch.tensor(cid, dtype=torch.long)))
    return samples


def evaluate(model, test_eeg_all, concept_ids, gallery, device):
    model.eval()
    sess_r1s = []
    n_ch, n_b = test_eeg_all.shape[2], test_eeg_all.shape[3]
    for sess in range(N_SESSIONS):
        flat = test_eeg_all[sess].reshape(200,-1)
        norm = StandardScaler().fit_transform(flat).reshape(200, n_ch, n_b)
        eeg_t = torch.tensor(norm, dtype=torch.float32).to(device)
        with torch.no_grad():
            embs = model(eeg_t)
        true_cids = torch.tensor(concept_ids[sess], dtype=torch.long, device=device)
        preds = (embs @ gallery.T).argmax(1)
        sess_r1s.append((preds==true_cids).float().mean().item())
    return float(np.mean(sess_r1s))


def train_and_eval(train_subs, test_sub_name, concept_embs, concept_ids, device):
    gallery = build_gallery(concept_embs, device)
    samples = make_train_samples(train_subs, concept_ids)
    loader  = DataLoader(samples, batch_size=BATCH_SIZE, shuffle=True,
                         drop_last=True, num_workers=0)
    n_ch, n_b = samples[0][0].shape
    model = EEGEncoder(n_channels=n_ch, n_time=n_b, embed_dim=512).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    for epoch in range(EPOCHS):
        model.train()
        for eeg_b, cid_b in loader:
            eeg_b,cid_b = eeg_b.to(device), cid_b.to(device)
            opt.zero_grad()
            logits = model(eeg_b) @ gallery.T / TEMPERATURE
            F.cross_entropy(logits, cid_b).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    test_eeg_all, _, _ = load_eeg(test_sub_name)
    test_cids = np.repeat(GT_LABEL, N_CLIPS, axis=1)
    return evaluate(model, test_eeg_all, test_cids, gallery, device)


def main():
    device = (torch.device("mps")  if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}  |  Test subject: {TEST_SUB}  |  Chance: {1/40:.4f}")

    concept_embs = torch.load(BOTH_CONC, weights_only=True)  # (7,40,512)
    concept_ids  = np.repeat(GT_LABEL, N_CLIPS, axis=1)      # (7,200)

    all_subs = sorted([f.replace(".npy","") for f in os.listdir(DE_DATA_DIR)
                       if f.endswith(".npy") and not f.startswith(TEST_SUB)])

    results = {}
    for n in N_TRAINING:
        seeds_r1 = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed * 100 + n)
            train_subs = rng.choice(all_subs, size=n, replace=False).tolist()
            t0 = time.time()
            r1 = train_and_eval(train_subs, TEST_SUB, concept_embs, concept_ids, device)
            seeds_r1.append(r1)
            print(f"  N={n:2d}  seed={seed}  R@1={r1:.4f}  ({time.time()-t0:.0f}s)")
        results[n] = {"mean": float(np.mean(seeds_r1)),
                      "std":  float(np.std(seeds_r1)),
                      "runs": seeds_r1}
        print(f"  N={n:2d}  → mean={np.mean(seeds_r1)*100:.2f}% ± {np.std(seeds_r1)*100:.2f}%\n")

    # Figure
    import matplotlib.pyplot as plt
    ns    = N_TRAINING
    means = [results[n]["mean"]*100 for n in ns]
    stds  = [results[n]["std"]*100  for n in ns]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ns, means, yerr=stds, fmt="o-", color="#7030a0",
                linewidth=2.5, markersize=8, capsize=6, label="Cross-subject R@1")
    ax.axhline(2.5, color="gray", linestyle="--", linewidth=1.5, label="Chance (2.5%)")
    ax.axhline(4.60, color="#70ad47", linestyle="--", linewidth=1.5,
               label="Within-subject ceiling (4.6%)")
    ax.fill_between(ns, [m-s for m,s in zip(means,stds)],
                        [m+s for m,s in zip(means,stds)],
                    color="#7030a0", alpha=0.15)
    ax.set_xlabel("Number of Training Subjects (N)", fontsize=12)
    ax.set_ylabel("Cross-Subject Concept R@1 (%)", fontsize=12)
    ax.set_title(f"Subject Scaling: How Many Brains Does NeuroCLIP Need?\n"
                 f"(Test subject: {TEST_SUB}, {N_SEEDS} seeds per N, {EPOCHS} epochs)",
                 fontsize=11)
    ax.set_xticks(ns)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(means) + max(stds) + 1.5)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "F14_subject_scaling.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved → {path}")

    with open(os.path.join(RESULTS_DIR, "results_subject_scaling.json"), "w") as f:
        json.dump({"n_values": N_TRAINING, "results": results,
                   "test_sub": TEST_SUB, "n_seeds": N_SEEDS}, f, indent=2)
    print(f"Saved → {RESULTS_DIR}/results_subject_scaling.json")

    print(f"\n=== Subject Scaling Results ===")
    for n in ns:
        print(f"  N={n:2d}: {results[n]['mean']*100:.2f}% ± {results[n]['std']*100:.2f}%")


if __name__ == "__main__":
    main()
