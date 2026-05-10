"""
Cross-subject NeuroCLIP — leave-one-subject-out evaluation.

For each held-out test subject:
  - Train on all 7 sessions × 19 other subjects (pooled, within-subject normalised)
  - Test on all 7 sessions of the held-out subject (never seen during training)

This directly tests whether CLIP-space EEG alignment generalises across brains,
not just across sessions of the same brain.

Uses NeuroCLIP-Both supervision (text + image averaged) — best performing condition.

Run from EEG2Video/:
    python neuroclip/train_neuroclip_crosssub.py
"""

import os, sys, json, argparse
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
BOTH_EMB     = "neuroclip/clip_both_embs_v2.pt"
CONCEPT_BOTH = "neuroclip/clip_concept_both_embs_v2.pt"
RESULTS_DIR  = "neuroclip/results"
N_CONCEPTS, N_CLIPS, N_SESSIONS = 40, 5, 7


# ---------------------------------------------------------------------------
# Dataset — pools multiple subjects, within-subject normalisation
# ---------------------------------------------------------------------------

class CrossSubDataset(Dataset):
    def __init__(self, sub_paths, both_embs, concept_embs, concept_ids):
        """
        sub_paths   : list of .npy paths for training subjects
        both_embs   : (7, 200, 512) CLIP both embeddings (shared across subjects)
        concept_embs: (7, 40,  512) concept-mean both embeddings
        concept_ids : (7, 200) concept_id per clip per session
        """
        self.samples = []
        for sub_path in sub_paths:
            raw = np.load(sub_path)
            # DE: (7,40,5,2,62,5) → average sub-segments → (7,200,62,5)
            n_sess, n_conc, n_cl, n_seg, n_ch, n_bands = raw.shape
            eeg = raw.mean(axis=3).reshape(n_sess, n_conc * n_cl, n_ch, n_bands)

            for sess in range(N_SESSIONS):
                flat = eeg[sess].reshape(200, -1)
                scaler = StandardScaler()
                norm = scaler.fit_transform(flat).reshape(200, n_ch, n_bands)
                for clip_idx in range(200):
                    cid = int(concept_ids[sess, clip_idx])
                    self.samples.append({
                        "eeg":         torch.tensor(norm[clip_idx], dtype=torch.float32),
                        "concept_emb": concept_embs[sess, clip_idx // N_CLIPS],
                        "concept_id":  cid,
                    })

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["eeg"], s["concept_emb"], torch.tensor(s["concept_id"], dtype=torch.long)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_gallery(concept_embs_tensor, sess, device):
    g = torch.zeros(40, 512)
    c = torch.zeros(40)
    for pos in range(N_CONCEPTS):
        cid = int(GT_LABEL[sess, pos])
        g[cid] += concept_embs_tensor[sess, pos]
        c[cid] += 1
    return F.normalize(g / c.clamp(min=1).unsqueeze(1), dim=-1).to(device)


def recall_at_k(embs, gallery, true_cids, ks=(1, 5, 10)):
    sim   = embs @ gallery.T
    ranks = sim.argsort(dim=1, descending=True)
    out   = {}
    for k in ks:
        top_k = gallery[ranks[:, :k]]  # unused — use label ranks
        top_k_labels = ranks[:, :k]    # (N, k) indices into gallery = concept_ids
        correct = (top_k_labels == true_cids.unsqueeze(1)).any(dim=1)
        out[k]  = correct.float().mean().item()
    return out


def evaluate_subject(model, sub_path, both_embs, concept_embs, concept_ids, device):
    raw = np.load(sub_path)
    n_sess, n_conc, n_cl, n_seg, n_ch, n_bands = raw.shape
    eeg = raw.mean(axis=3).reshape(n_sess, n_conc * n_cl, n_ch, n_bands)

    sess_r1s = []
    model.eval()
    for sess in range(N_SESSIONS):
        flat = eeg[sess].reshape(200, -1)
        norm = StandardScaler().fit_transform(flat).reshape(200, n_ch, n_bands)
        eeg_t = torch.tensor(norm, dtype=torch.float32).to(device)

        with torch.no_grad():
            embs = model(eeg_t)                    # (200, 512)

        gallery    = make_gallery(concept_embs, sess, device)   # (40, 512)
        true_cids  = torch.tensor(concept_ids[sess], dtype=torch.long, device=device)
        sim        = embs @ gallery.T
        top1_preds = sim.argmax(dim=1)
        r1 = (top1_preds == true_cids).float().mean().item()
        sess_r1s.append(r1)

    return np.mean(sess_r1s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=150)
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--temperature",type=float, default=0.07)
    p.add_argument("--verbose",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = (torch.device("mps")  if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")
    print(f"Chance: concept R@1={1/40:.4f}")

    both_embs    = torch.load(BOTH_EMB,     weights_only=True)  # (7,200,512)
    concept_embs = torch.load(CONCEPT_BOTH, weights_only=True)  # (7,40,512)
    concept_ids  = np.repeat(GT_LABEL, repeats=5, axis=1)       # (7,200)

    sub_files = sorted([f for f in os.listdir(DE_DATA_DIR) if f.endswith(".npy")])
    print(f"Total subjects found: {len(sub_files)}")

    # Build cross-session concept gallery (train sessions — all sessions, all subjects share same videos)
    gallery_all = torch.zeros(40, 512)
    counts_all  = torch.zeros(40)
    for s in range(N_SESSIONS):
        for pos in range(N_CONCEPTS):
            cid = int(GT_LABEL[s, pos])
            gallery_all[cid] += concept_embs[s, pos]
            counts_all[cid]  += 1
    gallery_all = F.normalize(gallery_all / counts_all.clamp(min=1).unsqueeze(1), dim=-1).to(device)

    all_sub_r1 = []

    for test_idx, test_sub in enumerate(sub_files):
        train_subs = [os.path.join(DE_DATA_DIR, f)
                      for i, f in enumerate(sub_files) if i != test_idx]
        test_path  = os.path.join(DE_DATA_DIR, test_sub)

        print(f"\n=== Test subject: {test_sub} ({test_idx+1}/{len(sub_files)}) ===")
        print(f"  Training on {len(train_subs)} subjects × 7 sessions = "
              f"{len(train_subs)*7*200} clips")

        train_ds = CrossSubDataset(train_subs, both_embs, concept_embs, concept_ids)
        loader   = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=0)

        # Sample one clip to get n_ch, n_time
        sample_eeg, _, _ = train_ds[0]
        n_ch, n_time = sample_eeg.shape

        model = EEGEncoder(n_channels=n_ch, n_time=n_time, embed_dim=512).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for eeg_b, concept_emb_b, cid_b in loader:
                eeg_b         = eeg_b.to(device)
                cid_b         = cid_b.to(device)
                opt.zero_grad()
                emb    = model(eeg_b)                                    # (B,512)
                logits = emb @ gallery_all.T / args.temperature          # (B,40)
                loss   = F.cross_entropy(logits, cid_b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
            sched.step()

            if args.verbose and ((epoch + 1) % 25 == 0 or epoch == 0):
                print(f"  epoch {epoch+1:4d}  loss={total_loss/len(loader):.4f}")

        # Use final epoch — no test-set model selection
        final_r1 = evaluate_subject(model, test_path, both_embs,
                                    concept_embs, concept_ids, device)
        all_sub_r1.append(final_r1)
        print(f"  {test_sub}: cross-subject R@1 = {final_r1:.3f}  (chance 0.025)")

        torch.save(model.state_dict(),
                   os.path.join(RESULTS_DIR, f"crosssub_test{test_sub.replace('.npy','')}_both.pt"))

    print(f"\n=== Cross-Subject Results (NeuroCLIP-Both) ===")
    print(f"Mean R@1 = {np.mean(all_sub_r1)*100:.3f}% ± {np.std(all_sub_r1)*100:.3f}%")
    print(f"Chance   = 2.500%")
    print(f"Within-subject Both = 4.599%")

    results = {
        "per_subject_r1": all_sub_r1,
        "mean_r1": float(np.mean(all_sub_r1)),
        "std_r1":  float(np.std(all_sub_r1)),
        "note": "final-epoch checkpoint only — no test-set model selection",
    }
    with open(os.path.join(RESULTS_DIR, "results_crosssub_both_final_epoch.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {RESULTS_DIR}/results_crosssub_both_final_epoch.json")


if __name__ == "__main__":
    main()
