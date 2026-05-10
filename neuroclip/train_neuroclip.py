"""
NeuroCLIP training script — concept-gallery InfoNCE.

Training objective
------------------
For each EEG segment, the positive target is the CLIP concept-mean embedding
(average of the 5 per-clip text embeddings for that concept).  The gallery
contains all 40 concept-mean embeddings.  This eliminates within-concept
false negatives entirely and gives a clean 40-way contrastive signal.

Evaluation
----------
Two retrieval tasks reported on the held-out test session:

1. Concept-gallery retrieval (primary):
   Query EEG vs 40 concept-mean embeddings.
   Success = correct concept is top-K.
   Chance: R@1 = 2.5%, R@5 = 12.5%
   Directly comparable to the classification baseline (4.37% / 17.17%).

2. Clip-gallery retrieval (secondary, harder):
   Query EEG vs all 200 per-clip text embeddings in the session.
   Success = correct clip is top-K.
   Chance: R@1 = 0.5%, R@5 = 2.5%, R@10 = 5%

Run from EEG2Video/:
    python neuroclip/train_neuroclip.py --verbose
    python neuroclip/train_neuroclip.py --feature de --verbose
    python neuroclip/train_neuroclip.py --chunks 4 --verbose
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_subject, NeuroCLIPDataset, GT_LABEL
from models_neuroclip import EEGEncoder, ChunkEEGEncoder, concept_infonce, infonce_loss


# ---------------------------------------------------------------------------
# Retrieval evaluation helpers
# ---------------------------------------------------------------------------

def recall_at_k(query_embs: torch.Tensor, gallery_embs: torch.Tensor,
                query_labels: torch.Tensor, gallery_labels: torch.Tensor,
                ks=(1, 5, 10)):
    """
    For each query, rank gallery items by cosine similarity and check if a
    gallery item with the correct label is in the top-K.

    Works for both concept retrieval (40-class gallery) and clip retrieval
    (200-item gallery where multiple gallery items share the same concept label).
    """
    sim   = query_embs @ gallery_embs.T       # (N_query, N_gallery)
    ranks = sim.argsort(dim=1, descending=True)
    results = {}
    for k in ks:
        top_k_labels = gallery_labels[ranks[:, :k]]   # (N_query, k)
        correct = (top_k_labels == query_labels.unsqueeze(1)).any(dim=1)
        results[k] = correct.float().mean().item()
    return results


# ---------------------------------------------------------------------------
# One fold training
# ---------------------------------------------------------------------------

def train_one_fold(eeg_data, text_embs, concept_embs, concept_ids,
                   train_sessions, val_session, test_session, args, device):

    n_ch, n_time = eeg_data.shape[2], eeg_data.shape[3]

    if args.chunks > 1:
        model = ChunkEEGEncoder(n_channels=n_ch, n_time=n_time,
                                k_chunks=args.chunks, embed_dim=512)
    else:
        model = EEGEncoder(n_channels=n_ch, n_time=n_time, embed_dim=512)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_ds = NeuroCLIPDataset(eeg_data, text_embs, concept_embs, concept_ids, train_sessions)
    val_ds   = NeuroCLIPDataset(eeg_data, text_embs, concept_embs, concept_ids, [val_session])
    test_ds  = NeuroCLIPDataset(eeg_data, text_embs, concept_embs, concept_ids, [test_session])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,  batch_size=200, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=200, shuffle=False)

    # Build a cross-session concept gallery indexed by concept_id (0-39).
    # gallery_by_id[concept_id] = mean of concept embeddings across all training sessions.
    # This eliminates within-concept false negatives entirely: the loss becomes
    # a clean 40-way cross-entropy in embedding space (equivalent to CLIP zero-shot).
    gallery_by_id = torch.zeros(40, 512)
    counts = torch.zeros(40)
    for s in train_sessions:
        for pos in range(40):
            cid = int(GT_LABEL[s, pos])
            gallery_by_id[cid] += concept_embs[s, pos]
            counts[cid] += 1
    gallery_by_id = gallery_by_id / counts.unsqueeze(1)
    gallery_by_id = torch.nn.functional.normalize(gallery_by_id, dim=-1).to(device)  # (40,512)

    # Evaluation galleries (session-specific, same building logic)
    def _make_eval_gallery(session):
        g = torch.zeros(40, 512)
        c = torch.zeros(40)
        for pos in range(40):
            cid = int(GT_LABEL[session, pos])
            g[cid] += concept_embs[session, pos]
            c[cid] += 1
        g = g / c.unsqueeze(1)
        return torch.nn.functional.normalize(g, dim=-1).to(device)

    val_concept_gallery  = _make_eval_gallery(val_session)    # (40,512) indexed by concept_id
    test_concept_gallery = _make_eval_gallery(test_session)
    gallery_label_ids    = torch.arange(40, device=device)    # concept_id 0-39 is its own label

    val_gallery_ids  = gallery_label_ids
    test_gallery_ids = gallery_label_ids

    best_val_r1 = -1
    best_state  = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            eeg_b, _, _, cid_b, _ = batch
            eeg_b = eeg_b.to(device)
            cid_b = cid_b.to(device)

            optimizer.zero_grad()

            if args.chunks > 1:
                eeg_out, _ = model(eeg_b)
            else:
                eeg_out = model(eeg_b)

            # Clean 40-way CE loss: eeg_out vs full concept gallery
            # No false negatives: concept_id uniquely indexes the gallery.
            logits = eeg_out @ gallery_by_id.T / args.temperature  # (B, 40)
            loss   = torch.nn.functional.cross_entropy(logits, cid_b)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_m = evaluate(model, val_loader, val_concept_gallery,
                             val_gallery_ids, text_embs[val_session].to(device),
                             concept_ids[val_session], device, args)
            if val_m["concept"][1] > best_val_r1:
                best_val_r1 = val_m["concept"][1]
                best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if args.verbose:
                c = val_m["concept"]
                cl = val_m["clip"]
                print(f"  epoch {epoch+1:4d}  loss={total_loss/len(train_loader):.4f}"
                      f"  val concept R@1={c[1]:.3f} R@5={c[5]:.3f}"
                      f"  clip R@1={cl[1]:.3f} R@10={cl[10]:.3f}")

    model.load_state_dict(best_state)
    test_m = evaluate(model, test_loader, test_concept_gallery,
                      test_gallery_ids, text_embs[test_session].to(device),
                      concept_ids[test_session], device, args)
    return test_m, model


def evaluate(model, loader, concept_gallery, gallery_ids,
             clip_text_embs, clip_concept_ids, device, args):
    """
    Returns dict with two sub-dicts:
      'concept': Recall@K against 40-class concept gallery
      'clip':    Recall@K against 200-clip text embedding gallery
    """
    model.eval()
    all_eeg   = []
    all_cids  = []

    with torch.no_grad():
        for eeg_b, _, _, cid_b, _ in loader:
            eeg_b = eeg_b.to(device)
            if args.chunks > 1:
                out, _ = model(eeg_b)
            else:
                out = model(eeg_b)
            all_eeg.append(out)
            all_cids.append(cid_b.to(device))

    all_eeg  = torch.cat(all_eeg,  dim=0)   # (200, 512)
    all_cids = torch.cat(all_cids, dim=0)   # (200,) concept ids

    # Concept-gallery retrieval: gallery = 40 concept-mean embeddings
    concept_m = recall_at_k(all_eeg, concept_gallery, all_cids, gallery_ids,
                             ks=(1, 5, 10))

    # Clip-gallery retrieval: gallery = 200 per-clip text embeddings
    clip_gallery = clip_text_embs.to(device)                  # (200, 512)
    clip_labels  = torch.tensor(clip_concept_ids, dtype=torch.long, device=device)  # (200,)
    clip_m = recall_at_k(all_eeg, clip_gallery, all_cids, clip_labels,
                          ks=(1, 5, 10))

    return {"concept": concept_m, "clip": clip_m}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature",     default="raw",  choices=["raw", "de", "psd"])
    p.add_argument("--chunks",      type=int,   default=1)
    p.add_argument("--epochs",      type=int,   default=150)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--sub",         default=None, help="e.g. sub1.npy; omit for all")
    p.add_argument("--output_dir",  default="neuroclip/results")
    p.add_argument("--text_emb",    default="neuroclip/clip_text_embeddings.pt")
    p.add_argument("--concept_emb", default="neuroclip/clip_concept_embeddings.pt")
    p.add_argument("--tag",         default="", help="suffix for output filenames, e.g. 'text' 'image' 'both'")
    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    device = (
        torch.device("mps")  if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}  |  feature: {args.feature}  |  chunks: {args.chunks}")
    print(f"Chance: concept R@1={1/40:.4f} R@5={5/40:.4f}  |  "
          f"clip R@1={1/200:.4f} R@5={5/200:.4f} R@10={10/200:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.feature == "raw":
        data_dir = "data/Segmented_Rawf_200Hz_2s"
    elif args.feature == "de":
        data_dir = "data/DE_1per1s"
    else:
        data_dir = "data/PSD_1per1s"

    sub_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])
    if args.sub:
        sub_files = [f for f in sub_files if args.sub in f]

    metrics_keys = ["concept_r1", "concept_r5", "concept_r10",
                    "clip_r1",    "clip_r5",    "clip_r10"]
    all_metrics  = {k: [] for k in metrics_keys}

    for sub_name in sub_files:
        print(f"\n=== Subject: {sub_name} ===")
        eeg_data, text_embs, concept_embs, concept_ids = load_subject(
            os.path.join(data_dir, sub_name),
            args.text_emb, args.concept_emb,
            feature=args.feature,
        )

        fold_metrics = {k: [] for k in metrics_keys}

        for test_fold in range(7):
            val_fold    = (test_fold - 1) % 7
            train_folds = [i for i in range(7) if i != test_fold and i != val_fold]

            m, model = train_one_fold(
                eeg_data, text_embs, concept_embs, concept_ids,
                train_folds, val_fold, test_fold,
                args, device,
            )

            fold_metrics["concept_r1"].append(m["concept"][1])
            fold_metrics["concept_r5"].append(m["concept"][5])
            fold_metrics["concept_r10"].append(m["concept"][10])
            fold_metrics["clip_r1"].append(m["clip"][1])
            fold_metrics["clip_r5"].append(m["clip"][5])
            fold_metrics["clip_r10"].append(m["clip"][10])

            print(f"  fold {test_fold}:"
                  f"  concept R@1={m['concept'][1]:.3f} R@5={m['concept'][5]:.3f}"
                  f"  |  clip R@1={m['clip'][1]:.3f} R@10={m['clip'][10]:.3f}")

            tag_str    = f"_{args.tag}" if args.tag else ""
            model_path = os.path.join(
                args.output_dir,
                f"{sub_name.replace('.npy','')}_fold{test_fold}_{args.feature}_k{args.chunks}{tag_str}.pt",
            )
            torch.save(model.state_dict(), model_path)

        for k in metrics_keys:
            mean_v = np.mean(fold_metrics[k])
            all_metrics[k].append(mean_v)

        print(f"  {sub_name} mean:"
              f"  concept R@1={np.mean(fold_metrics['concept_r1']):.3f}"
              f"  R@5={np.mean(fold_metrics['concept_r5']):.3f}"
              f"  |  clip R@1={np.mean(fold_metrics['clip_r1']):.3f}"
              f"  R@10={np.mean(fold_metrics['clip_r10']):.3f}")

    print("\n=== Overall (mean ± std across subjects) ===")
    print(f"Concept R@1  = {np.mean(all_metrics['concept_r1']):.4f} ± {np.std(all_metrics['concept_r1']):.4f}  "
          f"(chance {1/40:.4f})")
    print(f"Concept R@5  = {np.mean(all_metrics['concept_r5']):.4f} ± {np.std(all_metrics['concept_r5']):.4f}  "
          f"(chance {5/40:.4f})")
    print(f"Concept R@10 = {np.mean(all_metrics['concept_r10']):.4f} ± {np.std(all_metrics['concept_r10']):.4f}  "
          f"(chance {10/40:.4f})")
    print(f"Clip    R@1  = {np.mean(all_metrics['clip_r1']):.4f} ± {np.std(all_metrics['clip_r1']):.4f}  "
          f"(chance {1/200:.4f})")
    print(f"Clip    R@5  = {np.mean(all_metrics['clip_r5']):.4f} ± {np.std(all_metrics['clip_r5']):.4f}  "
          f"(chance {5/200:.4f})")
    print(f"Clip    R@10 = {np.mean(all_metrics['clip_r10']):.4f} ± {np.std(all_metrics['clip_r10']):.4f}  "
          f"(chance {10/200:.4f})")

    results = {
        "feature": args.feature, "chunks": args.chunks,
        "epochs": args.epochs, "temperature": args.temperature,
        "per_subject": {k: all_metrics[k] for k in metrics_keys},
        **{f"mean_{k}": float(np.mean(all_metrics[k])) for k in metrics_keys},
        **{f"std_{k}":  float(np.std(all_metrics[k]))  for k in metrics_keys},
    }
    tag_str  = f"_{args.tag}" if args.tag else ""
    out_path = os.path.join(args.output_dir, f"results_{args.feature}_k{args.chunks}{tag_str}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
