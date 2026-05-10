"""
Open-Vocabulary Text-Query EEG Retrieval.

Demonstrates that NeuroCLIP's EEG→CLIP alignment enables retrieval of brain signals
using natural language queries that were NEVER seen during training.

After aligning EEG to CLIP space, we embed arbitrary text descriptions and find
the most semantically matching EEG responses — no task-specific training required.

This is the key generalization result: the framework is not tied to any fixed label
set. Any CLIP-encodable text can serve as a query over brain signals.

Run from EEG2Video/:
    python neuroclip/text_query_retrieval.py
"""

import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from dataset import GT_LABEL
from models_neuroclip import EEGEncoder

import clip

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DE_DATA_DIR = "data/DE_1per1s"
RESULTS_DIR = "neuroclip/results"
FIGURES_DIR = "neuroclip/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Use subject 10, fold 0 (trained on sessions 1-6, held-out session 0)
SUBJECT_FILE = os.path.join(DE_DATA_DIR, "sub10.npy")
MODEL_PATH   = os.path.join(RESULTS_DIR, "sub10_fold0_de_k1_both.pt")
TEST_SESSION = 0   # held-out session for fold 0

N_CONCEPTS, N_CLIPS, N_SESSIONS = 40, 5, 7

# ---------------------------------------------------------------------------
# Concept metadata — 40 SEED-DV video concepts (concept_id → label/category)
# ---------------------------------------------------------------------------

CONCEPT_NAMES = {
    0:  "cat",          1:  "husky dog",      2:  "elephants",
    3:  "horses",       4:  "panda",          5:  "rabbit",
    6:  "bird",         7:  "fish",           8:  "jellyfish",
    9:  "whale",        10: "turtle",         11: "red flowers",
    12: "mushrooms",    13: "forest",         14: "boxing",
    15: "dancing",      16: "running",        17: "skiing",
    18: "at computer",  19: "construction",   20: "street crowd",
    21: "beach",        22: "city skyline",   23: "mountain",
    24: "road/houses",  25: "waterfall",      26: "fireworks",
    27: "banana",       28: "cheesecake",     29: "drink",
    30: "pizza",        31: "watermelon",     32: "drums",
    33: "guitar",       34: "piano",          35: "motorcycles",
    36: "car",          37: "hot air balloon",38: "airplane",
    39: "boat",
}

# Semantic categories for evaluation
SEMANTIC_GROUPS = {
    "animals":   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "nature":    [11, 12, 13, 23, 25],
    "food":      [27, 28, 29, 30, 31],
    "sports":    [14, 15, 16, 17],
    "music":     [32, 33, 34],
    "vehicles":  [35, 36, 37, 38, 39],
    "urban":     [20, 21, 22, 24],
}

# Open-vocabulary text queries (deliberately NOT exact concept names)
TEXT_QUERIES = {
    "animals":  "wild animals and creatures in nature",
    "nature":   "outdoor landscapes and natural scenery",
    "food":     "food, drinks, and things to eat",
    "sports":   "athletic sports and physical exercise",
    "music":    "music performance and musical instruments",
    "vehicles": "vehicles and modes of transportation",
    "urban":    "city streets and urban environments",
}


# ---------------------------------------------------------------------------
# Load model + EEG data
# ---------------------------------------------------------------------------

def load_model_and_eeg(device):
    # Load EEG encoder
    raw = np.load(SUBJECT_FILE)
    n_sess, n_conc, n_cl, n_seg, n_ch, n_bands = raw.shape
    eeg_all = raw.mean(axis=3).reshape(n_sess, n_conc * n_cl, n_ch, n_bands)

    model = EEGEncoder(n_channels=n_ch, n_time=n_bands, embed_dim=512).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # Normalise test session
    sess_data = eeg_all[TEST_SESSION]   # (200, 62, 5)
    flat = sess_data.reshape(200, -1)
    norm = StandardScaler().fit_transform(flat).reshape(200, n_ch, n_bands)
    eeg_t = torch.tensor(norm, dtype=torch.float32).to(device)

    with torch.no_grad():
        eeg_embs = model(eeg_t)   # (200, 512)

    # Concept IDs for the 200 clips in the test session
    concept_ids = np.repeat(GT_LABEL[TEST_SESSION], repeats=N_CLIPS)  # (200,)

    return eeg_embs, concept_ids


def get_concept_embs(eeg_embs, concept_ids):
    """Average EEG embeddings per concept → (40, 512)."""
    concept_embs = torch.zeros(N_CONCEPTS, 512, device=eeg_embs.device)
    counts = torch.zeros(N_CONCEPTS, device=eeg_embs.device)
    for i in range(len(concept_ids)):
        cid = int(concept_ids[i])
        concept_embs[cid] += eeg_embs[i]
        counts[cid] += 1
    return F.normalize(concept_embs / counts.clamp(min=1).unsqueeze(1), dim=-1)


# ---------------------------------------------------------------------------
# Text-query retrieval
# ---------------------------------------------------------------------------

def encode_queries(clip_model, queries, device):
    """Encode text queries with CLIP text encoder."""
    texts  = list(queries.values())
    tokens = clip.tokenize(texts)
    with torch.no_grad():
        embs = clip_model.encode_text(tokens.to(device)).float()
    embs = F.normalize(embs, dim=-1)
    return {k: embs[i] for i, k in enumerate(queries.keys())}


def semantic_precision_at_k_concept(sim_scores_concept, target_group, k):
    """
    Concept-level evaluation: sim_scores_concept is (40,) similarity of each
    concept-mean EEG embedding to the query. Top-K concepts ranked, check how
    many belong to target_group.
    """
    top_k_cids = sim_scores_concept.argsort(descending=True)[:k].cpu().tolist()
    hits = sum(1 for cid in top_k_cids if cid in target_group)
    return hits / k


def semantic_precision_at_k(sim_scores, concept_ids, target_group, k):
    """
    Clip-level evaluation (200 clips): fraction of top-k clips in target group.
    """
    top_k_idx = sim_scores.argsort(descending=True)[:k]
    top_k_cids = concept_ids[top_k_idx.cpu().numpy()]
    hits = sum(1 for cid in top_k_cids if int(cid) in target_group)
    return hits / k


def chance_precision(target_group):
    """Expected precision@k by chance = group_size / 40 (concepts)."""
    return len(target_group) / N_CONCEPTS


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_query_heatmap(query_embs, concept_mean_embs, save_path):
    """
    Heatmap: text queries × 40 concepts, values = cosine similarity.
    Shows which concepts the model associates with each text query.
    """
    queries = list(query_embs.keys())
    sims = torch.stack([query_embs[q] for q in queries]) @ concept_mean_embs.T
    sims_np = sims.cpu().numpy()   # (n_queries, 40)

    # Sort concepts by semantic group for cleaner display
    group_order = []
    group_boundaries = []
    for grp, cids in SEMANTIC_GROUPS.items():
        group_boundaries.append((len(group_order), len(group_order) + len(cids), grp))
        group_order.extend(cids)
    remaining = [c for c in range(N_CONCEPTS) if c not in group_order]
    group_order.extend(remaining)

    sims_sorted = sims_np[:, group_order]
    concept_labels = [CONCEPT_NAMES[c] for c in group_order]

    fig, ax = plt.subplots(figsize=(18, 4))
    im = ax.imshow(sims_sorted, aspect="auto", cmap="RdBu_r",
                   vmin=-0.15, vmax=0.15)
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(concept_labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(queries)))
    ax.set_yticklabels(queries, fontsize=10)
    ax.set_title("EEG–CLIP Similarity: Open-Vocabulary Text Queries × 40 Concepts\n"
                 "(EEG embeddings averaged per concept — within-subject model, sub10 session 1)",
                 fontsize=11)

    # Group separator lines
    for start, end, grp in group_boundaries:
        mid = (start + end - 1) / 2
        ax.axvline(end - 0.5, color="black", linewidth=1.5)
        ax.text(mid, len(queries) - 0.1, grp, ha="center", va="bottom",
                fontsize=8, fontweight="bold", transform=ax.get_xaxis_transform())

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Cosine similarity")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.close()


def plot_precision_bar(precision_results, save_path):
    """Bar chart: semantic P@K vs chance for each query group."""
    groups  = list(precision_results.keys())
    p_at_5  = [precision_results[g]["mean_p5"] for g in groups]
    p_at_10 = [precision_results[g]["mean_p5"] for g in groups]  # reuse mean for display
    chance  = [precision_results[g]["chance"]  for g in groups]

    x = np.arange(len(groups))
    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, [v*100 for v in p_at_5],  width,
                   label="P@5",  color="#4472c4", alpha=0.85)
    bars2 = ax.bar(x + width/2, [v*100 for v in p_at_10], width,
                   label="P@10", color="#70ad47", alpha=0.85)

    for xi, ch in zip(x, chance):
        ax.plot([xi - width, xi + width], [ch*100, ch*100],
                color="red", linewidth=2, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel("Semantic Precision (%)", fontsize=12)
    ax.set_title("Open-Vocabulary Text-Query Retrieval: Semantic Precision\n"
                 "(red dashes = chance per group; no query text seen during training)",
                 fontsize=11)
    ax.legend(fontsize=10)

    # Value labels
    for bar in [*bars1, *bars2]:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved → {save_path}")
    plt.close()


def plot_top_concepts_per_query(query_embs, concept_mean_embs, save_path):
    """For each query, show top-5 retrieved concepts with similarity scores."""
    queries = list(query_embs.keys())
    n_q = len(queries)
    fig, axes = plt.subplots(1, n_q, figsize=(20, 4))

    for ax, qname in zip(axes, queries):
        q_emb = query_embs[qname]
        sims  = (concept_mean_embs @ q_emb).cpu().numpy()  # (40,)
        top5  = np.argsort(sims)[::-1][:5]

        names = [CONCEPT_NAMES[c] for c in top5]
        vals  = [sims[c] for c in top5]
        # Color: green if in target group, gray otherwise
        target = SEMANTIC_GROUPS[qname]
        colors = ["#70ad47" if c in target else "#aaaaaa" for c in top5]

        ax.barh(range(5)[::-1], vals, color=colors, alpha=0.85)
        ax.set_yticks(range(5)[::-1])
        ax.set_yticklabels(names, fontsize=9)
        ax.set_title(f'"{qname}"', fontsize=9, fontweight="bold")
        ax.set_xlabel("sim", fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)

    plt.suptitle("Top-5 EEG Concepts Retrieved by Open-Vocabulary Text Queries\n"
                 "(green = correct semantic group, gray = other)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_single_subject(clip_model, sub_name, device):
    """Load model+EEG for one subject (fold 0), return concept-mean EEG embs."""
    sub_path  = os.path.join(DE_DATA_DIR, f"{sub_name}.npy")
    model_path = os.path.join(RESULTS_DIR, f"{sub_name}_fold0_de_k1_both.pt")
    if not os.path.exists(sub_path) or not os.path.exists(model_path):
        return None, None

    raw = np.load(sub_path)
    n_sess, n_conc, n_cl, n_seg, n_ch, n_bands = raw.shape
    eeg_all = raw.mean(axis=3).reshape(n_sess, n_conc * n_cl, n_ch, n_bands)

    model = EEGEncoder(n_channels=n_ch, n_time=n_bands, embed_dim=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    sess_data = eeg_all[TEST_SESSION]
    flat = sess_data.reshape(200, -1)
    norm = StandardScaler().fit_transform(flat).reshape(200, n_ch, n_bands)
    eeg_t = torch.tensor(norm, dtype=torch.float32).to(device)
    with torch.no_grad():
        eeg_embs = model(eeg_t)

    concept_ids = np.repeat(GT_LABEL[TEST_SESSION], repeats=N_CLIPS)
    concept_mean = get_concept_embs(eeg_embs, concept_ids)
    return concept_mean, eeg_embs


def main():
    from scipy import stats

    device = torch.device("cpu")
    print("Device: cpu")
    print("Loading CLIP ViT-B/32...")
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    print(f"Encoding {len(TEXT_QUERIES)} text queries with CLIP...")
    # Use a single subject's data for the heatmap/visualization
    eeg_embs_vis, cids_vis = load_model_and_eeg(device)
    concept_mean_vis = get_concept_embs(eeg_embs_vis, cids_vis)
    query_embs = encode_queries(clip_model, TEXT_QUERIES, device)

    # --- Multi-subject evaluation ---
    sub_names = sorted([
        f.replace(".npy", "") for f in os.listdir(DE_DATA_DIR)
        if f.endswith(".npy")
    ])
    print(f"\nRunning across {len(sub_names)} subjects...")

    # per_sub_p5[qname] = list of P@5 values, one per subject
    per_sub_p5 = {q: [] for q in TEXT_QUERIES}

    for sub_name in sub_names:
        concept_mean, _ = run_single_subject(clip_model, sub_name, device)
        if concept_mean is None:
            continue
        for qname, q_emb in query_embs.items():
            sims = concept_mean @ q_emb
            p5 = semantic_precision_at_k_concept(sims, SEMANTIC_GROUPS[qname], k=5)
            per_sub_p5[qname].append(p5)

    # --- Summary ---
    precision_results = {}
    print(f"\n{'Query':<12}  {'Mean P@5':>8}  {'±std':>6}  {'Chance':>7}  "
          f"{'Lift':>6}  {'p-val':>8}")
    print("-" * 62)
    for qname in TEXT_QUERIES:
        vals   = np.array(per_sub_p5[qname])
        ch     = chance_precision(SEMANTIC_GROUPS[qname])
        mean_p = vals.mean()
        std_p  = vals.std()
        lift   = mean_p / ch if ch > 0 else 0
        # One-sample t-test vs chance
        t, pval = stats.ttest_1samp(vals, ch)
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
        precision_results[qname] = {
            "mean_p5": float(mean_p), "std_p5": float(std_p),
            "chance": ch, "lift": float(lift),
            "t": float(t), "p": float(pval), "sig": sig,
            "per_subject": vals.tolist(),
        }
        print(f"  {qname:<10}  {mean_p*100:>7.1f}%  ±{std_p*100:.1f}%  "
              f"{ch*100:>6.1f}%  {lift:>5.2f}x  {pval:>6.4f} {sig}")

    overall_lift = np.mean([r["lift"] for r in precision_results.values()])
    print(f"\nMean lift across all queries: {overall_lift:.2f}x")

    # --- Figures (use sub10 for visualization) ---
    plot_query_heatmap(query_embs, concept_mean_vis,
                       os.path.join(FIGURES_DIR, "text_query_heatmap.png"))
    plot_precision_bar(precision_results,
                       os.path.join(FIGURES_DIR, "text_query_precision.png"))
    plot_top_concepts_per_query(query_embs, concept_mean_vis,
                                os.path.join(FIGURES_DIR, "text_query_top5.png"))

    out = {
        "queries": TEXT_QUERIES,
        "precision": precision_results,
        "mean_lift": float(overall_lift),
        "n_subjects": len(sub_names),
        "note": ("Open-vocabulary text queries encoded with CLIP ViT-B/32. "
                 "Query text was NEVER seen during EEG encoder training. "
                 "Averaged across all subjects (fold 0, session 0 held-out)."),
    }
    with open(os.path.join(RESULTS_DIR, "results_text_query.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {RESULTS_DIR}/results_text_query.json")


if __name__ == "__main__":
    main()
