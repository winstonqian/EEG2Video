"""
Optimal Stimulus Design: Using CLIP Geometry to Select More Decodable Concept Sets.

Finding: CLIP isolation predicts per-concept R@1 (r=0.484**).
Application: Greedily select N concepts from a large vocabulary to
maximize mean pairwise CLIP isolation → predict higher EEG R@1.

Uses CLIP to embed 500 common English nouns/concepts, then greedily
selects 40 that maximize mean pairwise cosine distance.

Compares:
  - SEED-DV (current): mean isolation, predicted R@1
  - Optimal-40 (selected): mean isolation, predicted R@1
  - Random-40 (baseline): mean isolation distribution

General contribution: prescriptive BCI design principle — stimulus
selection should maximize CLIP pairwise distance to optimize decodability.

Run from EEG2Video/:
    python neuroclip/optimal_stimulus_design.py
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F

RESULTS_DIR = "neuroclip/results"
FIGURES_DIR = "neuroclip/figures"

# Large vocabulary of common concrete nouns (easy to visualize, BCI-relevant)
VOCAB = [
    "airplane","apple","banana","bear","bed","bicycle","bird","boat","book","bottle",
    "bus","butterfly","cake","camera","car","cat","chair","clock","cloud","cow",
    "cup","deer","dinosaur","dog","dolphin","door","duck","eagle","elephant","fish",
    "flower","fork","frog","giraffe","guitar","hammer","horse","house","kangaroo",
    "keyboard","knife","lamp","lion","lobster","monkey","motorcycle","mountain","mouse",
    "mushroom","octopus","orange","owl","panda","penguin","piano","pig","pizza",
    "rabbit","refrigerator","rocket","rose","scissors","sheep","ship","skateboard",
    "skull","snake","sofa","spider","squirrel","strawberry","submarine","sunflower",
    "sword","table","telephone","tiger","tomato","train","tree","truck","trumpet",
    "turtle","umbrella","vest","violin","watch","waterfall","whale","wolf","zebra",
    "axe","balloon","basket","bell","bench","boot","bow","box","brain","brick",
    "bridge","broom","candle","castle","cave","chain","cheese","chess","chimp","chip",
    "coin","crab","crane","crown","crystal","dart","desk","diamond","dice","drum",
    "egg","fan","feather","fence","fire","flag","flame","flask","fox","fridge",
    "gate","gem","ghost","globe","glove","goat","gorilla","grape","gun","hammer",
    "hat","helmet","hippo","hook","horn","iceberg","igloo","island","jar","jellyfish",
    "kite","ladder","leaf","lemon","lighthouse","lizard","lock","map","mask","medal",
    "meteor","microscope","mirror","missile","nail","nest","net","oar","paddle","paw",
    "pear","pencil","pipe","planet","plate","plow","poison","pole","pot","prism",
    "pump","pyramid","radar","rail","rake","ram","ramp","reef","rifle","ring",
    "rod","rope","sail","saw","scale","screw","shell","shield","shoe","shovel",
    "silo","siren","sled","slide","sling","snail","sock","spear","spike","sponge",
    "spoon","spring","stamp","star","statue","stick","stone","storm","strap","straw",
    "stream","string","stripe","submarine","sundial","swing","switch","tank","tap",
    "thorn","toad","torch","tower","trap","tray","trophy","tube","tusk","valve",
    "vase","vine","volcano","wagon","wall","web","wedge","wheel","whip","whistle",
    "wig","wind","wing","wire","wrench","yarn",
]
VOCAB = list(dict.fromkeys(VOCAB))  # deduplicate while preserving order

# Current SEED-DV 40 concepts
SEED_CONCEPTS = [
    "airplane","bear","bird","boat","cat","chair","cow","cup","dinosaur","dog",
    "elephant","guitar","horse","house","kangaroo","knife","lion","monkey","motorcycle","person",
    "pizza","refrigerator","rocket","scissors","sheep","sofa","spider","table","telephone","tiger",
    "train","truck","turtle","umbrella","vest","violin","watch","waterfall","whale","zebra"
]


def embed_concepts(concept_list, device):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    embs = []
    batch_size = 64
    for i in range(0, len(concept_list), batch_size):
        batch = concept_list[i:i+batch_size]
        inputs = processor(text=[f"a photo of a {c}" for c in batch],
                          return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            feat = model.get_text_features(**inputs)
            feat = F.normalize(feat, dim=-1)
        embs.append(feat.cpu())
    return torch.cat(embs, dim=0)  # (N, 512)


def mean_isolation(embs_subset):
    """Mean pairwise cosine distance (1 - similarity) for a set of embeddings."""
    n = embs_subset.shape[0]
    sims = (embs_subset @ embs_subset.T).numpy()
    np.fill_diagonal(sims, 1.0)  # exclude self
    off_diag = sims[~np.eye(n, dtype=bool)]
    return float(1.0 - off_diag.mean())  # higher = more isolated


def greedy_max_isolation(all_embs, all_names, n_select=40, seed_indices=None):
    """Greedy selection: iteratively add concept that maximizes mean isolation."""
    n_total = all_embs.shape[0]
    selected = []

    if seed_indices:
        selected = list(seed_indices[:1])
    else:
        # Start with most isolated single concept
        sims = (all_embs @ all_embs.T).numpy()
        np.fill_diagonal(sims, 1.0)
        mean_sims = sims.mean(axis=1) - 1.0/(n_total)  # mean sim to others
        selected = [int(mean_sims.argmin())]

    while len(selected) < n_select:
        best_idx, best_iso = -1, -1
        for i in range(n_total):
            if i in selected: continue
            trial = selected + [i]
            iso = mean_isolation(all_embs[trial])
            if iso > best_iso:
                best_iso = iso
                best_idx = i
        selected.append(best_idx)
        if len(selected) % 10 == 0:
            print(f"  Selected {len(selected)}/{n_select}: mean_iso={best_iso:.4f}  last={all_names[best_idx]}")

    return selected, mean_isolation(all_embs[selected])


def main():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")
    print(f"Vocabulary size: {len(VOCAB)} concepts")

    # Check if cached embeddings exist
    cache_path = f"{RESULTS_DIR}/vocab_clip_embs.pt"
    if os.path.exists(cache_path):
        print("Loading cached vocab embeddings...")
        cache = torch.load(cache_path, weights_only=True)
        vocab_embs = cache["embs"]
        vocab_names = cache["names"]
    else:
        print("Embedding vocabulary via CLIP...")
        vocab_embs = embed_concepts(VOCAB, device)
        torch.save({"embs": vocab_embs, "names": VOCAB}, cache_path)
        vocab_names = VOCAB
        print(f"Embedded {len(vocab_names)} concepts")

    vocab_embs = F.normalize(vocab_embs, dim=-1)

    # Find SEED-DV concept indices in vocabulary
    seed_indices = [vocab_names.index(c) for c in SEED_CONCEPTS if c in vocab_names]
    missing = [c for c in SEED_CONCEPTS if c not in vocab_names]
    print(f"SEED-DV concepts found in vocab: {len(seed_indices)}/40  missing: {missing}")

    # SEED-DV isolation
    seed_iso = mean_isolation(vocab_embs[seed_indices])
    print(f"\nSEED-DV mean isolation: {seed_iso:.4f}")

    # Random baseline (1000 random 40-subsets)
    print("Computing random-40 baseline...")
    rng = np.random.default_rng(42)
    random_isos = []
    for _ in range(1000):
        idx = rng.choice(len(vocab_names), size=40, replace=False)
        random_isos.append(mean_isolation(vocab_embs[idx]))
    random_isos = np.array(random_isos)
    print(f"Random-40 mean isolation: {random_isos.mean():.4f} ± {random_isos.std():.4f}")

    # Greedy optimal selection
    print("\nGreedy selection for maximum isolation...")
    opt_indices, opt_iso = greedy_max_isolation(vocab_embs, vocab_names, n_select=40)
    opt_names = [vocab_names[i] for i in opt_indices]
    print(f"Optimal-40 mean isolation: {opt_iso:.4f}")

    # Predict R@1 using regression from concept_decodability
    conc_res = json.load(open(f"{RESULTS_DIR}/results_concept_decodability.json"))
    r = conc_res["r_isolation"]; p = conc_res["p_isolation"]

    # Simple linear model: R@1 = a * isolation + b
    isos = np.array(conc_res["clip_isolation"])
    r1s  = np.array(conc_res["per_concept_r1"])
    m, b_fit = np.polyfit(isos, r1s, 1)

    # Mean R@1 prediction for each set = model applied to mean isolation
    # (approximate: linear model on mean isolation)
    pred_seed_r1 = (m * seed_iso + b_fit) * 100
    pred_opt_r1  = (m * opt_iso + b_fit) * 100
    pred_rand_r1 = (m * random_isos.mean() + b_fit) * 100

    print(f"\n=== Predicted R@1 (linear extrapolation from isolation) ===")
    print(f"  Random-40:  isolation={random_isos.mean():.4f}  pred_R@1≈{pred_rand_r1:.2f}%")
    print(f"  SEED-DV 40: isolation={seed_iso:.4f}  pred_R@1≈{pred_seed_r1:.2f}%")
    print(f"  Optimal-40: isolation={opt_iso:.4f}  pred_R@1≈{pred_opt_r1:.2f}%")
    print(f"  Predicted improvement: +{pred_opt_r1-pred_seed_r1:.2f}pp over SEED-DV")

    print(f"\nOptimal-40 concepts:")
    for i, name in enumerate(opt_names):
        print(f"  {i+1:2d}. {name}")

    results = {
        "seed_iso": float(seed_iso), "seed_concepts": SEED_CONCEPTS,
        "opt_iso": float(opt_iso), "opt_concepts": opt_names,
        "random_iso_mean": float(random_isos.mean()), "random_iso_std": float(random_isos.std()),
        "pred_seed_r1": float(pred_seed_r1), "pred_opt_r1": float(pred_opt_r1),
        "pred_rand_r1": float(pred_rand_r1),
        "predicted_improvement_pp": float(pred_opt_r1 - pred_seed_r1),
        "regression_slope": float(m), "regression_intercept": float(b_fit),
        "r_isolation": float(r), "p_isolation": float(p),
    }
    with open(f"{RESULTS_DIR}/results_optimal_stimulus.json","w") as f:
        json.dump(results, f, indent=2)

    # ── Figure ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: isolation distribution random vs seed vs optimal
    ax = axes[0]
    ax.hist(random_isos, bins=30, color="#aaaaaa", alpha=0.7, label="Random-40 (1000 draws)", density=True)
    ax.axvline(seed_iso, color="#4472c4", linewidth=2.5, linestyle="-",
               label=f"SEED-DV (current): {seed_iso:.4f}")
    ax.axvline(opt_iso, color="#e74c3c", linewidth=2.5, linestyle="--",
               label=f"Optimal-40: {opt_iso:.4f}")
    ax.axvline(random_isos.mean(), color="gray", linewidth=1.5, linestyle=":",
               label=f"Random mean: {random_isos.mean():.4f}")
    z_seed = (seed_iso - random_isos.mean()) / random_isos.std()
    z_opt  = (opt_iso  - random_isos.mean()) / random_isos.std()
    ax.set_xlabel("Mean Pairwise CLIP Isolation", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"(A) Concept Set Isolation Distribution\nSEED-DV: z={z_seed:.1f}  Optimal: z={z_opt:.1f}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    # Panel B: concept-level scatter with regression line
    ax = axes[1]
    sc = ax.scatter(isos, r1s*100, c=isos, cmap="RdYlGn", s=60, alpha=0.85,
                    edgecolors="black", linewidths=0.5)
    x_ext = np.linspace(min(isos.min(), seed_iso, opt_iso)-0.005,
                         max(isos.max(), opt_iso)+0.005, 100)
    ax.plot(x_ext, (m*x_ext+b_fit)*100, "k-", linewidth=2, label=f"Linear fit (r={r:.3f})")
    ax.axvline(seed_iso, color="#4472c4", linewidth=2, linestyle="--",
               label=f"SEED-DV mean iso")
    ax.axvline(opt_iso, color="#e74c3c", linewidth=2, linestyle="--",
               label=f"Optimal mean iso")
    ax.plot(seed_iso, pred_seed_r1, "b^", markersize=12, zorder=5)
    ax.plot(opt_iso,  pred_opt_r1,  "r*", markersize=14, zorder=5)
    ax.annotate(f"SEED-DV\n{pred_seed_r1:.2f}%", (seed_iso, pred_seed_r1),
               xytext=(-35,10), textcoords="offset points", fontsize=9,
               fontweight="bold", color="#4472c4",
               arrowprops=dict(arrowstyle="->", color="#4472c4"))
    ax.annotate(f"Optimal\n{pred_opt_r1:.2f}%", (opt_iso, pred_opt_r1),
               xytext=(5,10), textcoords="offset points", fontsize=9,
               fontweight="bold", color="#e74c3c",
               arrowprops=dict(arrowstyle="->", color="#e74c3c"))
    ax.set_xlabel("CLIP Isolation (mean pairwise distance)", fontsize=10)
    ax.set_ylabel("Per-Concept R@1 (%)", fontsize=10)
    ax.set_title(f"(B) Isolation → Decodability: Predicted Improvement\n"
                 f"Optimal-40 predicted +{pred_opt_r1-pred_seed_r1:.2f}pp over SEED-DV",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label="CLIP Isolation")

    # Panel C: side-by-side bar of set-level R@1 prediction
    ax = axes[2]
    set_labels = ["Random-40\n(baseline)", "SEED-DV\n(current)", "Optimal-40\n(prescribed)"]
    set_vals   = [pred_rand_r1, pred_seed_r1, pred_opt_r1]
    set_cols   = ["#aaaaaa","#4472c4","#e74c3c"]
    bars = ax.bar(range(3), set_vals, color=set_cols, width=0.55, alpha=0.85)
    ax.axhline(1/40*100, color="gray", linestyle=":", linewidth=1.5, label="Chance (2.5%)")
    for bar, v in zip(bars, set_vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.03,
                f"{v:.2f}%", ha="center", fontsize=11, fontweight="bold")
    # Improvement bracket
    y_top = max(set_vals)+0.2
    ax.plot([1,1,2,2],[y_top,y_top+0.1,y_top+0.1,y_top],lw=1.5,color="black")
    ax.text(1.5, y_top+0.12, f"+{pred_opt_r1-pred_seed_r1:.2f}pp\nprescribed gain",
            ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(set_labels, fontsize=10)
    ax.set_ylabel("Predicted Mean Concept R@1 (%)", fontsize=10)
    ax.set_title("(C) Prescribed Improvement from Optimal Stimulus Selection\n"
                 "(via CLIP isolation maximization)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    plt.suptitle(
        "Optimal Stimulus Design: Maximize CLIP Isolation to Improve EEG Decodability\n"
        f"Greedy selection from {len(vocab_names)}-concept vocabulary  ·  "
        f"Predicted gain: +{pred_opt_r1-pred_seed_r1:.2f}pp over SEED-DV  ·  r={r:.3f} {sig(p)}",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = f"{FIGURES_DIR}/F30_optimal_stimulus_design.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nSaved → {path}")
    print(f"Saved → {RESULTS_DIR}/results_optimal_stimulus.json")

if __name__ == "__main__":
    main()
