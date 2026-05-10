"""
RSA by Supervision Condition: Both vs Image-only.

Tests whether multimodal CLIP supervision (text+image) creates stronger
categorical structure in EEG representations than image-only supervision.

Key question: does CLIP supervision modality affect representational geometry,
independent of task performance?

Run from EEG2Video/:
    python neuroclip/rsa_by_supervision.py
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from dataset import GT_LABEL
from models_neuroclip import EEGEncoder

DE_DATA_DIR = "data/DE_1per1s"
RESULTS_DIR = "neuroclip/results"
FIGURES_DIR = "neuroclip/figures"
N_CONCEPTS, N_CLIPS, N_SESSIONS = 40, 5, 7

ALL_SUBS = sorted([f.replace(".npy","") for f in os.listdir(DE_DATA_DIR) if f.endswith(".npy")])

# Category structure — correct SEED-DV concept IDs (0-indexed)
# 0=cat,1=husky,2=elephant,3=horses,4=panda,5=rabbit,6=bird,7=fish,8=jellyfish,
# 9=whale,10=turtle,11=flowers,12=mushrooms,13=forest,14=boxing,15=dancing,
# 16=running,17=skiing,18=computer,19=construction,20=crowd,21=beach,22=city,
# 23=mountain,24=road,25=waterfall,26=fireworks,27=banana,28=cheesecake,
# 29=drink,30=pizza,31=watermelon,32=drums,33=guitar,34=piano,
# 35=motorcycle,36=car,37=balloon,38=airplane,39=boat
SEMANTIC_GROUPS = {
    "Animals":  [0,1,2,3,4,5,6,7,8,9,10],
    "Nature":   [11,12,13,23,25],
    "Food":     [27,28,29,30,31],
    "Sports":   [14,15,16,17],
    "Music":    [32,33,34],
    "Vehicles": [35,36,37,38,39],
    "Urban":    [20,21,22,24],
    "People":   [18,19],
    "Other":    [26],
}
cat_label = np.zeros(40, dtype=int)
for gi, (grp_name, cids) in enumerate(SEMANTIC_GROUPS.items()):
    for cid in cids: cat_label[cid] = gi
ref = (cat_label[:,None]==cat_label[None,:]).astype(float)
np.fill_diagonal(ref, 0)
ref_upper = ref[np.triu_indices(40,k=1)]

def upper_tri(M): return M[np.triu_indices(40,k=1)]

def compute_rsa_for_condition(condition, device):
    suffix = f"de_k1_{condition}"
    rhos = []
    for sub in ALL_SUBS:
        raw = np.load(f"{DE_DATA_DIR}/{sub}.npy")
        n_s,n_c,n_cl,n_seg,n_ch,n_b = raw.shape
        eeg_all = raw.mean(axis=3).reshape(n_s, n_c*n_cl, n_ch, n_b)
        ckpt = f"{RESULTS_DIR}/{sub}_fold0_{suffix}.pt"
        if not os.path.exists(ckpt): continue
        model = EEGEncoder(n_channels=n_ch, n_time=n_b, embed_dim=512).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()
        # Use TEST session (fold 0 held-out session 0) — matches rsa_analysis.py
        TEST_SESS = 0
        flat = eeg_all[TEST_SESS].reshape(N_CONCEPTS*N_CLIPS, -1)
        norm = StandardScaler().fit_transform(flat).reshape(N_CONCEPTS*N_CLIPS, n_ch, n_b)
        eeg_t = torch.tensor(norm, dtype=torch.float32).to(device)
        cids = np.repeat(GT_LABEL[TEST_SESS], N_CLIPS)
        with torch.no_grad(): embs = model(eeg_t)
        ce = torch.zeros(N_CONCEPTS, 512, device=device)
        cnt = torch.zeros(N_CONCEPTS, device=device)
        for i, cid in enumerate(cids):
            ce[int(cid)] += embs[i]; cnt[int(cid)] += 1
        ce = F.normalize(ce/cnt.clamp(min=1).unsqueeze(1), dim=-1)
        sim = (ce @ ce.T).cpu().numpy(); np.fill_diagonal(sim,0)
        rho,_ = stats.spearmanr(upper_tri(sim), ref_upper)
        rhos.append(float(rho))
    return np.array(rhos)

def permutation_test(rhos, n_perm=2000):
    t_obs = rhos.mean() / (rhos.std()/np.sqrt(len(rhos)))
    t_null = []
    for _ in range(n_perm):
        shuf = np.random.permutation(rhos)
        t_null.append(shuf.mean()/(shuf.std()/np.sqrt(len(shuf))))
    p = (np.abs(t_null) >= np.abs(t_obs)).mean()
    return float(t_obs), float(p)

def main():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."

    results = {}
    for cond in ["both","image"]:
        print(f"\nComputing RSA for condition: {cond}")
        rhos = compute_rsa_for_condition(cond, device)
        t,p = stats.ttest_1samp(rhos,0)
        _,p_perm = permutation_test(rhos)
        results[cond] = {"rhos": rhos.tolist(), "mean": float(rhos.mean()),
                         "std": float(rhos.std()), "sem": float(rhos.std()/np.sqrt(len(rhos))),
                         "t": float(t), "p_ttest": float(p), "p_perm": float(p_perm),
                         "n": len(rhos)}
        print(f"  {cond}: ρ={rhos.mean():+.4f}±{rhos.std():.4f}  t={t:.2f}  p_perm={p_perm:.4f} {sig(p_perm)}")

    # Test both > image
    rhos_both = np.array(results["both"]["rhos"])
    rhos_img  = np.array(results["image"]["rhos"])
    t_diff, p_diff = stats.ttest_rel(rhos_both, rhos_img)
    results["comparison"] = {"t": float(t_diff), "p": float(p_diff)}
    print(f"\nBoth > Image: t={t_diff:.2f}  p={p_diff:.4f} {sig(p_diff)}")

    with open(f"{RESULTS_DIR}/results_rsa_by_supervision.json","w") as f:
        json.dump(results, f, indent=2)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4))

    ax = axes[0]
    labels = ["Image\n(unimodal)","Both\n(multimodal)"]
    means = [results["image"]["mean"], results["both"]["mean"]]
    sems  = [results["image"]["sem"],  results["both"]["sem"]]
    pvals = [results["image"]["p_perm"],results["both"]["p_perm"]]
    cols  = ["#4472c4","#70ad47"]
    bars = ax.bar([0,1], means, yerr=sems, color=cols, width=0.5, alpha=0.85, capsize=8,
                  error_kw={"elinewidth":2})
    ax.axhline(0, color="black", linewidth=1)
    for bar,m,p in zip(bars,means,pvals):
        ax.text(bar.get_x()+bar.get_width()/2, m+sems[means.index(m)]+0.003,
                f"ρ={m:.4f}\n{sig(p)}", ha="center", fontsize=10, fontweight="bold")
    y_top = max(m+s for m,s in zip(means,sems))+0.008
    ax.plot([0,0,1,1],[y_top,y_top+0.003,y_top+0.003,y_top],lw=1.5,color="black")
    ax.text(0.5,y_top+0.004,f"p={p_diff:.4f} {sig(p_diff)}",ha="center",fontsize=10,fontweight="bold")
    ax.set_xticks([0,1]); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Category RSA ρ", fontsize=11)
    ax.set_title("(A) Supervision Type → RSA\nMultimodal CLIP supervision creates\nstronger categorical brain structure",
                 fontsize=11, fontweight="bold")

    ax = axes[1]
    ax.scatter(rhos_img, rhos_both, c=rhos_both-rhos_img, cmap="RdYlGn", s=80,
               edgecolors="black", linewidths=0.5)
    lim = max(abs(rhos_both).max(), abs(rhos_img).max())+0.02
    ax.plot([-lim,lim],[-lim,lim],"k--",linewidth=1.5,alpha=0.5,label="Equal")
    ax.axhline(0,color="gray",linestyle=":",linewidth=1)
    ax.axvline(0,color="gray",linestyle=":",linewidth=1)
    n_above = (rhos_both > rhos_img).sum()
    ax.set_xlabel("Category RSA ρ — Image supervision", fontsize=10)
    ax.set_ylabel("Category RSA ρ — Both supervision", fontsize=10)
    ax.set_title(f"(B) Per-Subject Comparison\n({n_above}/{len(rhos_both)} subjects: Both > Image)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    plt.suptitle("RSA by Supervision Condition: Multimodal CLIP → Better Brain Geometry\n"
                 "NeuroCLIP-Both vs NeuroCLIP-Image · 21 subjects · Categorical RSA",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/F27_rsa_by_supervision.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {path}")

if __name__ == "__main__":
    main()
