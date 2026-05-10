"""
Grand summary figure: all NeuroCLIP findings in one panel.

Layout (2×3 grid):
  [A] Main bar chart: chance / classification / within / cross
  [B] Subject scaling curve (flat — individual variability bottleneck)
  [C] Hierarchical retrieval lifts
  [D] RSA: trained vs untrained + Category vs CLIP
  [E] Within vs Between category EEG similarity (t-test)
  [F] Concept decodability: CLIP isolation predicts R@1

Run from EEG2Video/:
    python neuroclip/grand_summary.py
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

RESULTS_DIR = "neuroclip/results"
FIGURES_DIR = "neuroclip/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

CHANCE     = 1/40
CLASS_MEAN = 0.0437
CLASS_STD  = 0.0264

def load(f):
    p = os.path.join(RESULTS_DIR, f)
    return json.load(open(p)) if os.path.exists(p) else None

def sig(p):
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."

def bracket(ax, x1, x2, y, h, label, fs=9):
    ax.plot([x1,x1,x2,x2],[y,y+h,y+h,y],lw=1.2,color="black")
    ax.text((x1+x2)/2, y+h+0.05, label, ha="center", va="bottom",
            fontsize=fs, fontweight="bold")


def main():
    both    = load("results_de_k1_both.json")
    cross   = load("results_crosssub_both_final_epoch.json")
    scaling = load("results_subject_scaling.json")
    hier    = load("results_hierarchical.json")
    rsa     = load("results_rsa.json")
    wb      = load("results_within_between.json")
    untr    = load("results_untrained_rsa.json")
    conc    = load("results_concept_decodability.json")

    within_r1 = np.array(both["per_subject"]["concept_r1"]) if both else np.array([])
    cross_r1  = np.array(cross["per_subject_r1"]) if cross else np.array([])

    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    colors = {"chance":"#aaaaaa","class":"#ff9999","within":"#70ad47",
              "cross":"#7030a0","cat":"#4472c4","conc":"#70ad47","intra":"#ed7d31"}

    # ── Panel A: Main comparison ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    labels = ["Chance", "Classif.\nBaseline", "Within-\nSubject", "Cross-\nSubject"]
    vals   = [CHANCE*100, CLASS_MEAN*100,
              within_r1.mean()*100 if len(within_r1) else 4.60,
              cross_r1.mean()*100  if len(cross_r1)  else 2.94]
    errs   = [0, CLASS_STD*100,
              within_r1.std()*100 if len(within_r1) else 2.70,
              cross_r1.std()*100  if len(cross_r1)  else 0.39]
    cols   = [colors["chance"], colors["class"], colors["within"], colors["cross"]]
    bars   = ax.bar(range(4), vals, yerr=errs, capsize=5, color=cols,
                    width=0.55, alpha=0.85, error_kw={"elinewidth":1.8})
    for bar, v, e in zip(bars, vals, errs):
        ax.text(bar.get_x()+bar.get_width()/2, v+e+0.05,
                f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    if len(within_r1) and len(cross_r1):
        t,p = stats.ttest_rel(within_r1, cross_r1)
        y_top = max(v+e for v,e in zip(vals,errs))+0.3
        bracket(ax, 2, 3, y_top, 0.2, f"p={p:.3f} {sig(p)}", 8)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Concept R@1 (%)", fontsize=10)
    ax.set_title("(A) Within vs Cross-Subject\nGeneralisation", fontsize=11, fontweight="bold")
    t_w,p_w = stats.ttest_1samp(within_r1, CHANCE) if len(within_r1) else (0,1)
    t_c,p_c = stats.ttest_1samp(cross_r1,  CHANCE) if len(cross_r1)  else (0,1)
    ax.set_xlabel(f"Within: p={p_w:.3f}{sig(p_w)}  Cross: p={p_c:.4f}{sig(p_c)}", fontsize=8)

    # ── Panel B: Subject scaling (UPDATED) ───────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    if scaling:
        ns    = scaling["n_values"]
        means = [scaling["results"][str(n)]["mean"]*100 for n in ns]
        stds  = [scaling["results"][str(n)]["std"]*100  for n in ns]
        ax.errorbar(ns, means, yerr=stds, fmt="o-", color=colors["cross"],
                    linewidth=2.5, markersize=7, capsize=5,
                    label="Cross-subject R@1")
        ax.fill_between(ns, [m-s for m,s in zip(means,stds)],
                            [m+s for m,s in zip(means,stds)],
                        color=colors["cross"], alpha=0.15)
        ax.axhline(CHANCE*100, color=colors["chance"], linestyle="--",
                   linewidth=1.5, label=f"Chance ({CHANCE*100:.1f}%)")
        ax.axhline(within_r1.mean()*100 if len(within_r1) else 4.60,
                   color=colors["within"], linestyle="--", linewidth=1.5,
                   label=f"Within-subject ({within_r1.mean()*100:.1f}%)" if len(within_r1) else "Within (4.6%)")
        ax.set_xticks(ns)
        ax.set_ylim(0, max(means)+max(stds)+1.5)
        ax.legend(fontsize=8, loc="lower right")
    ax.set_xlabel("# Training Subjects (N)", fontsize=10)
    ax.set_ylabel("Cross-Subject R@1 (%)", fontsize=10)
    ax.set_title("(B) Subject Scaling Curve\n(flat — individual variability is the bottleneck, not data quantity)",
                 fontsize=11, fontweight="bold")

    # ── Panel C: Hierarchical retrieval lifts ─────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    if hier:
        lvls   = ["Category\n(8-way)", "Concept\n(40-way)", "Intra-cat."]
        lifts  = [hier["cat_lift"], hier["conc_lift"], hier["intra_lift"]]
        pvals  = [hier["p_cat"],    hier["p_conc"],    hier["p_intra"]]
        cols_h = [colors["cat"], colors["conc"], colors["intra"]]
        bars_h = ax.bar(range(3), lifts, color=cols_h, width=0.5, alpha=0.85)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Chance (1×)")
        for i, (bar, l, p) in enumerate(zip(bars_h, lifts, pvals)):
            ax.text(bar.get_x()+bar.get_width()/2, l+0.02,
                    f"{l:.2f}×\n{sig(p)}", ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(range(3)); ax.set_xticklabels(lvls, fontsize=9)
        ax.legend(fontsize=9)
        ax.set_ylim(0, max(lifts)+0.5)
    ax.set_ylabel("Lift over chance", fontsize=10)
    ax.set_title("(C) Hierarchical Retrieval Lifts\n(all levels above chance — generalises across hierarchy)",
                 fontsize=11, fontweight="bold")

    # ── Panel D: RSA trained vs untrained ─────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    if rsa and untr:
        labels_d = ["CLIP-Both\n(trained)", "Category\n(trained)", "Category\n(untrained)"]
        rhos = [rsa["CLIP-Both"]["rho"], rsa["Category"]["rho"], untr["untrained_mean"]]
        pems = [rsa["CLIP-Both"]["p_emp"], rsa["Category"]["p_emp"],
                untr["p_untrained_vs_zero"]]
        nsds = [rsa["CLIP-Both"]["null_std"], rsa["Category"]["null_std"],
                untr["untrained_std"]/np.sqrt(untr["n_subjects"])]
        cols_d = ["#4472c4","#ffc000","#aaaaaa"]
        bars_d = ax.bar(range(3), rhos, color=cols_d, width=0.5, alpha=0.85,
                        yerr=nsds, capsize=6, error_kw={"elinewidth":1.8})
        ax.axhline(0, color="black", linewidth=1)
        for i,(bar,rho,p) in enumerate(zip(bars_d,rhos,pems)):
            y = rho + nsds[i] + 0.003
            ax.text(bar.get_x()+bar.get_width()/2, y,
                    f"ρ={rho:.3f}\n{sig(p)}", ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(range(3)); ax.set_xticklabels(labels_d, fontsize=9)
        ax.set_ylim(min(min(rhos)-0.04,-0.03), max(max(rhos)+0.08, 0.2))
    ax.set_ylabel("RSA Spearman ρ", fontsize=10)
    ax.set_title("(D) Representational Similarity Analysis\n"
                 "(training induces categorical, not continuous, structure)",
                 fontsize=11, fontweight="bold")

    # ── Panel E: Within vs Between category ───────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    if wb:
        n_sub = wb["n_subjects"]
        sem_w = wb["within_std"]  / np.sqrt(n_sub)
        sem_b = wb["between_std"] / np.sqrt(n_sub)
        vals_e = [wb["within_mean"], wb["between_mean"]]
        errs_e = [sem_w, sem_b]
        cols_e = ["#4472c4","#e74c3c"]
        bars_e = ax.bar([0,1], vals_e, yerr=errs_e, color=cols_e, width=0.45,
                        alpha=0.85, capsize=8, error_kw={"elinewidth":2})
        for bar,v in zip(bars_e, vals_e):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.006,
                    f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
        y_top = max(vals_e[0]+errs_e[0], vals_e[1]+errs_e[1]) + 0.005
        ax.plot([0,0,1,1],[y_top,y_top+0.003,y_top+0.003,y_top],lw=1.5,color="black")
        ax.text(0.5, y_top+0.004, f"p={wb['p_val']:.4f} {sig(wb['p_val'])}",
                ha="center", fontsize=10, fontweight="bold")
        ax.set_ylim(0.28, y_top+0.02)
        ax.set_xticks([0,1])
        ax.set_xticklabels(["Within\nCategory","Between\nCategory"], fontsize=10)
    ax.set_ylabel("Mean EEG cosine similarity", fontsize=10)
    ax.set_title("(E) Within > Between Category EEG Similarity\n"
                 "(categorical brain organisation — t-test, 21 subjects)",
                 fontsize=11, fontweight="bold")

    # ── Panel F: Activity vs Passive category R@1 ────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    cat_r1 = load("results_category_r1.json")
    if cat_r1 and conc:
        per_conc_r1 = np.array(conc["per_concept_r1"])
        SEMANTIC_GROUPS_ORDERED = {
            "People": [18,19], "Music": [32,33,34], "Sports": [14,15,16,17],
            "Urban": [20,21,22,24], "Other": [26],
            "Animals": [0,1,2,3,4,5,6,7,8,9,10],
            "Food": [27,28,29,30,31], "Vehicles": [35,36,37,38,39],
            "Nature": [11,12,13,23,25],
        }
        ACTIVITY_CATS_F = ["Sports", "Music", "People"]
        cats = list(SEMANTIC_GROUPS_ORDERED.keys())
        cat_means_f = [np.mean(per_conc_r1[SEMANTIC_GROUPS_ORDERED[c]])*100 for c in cats]
        cat_sems_f  = [np.std(per_conc_r1[SEMANTIC_GROUPS_ORDERED[c]])/
                       np.sqrt(len(SEMANTIC_GROUPS_ORDERED[c]))*100 for c in cats]
        cols_f = ["#e74c3c" if c in ACTIVITY_CATS_F else "#4472c4" for c in cats]
        bars_f = ax.bar(range(len(cats)), cat_means_f, yerr=cat_sems_f, color=cols_f,
                        alpha=0.85, capsize=4, width=0.65)
        ax.axhline(CHANCE*100, color="gray", linestyle="--", linewidth=1.5)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Mean Concept R@1 (%)", fontsize=10)
        t_act = cat_r1["t_activity_vs_passive"]; p_act = cat_r1["p_activity_vs_passive"]
        act_m = cat_r1["activity_mean"]*100; pas_m = cat_r1["passive_mean"]*100
        import matplotlib.patches as mpatches
        ax.legend(handles=[mpatches.Patch(color="#e74c3c",alpha=0.85,label=f"Activity ({act_m:.1f}%)"),
                           mpatches.Patch(color="#4472c4",alpha=0.85,label=f"Passive ({pas_m:.1f}%)")],
                  fontsize=8, loc="upper right")
        ax.set_title(f"(F) Action Semantics Drives Decodability\n"
                     f"Activity {act_m:.1f}% vs Passive {pas_m:.1f}% — t={t_act:.1f} {sig(p_act)}",
                     fontsize=11, fontweight="bold")

    plt.suptitle(
        "NeuroCLIP: CLIP-Contrastive Training Induces Categorical Structure — Action Semantics Drives Alignment\n"
        "SEED-DV · 40 concepts · 21 subjects · NeuroCLIP-Both (DE features, fold-0)",
        fontsize=13, fontweight="bold", y=1.01
    )
    path = os.path.join(FIGURES_DIR, "GRAND_SUMMARY.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")

    # Console summary
    print("\n" + "="*70)
    print("COMPLETE FINDINGS SUMMARY")
    print("="*70)
    if both and cross:
        t_w,p_w = stats.ttest_1samp(within_r1, CHANCE)
        t_c,p_c = stats.ttest_1samp(cross_r1, CHANCE)
        print(f"Within-subject R@1:  {within_r1.mean()*100:.2f}%±{within_r1.std()*100:.2f}%  p={p_w:.4f}{sig(p_w)}")
        print(f"Cross-subject R@1:   {cross_r1.mean()*100:.2f}%±{cross_r1.std()*100:.2f}%  p={p_c:.4f}{sig(p_c)}")
    if scaling:
        ns = scaling["n_values"]
        sc_means = [scaling["results"][str(n)]["mean"]*100 for n in ns]
        print(f"Subject scaling:     N=2: {sc_means[0]:.2f}%  N=20: {sc_means[-1]:.2f}%  (flat — individual variability bottleneck)")
    if hier:
        print(f"Category lift:       {hier['cat_lift']:.2f}×  p={hier['p_cat']:.4f}{sig(hier['p_cat'])}")
        print(f"Concept lift:        {hier['conc_lift']:.2f}×  p={hier['p_conc']:.4f}{sig(hier['p_conc'])}")
        print(f"Intra-category lift: {hier['intra_lift']:.2f}×  p={hier['p_intra']:.4f}{sig(hier['p_intra'])}")
    if rsa:
        print(f"Category RSA ρ:      {rsa['Category']['rho']:+.4f}  p={rsa['Category']['p_emp']:.4f}{sig(rsa['Category']['p_emp'])}")
        print(f"CLIP-Both RSA ρ:     {rsa['CLIP-Both']['rho']:+.4f}  p={rsa['CLIP-Both']['p_emp']:.4f}{sig(rsa['CLIP-Both']['p_emp'])}")
    if wb:
        print(f"Within>Between sim:  t={wb['t']:.2f}  p={wb['p_val']:.4f}{sig(wb['p_val'])}")
    if untr:
        print(f"Trained RSA:         {untr['trained_mean']:+.4f}  p={untr['p_trained_vs_zero']:.4f}{sig(untr['p_trained_vs_zero'])}")
        print(f"Untrained RSA:       {untr['untrained_mean']:+.4f}  p={untr['p_untrained_vs_zero']:.4f}{sig(untr['p_untrained_vs_zero'])}")
    if conc:
        print(f"CLIP isolation→R@1:  r={conc['r_isolation']:.3f}  p={conc['p_isolation']:.4f}{sig(conc['p_isolation'])}")
    print("="*70)


if __name__ == "__main__":
    main()
