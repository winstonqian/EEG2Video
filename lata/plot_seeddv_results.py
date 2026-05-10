"""
Plot LATA SEED-DV training results.
Produces: lata_seeddv_results.png
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── Data from training run ────────────────────────────────────────────────────
# Subject 1, 200 epochs, K=4 chunks (0.5s each), max_delay=3
delay_dist_final = np.array([0.172, 0.276, 0.312, 0.239])   # learned w at epoch 200
expected_delay   = 1.619   # chunks
best_val_epoch   = 1       # best val was at epoch 1 (overfitting observed)

# Epoch logs (every 20 epochs + epoch 1)
log_epochs      = [1,   20,   40,   60,   80,   100,  120,  140,  160,  180,  200]
log_train_loss  = [5.567, 4.990, 4.466, 4.027, 3.711, 3.445, 3.251, 3.118, 3.039, 3.004, 3.008]
log_val_loss    = [5.037, 5.687, 5.996, 6.194, 6.410, 6.768, 6.773, 6.415, 6.574, 6.548, 6.536]

# Delay distribution at key epochs (hand-traced from logs)
delay_history = {
    1:   [0.249, 0.249, 0.251, 0.251],
    100: [0.177, 0.281, 0.308, 0.234],
    200: [0.172, 0.276, 0.312, 0.239],
}

C_BLUE  = "#005b96"
C_LIGHT = "#b3cde0"
C_RED   = "#c0392b"
C_GRAY  = "#888888"
C_GREEN = "#27ae60"

fig = plt.figure(figsize=(14, 6))
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(1, 3, wspace=0.38)

delay_ticks = list(range(4))   # δ ∈ {0,1,2,3}
delay_ms    = [d * 500 for d in delay_ticks]   # ms at 0.5s/chunk

# ── Panel 1: Final learned delay distribution ─────────────────────────────────
ax1 = fig.add_subplot(gs[0])
peak = int(np.argmax(delay_dist_final))
colors = [C_BLUE if i == peak else C_LIGHT for i in delay_ticks]
bars = ax1.bar(delay_ticks, delay_dist_final, color=colors, edgecolor="white",
               linewidth=0.8, zorder=3, width=0.6)
ax1.axvline(peak, color=C_RED, linestyle="--", linewidth=1.8,
            label=f"Peak δ = {peak} ({peak * 500} ms)", zorder=4)
# Annotate values on bars
for bar, v in zip(bars, delay_dist_final):
    ax1.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.3f}",
             ha="center", va="bottom", fontsize=8, color="#333")
ax1.set_title("Learned Delay Distribution\n(Subject 1, SEED-DV, epoch 200)",
              fontsize=11, fontweight="bold", color="#011f4b")
ax1.set_xlabel("Delay δ (chunks)", fontsize=9)
ax1.set_ylabel("P(δ)", fontsize=9)
ax1.set_xticks(delay_ticks)
ax1.set_xticklabels([f"δ={d}\n({d*500}ms)" for d in delay_ticks], fontsize=8)
ax1.set_ylim(0, 0.45)
ax1.legend(fontsize=8.5, framealpha=0.8)
ax1.grid(axis="y", alpha=0.3, linestyle="--")
ax1.spines[["top", "right"]].set_visible(False)
ax1.text(0.97, 0.97, f"E[δ] = {expected_delay:.2f} chunks\n≈ {expected_delay*500:.0f} ms",
         transform=ax1.transAxes, ha="right", va="top", fontsize=8.5,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))

# ── Panel 2: Training / validation loss ───────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.plot(log_epochs, log_train_loss, color=C_BLUE,  linewidth=2.0, marker="o",
         markersize=4, label="Train InfoNCE")
ax2.plot(log_epochs, log_val_loss,   color=C_RED,   linewidth=2.0, marker="s",
         markersize=4, linestyle="--", label="Val InfoNCE")
ax2.axhline(np.log(4 * 7 * 40 * 5),   color=C_GRAY, linestyle=":", linewidth=1, alpha=0.6,
            label="Chance (log N)")
ax2.set_title("InfoNCE Loss vs Epoch\n(Subject 1, 6 train sessions)",
              fontsize=11, fontweight="bold", color="#011f4b")
ax2.set_xlabel("Epoch", fontsize=9)
ax2.set_ylabel("InfoNCE Loss", fontsize=9)
ax2.legend(fontsize=8.5, framealpha=0.8)
ax2.grid(alpha=0.3, linestyle="--")
ax2.spines[["top", "right"]].set_visible(False)

# ── Panel 3: Delay distribution evolution ─────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
epochs_shown = [1, 100, 200]
colors_ev    = ["#cce5ff", "#6699cc", C_BLUE]
x = np.array(delay_ticks, dtype=float)
bar_w = 0.22
for i, ep in enumerate(epochs_shown):
    ax3.bar(x + (i - 1) * bar_w, delay_history[ep], width=bar_w,
            color=colors_ev[i], edgecolor="white", linewidth=0.5,
            label=f"Epoch {ep}", zorder=3)
ax3.axhline(0.25, color=C_GRAY, linestyle=":", linewidth=1.2, label="Uniform (0.25)", alpha=0.7)
ax3.axvline(peak, color=C_RED, linestyle="--", linewidth=1.5, alpha=0.7, zorder=4)
ax3.set_title("Delay Distribution Evolution\n(δ=2 preference emerges with training)",
              fontsize=11, fontweight="bold", color="#011f4b")
ax3.set_xlabel("Delay δ (chunks)", fontsize=9)
ax3.set_ylabel("P(δ)", fontsize=9)
ax3.set_xticks(delay_ticks)
ax3.set_xticklabels([f"δ={d}" for d in delay_ticks], fontsize=9)
ax3.set_ylim(0, 0.42)
ax3.legend(fontsize=8.5, framealpha=0.8, loc="upper right")
ax3.grid(axis="y", alpha=0.3, linestyle="--")
ax3.spines[["top", "right"]].set_visible(False)

fig.suptitle(
    "LATA on SEED-DV: EEG–Video Latency Learning (Subject 1, 1400 training clips)",
    fontsize=13, fontweight="bold", color="#011f4b", y=1.02
)

out = os.path.join(os.path.dirname(__file__), "lata_seeddv_results.png")
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
plt.show()
