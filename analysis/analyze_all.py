import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
from sklearn.metrics import confusion_matrix

try:
    file_prefix = sys.argv[1]
except IndexError:
    file_prefix = 'PSD_no_data_leak'

out_dir = f'analysis/{file_prefix}'
os.makedirs(out_dir, exist_ok=True)

pred_files = glob.glob(f'./ClassificationResults/40c_top1/{file_prefix}_Predict_Label_*.npy')
pred_files = [f for f in pred_files if 'All_subject_acc' not in f]

# ---------------------------------------------------------
# Part 1: Sequence Shortcut Audit (Run Position)
# ---------------------------------------------------------
pos_accuracies = {1: [], 2: [], 3: [], 4: [], 5: []}

all_preds = []
all_labels = []

for f in pred_files:
    data = np.load(f)
    preds = data[0]
    labels = data[1]
    
    all_preds.extend(preds)
    all_labels.extend(labels)
    
    if len(preds) == 1400:
        preds_reshaped = preds.reshape(280, 5)
        labels_reshaped = labels.reshape(280, 5)
        for clip_idx in range(5):
            acc = np.mean(preds_reshaped[:, clip_idx] == labels_reshaped[:, clip_idx])
            pos_accuracies[clip_idx+1].append(acc)
    else:
        preds_reshaped = preds.reshape(280, 5, 2)
        labels_reshaped = labels.reshape(280, 5, 2)
        for clip_idx in range(5):
            acc = np.mean(preds_reshaped[:, clip_idx, :] == labels_reshaped[:, clip_idx, :])
            pos_accuracies[clip_idx+1].append(acc)

avg_acc_pos = [np.mean(pos_accuracies[i]) for i in range(1, 6)]
std_acc_pos = [np.std(pos_accuracies[i]) for i in range(1, 6)]
print("Run Position Accuracies (Clip 1 to 5):", avg_acc_pos)

plt.figure(figsize=(8, 5))
plt.errorbar(range(1, 6), avg_acc_pos, yerr=std_acc_pos, marker='o', linestyle='-', linewidth=2, capsize=5)
plt.title('RQ1: Top-1 Accuracy by Clip Run-Position')
plt.xlabel('Clip Sequence Position (1st to 5th)')
plt.ylabel('Top-1 Classification Accuracy')
plt.xticks(range(1, 6))
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f'{out_dir}/Audit_Sequence_Shortcut_{file_prefix}.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# Part 2: Dataset Biases Analysis
# ---------------------------------------------------------
block_accuracies = {i: [] for i in range(7)}

for f in pred_files:
    data = np.load(f)
    preds = data[0]
    labels = data[1]
    
    segments_per_block = 200 if len(preds) == 1400 else 400
    preds_blocks = preds.reshape(7, segments_per_block)
    labels_blocks = labels.reshape(7, segments_per_block)
    
    for b in range(7):
        acc = np.mean(preds_blocks[b] == labels_blocks[b])
        block_accuracies[b].append(acc)

# Plot 1: Cognitive Fatigue
avg_block = [np.mean(block_accuracies[i]) for i in range(7)]
std_block = [np.std(block_accuracies[i]) for i in range(7)]

plt.figure(figsize=(8, 5))
plt.errorbar(range(1, 8), avg_block, yerr=std_block, marker='s', color='orange', linestyle='-', linewidth=2, capsize=5)
plt.title('Cognitive Fatigue (Accuracy by Test Session)')
plt.xlabel('Session Block (1st to 7th)')
plt.ylabel('Top-1 Classification Accuracy')
plt.axhline(y=0.025, color='r', linestyle='--', label='Random Chance (2.5%)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f'{out_dir}/Audit_Cognitive_Fatigue_{file_prefix}.png', dpi=300)
plt.close()

# Plot 2: Individual Subject Variation
sub_acc_data = np.load(f'./ClassificationResults/40c_top1/{file_prefix}_All_subject_acc.npy')
plt.figure(figsize=(8, 5))
plt.hist(sub_acc_data * 100, bins=10, color='teal', edgecolor='black')
plt.axvline(np.mean(sub_acc_data)*100, color='red', linestyle='dashed', linewidth=2, label=f"Mean: {np.mean(sub_acc_data)*100:.2f}%")
plt.title('Subject Variability Check')
plt.xlabel('Subject Top-1 Accuracy (%)')
plt.ylabel('Number of Subjects')
plt.legend()
plt.tight_layout()
plt.savefig(f'{out_dir}/Audit_Subject_Variability_{file_prefix}.png', dpi=300)
plt.close()

# Plot 3: 40-Class Confusion Matrix Heatmap
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.title('40-Class Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.tight_layout()
plt.savefig(f'{out_dir}/Audit_Confusion_Matrix_{file_prefix}.png', dpi=300)
plt.close()

print(f"Analyses complete! 4 Plots saved to: {out_dir}/")
print("Mean per block:", avg_block)