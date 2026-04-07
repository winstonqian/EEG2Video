import numpy as np
import glob
import os
from collections import Counter

def get_random_baseline(chunk_size, num_classes=40, trials=10000):
    max_counts = []
    for _ in range(trials):
        random_preds = np.random.randint(0, num_classes, chunk_size)
        max_counts.append(Counter(random_preds).most_common(1)[0][1])
    return np.mean(max_counts)

def generate_tables():
    # ---------------------------------------------------------
    # 1. Consistency Table
    # ---------------------------------------------------------
    pred_files = glob.glob('ClassificationResults/40c_top1/*_Predict_Label_sub*.npy')
    experiments = {}
    for f in pred_files:
        exp_name = os.path.basename(f).split('_Predict_Label_')[0].replace('GLMNet_', '')
        if exp_name not in experiments: experiments[exp_name] = []
        experiments[exp_name].append(f)

    with open('analysis/summary_tables.md', 'w') as out:
        out.write("### 1. Prediction Consistency within Single-Concept Chunks\n\n")
        out.write("| Experiment Name | Chunk Size | Random Max Freq (Expected) | Actual Max Freq | Actual Consistency % |\n")
        out.write("|---|---|---|---|---|\n")
        for exp_name, files in sorted(experiments.items()):
            all_max_counts, chunk_sizes = [], []
            for f in files:
                data = np.load(f)
                preds, labels = data[0], data[1]
                changes = np.where(labels[:-1] != labels[1:])[0] + 1
                starts = np.insert(changes, 0, 0)
                ends = np.append(changes, len(labels))
                for start, end in zip(starts, ends):
                    chunk_preds = preds[start:end]
                    cnt = Counter(chunk_preds)
                    all_max_counts.append(cnt.most_common(1)[0][1])
                    chunk_sizes.append(end - start)
            if not chunk_sizes: continue
            mean_sz = int(np.median(chunk_sizes))
            actual_mean = np.mean(all_max_counts)
            random_mean = get_random_baseline(mean_sz)
            
            out.write(f"| **{exp_name}** | {mean_sz} | {random_mean:.2f} ({(random_mean/mean_sz)*100:.1f}%) | {actual_mean:.2f} | **{(actual_mean/mean_sz)*100:.1f}%** |\n")
        
        # ---------------------------------------------------------
        # 2. Accuracy Table
        # ---------------------------------------------------------
        out.write("\n### 2. Classification Accuracy Across Experiments\n\n")
        out.write("| Experiment Name | Top-1 Accuracy (Mean ± Std) | Top-5 Accuracy (Mean ± Std) |\n")
        out.write("|---|---|---|\n")
        top1_files = glob.glob('ClassificationResults/40c_top1/*_All_subject_acc.npy')
        for f in sorted(top1_files):
            exp_name = os.path.basename(f).replace('_All_subject_acc.npy', '').replace('GLMNet_', '')
            t1 = np.load(f) * 100
            
            f5 = f.replace('40c_top1', '40c_top5')
            if os.path.exists(f5):
                t5 = np.load(f5) * 100
                top5_str = f"{np.mean(t5):.2f}% ± {np.std(t5):.2f}%"
            else:
                top5_str = "N/A"
                
            out.write(f"| **{exp_name}** | {np.mean(t1):.2f}% ± {np.std(t1):.2f}% | {top5_str} |\n")

if __name__ == "__main__":
    generate_tables()
    print("Markdown tables successfully generated in analysis/summary_tables.md")
