### 1. Prediction Consistency within Single-Concept Chunks

| Experiment Name | Chunk Size | Random Max Freq (Expected) | Actual Max Freq | Actual Consistency % |
|---|---|---|---|---|
| **DE** | 10 | 1.78 (17.8%) | 2.43 | **24.3%** |
| **DE_no_data_leak** | 10 | 1.77 (17.7%) | 2.75 | **27.5%** |
| **DE_no_early_stop** | 10 | 1.78 (17.8%) | 2.56 | **25.6%** |
| **DE_run2** | 10 | 1.77 (17.7%) | 2.44 | **24.4%** |
| **PSD** | 10 | 1.77 (17.7%) | 2.48 | **24.8%** |
| **PSD_no_data_leak** | 10 | 1.78 (17.8%) | 2.67 | **26.7%** |
| **PSD_no_early_stop** | 10 | 1.78 (17.8%) | 2.45 | **24.5%** |
| **Raw_EEG** | 5 | 1.23 (24.6%) | 1.36 | **27.2%** |

### 2. Classification Accuracy Across Experiments

| Experiment Name | Top-1 Accuracy (Mean ± Std) | Top-5 Accuracy (Mean ± Std) |
|---|---|---|
| **DE** | 4.37% ± 2.64% | 17.17% ± 5.44% |
| **DE_no_data_leak** | 4.09% ± 2.62% | 16.84% ± 5.19% |
| **DE_no_early_stop** | 4.07% ± 2.55% | 16.99% ± 5.34% |
| **DE_run2** | 4.27% ± 2.57% | 17.19% ± 5.04% |
| **PSD** | 4.15% ± 2.60% | 17.15% ± 4.89% |
| **PSD_no_data_leak** | 4.29% ± 2.29% | 17.13% ± 4.87% |
| **PSD_no_early_stop** | 4.33% ± 2.34% | 16.98% ± 4.58% |
| **Raw_EEG** | 3.79% ± 1.56% | 15.79% ± 3.54% |
