# EEG2Video [![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://bcmi.sjtu.edu.cn/home/eeg2video/)

This repository is the official implementation of our NeurIPS 24 paper: [EEG2Video](https://nips.cc/virtual/2024/poster/95156).

**[EEG2Video: Towards Decoding Dynamic Visual Perception from EEG Signals](https://nips.cc/virtual/2024/poster/95156)**
<br/>
[Xuan-Hao Liu](https://scholar.google.com/citations?hl=zh-CN&user=99yIdXAAAAAJ), 
[Yan-Kai Liu](https://scholar.google.com/citations?user=ya-8ObcAAAAJ&hl=zh-CN), 
[Yansen Wang](https://scholar.google.com/citations?user=Hvbzb1kAAAAJ&hl=en), 
[Kan Ren](https://www.saying.ren/), 
[Hanwen Shi](https://github.com/IvyCharon), 
[Zilong Wang](https://scholar.google.com/citations?hl=en&user=gOaxHvMAAAAJ),
[Dongsheng Li](http://recmind.cn/), 
[Bao-Liang Lu](https://bcmi.sjtu.edu.cn/home/blu/), 
[Wei-Long Zheng](https://weilongzheng.github.io/)
<br/>

## 📣 News
- Apr. 24, 2025. We are excited to release the new version of **EEG2Video**.
- Dec. 14, 2024. Our SEED-DV Dataset release.
- Dec. 13, 2024. EEG2Video code release.
- Nov. 25, 2024. EEG-VP code release.
- Sep. 26, 2024. Accepted by NeurIPS 2024.

## Installation

1. Fill out the SEED-DV's [License file](https://cloud.bcmi.sjtu.edu.cn/sharing/o64PBIsIc) and [Apply](https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/) the dataset.

2. Download this repository: ``git clone https://github.com/XuanhaoLiu/EEG2Video.git``

3. Create a conda environment and install the packages necessary to run the code.

```bash
conda create -n eegvideo
conda activate eegvideo
pip install -r requirements.txt
```

## 🖼️ Reconstruction Demos
<table class="center">
      <tr style="line-height: 0">
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      </tr>
      <td style="border: none"><img src="assets/origif/image1.GIF"></td>
      <td style="border: none"><img src="assets/recgif/image1.GIF"></td>
      <td style="border: none"><img src="assets/origif/image2.GIF"></td>
      <td style="border: none"><img src="assets/recgif/image2.GIF"></td>
      <td style="border: none"><img src="assets/origif/image3.GIF"></td>
      <td style="border: none"><img src="assets/recgif/image3.GIF"></td>
      </tr>
      <tr style="line-height: 0">
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      </tr>
      <td style="border: none"><img src="assets/origif/image4.GIF"></td>
      <td style="border: none"><img src="assets/recgif/image4.GIF"></td>
      <td style="border: none"><img src="assets/origif/image7.GIF"></td>
      <td style="border: none"><img src="assets/recgif/image7.GIF"></td>
      <td style="border: none"><img src="assets/origif/image8.GIF"></td>
      <td style="border: none"><img src="assets/recgif/image8.GIF"></td>
      </tr>
      <tr style="line-height: 0">
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      </tr>
      <td style="border: none"><img src="assets/origif/image10.GIF"></td>
      <td style="border: none"><img src="assets/recgif/image10.GIF"></td>
      <td style="border: none"><img src="assets/origif/image15.GIF"></td>
      <td style="border: none"><img src="assets/recgif/image15.GIF"></td>
      <td style="border: none"><img src="assets/origif/image33.GIF"></td>
      <td style="border: none"><img src="assets/recgif/image33.GIF"></td>
      </tr>
  </table>

## 😞 Fail Cases
We present some failure samples, these failures are typically caused by the inability of the model to infer either the semantic information or the low-level visual information correctly, resulting the irrelevantly generated videos.
<table class="center">
      <tr style="line-height: 0">
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      </tr>
      <td style="border: none"><img src="assets/origif/image41.GIF"></td>
      <td style="border: none"><img src="assets/recgif/image41.GIF"></td>
      <td style="border: none"><img src="assets/origif/image43.GIF"></td>
      <td style="border: none"><img src="assets/recgif/image43.GIF"></td>
      <td style="border: none"><img src="assets/origif/image42.GIF"></td>
      <td style="border: none"><img src="assets/recgif/image42.GIF"></td>
      </tr>
  </table>

## Reproduction Process

1. Segment the original videos to 2-second video clips and downsample all 2-second videos to the targeted FPS, e.g., 3 FPS in our paper, resulting in 6 frames of each video.
2. Deploy Tune-A-Video: https://github.com/showlab/Tune-A-Video. After you can use one Input Video to fine-tune Stable Diffusion, trying to use more than one Input Video to fine-tune, in our study, we use 80 (each class 2 videos) Input Videos to fine-tune Stable Diffusion. Also, you can download the fine-tuned weights on SEED-DV at our new paper MindCine's Github: https://github.com/KevinZhou6/MindCine. Remember to save the negative prompts' embeddings.
3. Use Stable Diffusion's VAE encoder to get the visual latents of each video: it will be like shape: frames × channels × H × W.
4. Use Stable Diffusion's Text encoder to get the text embeddings of each video's captions: it will be like shape: 77 × 768.
5. Train a vanilla Transformer, the input is EEG signal (7×C×100), and the output is visual latents: frames × channels × H × W.
6. Train a simple MLP, the input is EEG signal/features, mayebe (C×d), and the output is text embeddings: 77 × 768. You only select one caption for each class, and may adopt pretraining method on a fake dataset (replacing EEG signals by np.ones_like(EEG data) × [1, 2, ..., 40]) to train this MLP.
7. Train a binary classifiers called dyanmic predictor (also an MLP), the input is EEG signal/features, mayebe (C×d), and the output is Fast/Slow, corresponding to the Fast/Slow task in the EEG-VP benchmark.
8. Inference Stage: get one EEG signal, use vanilla Transformer and simple MLP to get the predicted visual latents and text embeddings. Use binary classifiers to get the Fast/Slow information, on which the DANA module's $\beta$ is depend. If you use DANA module, than add the same noise and diverse noise to the visual latents now to get noise_added visual latents. Than put text embeddings, visual latents or noise_added visual latents, and negative prompts' embeddings into fine-tuned Tune-A-Video model to generate videos.
9. Calculate the metrics: prepare the 200 generated videos inferred from EEG signals of the test set and the 200 original videos of the test set. Than use 40_class_run_metrics.py to calculate the video's semantic-level and pixel-level metrics.

## BibTeX
```
@inproceedings{liu2024eegvideo,
    title={{EEG}2Video: Towards Decoding Dynamic Visual Perception from {EEG} Signals},
    author={Liu, Xuan-Hao and Liu, Yan-Kai and Wang, Yansen and Ren, Kan and Shi, Hanwen and Wang, Zilong and Li, Dongsheng and Lu, Bao-Liang and Zheng, Wei-Long},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS)},
    year={2024},
    url={https://openreview.net/forum?id=RfsfRn9OFd}
}
```

## Acknowledgement
Huge thanks to the [Stable Diffusion Team](https://stablediffusionweb.com/) for opensourcing their high-quality AIGC models. Gratitude to the [Tune-A-Video Team](https://tuneavideo.github.io/) for their elegant text-to-video model. And kudos to the [Mind-Video Team](https://www.mind-video.com/) for their pioneering and excellent fMRI-to-video work.

Great thanks to our intelligent friend [**Tianyi Zhou**](https://scholar.google.com/citations?user=VyLD9McAAAAJ) for creating the new version of EEG2Video!

<div align="center">
    <img src="assets/galaxy_brain.gif" alt="galaxy brain" height=100 />
</div>
