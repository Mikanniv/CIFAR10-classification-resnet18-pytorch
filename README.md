# CIFAR-10 Image Classification Baseline (PyTorch + ResNet18 + MPS)

A clean, minimal, reproducible baseline for CIFAR-10 image classification using PyTorch.  
Designed to be the *starter foundation* before scaling toward stronger training techniques  
(Mixup / CutMix / Label Smoothing / Cosine LR / Strong Augs...).

---

## Motivation

- 用一个最小可控 baseline 熟悉 PyTorch 训练 pipeline  
- 使用 Apple Silicon 上的 **MPS** backend  
- 逐渐加入 SOTA trick，看清楚单独增加每个 trick 带来的 performance delta  
- 模块化拆分

---

## Key Features

- ResNet-18, trained from scratch
- CIFAR-10 dataset
- MPS acceleration support (for Mac)
- Visualization script to show **Target vs Predicted**
- Extremely clean minimal codebase

---

## Project Structure
- .
- ├── main.py # training script
- ├── predict.py # inference + sample visualization
- ├── runs/ # saved model weights
- ├── images/ # inference examples auto saved here
- └── README.md

---

## Environment
- Python 3.10
- PyTorch (with MPS support)
- torchvision
- tensorboard (optional)

## References
- Dataset: CIFAR-10
- Backbone Paper: Deep Residual Learning for Image Recognition (ResNet), He et al.
