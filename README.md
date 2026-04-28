#Deep Learning Optimization for Single-Channel EEG Sleep Staging

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An optimized, end-to-end Deep Learning pipeline for automated sleep stage classification using **single-channel EEG (Fpz-Cz)**. This project redesigns the traditional multi-CNN approach by replacing 15 shallow feature extractors with a unified **ResNet-1D** backbone, then evaluates two advanced sequence models — **TCN** and **SleepConformer** — on the Sleep-EDF-18 benchmark.

---

## Key Contributions

- **Amplitude-Preserving Preprocessing:** Replaced standard Z-score normalization with a 0.5–30Hz bandpass filter, 50Hz notch filter, and static scaling (/100) to preserve physiological amplitude differences (e.g., Delta waves).
- **XAI-Driven Architecture Simplification:** Applied Mutual Information, Random Forest MDI, and Permutation Importance across 75 spatial-temporal features — revealing 66% were highly correlated (|r| > 0.95) and contributed negligible predictive value. Replaced 15 redundant shallow CNNs with a single ResNet-1D extracting 128-dimensional embeddings.
- **Advanced Sequence Modeling:** Designed and benchmarked two architectures:
  - **TCN (Vanilla):** Captures long-range local patterns with a receptive field of 249 epochs.
  - **SleepConformer:** Hybrid CNN-Attention model augmented with **Supervised Contrastive Learning (SCL)** and Cross-Entropy class-weighting to handle severe N1 stage imbalance.
- **Strict Evaluation Protocol:** Subject-wise 10-Fold Cross-Validation with hard temporal boundaries — eliminating cross-record data leakage.

---

## Results on Sleep-EDF-18

| Architecture | Preprocessing | Accuracy | Macro F1 | F1 N1 | F1 REM |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ResNet-1D + TCN** | Bandpass + Scale | **83.89%** | **78.53%** | **49.40%** | 83.20% |
| **ResNet-1D + SleepConformer** | Bandpass + Scale | 83.84% | 78.28% | 48.25% | **83.28%** |

> Both models use identical 128-dim spatial embeddings extracted from the pre-trained ResNet-1D backbone.

---

## Pipeline Architecture

```
Raw EEG
  └─► Preprocessing (preprocess.py)
        Bandpass 0.5–30Hz → Notch 50Hz → Scale /100 → Epoch 30s
            │
            ▼
  Phase 1: Representation Learning (extract_resnet.py)
        ResNet-1D + Early Stopping → (B, T, 128) embeddings
            │
            ▼
  Phase 2: Sequence Modeling
        ├── TCN Vanilla       (train_tcn.py)
        └── SleepConformer    (train_conformer.py)
              SCL Loss + Sequence Augmentation + N1 Class-Weighting
```

---

## Repository Structure

```
SleepEEG/
├── data/                          # Dataset directory (NPZ format)
├── preprocessing/
│   └── preprocess.py              # Signal filtering and epoch extraction
├── models/
│   ├── resnet1d.py                # ResNet-1D backbone
│   ├── tcn.py                     # Temporal Convolutional Network
│   └── conformer.py               # SleepConformer with SCL projection head
├── extract_resnet.py              # Train ResNet & extract 128-dim features
├── train_tcn.py                   # Train TCN sequence model
├── train_conformer.py             # Train SleepConformer with SCL & reweighting
└── README.md
```

---

## Getting Started

### Requirements

```bash
pip install torch torchvision torchaudio numpy pandas scipy scikit-learn seaborn matplotlib tqdm
```

### 1. Data Preparation

Download the [Sleep-EDF-18 dataset](https://physionet.org/content/sleep-edfx/1.0.0/) from PhysioNet, then run preprocessing:

```bash
python preprocessing/preprocess.py --data_dir /path/to/raw/edf --output_dir ./data
```

### 2. Phase 1 — Feature Extraction

Train the ResNet-1D backbone and extract sequence embeddings:

```bash
python extract_resnet.py
```

Checkpoints and `.npy` feature files will be saved to `./outputs/`.

### 3. Phase 2 — Sequence Training

**TCN (Vanilla):**
```bash
python train_tcn.py
```

**SleepConformer (with SCL + SeqAug):**
```bash
python train_conformer.py
```

---

## Visualizations

| Confusion Matrix (TCN) | Hypnogram |
|:---:|:---:|
| *(Insert cm_tcn.png)* | *(Insert hypnogram.png)* |

> Tip: Upload your confusion matrix and hypnogram plots to the repo and replace the placeholders above with the image paths.

---

## 📝 Acknowledgments

- Dataset: [PhysioNet Sleep-EDF Database](https://physionet.org/content/sleep-edfx/1.0.0/)
- Conformer architecture inspired by [Gulati et al., 2020](https://arxiv.org/abs/2005.08100)
- Supervised Contrastive Learning adapted from [Khosla et al., 2020](https://arxiv.org/abs/2004.11362)
