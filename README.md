# Deep Learning Optimization for Single-Channel EEG Sleep Stage Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository presents an optimized, end-to-end deep learning pipeline for automated sleep stage classification utilizing single-channel EEG (Fpz-Cz) signals. The project reproduces and significantly enhances the traditional ZleepAnlystNet methodology. By replacing the conventional two-stage architecture—which previously relied on 15 shallow Convolutional Neural Networks (CNNs) and a Bidirectional Long Short-Term Memory (BiLSTM) network—this pipeline introduces a unified ResNet-1D backbone for robust feature extraction paired with a non-causal SleepTCN for sequence modeling.

---

## Key Contributions

* **Amplitude-Preserving Preprocessing Pipeline:** The standard Z-score normalization has been replaced with a comprehensive signal processing approach. This includes a 0.5–30Hz Butterworth bandpass filter to eliminate DC drift and high-frequency noise, a 50Hz notch filter for powerline interference, clipping at ±800µV to remove extreme artifacts, and constant scaling (/100). This strategy preserves the absolute physiological amplitude differences essential for distinguishing sleep stages, particularly the high-amplitude Delta waves in the N3 stage.
* **XAI-Driven Architecture Simplification:** Quantitative feature analysis revealed that 10 out of the 15 original CNNs contributed only 12% of the total predictive information, with severe redundancy across features (correlation > 0.95). The 15 CNNs were systematically replaced with a single ResNet-1D model containing approximately 475,000 parameters. This consolidation generates a continuous 128-dimensional embedding per epoch and reduces the overall training time by a factor of 8.2 (from 12.35 hours to 1.5 hours).
* **Advanced Sequence Modeling via SleepTCN:** The sequential processing bottleneck of the BiLSTM was addressed by implementing a non-causal Temporal Convolutional Network (SleepTCN). Configured with 6 blocks and symmetric padding, the SleepTCN achieves a receptive field of 253 epochs (approximately 126.5 minutes), effectively capturing bidirectional temporal context while enabling full parallelization during training.
* **Label Noise Mitigation:** Epochs marked as Movement or Unknown are explicitly handled using an `ignore_index=-100` mechanism during cross-entropy loss computation. This preserves the temporal sequence continuity without erroneously polluting the Wake class distribution with signal artifacts.
* **Cross-Dataset Generalization:** The architecture's robustness was rigorously validated on the cross-domain Sleep Heart Health Study (SHHS) dataset, demonstrating stable performance on clinical data from patients with sleep apnea using different recording equipment and montages.

---

## Experimental Results

### Performance on Sleep-EDF-78 (10-Fold Subject-Wise Cross-Validation)

| Architecture | Preprocessing | Overall Accuracy | Macro F1 Score | N1 F1 Score | REM F1 Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ResNet-1D + SleepTCN** | Bandpass + Notch + Scale | **83.89%** | **78.53%** | **49.40%** | **83.20%** |
| ResNet-1D + SleepConformer | Bandpass + Notch + Scale | 83.84% | 78.28% | 48.25% | 83.28% |
| Baseline (15 CNN + BiLSTM) | None (Raw Signal) | 83.00% | 76.67% | 48.31% | 82.85% |

### Cross-Dataset Evaluation on SHHS

| Evaluation Scenario | Preprocessing | Accuracy | Observations |
| :--- | :--- | :---: | :--- |
| Zero-shot (Sleep-EDF weights) | Applied | 73.89% | Demonstrates strong generalization across different demographic and hardware domains. |
| Zero-shot (Sleep-EDF weights) | None | 22.54% | Total model failure; quantitatively proves the critical necessity of the preprocessing pipeline. |
| Retraining (5-Fold CV) | Applied | 84.17% | Confirms the proposed architecture's capacity is not restricted to a specific dataset. |

---

## Repository Structure

```text
PipelineEEG/
├── app.py                    # Main Streamlit user interface
├── pipeline/
│   ├── __init__.py
│   ├── models.py             # ResNet-1D and SleepTCN neural network architectures
│   ├── preprocess.py         # Signal filtering, clipping, and EDF to NPZ conversion
│   └── trainer.py            # Orchestrator for cross-validation splits, training, and evaluation
└── weight/
weight for resnet and tcn model
```

---

## Getting Started

### 1. Prerequisites

Ensure all dependencies are installed via pip:

```bash
pip install streamlit torch torchvision torchaudio numpy scipy pyedflib scikit-learn matplotlib seaborn pandas tqdm
```

### 2. Launching the User Interface

A comprehensive Streamlit UI is provided for seamless execution of the entire pipeline. The interface includes dedicated tabs for Preprocessing, Training, Evaluation, and Results analysis:

```bash
cd sleep_pipeline
streamlit run app.py
```

### 3. Programmatic Execution (Python / Jupyter)

For integration into automated workflows or custom scripts, the modules can be invoked directly:

**Step 3.1: Data Preprocessing**

```python
from pipeline.preprocess import PreprocessConfig, run_preprocess

pp_cfg = PreprocessConfig(
    data_dir="./data/sleepedf/sleep-cassette",
    output_dir="./preprocessed",
    select_ch="EEG Fpz-Cz",
    apply_bandpass=True, bp_low=0.5, bp_high=30.0,
    apply_notch=True, notch_freq=50.0,
)
run_preprocess(pp_cfg)
```

**Step 3.2: Model Training**

```python
from pipeline.trainer import PipelineConfig, run_full_pipeline

cfg = PipelineConfig(
    npz_dir="./preprocessed",
    output_dir="./results",
    ckpt_dir="./results/checkpoints",
    n_folds=10,
    seed=123
)
results = run_full_pipeline(cfg, train_resnet=True, train_tcn=True)
```

**Step 3.3: Inference and Evaluation**

```python
cfg_new = PipelineConfig(npz_dir="./new_data", output_dir="./results_new")
results = run_full_pipeline(
    cfg_new,
    resnet_ckpt_dir="./results/checkpoints",
    tcn_ckpt_dir="./results/checkpoints",
    train_resnet=False,
    train_tcn=False,
)
```

---

## Acknowledgments

* **Datasets:** The research utilized the [PhysioNet Sleep-EDF Database Expanded](https://physionet.org/content/sleep-edfx/1.0.0/) and a subset from the [Sleep Heart Health Study (SHHS)](https://sleepdata.org/datasets/shhs).
* **Architecture Foundations:** The Temporal Convolutional Network implementation is based on the theoretical frameworks introduced by Bai et al. (2018), and the residual connections are adapted from He et al. (2016).
005.08100)
- Supervised Contrastive Learning adapted from [Khosla et al., 2020](https://arxiv.org/abs/2004.11362)
