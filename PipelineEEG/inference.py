"""
pipeline/inference.py
─────────────────────
End-to-end inference pipeline:
  Raw EDF → Preprocess → ResNet Feature Extraction → TCN Classification

Hỗ trợ:
  - File EDF đơn (không có nhãn) → inference thuần
  - Cặp PSG.edf + Hypnogram.edf → inference + so sánh với nhãn thực

Kết quả: dict chứa predicted stages, raw signal, features, v.v.
"""

import gc
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import resample

try:
    import pyedflib
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyedflib", "-q"])
    import pyedflib

import torch
import torch.nn as nn

from pipeline.models import EEG_ResNet1D, SleepTCN
from pipeline.preprocess import (
    TARGET_FS, EPOCH_SEC,
    bandpass_filter, notch_filter, clip_and_scale,
    read_labels_sleepedf, STAGE_DICT,
    list_edf_channels,
)

SLEEP_STAGES   = ["W", "N1", "N2", "N3", "REM"]
STAGE_COLORS   = {
    "W":   "#e74c3c",
    "N1":  "#f39c12",
    "N2":  "#3498db",
    "N3":  "#2c3e50",
    "REM": "#2ecc71",
}
N_CLASSES      = 5
EPOCH_SAMPLES  = int(EPOCH_SEC * TARGET_FS)   # 3000


# ══════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════

@dataclass
class InferenceConfig:
    """Cấu hình inference pipeline."""
    # EDF channel
    select_ch: str = "EEG Fpz-Cz"

    # Filter
    apply_bandpass: bool  = True
    bp_low:         float = 0.5
    bp_high:        float = 30.0
    bp_order:       int   = 4
    apply_notch:    bool  = True
    notch_freq:     float = 50.0
    notch_q:        float = 30.0

    # Normalization
    clip_uv:      float = 800.0
    scale_factor: float = 100.0

    # Model architecture (phải khớp với checkpoint)
    n_feat:          int   = 128
    tcn_dim:         int   = 128
    tcn_kernel_size: int   = 3
    tcn_blocks:      int   = 6
    tcn_dropout:     float = 0.2

    # Callback
    log_cb: Optional[Callable[[str], None]] = field(default=None, repr=False)


# ══════════════════════════════════════════════════════════════════
# Preprocess single EDF (không cần nhãn)
# ══════════════════════════════════════════════════════════════════

def preprocess_raw_edf(
    edf_path: str,
    cfg: InferenceConfig,
) -> Tuple[np.ndarray, float, int, np.ndarray]:
    """
    Đọc và tiền xử lý 1 file EDF.

    Returns:
        signals     : (n_epochs, 3000) float32 — epochs đã lọc + chuẩn hoá
        fs_orig     : tần số lấy mẫu gốc
        n_all_epochs: tổng số epoch gốc
        raw_signal  : tín hiệu thô (chưa lọc) để trực quan hoá
    """
    _log(cfg.log_cb, f"Đọc EDF: {Path(edf_path).name}")

    f = pyedflib.EdfReader(edf_path)
    ch_names = f.getSignalLabels()

    ch_idx = -1
    for s in range(f.signals_in_file):
        if ch_names[s] == cfg.select_ch:
            ch_idx = s
            break

    if ch_idx == -1:
        f.close()
        raise ValueError(
            f"Channel '{cfg.select_ch}' không tồn tại trong file.\n"
            f"Các channel có sẵn: {', '.join(ch_names)}"
        )

    fs_orig      = f.getSampleFrequency(ch_idx)
    n_all_epochs = int(f.getFileDuration() / EPOCH_SEC)
    raw_signal   = f.readSignal(ch_idx).astype(np.float64)
    f.close()

    _log(cfg.log_cb, f"  Channel: {cfg.select_ch}  |  fs={fs_orig}Hz  |  "
                     f"{n_all_epochs} epochs  |  {len(raw_signal)} samples")

    # Lưu bản gốc để trực quan hoá
    raw_for_viz = raw_signal.copy()

    # Lọc tín hiệu
    if cfg.apply_bandpass:
        raw_signal = bandpass_filter(raw_signal, fs_orig, cfg.bp_low, cfg.bp_high, cfg.bp_order)
        _log(cfg.log_cb, f"  ✓ Bandpass {cfg.bp_low}-{cfg.bp_high} Hz (order {cfg.bp_order})")
    if cfg.apply_notch:
        raw_signal = notch_filter(raw_signal, fs_orig, cfg.notch_freq, cfg.notch_q)
        _log(cfg.log_cb, f"  ✓ Notch {cfg.notch_freq} Hz (Q={cfg.notch_q})")

    # Resample về TARGET_FS
    if fs_orig != TARGET_FS:
        n_target   = round(len(raw_signal) * TARGET_FS / fs_orig)
        raw_signal = resample(raw_signal, n_target)
        # Resample raw_for_viz cũng vậy
        raw_for_viz = resample(raw_for_viz, n_target)
        _log(cfg.log_cb, f"  ✓ Resample {fs_orig}Hz → {TARGET_FS}Hz")

    # Chia epoch
    n_ep_samples = int(EPOCH_SEC * TARGET_FS)  # 3000
    n_complete   = len(raw_signal) // n_ep_samples
    raw_signal   = raw_signal[:n_complete * n_ep_samples]
    raw_for_viz  = raw_for_viz[:n_complete * n_ep_samples]

    signals = np.stack([
        clip_and_scale(s, cfg.clip_uv, cfg.scale_factor)
        for s in raw_signal.reshape(-1, n_ep_samples)
    ])

    raw_epochs = raw_for_viz.reshape(-1, n_ep_samples)

    _log(cfg.log_cb, f"  ✓ {len(signals)} epochs ({n_ep_samples} samples/epoch)")
    return signals, fs_orig, n_all_epochs, raw_epochs


# ══════════════════════════════════════════════════════════════════
# Inference pipeline
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(
    signals: np.ndarray,
    resnet_ckpt: str,
    tcn_ckpt: str,
    cfg: InferenceConfig,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    signals: (n_epochs, 3000) float32
    Returns dict with:
      - predictions: np.ndarray (n_epochs,) — predicted stage indices
      - stage_names: list of str — ["W", "N1", ...]
      - probabilities: np.ndarray (n_epochs, 5) — softmax probs
      - features: np.ndarray (n_epochs, n_feat) — ResNet features
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _log(cfg.log_cb, f"Device: {device}")

    # ── Load ResNet ───────────────────────────────────────────────
    resnet = EEG_ResNet1D(cfg.n_feat, N_CLASSES).to(device)
    try:
        # Try weights_only first (PyTorch 2.0+)
        resnet.load_state_dict(
            torch.load(resnet_ckpt, map_location=device, weights_only=True)
        )
    except TypeError:
        # Fallback for older PyTorch versions
        resnet.load_state_dict(
            torch.load(resnet_ckpt, map_location=device)
        )
    resnet.eval()
    _log(cfg.log_cb, f"  ✓ ResNet loaded: {Path(resnet_ckpt).name}")

    # ── Extract features ──────────────────────────────────────────
    _log(cfg.log_cb, "  Extracting features ...")
    feats_chunks = []
    batch_size   = 128
    for i in range(0, len(signals), batch_size):
        xb = torch.from_numpy(signals[i:i+batch_size]).unsqueeze(1).to(device)
        feats_chunks.append(resnet(xb, extract_features=True).cpu().numpy())
    features = np.concatenate(feats_chunks, axis=0)
    _log(cfg.log_cb, f"  ✓ Features: {features.shape}")

    del resnet
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Load TCN ──────────────────────────────────────────────────
    tcn = SleepTCN(
        input_size=cfg.n_feat, dim=cfg.tcn_dim,
        kernel_size=cfg.tcn_kernel_size, n_blocks=cfg.tcn_blocks,
        dropout=cfg.tcn_dropout, n_classes=N_CLASSES,
    ).to(device)
    try:
        # Try weights_only first (PyTorch 2.0+)
        tcn.load_state_dict(
            torch.load(tcn_ckpt, map_location=device, weights_only=True)
        )
    except TypeError:
        # Fallback for older PyTorch versions
        tcn.load_state_dict(
            torch.load(tcn_ckpt, map_location=device)
        )
    tcn.eval()
    _log(cfg.log_cb, f"  ✓ TCN loaded: {Path(tcn_ckpt).name}")

    # ── Classify ──────────────────────────────────────────────────
    _log(cfg.log_cb, "  Classifying ...")
    xb     = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(device)
    logits = tcn(xb).squeeze(0).cpu().numpy()          # (T, 5)

    # Softmax
    exp_logits    = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    predictions = logits.argmax(axis=1)                # (T,)
    stage_names = [SLEEP_STAGES[p] for p in predictions]

    del tcn
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    _log(cfg.log_cb, f"  ✓ Phân lớp xong: {len(predictions)} epochs")

    # ── Summary ───────────────────────────────────────────────────
    unique, counts = np.unique(predictions, return_counts=True)
    dist = {SLEEP_STAGES[u]: int(c) for u, c in zip(unique, counts)}
    _log(cfg.log_cb, f"  Phân bố: {dist}")

    return {
        "predictions":   predictions,
        "stage_names":   stage_names,
        "probabilities": probabilities,
        "features":      features,
        "stage_dist":    dist,
    }


# ══════════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════════

def _log(cb, msg: str):
    if cb:
        cb(msg)
    else:
        print(msg)
