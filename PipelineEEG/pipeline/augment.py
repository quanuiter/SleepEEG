"""
pipeline/augment.py
───────────────────
Signal-level augmentation cho raw EEG epochs (3000 samples @ 100Hz).

Mỗi hàm nhận (signal, rng) và trả về signal mới cùng shape (3000,) — KHÔNG
modify input. Dùng `rng = np.random.default_rng(derive_seed(...))` để
reproducible.

Các hàm đều pure NumPy (chỉ augment.py độc lập với torch):

  1. gaussian_noise    : N(0, σ) — σ được random hoá mỗi epoch từ std_range
  2. time_shift        : circular shift ±max_samples (dùng reflect padding)
  3. amplitude_scale   : multiply hằng số ngẫu nhiên scale_range
  4. resnet_mixup      : Mixup trên (B, 1, 3000) batch — soft target labels
                          cho cả CE và Focal.

Lưu ý: augmentation áp lên RAW signal, KHÔNG lên features 128-dim của
ResNet (đó là đầu vào TCN). Vì vậy dataset `_EpochDataset.__getitem__` cần
nhận `transform` callback.

Usage:
    rng = np.random.default_rng(derive_seed(seed, fold, "augment"))
    x_aug = apply_random_augment(x, rng, config=AugmentConfig(...))
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════


@dataclass
class AugmentConfig:
    """Cấu hình signal-level augmentation. Mặc định tất cả OFF."""

    enabled: bool = False

    # gaussian noise (relative to per-signal std)
    apply_gaussian:   bool  = False
    gaussian_std_min: float = 0.0
    gaussian_std_max: float = 0.05

    # time shift (max_seconds ở 100Hz → samples)
    apply_time_shift: bool  = False
    max_shift_seconds: float = 2.0

    # amplitude scale
    apply_amplitude:  bool  = False
    amplitude_min:    float = 0.9
    amplitude_max:    float = 1.1

    # Mixup (chỉ áp lúc training ResNet — không phải epoch-level)
    apply_mixup:      bool  = False
    mixup_alpha:      float = 0.2
    mixup_prob:       float = 0.5

    # Áp mỗi augment với xác suất riêng (nếu None → luôn áp khi enabled)
    gaussian_p: Optional[float] = 1.0
    time_shift_p: Optional[float] = 1.0
    amplitude_p: Optional[float] = 1.0

    def enabled_methods(self) -> List[str]:
        out = []
        if self.apply_gaussian:
            out.append("gaussian")
        if self.apply_time_shift:
            out.append("time_shift")
        if self.apply_amplitude:
            out.append("amplitude")
        return out


# ══════════════════════════════════════════════════════════════════
# Atomic augmentations (pure NumPy, in-place safety: trả về mới)
# ══════════════════════════════════════════════════════════════════


def gaussian_noise(
    signal: np.ndarray,
    rng: np.random.Generator,
    std_min: float = 0.0,
    std_max: float = 0.05,
) -> np.ndarray:
    """
    Thêm Gaussian noise với std được random hoá từ [std_min, std_max] *
    per-signal std. Tỉ lệ theo per-signal std giúp noise thích ứng với
    biên độ tín hiệu (EEG có thể dao động ±100µV nhưng cũng có thể
    ±10µV tuỳ subject).
    """
    sig_std = float(np.std(signal))
    if sig_std <= 0:
        return signal.copy()
    sigma = rng.uniform(std_min, std_max) * sig_std
    noise = rng.normal(0.0, sigma, size=signal.shape).astype(signal.dtype)
    return signal + noise


def time_shift(
    signal: np.ndarray,
    rng: np.random.Generator,
    max_shift_seconds: float = 2.0,
    fs: int = 100,
) -> np.ndarray:
    """
    Dịch tín hiệu ±max_samples. Dùng reflect padding (không `np.roll`)
    để không tạo nhiễu ở biên.

    max_samples = round(max_shift_seconds * fs).
    """
    max_samples = int(round(max_shift_seconds * fs))
    if max_samples <= 0:
        return signal.copy()
    shift = int(rng.integers(-max_samples, max_samples + 1))
    if shift == 0:
        return signal.copy()

    out = np.empty_like(signal)
    n = signal.shape[0]
    if shift > 0:
        # Tín hiệu dịch phải: phần đầu lấy từ reflect(0..shift)
        # reflect values: 0, 1, 2, ... shift-1 → source: shift, shift-1, ..., 1
        src_left  = signal[shift:0:-1] if shift > 1 else signal[1:2]
        # pad nếu shift quá lớn
        if len(src_left) < shift:
            extra = np.pad(src_left, (0, shift - len(src_left)), mode="edge")
            src_left = extra
        out[:shift] = src_left[:shift]
        out[shift:] = signal[:n - shift]
    else:
        k = -shift
        # Tín hiệu dịch trái: phần cuối lấy từ reflect
        # signal: 0, 1, 2, ..., n-1
        # cần lấp: out[n-k:] = signal[n-k-2 : n-k-2-k : -1]
        end_src = signal[n - k - 2 : n - k - 2 - k : -1] if k > 1 else signal[n - k - 2 : n - k - 1]
        if len(end_src) < k:
            extra = np.pad(end_src, (0, k - len(end_src)), mode="edge")
            end_src = extra
        out[n - k:] = end_src[:k]
        out[:n - k] = signal[k:]
    return out


def amplitude_scale(
    signal: np.ndarray,
    rng: np.random.Generator,
    scale_min: float = 0.9,
    scale_max: float = 1.1,
) -> np.ndarray:
    """Multiply signal với 1 scalar trong [scale_min, scale_max]."""
    s = float(rng.uniform(scale_min, scale_max))
    return signal * s


# ══════════════════════════════════════════════════════════════════
# Composite epoch-level augment
# ══════════════════════════════════════════════════════════════════


def _maybe_apply(
    signal: np.ndarray,
    rng: np.random.Generator,
    p: Optional[float],
    fn: Callable[..., np.ndarray],
    **kwargs,
) -> np.ndarray:
    """Apply fn với xác suất p (nếu p không None), luôn apply nếu p is None."""
    if p is None or rng.random() < p:
        return fn(signal, rng, **kwargs)
    return signal


def apply_epoch_augment(
    signal: np.ndarray,
    rng: np.random.Generator,
    cfg: AugmentConfig,
    fs: int = 100,
) -> np.ndarray:
    """
    Áp dụng chain augmentation theo AugmentConfig lên 1 epoch.

    Args:
        signal : (n_samples,) float32/float64
        rng    : np.random.Generator (để reproducible)
        cfg    : AugmentConfig
        fs     : sample rate (mặc định 100Hz = TARGET_FS)

    Returns:
        augmented signal cùng shape/dtype.
    """
    if not cfg.enabled:
        return signal.copy()

    out = signal.copy()
    if cfg.apply_gaussian:
        out = _maybe_apply(
            out, rng, cfg.gaussian_p, gaussian_noise,
            std_min=cfg.gaussian_std_min, std_max=cfg.gaussian_std_max,
        )
    if cfg.apply_time_shift:
        out = _maybe_apply(
            out, rng, cfg.time_shift_p, time_shift,
            max_shift_seconds=cfg.max_shift_seconds, fs=fs,
        )
    if cfg.apply_amplitude:
        out = _maybe_apply(
            out, rng, cfg.amplitude_p, amplitude_scale,
            scale_min=cfg.amplitude_min, scale_max=cfg.amplitude_max,
        )
    return out


# ══════════════════════════════════════════════════════════════════
# Mixup (batch-level, dùng cho ResNet training)
# ══════════════════════════════════════════════════════════════════


def mixup_batch(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    alpha: float = 0.2,
    prob: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mixup trên batch. Trả về (x_mixed, y_soft) — y_soft là phân phối mềm
    `(N, C)` tương thích với cả CrossEntropy (dùng one-hot) và Focal
    (label_smoothing).

    Args:
        x     : (N, ...) input batch
        y     : (N,) int64 class indices
        rng   : np.random.Generator
        alpha : Beta distribution α. α=0 → no mixup, α lớn → mạnh hơn.
                Paper đề xuất α=0.2 cho supervised.
        prob  : xác suất áp mixup cho batch đó. prob=1.0 → luôn áp.

    Returns:
        x_mixed : cùng shape với x
        y_soft  : (N, n_classes) one-hot soft (kết hợp original label + shuffled label)
    """
    if rng.random() >= prob or alpha <= 0:
        # Không mixup → one-hot
        n = x.shape[0]
        # Suy ra n_classes từ max(y)+1, fallback 5
        n_classes = int(y.max()) + 1 if y.size > 0 else 5
        y_soft = np.zeros((n, n_classes), dtype=np.float32)
        y_soft[np.arange(n), y] = 1.0
        return x, y_soft

    lam = float(rng.beta(alpha, alpha))
    n = x.shape[0]
    perm = rng.permutation(n)

    # Mix
    x_mixed = lam * x + (1.0 - lam) * x[perm]
    # Soft target = lam * one_hot(y) + (1-lam) * one_hot(y[perm])
    n_classes = int(max(y.max(), y[perm].max())) + 1
    y_soft = np.zeros((n, n_classes), dtype=np.float32)
    y_soft[np.arange(n), y]         = lam
    y_soft[np.arange(n), y[perm]]  += (1.0 - lam)
    return x_mixed, y_soft


def mixup_soft_to_focal_targets(
    y_soft: np.ndarray,
    smoothing: float = 0.0,
) -> np.ndarray:
    """
    (Optional) thêm label_smoothing vào soft target trước khi đưa vào
    FocalLoss. smoothing=0 → identity.
    """
    if smoothing <= 0:
        return y_soft
    n_classes = y_soft.shape[1]
    return (1.0 - smoothing) * y_soft + smoothing / n_classes


__all__ = [
    "AugmentConfig",
    "gaussian_noise",
    "time_shift",
    "amplitude_scale",
    "apply_epoch_augment",
    "mixup_batch",
    "mixup_soft_to_focal_targets",
]
