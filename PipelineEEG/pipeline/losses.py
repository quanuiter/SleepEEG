"""
pipeline/losses.py
──────────────────
Loss functions cho pipeline SleepEEG:

  - FocalLoss       : class-imbalanced focal loss, hữu ích với N1 (chỉ ~6% data)
  - compute_class_weights : inverse-frequency weights cho CE (baseline cũ)
  - build_loss      : factory chọn CE / Focal theo config

Công thức Focal Loss (multi-class, từ Lin et al. 2017 + multi-class
extension phổ biến):
    loss = -alpha_c * (1 - p_c)^gamma * log(p_c)

Trong đó:
  - alpha_c  : per-class weight (α trong paper) — thường là class weight
  - gamma    : focusing param, γ=0 → thuần CE
  - p_c      : softmax probability của đúng lớp c

Khi gamma=0 và alpha=None, FocalLoss ≡ CrossEntropyLoss(weight=None)
(sai số do phép tính power/exp).
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════
# Class weight helpers
# ══════════════════════════════════════════════════════════════════


def compute_class_weights(
    labels: np.ndarray,
    n_classes: int = 5,
    scheme: str = "inverse_sqrt",
    smoothing: float = 1.0,
) -> np.ndarray:
    """
    Tính per-class weight cho CE/Focal.

    Args:
        labels    : 1D int array ground-truth (giá trị trong [0, n_classes-1])
        n_classes : số lớp (5 cho sleep stage)
        scheme    : "inverse"        w_c = N / (n_classes * count_c)
                    "inverse_sqrt"   w_c = 1 / sqrt(count_c)  (gentler)
                    "effective"      w_c = (1 - β) / (1 - β^count_c)
                                     với β = 0.999
        smoothing : additive Laplace smoothing trên count

    Returns:
        (n_classes,) float32 vector. Khi không có sample cho class c,
        weight c được set = median để tránh NaN/inf.
    """
    labels = labels.astype(np.int64).ravel()
    counts = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        counts[c] = float((labels == c).sum())
    counts += smoothing

    if scheme == "inverse":
        w = labels.size / (n_classes * counts) if labels.size > 0 else np.ones(n_classes)
    elif scheme == "inverse_sqrt":
        w = 1.0 / np.sqrt(counts)
    elif scheme == "effective":
        beta = 0.999
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    # Đảm bảo weight hữu hạn và dương
    w = np.where(np.isfinite(w) & (w > 0), w, np.median(w[np.isfinite(w) & (w > 0)]))

    # Normalize về mean=1 (giúp learning rate không bị scale)
    w = w / w.mean()
    return w.astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# Focal Loss
# ══════════════════════════════════════════════════════════════════


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss (Cross-Entropy based, supports label smoothing).

    Args:
        gamma           : focusing param. γ=0 → thuần CE weighted.
                           Paper đề xuất 2.0; cho sleep stage thường 1.5-3.0.
        alpha            : (n_classes,) per-class weight, hoặc None
        ignore_index    : label bị bỏ qua (mặc định -100, match PyTorch CE)
        reduction       : 'mean' (default) | 'sum' | 'none'
        label_smoothing : epsilon cho label smoothing (0 = none)

    Input:
        logits  : (N, C) raw logits
        targets : (N,)  int64 class indices

    Lưu ý numerical stability: dùng log_softmax thay vì log(softmax) trực tiếp.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[torch.Tensor, np.ndarray, Sequence[float]]] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma phải ≥0, nhận {gamma}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction không hợp lệ: {reduction}")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing phải trong [0, 1), nhận {label_smoothing}")

        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

        if alpha is None:
            self.alpha: Optional[nn.Parameter] = None
        else:
            if isinstance(alpha, np.ndarray):
                alpha = torch.from_numpy(alpha.astype(np.float32))
            elif not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(list(alpha), dtype=torch.float32)
            self.alpha = nn.Parameter(alpha, requires_grad=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # ── Mask ignore_index TRƯỚC khi gather để tránh index out-of-bounds
        valid = (targets != self.ignore_index)
        # Safe targets cho gather (clamp ignore → 0, mask out sau)
        safe_targets = targets.clamp(min=0).long()

        # (N, C) → log softmax
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        # Label smoothing (optional): phân bố mềm targets
        if self.label_smoothing > 0:
            n_classes = logits.size(1)
            with torch.no_grad():
                soft_targets = torch.full_like(log_probs, self.label_smoothing / (n_classes - 1))
                soft_targets.scatter_(1, safe_targets.unsqueeze(1), 1.0 - self.label_smoothing)
            focal_per_class = (1.0 - probs).clamp(min=0.0).pow(self.gamma)
            loss_per_sample = -(soft_targets * focal_per_class * log_probs)
            if self.alpha is not None:
                loss_per_sample = loss_per_sample * self.alpha.view(1, -1).to(logits.device)
            loss = loss_per_sample.sum(dim=1)
            loss = loss * valid.float()
        else:
            log_p_t = log_probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)   # (N,)
            p_t     = probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)       # (N,)
            loss    = -((1.0 - p_t).clamp(min=0.0).pow(self.gamma) * log_p_t)
            if self.alpha is not None:
                alpha_t = self.alpha.to(logits.device).gather(0, safe_targets)
                loss = loss * alpha_t
            loss = loss * valid.float()

        if self.reduction == "mean":
            n_valid = valid.sum().clamp(min=1).float()
            return loss.sum() / n_valid
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def extra_repr(self) -> str:
        return (f"gamma={self.gamma}, alpha={None if self.alpha is None else 'per-class'}, "
                f"ignore_index={self.ignore_index}, reduction={self.reduction}, "
                f"label_smoothing={self.label_smoothing}")


# ══════════════════════════════════════════════════════════════════
# Loss factory
# ══════════════════════════════════════════════════════════════════


@dataclass
class LossConfig:
    """Config cho loss function. Tương thích ngược khi loss_type='ce'."""
    loss_type: str = "ce"        # "ce" | "focal"
    # class weights
    use_class_weights: bool = True
    weight_scheme:      str  = "inverse_sqrt"     # cho CE/Focal
    # Focal params
    focal_gamma:        float = 2.0
    focal_label_smoothing: float = 0.0


def build_loss(cfg: LossConfig, class_weights: Optional[np.ndarray] = None) -> nn.Module:
    """
    Tạo loss module dựa trên config.

    Args:
        cfg           : LossConfig
        class_weights : (n_classes,) numpy array hoặc None
                        Nếu cfg.use_class_weights=False → bỏ qua, dùng None
                        Nếu None và use_class_weights=True → warn, fallback uniform
    """
    weights = None
    if cfg.use_class_weights and class_weights is not None:
        weights = class_weights
    # else: alpha=None → không scale per-class

    if cfg.loss_type == "ce":
        # CrossEntropyLoss(weight=...) ưu tiên hơn Focal khi gamma=0
        # (CE bỏ qua label_smoothing vì CEloss đã không smooth ở đây)
        return nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    elif cfg.loss_type == "focal":
        return FocalLoss(
            gamma=cfg.focal_gamma,
            alpha=weights,
            ignore_index=-100,
            label_smoothing=cfg.focal_label_smoothing,
        )
    else:
        raise ValueError(f"Unknown loss_type: {cfg.loss_type}")


__all__ = [
    "FocalLoss",
    "LossConfig",
    "build_loss",
    "compute_class_weights",
]
