"""
pipeline/models.py
──────────────────
Định nghĩa kiến trúc ResNet-1D (feature extractor) và SleepTCN (sequence model).
Giữ nguyên 100% so với extract_new.ipynb và tcn_sleep_resnet.ipynb.
"""

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════
# ResNet-1D — Feature Extractor
# ══════════════════════════════════════════════════════════════════

class ResNetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 7, stride=stride, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 7, stride=1, padding=3, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        res = self.shortcut(x)
        x   = self.relu(self.bn1(self.conv1(x)))
        x   = self.bn2(self.conv2(x))
        return self.relu(x + res)


class EEG_ResNet1D(nn.Module):
    """
    ResNet-1D nhận epoch EEG (1, 3000) → feature vector (feature_dim,).
    Khi extract_features=True trả về feature; ngược lại trả về logit.
    """
    def __init__(self, feature_dim: int = 128, num_classes: int = 5):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes  = num_classes

        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 50, stride=5, padding=25, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layer1 = ResNetBlock1D(32,          64,          stride=1)
        self.layer2 = ResNetBlock1D(64,          128,         stride=2)
        self.layer3 = ResNetBlock1D(128,         feature_dim, stride=2)
        self.gap     = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc      = nn.Linear(feature_dim, num_classes)

    def forward(self, x, extract_features: bool = False):
        x        = self.stem(x)
        x        = self.layer1(x)
        x        = self.layer2(x)
        x        = self.layer3(x)
        features = self.gap(x).squeeze(-1)          # (B, feature_dim)
        if extract_features:
            return features
        return self.fc(self.dropout(features))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════
# SleepTCN — Sequence Classifier
# ══════════════════════════════════════════════════════════════════

class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel_size, padding=pad, dilation=dilation, bias=False)
        self.norm1 = nn.LayerNorm(out_ch)
        self.act1  = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation, bias=False)
        self.norm2 = nn.LayerNorm(out_ch)
        self.act2  = nn.GELU()
        self.drop2 = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.conv1(x)
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = self.drop1(self.act1(out))
        out = self.conv2(out)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = self.drop2(self.act2(out))
        return out + res


class SleepTCN(nn.Module):
    """
    TCN nhận sequence features (B, T, input_size) → logits (B, T, n_classes).
    """
    def __init__(
        self,
        input_size: int = 128,
        dim:        int = 128,
        kernel_size:int = 3,
        n_blocks:   int = 6,
        dropout:    float = 0.2,
        n_classes:  int = 5,
    ):
        super().__init__()
        self.input_size = input_size
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.tcn = nn.Sequential(
            *[TemporalBlock(dim, dim, kernel_size, dilation=2**i, dropout=dropout)
              for i in range(n_blocks)]
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x, key_padding_mask=None):
        # x: (B, T, input_size)
        h = self.input_proj(x)           # (B, T, dim)
        h = h.transpose(1, 2)            # (B, dim, T)
        h = self.tcn(h)
        h = h.transpose(1, 2)            # (B, T, dim)
        return self.classifier(self.dropout(h))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
