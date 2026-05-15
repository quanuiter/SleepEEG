#!/usr/bin/env python3
"""
test_pipeline.py - Test script để kiểm tra pipeline trước deployment
Chạy: python test_pipeline.py
"""

import sys
from pathlib import Path

# Add current dir to path
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

print("=" * 60)
print("🧪 KIỂM TRA PIPELINE TRƯỚC DEPLOYMENT")
print("=" * 60)

# ── Check 1: Python & Core Libraries ──────────────────────────────
print("\n1️⃣  Kiểm tra Python packages...")
try:
    import numpy as np
    print(f"  ✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"  ✗ NumPy: {e}")

try:
    import scipy
    print(f"  ✓ SciPy {scipy.__version__}")
except ImportError as e:
    print(f"  ✗ SciPy: {e}")

try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    print(f"    CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"  ✗ PyTorch: {e}")

try:
    import streamlit as st
    print(f"  ✓ Streamlit {st.__version__}")
except ImportError as e:
    print(f"  ✗ Streamlit: {e}")

try:
    import pyedflib
    print(f"  ✓ pyedflib {pyedflib.__version__}")
except ImportError as e:
    print(f"  ✗ pyedflib: {e}")

# ── Check 2: Pipeline Modules ──────────────────────────────────────
print("\n2️⃣  Kiểm tra pipeline modules...")
try:
    from pipeline.models import EEG_ResNet1D, SleepTCN
    print("  ✓ pipeline.models")
except ImportError as e:
    print(f"  ✗ pipeline.models: {e}")

try:
    from pipeline.preprocess import PreprocessConfig, bandpass_filter
    print("  ✓ pipeline.preprocess")
except ImportError as e:
    print(f"  ✗ pipeline.preprocess: {e}")

try:
    from inference import InferenceConfig, preprocess_raw_edf, run_inference
    print("  ✓ inference module")
except ImportError as e:
    print(f"  ✗ inference module: {e}")

# ── Check 3: Model Weights ────────────────────────────────────────
print("\n3️⃣  Kiểm tra model weights...")
weight_dir = APP_DIR / "weight"
if weight_dir.exists():
    print(f"  ✓ Thư mục weight/ tồn tại")
    resnet_files = list(weight_dir.glob("resnet_fold*.pth"))
    tcn_files = list(weight_dir.glob("tcn_fold*.pth"))
    print(f"    ResNet files: {len(resnet_files)}")
    print(f"    TCN files: {len(tcn_files)}")
    
    if resnet_files:
        print(f"    🔹 {resnet_files[0].name}")
    if tcn_files:
        print(f"    🔹 {tcn_files[0].name}")
    
    if len(resnet_files) == 0 or len(tcn_files) == 0:
        print("  ⚠️  Không tìm thấy đủ model weights!")
else:
    print(f"  ✗ Thư mục weight/ không tồn tại tại {weight_dir}")

# ── Check 4: Config Files ──────────────────────────────────────────
print("\n4️⃣  Kiểm tra config files...")
config_files = [
    ".streamlit/config.toml",
    "requirements.txt",
    "app.py",
    "DEPLOYMENT.md",
]
for cf in config_files:
    path = APP_DIR / cf
    if path.exists():
        print(f"  ✓ {cf}")
    else:
        print(f"  ✗ {cf} không tìm thấy")

# ── Check 5: Model Architecture Test ────────────────────────────────
print("\n5️⃣  Kiểm tra model architecture...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models
    resnet = EEG_ResNet1D(feature_dim=128, num_classes=5).to(device)
    tcn = SleepTCN(input_size=128, dim=128, kernel_size=3, n_blocks=6, 
                   dropout=0.2, n_classes=5).to(device)
    
    # Test forward pass
    x_resnet = torch.randn(2, 1, 3000).to(device)
    feat = resnet(x_resnet, extract_features=True)
    print(f"  ✓ ResNet output shape: {feat.shape}")
    
    x_tcn = torch.randn(2, 10, 128).to(device)
    out_tcn = tcn(x_tcn)
    print(f"  ✓ TCN output shape: {out_tcn.shape}")
    
except Exception as e:
    print(f"  ✗ Model test failed: {e}")

print("\n" + "=" * 60)
print("✅ KIỂM TRA XONG!")
print("=" * 60)
print("\n📝 Tiếp theo:")
print("  1. Chắc chắn tất cả ✓ trên")
print("  2. Git push lên GitHub")
print("  3. Deploy trên Streamlit Cloud")
print("\nXem DEPLOYMENT.md để chi tiết.")
