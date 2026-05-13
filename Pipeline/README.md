# 🧠 Sleep Stage Classification Pipeline

Pipeline hoàn chỉnh: **EDF → Preprocess → ResNet-1D → SleepTCN → Sleep Staging**

## Cấu trúc

```
sleep_pipeline/
├── app.py                    ← Streamlit UI chính
├── pipeline/
│   ├── __init__.py
│   ├── models.py             ← ResNet-1D + SleepTCN (giữ nguyên kiến trúc)
│   ├── preprocess.py         ← EDF → NPZ (refactor từ preprocess.py gốc)
│   └── trainer.py            ← Fold split, train, evaluate, orchestrator
└── README.md
```

## Cài đặt

```bash
pip install streamlit torch torchvision numpy scipy pyedflib \
            scikit-learn matplotlib seaborn pandas
```

## Chạy UI

```bash
cd sleep_pipeline
streamlit run app.py
```

## Dùng từ Jupyter / Python thuần

```python
from pipeline.preprocess import PreprocessConfig, run_preprocess
from pipeline.trainer import PipelineConfig, run_full_pipeline

# 1. Preprocess
pp_cfg = PreprocessConfig(
    data_dir="./data/sleepedf/sleep-cassette",
    output_dir="./preprocessed",
    select_ch="EEG Fpz-Cz",
    apply_bandpass=True, bp_low=0.5, bp_high=30.0,
    apply_notch=True,   notch_freq=50.0,
)
run_preprocess(pp_cfg)

# 2. Train
cfg = PipelineConfig(
    npz_dir="./preprocessed",
    output_dir="./results",
    ckpt_dir="./results/checkpoints",
    n_folds=10, seed=123,
)
results = run_full_pipeline(cfg, train_resnet=True, train_tcn=True)

# 3. Evaluate trên data mới với weights cũ
cfg_new = PipelineConfig(npz_dir="./new_data", output_dir="./results_new")
results = run_full_pipeline(
    cfg_new,
    resnet_ckpt_dir="./results/checkpoints",
    tcn_ckpt_dir="./results/checkpoints",
    train_resnet=False,
    train_tcn=False,
)
```

## Các tính năng chính

| Tab | Chức năng |
|-----|-----------|
| **Preprocess** | Chọn channel EEG, cấu hình bandpass/notch/clip/scale, log realtime |
| **Train** | Cấu hình K-Fold, siêu tham số ResNet + TCN, chọn fold cụ thể |
| **Evaluate** | Load checkpoint ResNet/TCN cụ thể, cross-val dataset mới |
| **Results** | Metrics tổng hợp, per-fold table, biểu đồ, download JSON |

## Lưu ý về fold

- Seed = 123 → fold split **khớp 100%** với `extract_new.ipynb` và `tcn_sleep_resnet.ipynb`
- Checkpoint lưu theo format `resnet_fold{N}.pth` và `tcn_fold{N}.pth`
- Có thể chọn chạy subset fold (vd: chỉ fold 1,3,5) để debug nhanh
