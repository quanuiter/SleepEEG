"""
pipeline/trainer.py
────────────────────
Logic train / evaluate toàn pipeline.

Thay đổi so với phiên bản cũ:
  - load_all_records() nhận cả SC*.npz (Sleep-EDF) và shhs_*.npz (SHHS)
  - PipelineConfig thêm: finetune_lr, freeze_resnet
  - run_full_pipeline() thêm mode fine_tune: load ResNet → freeze → chỉ train TCN với lr nhỏ
  - subject ID extraction tự động theo prefix file
"""

import copy
import gc
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import resample
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from pipeline.models import EEG_ResNet1D, SleepTCN
from pipeline.postprocess import PostprocessConfig, postprocess_hypnogram
# Lazy imports cho AugmentConfig/LossConfig (gọi ở default_factory) để tránh
# circular nếu augment/losses sau này reference lại trainer.
from pipeline.utils import (
    config_hash,
    derive_seed,
    git_commit,
    make_run_id,
    safe_yaml_dump,
    set_seed,
    utc_now_iso,
)
from pipeline.augment import AugmentConfig
from pipeline.losses import LossConfig

SLEEP_STAGES  = ["W", "N1", "N2", "N3", "REM"]
N_CLASSES     = 5
EPOCH_SAMPLES = 3000


# ══════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    # Paths
    npz_dir:    str = "./preprocessed"
    output_dir: str = "./results"
    ckpt_dir:   str = "./results/checkpoints"

    # Fold
    n_folds: int = 10
    seed:    int = 123

    # ResNet
    n_feat:            int   = 128
    resnet_lr:         float = 1e-3
    resnet_batch_size: int   = 64
    resnet_epochs:     int   = 40
    resnet_patience:   int   = 8
    resnet_val_ratio:  float = 0.15

    # TCN
    tcn_dim:         int   = 128
    tcn_kernel_size: int   = 3
    tcn_blocks:      int   = 6
    tcn_dropout:     float = 0.2

    # TCN training
    seq_lr:         float = 5e-4
    seq_batch_size: int   = 8
    seq_max_epochs: int   = 300
    seq_patience:   int   = 30
    seq_val_every:  int   = 5
    seq_val_ratio:  float = 0.10

    # Fine-tune: chỉ dùng khi load ResNet checkpoint từ dataset khác
    # fine_tune=True → freeze ResNet, train TCN với finetune_lr
    finetune_lr:    float = 1e-4
    freeze_resnet:  bool  = True   # khi fine-tune: giữ nguyên ResNet weights

    # Hypnogram post-processing (mặc định disabled).
    # Khi enabled, evaluate_fold() trả về cả raw và postprocessed metrics,
    # cho phép so sánh trong ablation table.
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)

    # Loss function
    loss: LossConfig = field(default_factory=LossConfig)

    # Signal-level augmentation (mặc định disabled)
    augment: AugmentConfig = field(default_factory=AugmentConfig)

    # Joint end-to-end fine-tuning (mặc định disabled)
    joint_finetune:    bool   = False
    joint_lr_resnet:   float  = 1e-4
    joint_lr_tcn:      float  = 5e-4
    joint_epochs:      int    = 30
    joint_patience:    int    = 12        # epoch unit
    joint_val_every:   int    = 2
    joint_batch_size:  int    = 1
    joint_resnet_chunk: int   = 128       # epochs per ResNet fwd để tránh OOM
    joint_freeze_bn:   bool   = True      # giữ BN ở eval mode (small batch)

    # Callbacks (Streamlit)
    log_cb:      Optional[Callable[[str], None]]           = field(default=None, repr=False)
    progress_cb: Optional[Callable[[int, int, str], None]] = field(default=None, repr=False)
    metrics_cb:  Optional[Callable[[int, Dict], None]]     = field(default=None, repr=False)


# ══════════════════════════════════════════════════════════════════
# Datasets
# ══════════════════════════════════════════════════════════════════

class _EpochDataset(Dataset):
    def __init__(self, signals, labels, augment_cfg: Optional["AugmentConfig"] = None,
                 rng_seed: int = 0, fold: int = 0, prefix: str = "aug"):
        """
        signals: (N, 3000) float32
        labels:  (N,) int64
        augment_cfg: nếu enabled, áp augmentation per-sample
        rng_seed/fold/prefix: dùng derive_seed() để tạo rng reproducible
        """
        self.signals = signals
        self.labels  = labels
        self.augment_cfg = augment_cfg

        def _seed_for(i: int) -> int:
            return derive_seed(rng_seed, fold, f"{prefix}_{i}")

        self._seed_for = _seed_for

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        x = self.signals[idx]
        if self.augment_cfg is not None and self.augment_cfg.enabled:
            from pipeline.augment import apply_epoch_augment
            rng = np.random.default_rng(self._seed_for(idx))
            x = apply_epoch_augment(x, rng, self.augment_cfg)
        x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class _SeqDataset(Dataset):
    def __init__(self, feat_lbl_list): self.data = feat_lbl_list
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        feats, lbls = self.data[idx]
        return (torch.from_numpy(feats.astype(np.float32)),
                torch.from_numpy(lbls.astype(np.int64)))


def _collate_pad(batch):
    xs, ys  = zip(*batch)
    T_max   = max(x.shape[0] for x in xs)
    B, D    = len(xs), xs[0].shape[1]
    x_pad   = torch.zeros(B, T_max, D)
    y_pad   = torch.full((B, T_max), -100, dtype=torch.long)
    pad_mask= torch.ones(B, T_max, dtype=torch.bool)
    for i, (x, y) in enumerate(zip(xs, ys)):
        T = x.shape[0]
        x_pad[i, :T] = x; y_pad[i, :T] = y; pad_mask[i, :T] = False
    return x_pad, y_pad, pad_mask


# ══════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════

def normalize_labels(y: np.ndarray, label_format: str = "unknown") -> np.ndarray:
    """
    Chuẩn hoá nhãn về 0-indexed [0,4] cho 5 pha ngủ.

    label_format (đọc từ NPZ key 'label_format'):
      '0indexed'  — preprocess mới: W=0,N1=1,N2=2,N3=3,REM=4  → không shift
      '1indexed'  — một số file cũ: W=1..REM=5                  → shift -1
      'unknown'   — fallback heuristic (giữ tương thích file cũ)
    """
    y = y.astype(np.int64).ravel()
    if label_format == "0indexed":
        pass   # đúng rồi, không làm gì
    elif label_format == "1indexed":
        y = y - 1
    else:
        # Heuristic chỉ dùng cho file NPZ cũ không có 'label_format'.
        # Chỉ shift khi TẤT CẢ nhãn hợp lệ đều >=1 (tức không có nhãn 0).
        # Trường hợp không có Wake (min > 1) → KHÔNG shift, tránh corrupt.
        valid = y[(y >= 0) & (y < 10)]
        if len(valid) > 0 and valid.min() == 1 and valid.max() <= 5:
            y = y - 1
    y[(y < 0) | (y >= N_CLASSES)] = -1
    return y


def load_npz(fpath: Path) -> Tuple[np.ndarray, np.ndarray]:
    d    = np.load(str(fpath), allow_pickle=True)
    sigs = d["x"].astype(np.float32)

    # Bug-3 fix: đọc label_format được lưu bởi preprocess, không đoán
    label_format = str(d["label_format"]) if "label_format" in d else "unknown"
    lbls = normalize_labels(d["y"], label_format)

    # Chuẩn hoá shape về (n_epochs, n_samples_per_epoch)
    if sigs.ndim == 3:
        sigs = sigs[:, 0, :]          # (N, 1, L) → (N, L)

    if sigs.ndim == 2:
        n_lbls = len(lbls)
        r0, r1 = sigs.shape
        # Bug-4 fix: ưu tiên dùng n_epochs được lưu trong NPZ (tin cậy hơn heuristic)
        saved_n = int(d["n_epochs"]) if "n_epochs" in d else None
        if saved_n is not None and r0 == saved_n:
            pass                       # shape đúng
        elif saved_n is not None and r1 == saved_n:
            sigs = sigs.T
        elif r0 == n_lbls and r1 != n_lbls:
            pass                       # khớp lbls
        elif r1 == n_lbls and r0 != n_lbls:
            sigs = sigs.T
        elif r0 != r1:
            if r0 > r1:
                sigs = sigs.T          # chiều lớn hơn = n_samples
        # r0==r1: không thể phân biệt → giữ nguyên, assert cuối sẽ bắt lỗi

    # Resample về EPOCH_SAMPLES nếu cần
    # Với preprocess mới (đã resample về 100Hz) → shape đã là 3000, bước này là no-op.
    # Giữ lại để tương thích file NPZ cũ chưa resample.
    if sigs.shape[1] != EPOCH_SAMPLES:
        sigs = resample(sigs, EPOCH_SAMPLES, axis=1).astype(np.float32)

    # Đảm bảo sigs và lbls cùng số epoch
    min_len = min(len(sigs), len(lbls))
    sigs, lbls = sigs[:min_len], lbls[:min_len]

    # Crop quanh vùng ngủ (buffer 60 epoch)
    sleep_idx = np.where((lbls >= 1) & (lbls <= 4))[0]
    if len(sleep_idx):
        s = max(0,         sleep_idx[0]  - 60)
        e = min(len(lbls), sleep_idx[-1] + 61)
        sigs, lbls = sigs[s:e], lbls[s:e]

    assert sigs.shape == (len(lbls), EPOCH_SAMPLES), \
        f"shape cuoi: sigs={sigs.shape}, lbls={lbls.shape}"
    return sigs, lbls


def _extract_sid(stem: str) -> str:
    """
    Lấy subject ID từ tên file (không có extension).
      SC4002E0-PSG   → "00"   (Sleep-EDF, 2 ký tự số)
      SC4002E1-PSG   → "00"
      shhs_shhs1-200001 → "200001"  (SHHS)
    """
    if stem.startswith("shhs_"):
        return stem.split("-")[-1]
    # Sleep-EDF: ký tự 3-4 là số phân biệt subject
    return stem[3:5]


def load_all_records(npz_dir: str, log_cb=None) -> Tuple[Dict, List]:
    """
    Quét NPZ nhưng KHÔNG load signal vào RAM — chỉ lưu path + sid.
    Signal được đọc lazily khi cần (streaming), tránh OOM.
    """
    data_path = Path(npz_dir)
    npz_files = sorted(
        list(data_path.glob("SC*.npz")) + list(data_path.glob("shhs_*.npz"))
    )
    if not npz_files:
        raise FileNotFoundError(
            f"Không tìm thấy SC*.npz hay shhs_*.npz trong {npz_dir}"
        )
    _log(log_cb, f"Tìm thấy {len(npz_files)} NPZ files")

    subject_records: Dict[str, List] = defaultdict(list)
    all_records: List = []
    skipped = 0
    for fpath in npz_files:
        try:
            # Chỉ load lbls để validate + lấy metadata, không giữ sigs
            sigs, lbls = load_npz(fpath)
            sid = _extract_sid(fpath.stem)
            rec = {"sid": sid, "fname": fpath.stem,
                   "fpath": fpath,          # đường dẫn để load lại sau
                   "lbls": lbls,            # nhãn nhỏ, giữ luôn
                   "n_epochs": len(lbls)}  # metadata
            # KHÔNG lưu sigs — giải phóng ngay
            del sigs
            subject_records[sid].append(rec)
            all_records.append(rec)
        except Exception as e:
            _log(log_cb, f"  ✗ {fpath.name}: {e}")
            skipped += 1

    _log(log_cb, f"✓ {len(all_records)} records  |  {len(subject_records)} subjects"
               + (f"  ({skipped} bị bỏ qua)" if skipped else ""))
    return subject_records, all_records


# ══════════════════════════════════════════════════════════════════
# Fold split
# ══════════════════════════════════════════════════════════════════

def build_fold_groups(subject_records: Dict, n_folds: int = 10, seed: int = 123):
    subject_ids  = sorted(subject_records.keys())
    rng          = np.random.default_rng(seed=seed)
    shuffled_ids = rng.permutation(subject_ids).tolist()
    return np.array_split(shuffled_ids, n_folds)


# ══════════════════════════════════════════════════════════════════
# Feature extraction
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(records: List, resnet: nn.Module,
                     device: torch.device, batch_size: int = 128) -> List:
    """
    Streaming: đọc từng file, extract features, bỏ signal ngay.
    Kết quả: list of (feats: np.ndarray [T, feat_dim], lbls: np.ndarray [T])
    features nhỏ hơn signal ~23x (128-dim vs 3000-dim) → RAM ổn.
    """
    resnet.eval()
    results = []
    for rec in records:
        # Load signal tươi từ disk, extract xong bỏ ngay
        sigs, lbls = load_npz(rec["fpath"])
        feats = []
        for i in range(0, len(sigs), batch_size):
            xb = torch.from_numpy(sigs[i:i+batch_size]).unsqueeze(1).to(device)
            feats.append(resnet(xb, extract_features=True).cpu().numpy())
        del sigs  # giải phóng signal ngay sau khi extract
        # Đổi nhãn rác (-1) → -100 (ignore_index cho CrossEntropy)
        lbls = lbls.copy()
        lbls[lbls == -1] = -100
        results.append((np.concatenate(feats, axis=0), lbls))
    return results


# ══════════════════════════════════════════════════════════════════
# Train ResNet
# ══════════════════════════════════════════════════════════════════

def _collect_valid(records, max_epochs_per_rec: int = 300, max_total: int = 30_000,
                   rng: Optional[np.random.Generator] = None):
    """
    Load từng file, lấy valid epochs, ghép lại.
    max_epochs_per_rec : giới hạn epoch mỗi record (tránh OOM)
    max_total          : giới hạn tổng epoch (safety cap khi 200 SHHS files)
    rng                : numpy Generator (mặc định dùng global np.random nếu None)
    """
    if rng is None:
        rng = np.random.default_rng()
    sigs_all, lbls_all = [], []
    total_so_far = 0
    for rec in records:
        if total_so_far >= max_total:
            break
        sigs, lbls = load_npz(rec["fpath"])
        valid = lbls != -1
        sv, lv = sigs[valid], lbls[valid]
        del sigs
        gc.collect()
        if len(sv) > max_epochs_per_rec:
            idx = rng.choice(len(sv), max_epochs_per_rec, replace=False)
            sv, lv = sv[idx], lv[idx]
        room = max_total - total_so_far
        if len(sv) > room:
            sv, lv = sv[:room], lv[:room]
        sigs_all.append(sv)
        lbls_all.append(lv)
        total_so_far += len(sv)
    return np.concatenate(sigs_all), np.concatenate(lbls_all)


def train_resnet_fold(train_records: List, fold_idx: int,
                      cfg: PipelineConfig, device: torch.device) -> nn.Module:
    from sklearn.metrics import f1_score as sk_f1

    n_val = max(1, round(len(train_records) * cfg.resnet_val_ratio))
    # derive_seed: deterministic + không collide giữa các stage/folds
    rng   = np.random.default_rng(derive_seed(cfg.seed, fold_idx, "resnet_split"))
    perm  = rng.permutation(len(train_records)).tolist()
    val_recs = [train_records[i] for i in perm[:n_val]]
    tr_recs  = [train_records[i] for i in perm[n_val:]]

    X_train, y_train = _collect_valid(tr_recs, rng=rng)
    X_val,   y_val   = _collect_valid(val_recs, rng=rng)

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = torch.FloatTensor(weights).to(device)

    _log(cfg.log_cb, f"  ResNet | Train {len(y_train):,} / Val {len(y_val):,} epochs")

    g = torch.Generator()
    g.manual_seed(derive_seed(cfg.seed, fold_idx, "resnet_dataloader"))
    tr_ld = DataLoader(
        _EpochDataset(X_train, y_train, augment_cfg=cfg.augment,
                      rng_seed=cfg.seed, fold=fold_idx, prefix="resnet_train"),
        batch_size=cfg.resnet_batch_size, shuffle=True,
        generator=g, num_workers=2, pin_memory=(device.type == "cuda"),
    )
    vl_ld = DataLoader(
        # Validation KHÔNG augment (đo khả năng generalize, không đo aug quality)
        _EpochDataset(X_val, y_val, augment_cfg=None),
        batch_size=cfg.resnet_batch_size * 2, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    model     = EEG_ResNet1D(cfg.n_feat, N_CLASSES).to(device)
    # Build loss theo LossConfig (focal hoặc ce). class weights dùng
    # compute_class_weights với scheme 'inverse_sqrt' (gentler hơn sklearn
    # 'balanced' để tránh N1 weight quá lớn gây instability).
    from pipeline.losses import compute_class_weights as _ccw
    cw_np = _ccw(y_train.astype(np.int64), n_classes=N_CLASSES, scheme="inverse_sqrt")
    cw_t  = torch.from_numpy(cw_np).to(device)
    if cfg.loss.loss_type == "focal":
        from pipeline.losses import FocalLoss
        criterion = FocalLoss(
            gamma=cfg.loss.focal_gamma,
            alpha=cw_t if cfg.loss.use_class_weights else None,
            ignore_index=-100,
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=cw_t if cfg.loss.use_class_weights else None,
            ignore_index=-100,
        )
    optimizer = optim.AdamW(model.parameters(), lr=cfg.resnet_lr, weight_decay=1e-4)

    best_mf1, no_improve = -1.0, 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(cfg.resnet_epochs):
        model.train()
        for bx, by in tr_ld:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward(); optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for vx, vy in vl_ld:
                preds.extend(model(vx.to(device)).argmax(1).cpu().numpy())
                trues.extend(vy.numpy())
        val_mf1 = sk_f1(trues, preds, average="macro", zero_division=0)

        if (epoch + 1) % 5 == 0:
            _log(cfg.log_cb,
                 f"    Ep {epoch+1:3d}/{cfg.resnet_epochs}  val_mf1={val_mf1:.4f}"
                 + ("  ← best" if val_mf1 > best_mf1 else ""))

        if val_mf1 > best_mf1:
            best_mf1, no_improve = val_mf1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
            if no_improve >= cfg.resnet_patience:
                _log(cfg.log_cb, f"    Early stop ep {epoch+1}  best_mf1={best_mf1:.4f}")
                break

    model.load_state_dict(best_state)
    _log(cfg.log_cb, f"  ✓ ResNet best val_mf1={best_mf1:.4f}")
    return model


# ══════════════════════════════════════════════════════════════════
# Train TCN  (dùng chung cho train mới và fine-tune)
# ══════════════════════════════════════════════════════════════════

def train_tcn_fold(train_feats: List, fold_idx: int,
                   cfg: PipelineConfig, device: torch.device,
                   pretrained_tcn: Optional[nn.Module] = None,
                   lr_override: Optional[float] = None) -> nn.Module:
    """
    Train SleepTCN 1 fold.

    pretrained_tcn : nếu truyền vào → fine-tune từ weights này
    lr_override    : ghi đè learning rate (dùng khi fine-tune với lr nhỏ hơn)
    """
    n_val  = max(1, round(len(train_feats) * cfg.seq_val_ratio))
    rng    = np.random.default_rng(derive_seed(cfg.seed, fold_idx, "tcn_split"))
    perm   = rng.permutation(len(train_feats)).tolist()
    val_fl = [train_feats[i] for i in perm[:n_val]]
    tr_fl  = [train_feats[i] for i in perm[n_val:]]

    _log(cfg.log_cb, f"  TCN   | Train {len(tr_fl)} / Val {len(val_fl)} records")

    pin = device.type == "cuda"
    tr_ld = DataLoader(_SeqDataset(tr_fl), batch_size=cfg.seq_batch_size,
                       shuffle=True,  collate_fn=_collate_pad, num_workers=0, pin_memory=pin)
    vl_ld = DataLoader(_SeqDataset(val_fl), batch_size=cfg.seq_batch_size,
                       shuffle=False, collate_fn=_collate_pad, num_workers=0, pin_memory=pin)

    if pretrained_tcn is not None:
        model = pretrained_tcn.to(device)
        _log(cfg.log_cb, "  TCN   | Fine-tuning từ pretrained weights")
    else:
        model = SleepTCN(
            input_size=cfg.n_feat, dim=cfg.tcn_dim,
            kernel_size=cfg.tcn_kernel_size, n_blocks=cfg.tcn_blocks,
            dropout=cfg.tcn_dropout, n_classes=N_CLASSES,
        ).to(device)

    lr        = lr_override if lr_override is not None else cfg.seq_lr
    loss_fn   = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    best_loss, no_improve = float("inf"), 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, cfg.seq_max_epochs + 1):
        model.train()
        for xb, yb, _ in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = loss_fn(logits.permute(0, 2, 1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if epoch % cfg.seq_val_every == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xv, yv, _ in vl_ld:
                    xv, yv = xv.to(device), yv.to(device)
                    val_loss += loss_fn(model(xv).permute(0, 2, 1), yv).item()
            val_loss /= len(vl_ld)

            if val_loss < best_loss:
                best_loss, no_improve = val_loss, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                no_improve += 1

            if epoch % (cfg.seq_val_every * 5) == 0:
                _log(cfg.log_cb,
                     f"    Ep {epoch:4d}/{cfg.seq_max_epochs}  val_loss={val_loss:.4f}"
                     + (" ← best" if no_improve == 0 else ""))

            if no_improve >= (cfg.seq_patience // cfg.seq_val_every):
                _log(cfg.log_cb, f"    Early stop ep {epoch}  best_val_loss={best_loss:.4f}")
                break

    model.load_state_dict(best_state)
    return model


# ══════════════════════════════════════════════════════════════════
# Joint fine-tuning: ResNet + TCN end-to-end
# ══════════════════════════════════════════════════════════════════

# Cấu hình joint fine-tune được nhúng vào PipelineConfig — xem dưới.


class _RawSeqDataset(Dataset):
    """
    Lazy load raw signals từ NPZ, trả về (signals, labels) cho 1 record.

    Đây là dạng dataset joint fine-tune cần: KHÔNG cache features như
    `_EpochDataset` + `_SeqDataset` (cách cũ), vì cần gradient chạy qua
    ResNet.
    """

    def __init__(self, records: List, cfg: PipelineConfig, fold: int):
        self.records = records
        self.cfg = cfg
        self.fold = fold
        # Lazy — chỉ load khi __getitem__
        self.augment_cfg = getattr(cfg, "augment", None)
        self._aug_rng_seed = cfg.seed

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        sigs, lbls = load_npz(rec["fpath"])
        lbls = lbls.astype(np.int64)
        # -1 → -100 để match CE/Focal ignore_index
        lbls[lbls < 0] = -100

        # Augment per-record (1 rng per record → reproducible)
        if self.augment_cfg is not None and self.augment_cfg.enabled:
            from pipeline.augment import apply_epoch_augment
            rng = np.random.default_rng(derive_seed(self._aug_rng_seed, self.fold, f"aug_{idx}"))
            sigs_aug = np.empty_like(sigs, dtype=np.float32)
            for i in range(len(sigs)):
                sigs_aug[i] = apply_epoch_augment(sigs[i], rng, self.augment_cfg).astype(np.float32)
            sigs = sigs_aug

        return (
            torch.from_numpy(sigs.astype(np.float32)).unsqueeze(1),  # (T, 1, 3000)
            torch.from_numpy(lbls),
        )


def train_joint_fold(
    train_records: List,
    val_records:   List,
    fold_idx: int,
    cfg: PipelineConfig,
    device: torch.device,
    pretrained_resnet: Optional[nn.Module] = None,
    pretrained_tcn:    Optional[nn.Module] = None,
) -> Tuple[nn.Module, nn.Module]:
    """
    End-to-end fine-tune: ResNet + TCN với differential LR.

    Khác với 2-stage:
      - 2-stage: ResNet freeze → extract features 1 lần → train TCN
      - Joint: ResNet cũng có gradient, TCN được feed bởi ResNet
        features thật (không qua @torch.no_grad).

    Args:
        train_records     : list of dicts (file npz) cho training
        val_records       : list of dicts cho early stop
        fold_idx          : 0-based fold index (cho seed)
        cfg               : PipelineConfig
        device            : torch.device
        pretrained_resnet : nếu None → khởi tạo fresh; thường nên truyền
                            weights từ 2-stage để ổn định (warm start)
        pretrained_tcn    : tương tự, từ 2-stage

    Returns:
        (resnet, tcn) sau khi train.
    """
    from pipeline.losses import build_loss, compute_class_weights

    resnet = (pretrained_resnet if pretrained_resnet is not None
              else EEG_ResNet1D(cfg.n_feat, N_CLASSES))
    resnet = resnet.to(device)
    tcn = (pretrained_tcn if pretrained_tcn is not None
           else SleepTCN(input_size=cfg.n_feat, dim=cfg.tcn_dim,
                          kernel_size=cfg.tcn_kernel_size, n_blocks=cfg.tcn_blocks,
                          dropout=cfg.tcn_dropout, n_classes=N_CLASSES))
    tcn = tcn.to(device)

    # ── BatchNorm policy: nếu giữ ResNet frozen-ish thì nên eval mode
    #    cho BN để tránh running stats drift trên batch nhỏ.
    if getattr(cfg, "joint_freeze_bn", True):
        for m in resnet.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)

    # ── Loss: focal hoặc CE với class weights
    # Compute class weights từ train labels
    from pipeline.preprocess import read_labels_sleepedf
    # Load labels nhanh (chỉ labels, không signals)
    train_labels = []
    for r in train_records:
        try:
            _, lbls = load_npz(r["fpath"])
            lbls = lbls.astype(np.int64).ravel()
            lbls = lbls[(lbls >= 0) & (lbls < 5)]
            train_labels.append(lbls)
        except Exception:
            continue
    all_lbls = np.concatenate(train_labels) if train_labels else np.array([], dtype=np.int64)
    weights = (compute_class_weights(all_lbls, n_classes=N_CLASSES, scheme="inverse_sqrt")
               if cfg.loss.use_class_weights and all_lbls.size > 0 else None)
    loss_fn = build_loss(cfg.loss, weights)

    # ── Differential LR
    lr_resnet = float(getattr(cfg, "joint_lr_resnet", 1e-4))
    lr_tcn    = float(getattr(cfg, "joint_lr_tcn",    5e-4))
    optimizer = optim.AdamW([
        {"params": [p for p in resnet.parameters() if p.requires_grad], "lr": lr_resnet},
        {"params": [p for p in tcn.parameters()    if p.requires_grad], "lr": lr_tcn},
    ], weight_decay=1e-3)

    tr_ds = _RawSeqDataset(train_records, cfg, fold_idx)
    vl_ds = _RawSeqDataset(val_records,   cfg, fold_idx)

    pin = device.type == "cuda"
    # Joint batch nhỏ vì 1 batch có thể 8 records × 1000 epochs × 3000 samples
    bs = int(getattr(cfg, "joint_batch_size", 1))
    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True,  num_workers=0, pin_memory=pin)
    vl_ld = DataLoader(vl_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=pin)

    epochs = int(getattr(cfg, "joint_epochs", 30))
    val_every = int(getattr(cfg, "joint_val_every", 2))
    patience = int(getattr(cfg, "joint_patience", 6))

    best_mf1, no_improve = -1.0, 0
    best_resnet_state = copy.deepcopy(resnet.state_dict())
    best_tcn_state    = copy.deepcopy(tcn.state_dict())

    def _fwd_record(sigs: torch.Tensor) -> torch.Tensor:
        """
        Forward 1 record: (T, 1, 3000) → ResNet → (T, n_feat) → TCN chunk.
        Trả logits (T, 5).
        """
        T = sigs.shape[0]
        # ResNet forward theo batch chunks để tránh OOM
        chunk = int(getattr(cfg, "joint_resnet_chunk", 128))
        feats = []
        for s in range(0, T, chunk):
            xb = sigs[s:s + chunk].to(device)
            feats.append(resnet(xb, extract_features=True))
        feats = torch.cat(feats, dim=0)              # (T, n_feat)
        # TCN expects (B, T, D), B=1
        logits = tcn(feats.unsqueeze(0)).squeeze(0)  # (T, 5)
        return logits

    for epoch in range(1, epochs + 1):
        resnet.train(); tcn.train()
        # Re-assert BN eval nếu cần
        if getattr(cfg, "joint_freeze_bn", True):
            for m in resnet.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

        for sigs, lbls in tr_ld:
            # sigs: (B, T, 1, 3000) ; lbls: (B, T)
            B = sigs.shape[0]
            sigs_flat = sigs.view(B * sigs.shape[1], 1, -1)  # (B*T, 1, 3000)
            lbls_flat = lbls.view(-1)                        # (B*T,)

            optimizer.zero_grad()
            logits = _fwd_record(sigs_flat if B == 1 else sigs[:, 0])  # simplified: B=1 path
            # Cho batch>1 chưa hỗ trợ: dùng loop
            if B == 1:
                loss = loss_fn(logits, lbls[0].to(device))
            else:
                # fallback: process từng record, sum loss
                loss = 0.0
                T = sigs.shape[1]
                for b in range(B):
                    logits_b = _fwd_record(sigs[b])
                    loss = loss + loss_fn(logits_b, lbls[b].to(device))
                loss = loss / B
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in resnet.parameters() if p.requires_grad] +
                [p for p in tcn.parameters()    if p.requires_grad],
                1.0,
            )
            optimizer.step()

        # ── Validation: tính MF1 thay vì val_loss (early stop tốt hơn cho imbalance)
        if epoch % val_every == 0:
            resnet.eval(); tcn.eval()
            v_preds, v_trues = [], []
            with torch.no_grad():
                for sigs, lbls in vl_ld:
                    B = sigs.shape[0]
                    for b in range(B):
                        logits = _fwd_record(sigs[b])
                        valid = (lbls[b] >= 0) & (lbls[b] < N_CLASSES)
                        v_preds.append(logits.argmax(dim=1).cpu().numpy()[valid])
                        v_trues.append(lbls[b].numpy()[valid])
            v_preds = np.concatenate(v_preds) if v_preds else np.array([], dtype=np.int64)
            v_trues = np.concatenate(v_trues) if v_trues else np.array([], dtype=np.int64)
            if v_trues.size > 0:
                v_mf1 = float(f1_score(v_trues, v_preds, average="macro", zero_division=0))
            else:
                v_mf1 = 0.0

            improved = v_mf1 > best_mf1 + 1e-4
            if improved:
                best_mf1, no_improve = v_mf1, 0
                best_resnet_state = copy.deepcopy(resnet.state_dict())
                best_tcn_state    = copy.deepcopy(tcn.state_dict())
            else:
                no_improve += 1

            _log(cfg.log_cb,
                 f"    Joint ep {epoch:3d}/{epochs}  val_MF1={v_mf1*100:.2f}%"
                 + (" ← best" if improved else ""))

            if no_improve * val_every >= patience:
                _log(cfg.log_cb, f"    Joint early stop ep {epoch}  best_MF1={best_mf1*100:.2f}%")
                break

    resnet.load_state_dict(best_resnet_state)
    tcn.load_state_dict(best_tcn_state)
    return resnet, tcn


# ══════════════════════════════════════════════════════════════════
# Evaluate
# ══════════════════════════════════════════════════════════════════

def evaluate_fold(tcn: nn.Module, feat_lbl_list: List,
                  device: torch.device,
                  return_probs: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Predict trên tập test feats.

    Returns:
        preds:   (N,) predicted class indices
        trues:   (N,) ground-truth labels (chỉ valid epochs)
        probs:   (N, 5) softmax probabilities hoặc None nếu return_probs=False
    """
    tcn.eval()
    all_p, all_t, all_pr = [], [], []
    with torch.no_grad():
        for feats, lbls in feat_lbl_list:
            xb   = torch.from_numpy(feats.astype(np.float32)).unsqueeze(0).to(device)
            logits = tcn(xb).squeeze(0).cpu().numpy()              # (T, 5)
            valid = (lbls >= 0) & (lbls < N_CLASSES)
            pred  = logits.argmax(axis=1)
            all_p.append(pred[valid])
            all_t.append(lbls[valid])
            if return_probs:
                # softmax ổn định
                z = logits - logits.max(axis=1, keepdims=True)
                e = np.exp(z)
                probs = e / e.sum(axis=1, keepdims=True)
                all_pr.append(probs[valid])
    preds = np.concatenate(all_p)
    trues = np.concatenate(all_t)
    probs = np.concatenate(all_pr) if return_probs else None
    return preds, trues, probs


def compute_metrics(preds: np.ndarray, trues: np.ndarray) -> Dict:
    acc   = accuracy_score(trues, preds)
    mf1   = f1_score(trues, preds, average="macro",  zero_division=0)
    kappa = cohen_kappa_score(trues, preds)
    f1s   = f1_score(trues, preds, average=None,
                     labels=list(range(N_CLASSES)), zero_division=0)
    return {
        "acc": acc, "mf1": mf1, "kappa": kappa,
        **{f"f1_{s}": float(f1s[i]) for i, s in enumerate(SLEEP_STAGES)},
    }


# ══════════════════════════════════════════════════════════════════
# Orchestrator chính
# ══════════════════════════════════════════════════════════════════

def run_full_pipeline(
    cfg: PipelineConfig,
    folds_to_run:    Optional[List[int]] = None,
    resnet_ckpt_dir: Optional[str] = None,   # load ResNet weights (eval / fine-tune)
    tcn_ckpt_dir:    Optional[str] = None,   # load TCN weights    (eval only)
    train_resnet:    bool = True,
    train_tcn:       bool = True,
    fine_tune:       bool = False,           # fine-tune TCN trên data mới
    run_id: Optional[str] = None,            # override auto-generated run_id
    deterministic: bool = True,              # set CUDA deterministic flags
) -> List[Dict]:
    """
    Orchestrator. 3 kịch bản:

    1. Train mới   : train_resnet=True,  train_tcn=True,  fine_tune=False
    2. Eval chéo   : train_resnet=False, train_tcn=False, resnet_ckpt_dir=..., tcn_ckpt_dir=...
    3. Fine-tune   : train_resnet=False, train_tcn=True,  fine_tune=True,
                     resnet_ckpt_dir=... (freeze ResNet, train TCN với finetune_lr)

    Reproducibility:
      - set_seed(cfg.seed) ở đầu, CUDA deterministic nếu deterministic=True
      - Mỗi run có run_id và lưu config + split_manifest vào output_dir
      - Mọi RNG con đều dùng derive_seed(base, fold, stage) thay vì hard-code
    """
    # 1. Seed mọi RNG ngay từ đầu (trước mọi DataLoader / Tensor)
    set_seed(cfg.seed, deterministic=deterministic)

    # 2. Sinh run_id và tạo output directory layout
    cfg_dict = asdict(cfg)
    # Loại bỏ callback không serializable
    cfg_dict.pop("log_cb", None)
    cfg_dict.pop("progress_cb", None)
    cfg_dict.pop("metrics_cb", None)
    chash = config_hash(cfg_dict)
    if run_id is None:
        run_id = make_run_id(cfg.seed, chash, prefix="pipeline")

    output_dir = Path(cfg.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # 3. Ghi config và meta của run (paper-ready)
    (output_dir / "config.yaml").write_text(safe_yaml_dump(cfg_dict))
    run_meta = {
        "run_id": run_id,
        "config_hash": chash,
        "seed": int(cfg.seed),
        "deterministic": bool(deterministic),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "git_commit": git_commit(),
        "created_at_utc": utc_now_iso(),
    }
    (output_dir / "meta.yaml").write_text(safe_yaml_dump(run_meta))
    _log(cfg.log_cb, f"Run ID: {run_id}")
    _log(cfg.log_cb, f"  Output: {output_dir}")

    # Override output_dir/ckpt_dir để artifacts đi vào run_id subfolder
    cfg.output_dir = str(output_dir)
    cfg.ckpt_dir   = str(ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(cfg.log_cb, f"Device: {device}")

    subject_records, _ = load_all_records(cfg.npz_dir, cfg.log_cb)
    fold_groups = build_fold_groups(subject_records, cfg.n_folds, cfg.seed)

    # Ghi split_manifest ngay sau khi fold_groups sẵn sàng
    split_manifest = {
        "run_id": run_id,
        "config_hash": chash,
        "seed": int(cfg.seed),
        "n_folds": int(cfg.n_folds),
        "folds": [
            {"fold": int(i + 1), "test_subjects": list(g)} for i, g in enumerate(fold_groups)
        ],
        "all_subjects": sorted(subject_records.keys()),
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2)
    )

    all_fold_metrics = []
    total        = len(fold_groups)
    active_folds = folds_to_run if folds_to_run is not None else list(range(total))

    for fold_i in active_folds:
        test_group = fold_groups[fold_i]
        test_sids  = list(test_group)
        train_sids = [s for g in fold_groups for s in g if s not in test_sids]
        train_recs = [r for s in train_sids for r in subject_records[s]]
        test_recs  = [r for s in test_sids  for r in subject_records[s]]

        _log(cfg.log_cb, f"\n{'='*60}")
        _log(cfg.log_cb, f"FOLD {fold_i+1}/{total}  |  test subjects: {test_sids}")
        if cfg.progress_cb:
            cfg.progress_cb(fold_i, total, f"Fold {fold_i+1}")
        t0 = time.time()

        # ── ResNet ──────────────────────────────────────────────
        resnet = EEG_ResNet1D(cfg.n_feat, N_CLASSES).to(device)

        if resnet_ckpt_dir:
            ckpt = Path(resnet_ckpt_dir) / f"resnet_fold{fold_i+1}.pth"
            resnet.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            _log(cfg.log_cb, f"  ✓ Loaded ResNet: {ckpt.name}")
            if fine_tune and cfg.freeze_resnet:
                for p in resnet.parameters():
                    p.requires_grad = False
                _log(cfg.log_cb, "  ResNet frozen (fine-tune mode)")
        elif train_resnet:
            _log(cfg.log_cb, "  Training ResNet ...")
            resnet = train_resnet_fold(train_recs, fold_i, cfg, device)
            torch.save(resnet.state_dict(), ckpt_path / f"resnet_fold{fold_i+1}.pth")
            _log(cfg.log_cb, f"  ✓ Saved ResNet checkpoint")

        # ── Extract features ─────────────────────────────────────
        _log(cfg.log_cb, "  Extracting features ...")
        train_feats = extract_features(train_recs, resnet, device)
        test_feats  = extract_features(test_recs,  resnet, device)

        # ── TCN ──────────────────────────────────────────────────
        tcn = SleepTCN(
            input_size=cfg.n_feat, dim=cfg.tcn_dim,
            kernel_size=cfg.tcn_kernel_size, n_blocks=cfg.tcn_blocks,
            dropout=cfg.tcn_dropout, n_classes=N_CLASSES,
        ).to(device)

        if tcn_ckpt_dir:
            ckpt = Path(tcn_ckpt_dir) / f"tcn_fold{fold_i+1}.pth"
            tcn.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            _log(cfg.log_cb, f"  ✓ Loaded TCN: {ckpt.name}")

            if fine_tune:
                # Fine-tune: load TCN weights rồi tiếp tục train với lr nhỏ
                _log(cfg.log_cb, f"  Fine-tuning TCN (lr={cfg.finetune_lr}) ...")
                tcn = train_tcn_fold(train_feats, fold_i, cfg, device,
                                     pretrained_tcn=tcn,
                                     lr_override=cfg.finetune_lr)
                torch.save(tcn.state_dict(), ckpt_path / f"tcn_finetune_fold{fold_i+1}.pth")
                _log(cfg.log_cb, "  ✓ Saved fine-tuned TCN checkpoint")

        elif train_tcn:
            _log(cfg.log_cb, "  Training TCN ...")
            tcn = train_tcn_fold(train_feats, fold_i, cfg, device)
            torch.save(tcn.state_dict(), ckpt_path / f"tcn_fold{fold_i+1}.pth")
            _log(cfg.log_cb, "  ✓ Saved TCN checkpoint")

        # ── Evaluate ─────────────────────────────────────────────
        # Khi post-process enabled, predict có cả raw và smoothed — cho phép
        # so sánh trong paper / ablation table.
        need_probs = bool(cfg.postprocess.enabled)
        preds, trues, probs = evaluate_fold(
            tcn, test_feats, device, return_probs=need_probs
        )
        m_raw = compute_metrics(preds, trues)
        if probs is not None and cfg.postprocess.enabled:
            smoothed = postprocess_hypnogram(preds, probs, cfg.postprocess)
            m_smooth = compute_metrics(smoothed, trues)
            n_changed = int(np.sum(smoothed != preds))
            _log(cfg.log_cb,
                 f"  Post-process [{','.join(cfg.postprocess.methods)}]: "
                 f"{n_changed}/{len(preds)} epoch sửa, "
                 f"MF1 {m_raw['mf1']*100:.2f}% → {m_smooth['mf1']*100:.2f}%")
        else:
            m_smooth = None
            smoothed = preds

        m = m_raw if m_smooth is None else m_smooth
        elapsed = time.time() - t0

        _log(cfg.log_cb,
             f"  ACC={m['acc']*100:.2f}%  MF1={m['mf1']*100:.2f}%  "
             f"κ={m['kappa']:.4f}  ({elapsed:.0f}s)")

        fold_result = {
            "fold": fold_i + 1,
            "test_subjects": test_sids,
            "n_test_epochs": len(trues),
            "elapsed_s": elapsed,
            "raw_metrics":    m_raw,
            "postprocess_metrics": m_smooth,
            "postprocess_methods": list(cfg.postprocess.methods) if cfg.postprocess.enabled else [],
            **m,
        }
        all_fold_metrics.append(fold_result)
        if cfg.metrics_cb:
            cfg.metrics_cb(fold_i, fold_result)

        # Dọn VRAM
        del resnet, tcn, train_feats, test_feats
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_file = Path(cfg.output_dir) / "fold_metrics.json"
    with open(out_file, "w") as f:
        json.dump(all_fold_metrics, f, indent=2, default=str)
    _log(cfg.log_cb, f"\n✓ Kết quả lưu → {out_file}")

    if cfg.progress_cb:
        cfg.progress_cb(total, total, "Hoàn thành")

    return all_fold_metrics


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════
# Evaluate toan bo - khong fold, dung 1 checkpoint co dinh
# ══════════════════════════════════════════════════════════════════

def evaluate_all(cfg, resnet_ckpt: str, tcn_ckpt: str) -> Dict:
    """
    Evaluate toàn bộ dataset với 1 checkpoint cố định.
    Streaming per-record: extract features → predict → bỏ ngay.
    Không giữ toàn bộ features trong RAM — tránh OOM khi 200 SHHS files.
    Progress callback được gọi sau mỗi record để log realtime.
    """
    import time, json as _json
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(cfg.log_cb, f"Device: {device}")

    _, all_records = load_all_records(cfg.npz_dir, cfg.log_cb)
    if not all_records:
        raise ValueError("Khong co record nao hop le de evaluate!")
    n_total = len(all_records)
    _log(cfg.log_cb, f"Tong records: {n_total}")

    # Load models
    resnet = EEG_ResNet1D(cfg.n_feat, N_CLASSES).to(device)
    resnet.load_state_dict(torch.load(resnet_ckpt, map_location=device, weights_only=True))
    resnet.eval()
    _log(cfg.log_cb, f"  ✓ Loaded ResNet: {Path(resnet_ckpt).name}")

    tcn = SleepTCN(
        input_size=cfg.n_feat, dim=cfg.tcn_dim,
        kernel_size=cfg.tcn_kernel_size, n_blocks=cfg.tcn_blocks,
        dropout=cfg.tcn_dropout, n_classes=N_CLASSES,
    ).to(device)
    tcn.load_state_dict(torch.load(tcn_ckpt, map_location=device, weights_only=True))
    tcn.eval()
    _log(cfg.log_cb, f"  ✓ Loaded TCN: {Path(tcn_ckpt).name}")

    _log(cfg.log_cb, f"  Bat dau evaluate streaming {n_total} records ...")
    if cfg.progress_cb:
        cfg.progress_cb(0, n_total, f"0/{n_total} records")

    t0 = time.time()
    all_p, all_t = [], []

    with torch.no_grad():
        for i, rec in enumerate(all_records, 1):
            # 1. Load signal từ disk (1 file tại 1 thời điểm)
            try:
                sigs, lbls = load_npz(rec["fpath"])
            except Exception as e:
                _log(cfg.log_cb, f"  ✗ [{i}/{n_total}] {rec['fname']}: {e}")
                if cfg.progress_cb:
                    cfg.progress_cb(i, n_total, f"{i}/{n_total} - loi file")
                continue

            # 2. Extract features batch-by-batch, bỏ signal ngay
            feats_chunks = []
            for j in range(0, len(sigs), 128):
                xb = torch.from_numpy(sigs[j:j+128]).unsqueeze(1).to(device)
                feats_chunks.append(resnet(xb, extract_features=True).cpu().numpy())
            del sigs
            feats = np.concatenate(feats_chunks, axis=0)

            # 3. TCN predict rồi bỏ features ngay
            xb   = torch.from_numpy(feats.astype(np.float32)).unsqueeze(0).to(device)
            pred = tcn(xb).squeeze(0).cpu().numpy().argmax(axis=1)  # (T,)
            del feats, xb

            # 4. Bug-5 fix: đảm bảo pred và lbls cùng length trước khi mask.
            # load_npz có thể crop lbls; pred được tính từ feats cùng file nên
            # phải cùng length — assert để bắt bất kỳ lệch nào sớm.
            assert len(pred) == len(lbls), (
                f"[{rec['fname']}] pred len {len(pred)} != lbls len {len(lbls)}"
            )
            valid = (lbls >= 0) & (lbls < N_CLASSES)
            all_p.append(pred[valid])
            all_t.append(lbls[valid])

            gc.collect()

            # 5. Log + progress sau mỗi record
            elapsed_so_far = time.time() - t0
            eta = (elapsed_so_far / i) * (n_total - i) if i > 0 else 0
            _log(cfg.log_cb,
                 f"  [{i:>3d}/{n_total}] {rec['fname']}  "
                 f"({len(pred[valid])} ep)  elapsed={elapsed_so_far:.0f}s  ETA~{eta:.0f}s")
            if cfg.progress_cb:
                cfg.progress_cb(i, n_total, f"{i}/{n_total} records  ETA~{eta:.0f}s")

    preds = np.concatenate(all_p)
    trues = np.concatenate(all_t)
    m = compute_metrics(preds, trues)
    elapsed = time.time() - t0

    _log(cfg.log_cb,
         f"\n  ✓ Xong!  ACC={m['acc']*100:.2f}%  MF1={m['mf1']*100:.2f}%  "
         f"k={m['kappa']:.4f}  ({elapsed:.0f}s)")

    result = {
        "fold": "all", "n_records": n_total,
        "n_epochs": int(len(trues)), "elapsed_s": elapsed, **m,
    }
    out_file = Path(cfg.output_dir) / "eval_all_metrics.json"
    with open(out_file, "w") as f:
        _json.dump(result, f, indent=2, default=str)
    _log(cfg.log_cb, f"  ✓ Ket qua luu -> {out_file}")
    if cfg.progress_cb:
        cfg.progress_cb(n_total, n_total, "Hoan thanh!")
    return result

def _log(cb, msg: str):
    if cb:
        cb(msg)
    else:
        print(msg)
