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
from dataclasses import dataclass, field
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

    # Callbacks (Streamlit)
    log_cb:      Optional[Callable[[str], None]]           = field(default=None, repr=False)
    progress_cb: Optional[Callable[[int, int, str], None]] = field(default=None, repr=False)
    metrics_cb:  Optional[Callable[[int, Dict], None]]     = field(default=None, repr=False)


# ══════════════════════════════════════════════════════════════════
# Datasets
# ══════════════════════════════════════════════════════════════════

class _EpochDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals, self.labels = signals, labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        x = torch.from_numpy(self.signals[idx].astype(np.float32)).unsqueeze(0)
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

def _collect_valid(records, max_epochs_per_rec: int = 300, max_total: int = 30_000):
    """
    Load từng file, lấy valid epochs, ghép lại.
    max_epochs_per_rec : giới hạn epoch mỗi record (tránh OOM)
    max_total          : giới hạn tổng epoch (safety cap khi 200 SHHS files)
    """
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
            idx = np.random.choice(len(sv), max_epochs_per_rec, replace=False)
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
    rng   = np.random.default_rng(42 + fold_idx)
    perm  = rng.permutation(len(train_records)).tolist()
    val_recs = [train_records[i] for i in perm[:n_val]]
    tr_recs  = [train_records[i] for i in perm[n_val:]]

    X_train, y_train = _collect_valid(tr_recs)
    X_val,   y_val   = _collect_valid(val_recs)

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = torch.FloatTensor(weights).to(device)

    _log(cfg.log_cb, f"  ResNet | Train {len(y_train):,} / Val {len(y_val):,} epochs")

    g = torch.Generator(); g.manual_seed(cfg.seed + fold_idx)
    tr_ld = DataLoader(_EpochDataset(X_train, y_train),
                       batch_size=cfg.resnet_batch_size, shuffle=True,
                       generator=g, num_workers=2, pin_memory=(device.type == "cuda"))
    vl_ld = DataLoader(_EpochDataset(X_val, y_val),
                       batch_size=cfg.resnet_batch_size * 2, shuffle=False,
                       num_workers=2, pin_memory=(device.type == "cuda"))

    model     = EEG_ResNet1D(cfg.n_feat, N_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
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
    rng    = np.random.default_rng(99 + fold_idx)
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
# Evaluate
# ══════════════════════════════════════════════════════════════════

def evaluate_fold(tcn: nn.Module, feat_lbl_list: List,
                  device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    tcn.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for feats, lbls in feat_lbl_list:
            xb   = torch.from_numpy(feats.astype(np.float32)).unsqueeze(0).to(device)
            pred = tcn(xb).squeeze(0).cpu().numpy().argmax(axis=1)
            valid = (lbls >= 0) & (lbls < N_CLASSES)
            all_p.append(pred[valid])
            all_t.append(lbls[valid])
    return np.concatenate(all_p), np.concatenate(all_t)


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
) -> List[Dict]:
    """
    Orchestrator. 3 kịch bản:

    1. Train mới   : train_resnet=True,  train_tcn=True,  fine_tune=False
    2. Eval chéo   : train_resnet=False, train_tcn=False, resnet_ckpt_dir=..., tcn_ckpt_dir=...
    3. Fine-tune   : train_resnet=False, train_tcn=True,  fine_tune=True,
                     resnet_ckpt_dir=... (freeze ResNet, train TCN với finetune_lr)
    """
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(cfg.ckpt_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(cfg.log_cb, f"Device: {device}")

    subject_records, _ = load_all_records(cfg.npz_dir, cfg.log_cb)
    fold_groups = build_fold_groups(subject_records, cfg.n_folds, cfg.seed)

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
        preds, trues = evaluate_fold(tcn, test_feats, device)
        m = compute_metrics(preds, trues)
        elapsed = time.time() - t0

        _log(cfg.log_cb,
             f"  ACC={m['acc']*100:.2f}%  MF1={m['mf1']*100:.2f}%  "
             f"κ={m['kappa']:.4f}  ({elapsed:.0f}s)")

        fold_result = {
            "fold": fold_i + 1,
            "test_subjects": test_sids,
            "n_test_epochs": len(trues),
            "elapsed_s": elapsed,
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
