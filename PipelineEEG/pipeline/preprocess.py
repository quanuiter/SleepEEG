"""
pipeline/preprocess.py
──────────────────────
Tiền xử lý EDF → NPZ, hỗ trợ cả Sleep-EDF và SHHS.

Sleep-EDF : *PSG.edf + *Hypnogram.edf  (nhãn trong EDF hypnogram)
SHHS      : edfs/*.edf + annotations-events-nsrr/*-nsrr.xml

Cả hai đều xuất ra NPZ cùng schema (x, y, fs, ...) nên
toàn bộ pipeline train/eval phía sau dùng chung không đổi.
"""

import glob
import logging
import ntpath
import os
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np
try:
    import pyedflib
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyedflib", "-q"])
    import pyedflib
from scipy.signal import butter, iirnotch, lfilter, resample, sosfilt

# Hằng số chuẩn hoá — KHÔNG thay đổi, phải nhất quán với trainer.py
TARGET_FS  = 100    # Hz — tần số lấy mẫu mục tiêu
EPOCH_SEC  = 30.0   # giây — độ dài 1 epoch ngủ (cố định theo AASM)

# ── Nhãn giấc ngủ ───────────────────────────────────────────────
STAGE_DICT = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "MOVE": 5, "UNK": 6}

SLEEPEDF_ANN2LABEL = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 6,
    "Movement time": 5,
}

# NSRR XML EventConcept strings (bao phủ cả 2 cách viết thường gặp)
SHHS_ANN2LABEL = {
    "Wake|0":            0,
    "NREM1|1":           1,
    "NREM2|2":           2,
    "NREM3|3":           3,
    "NREM4|4":           3,
    "REM|5":             4,
    "Unscored|9":        6,
    "Wake":              0,
    "Stage 1 sleep|1":   1,
    "Stage 2 sleep|2":   2,
    "Stage 3 sleep|3":   3,
    "Stage 4 sleep|4":   3,
    "REM sleep|5":       4,
    "Unscored":          6,
    "Movement|6":        5,
}


# ══════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════

@dataclass
class PreprocessConfig:
    # ── Dataset ─────────────────────────────────────────────────
    dataset: Literal["sleepedf", "shhs"] = "sleepedf"

    # Sleep-EDF: thư mục chứa *PSG.edf + *Hypnogram.edf
    # SHHS     : thư mục gốc chứa edfs/ và annotations-events-nsrr/
    data_dir:   str = "./data/sleepedf/sleep-cassette"
    output_dir: str = "./preprocessed"
    select_ch:  str = "EEG Fpz-Cz"

    # ── Chọn subset subject ──────────────────────────────────────
    # max_subjects=None  → dùng tất cả
    # max_subjects=N     → chọn ngẫu nhiên N subject (theo random_seed)
    max_subjects:  Optional[int]       = None
    random_seed:   int                 = 42
    # Danh sách subject ID muốn xử lý (None = không lọc)
    # Sleep-EDF: ["SC4002E", ...]   SHHS: ["200001", ...]
    subject_ids:   Optional[List[str]] = None

    # ── Bộ lọc ──────────────────────────────────────────────────
    apply_bandpass: bool  = True
    bp_low:         float = 0.5
    bp_high:        float = 30.0
    bp_order:       int   = 4

    apply_notch:    bool  = True
    notch_freq:     float = 50.0
    notch_q:        float = 30.0

    # ── Chuẩn hoá ───────────────────────────────────────────────
    clip_uv:      float = 800.0
    scale_factor: float = 100.0

    # ── Window cạnh vùng ngủ (phút) ─────────────────────────────
    w_edge_mins: int = 30

    # ── Callback UI ──────────────────────────────────────────────
    progress_cb: Optional[Callable[[int, int, str], None]] = field(default=None, repr=False)


# ══════════════════════════════════════════════════════════════════
# Bộ lọc tín hiệu
# ══════════════════════════════════════════════════════════════════

def bandpass_filter(signal, fs, low=0.5, high=30.0, order=4):
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfilt(sos, signal)


def notch_filter(signal, fs, freq=50.0, Q=30.0):
    b, a = iirnotch(freq / (fs / 2), Q)
    return lfilter(b, a, signal)


def clip_and_scale(signal, clip_uv=800.0, scale=100.0):
    return (np.clip(signal, -clip_uv, clip_uv) / scale).astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# Đọc nhãn
# ══════════════════════════════════════════════════════════════════

def read_labels_sleepedf(ann_fp: str, epoch_duration: float) -> np.ndarray:
    """Nhãn từ Hypnogram EDF."""
    f = pyedflib.EdfReader(ann_fp)
    onsets, durations, stages = f.readAnnotations()
    f.close()
    labels, total_dur = [], 0
    for onset, dur, stage in zip(onsets, durations, stages):
        onset, dur = int(onset), int(dur)
        assert onset == total_dur
        label = SLEEPEDF_ANN2LABEL["".join(stage)]
        labels.append(np.ones(int(dur / epoch_duration), dtype=np.int32) * label)
        total_dur += dur
    return np.hstack(labels)


def read_labels_shhs(xml_fp: str, epoch_duration: float = 30.0) -> np.ndarray:
    """
    Nhãn từ NSRR XML của SHHS.
    Chỉ lấy ScoredEvent có EventType chứa "Stages".
    """
    tree = ET.parse(xml_fp)
    events = []
    for ev in tree.getroot().iter("ScoredEvent"):
        if "Stages" not in ev.findtext("EventType", ""):
            continue
        concept  = ev.findtext("EventConcept", "").strip()
        start    = float(ev.findtext("Start", "0"))
        duration = float(ev.findtext("Duration", "0"))
        events.append((start, duration, concept))

    if not events:
        raise ValueError(f"Không tìm thấy sleep stage events trong {xml_fp}")

    events.sort(key=lambda e: e[0])
    labels = []
    for _, duration, concept in events:
        label   = SHHS_ANN2LABEL.get(concept, 6)
        n_epoch = max(1, round(duration / epoch_duration))
        labels.append(np.ones(n_epoch, dtype=np.int32) * label)
    return np.hstack(labels)


# ══════════════════════════════════════════════════════════════════
# Ghép cặp file
# ══════════════════════════════════════════════════════════════════

def _pair_sleepedf(data_dir, logger, subject_ids=None, max_subjects=None,
                   random_seed=42) -> List[Tuple]:
    raw_psg = sorted(glob.glob(os.path.join(data_dir, "*PSG.edf")))
    raw_ann = glob.glob(os.path.join(data_dir, "*Hypnogram.edf"))
    pairs = []
    for psg in raw_psg:
        sid = os.path.basename(psg)[:7]
        matched = [f for f in raw_ann if os.path.basename(f).startswith(sid)]
        if matched:
            pairs.append((psg, matched[0], sid))
        else:
            logger.warning(f"Bỏ qua: không tìm thấy nhãn cho {os.path.basename(psg)}")
    return _filter_subjects(pairs, subject_ids, max_subjects, random_seed, logger)


def _pair_shhs(data_dir, logger, subject_ids=None, max_subjects=None,
               random_seed=42) -> List[Tuple]:
    edf_dir = os.path.join(data_dir, "edfs")
    xml_dir = os.path.join(data_dir, "annotations-events-nsrr")
    if not os.path.isdir(edf_dir):
        raise FileNotFoundError(f"Không tìm thấy: {edf_dir}")
    if not os.path.isdir(xml_dir):
        raise FileNotFoundError(f"Không tìm thấy: {xml_dir}")

    # Hỗ trợ cả flat layout và subdirectory layout (shhs1/, shhs2/, ...)
    # edfs/shhs*.edf  hoặc  edfs/shhs1/shhs*.edf, edfs/shhs2/shhs*.edf
    edf_files = sorted(
        glob.glob(os.path.join(edf_dir, "shhs*.edf")) +
        glob.glob(os.path.join(edf_dir, "**", "shhs*.edf"), recursive=True)
    )
    edf_files = sorted(set(edf_files))  # deduplicate

    # Build index: basename → xml_fp (tìm trong xml_dir và các subdir)
    all_xml = (
        glob.glob(os.path.join(xml_dir, "*-nsrr.xml")) +
        glob.glob(os.path.join(xml_dir, "**", "*-nsrr.xml"), recursive=True)
    )
    xml_index = {os.path.basename(x): x for x in all_xml}

    pairs = []
    for edf_fp in edf_files:
        basename    = os.path.basename(edf_fp)
        sid         = basename.replace(".edf", "").split("-")[-1]   # "200001"
        xml_name    = basename.replace(".edf", "-nsrr.xml")
        xml_fp      = xml_index.get(xml_name)
        if xml_fp:
            pairs.append((edf_fp, xml_fp, sid))
        else:
            logger.warning(f"Bỏ qua: không tìm thấy XML cho {basename}")
    return _filter_subjects(pairs, subject_ids, max_subjects, random_seed, logger)


def _filter_subjects(pairs, subject_ids, max_subjects, random_seed, logger):
    if subject_ids:
        before = len(pairs)
        pairs  = [(p, a, s) for p, a, s in pairs if s in subject_ids]
        logger.info(f"Lọc subject_ids: {before} → {len(pairs)}")
    if max_subjects and len(pairs) > max_subjects:
        rng   = random.Random(random_seed)
        pairs = rng.sample(pairs, max_subjects)
        pairs.sort(key=lambda t: t[2])
        logger.info(f"Giới hạn {max_subjects} subjects (seed={random_seed})")
    return pairs


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def run_preprocess(cfg: PreprocessConfig,
                   logger: Optional[logging.Logger] = None) -> List[str]:
    if logger is None:
        logger = _make_console_logger()

    out_path = Path(cfg.output_dir)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True)

    logger.info(f"Dataset: {cfg.dataset.upper()}")

    if cfg.dataset == "sleepedf":
        pairs         = _pair_sleepedf(cfg.data_dir, logger, cfg.subject_ids,
                                       cfg.max_subjects, cfg.random_seed)
        label_reader  = lambda ann, ep: read_labels_sleepedf(ann, ep)
        make_name     = lambda fp: ntpath.basename(fp).replace("-PSG.edf", ".npz")

    elif cfg.dataset == "shhs":
        pairs         = _pair_shhs(cfg.data_dir, logger, cfg.subject_ids,
                                   cfg.max_subjects, cfg.random_seed)
        label_reader  = lambda ann, ep: read_labels_shhs(ann, ep)
        make_name     = lambda fp: "shhs_" + ntpath.basename(fp).replace(".edf", ".npz")

    else:
        raise ValueError(f"dataset không hợp lệ: {cfg.dataset!r}")

    if not pairs:
        logger.error("Không có file nào để xử lý!")
        return []

    logger.info(f"Tổng subjects: {len(pairs)}")
    saved, total = [], len(pairs)

    for i, (psg_fp, ann_fp, sid) in enumerate(pairs):
        if cfg.progress_cb:
            cfg.progress_cb(i, total, os.path.basename(psg_fp))
        logger.info(f"[{i+1}/{total}] Subject {sid}")
        try:
            npz_path = _process_one(psg_fp, ann_fp, cfg, logger, out_path,
                                    label_reader, make_name)
            saved.append(str(npz_path))
        except Exception as e:
            logger.error(f"  ✗ Lỗi: {e}")

    if cfg.progress_cb:
        cfg.progress_cb(total, total, "Hoàn thành")
    logger.info(f"\n✓ Xong: {len(saved)}/{total} files → {cfg.output_dir}")
    return saved


# ══════════════════════════════════════════════════════════════════
# Xử lý 1 cặp file
# ══════════════════════════════════════════════════════════════════

def _process_one(psg_fp, ann_fp, cfg, logger, out_path, label_reader, make_name) -> Path:
    psg_f = pyedflib.EdfReader(psg_fp)

    ch_names      = psg_f.getSignalLabels()
    select_ch_idx = next(
        (s for s in range(psg_f.signals_in_file) if ch_names[s] == cfg.select_ch), -1
    )
    if select_ch_idx == -1:
        raise ValueError(
            f"Channel '{cfg.select_ch}' không có. "
            f"Có sẵn: {', '.join(ch_names)}"
        )

    fs_orig        = psg_f.getSampleFrequency(select_ch_idx)
    # BUG-2 FIX: epoch_duration luôn là 30s (chuẩn AASM).
    # datarecord_duration là thời gian 1 EDF data record — có thể là 1s, 10s, 60s,
    # KHÔNG phải độ dài epoch ngủ. Dùng hằng số EPOCH_SEC thay vì đọc từ header.
    epoch_duration = EPOCH_SEC
    n_ep_samples   = int(EPOCH_SEC * fs_orig)   # số mẫu/epoch tại fs gốc
    n_all_epochs   = int(psg_f.getFileDuration() / EPOCH_SEC)

    raw = psg_f.readSignal(select_ch_idx).astype(np.float64)
    psg_f.close()

    # Lọc tín hiệu tại fs gốc (quan trọng: phải lọc TRƯỚC khi resample
    # để tránh aliasing và giữ đúng tần số sinh lý)
    if cfg.apply_bandpass:
        raw = bandpass_filter(raw, fs_orig, cfg.bp_low, cfg.bp_high, cfg.bp_order)
    if cfg.apply_notch:
        raw = notch_filter(raw, fs_orig, cfg.notch_freq, cfg.notch_q)

    # BUG-1 FIX: resample về TARGET_FS (100 Hz) trước khi reshape.
    # Nếu bỏ qua bước này, SHHS (125 Hz → 3750 mẫu/epoch) sẽ lệch tần số
    # so với Sleep-EDF (100 Hz → 3000 mẫu/epoch) mà mô hình đã train.
    fs = fs_orig
    if fs_orig != TARGET_FS:
        n_target = round(len(raw) * TARGET_FS / fs_orig)
        raw      = resample(raw, n_target)
        fs       = TARGET_FS

    n_ep_samples_target = int(EPOCH_SEC * TARGET_FS)   # = 3000

    n_complete = len(raw) // n_ep_samples_target
    raw        = raw[:n_complete * n_ep_samples_target]
    signals    = np.stack([
        clip_and_scale(s, cfg.clip_uv, cfg.scale_factor)
        for s in raw.reshape(-1, n_ep_samples_target)
    ])

    labels  = label_reader(ann_fp, epoch_duration)
    min_len = min(len(signals), len(labels))
    x, y    = signals[:min_len].astype(np.float32), labels[:min_len].astype(np.int32)

    # Cắt cạnh vùng ngủ
    nw_idx = np.where(y != STAGE_DICT["W"])[0]
    if len(nw_idx) == 0:
        raise ValueError("Không có epoch non-Wake")
    s_idx = max(0,          nw_idx[0]  - cfg.w_edge_mins * 2)
    e_idx = min(len(y) - 1, nw_idx[-1] + cfg.w_edge_mins * 2)
    x, y  = x[s_idx:e_idx+1], y[s_idx:e_idx+1]

    # MOVE / UNK → label -1 (giữ epoch tại chỗ, bảo toàn chuỗi thời gian)
    # normalize_labels() trong trainer.py sẽ giữ -1;
    # extract_features() đổi -1 → -100 → ignore_index trong CE loss bỏ qua.
    mask = (y == STAGE_DICT["MOVE"]) | (y == STAGE_DICT["UNK"])
    if mask.any():
        y[mask] = -1

    save_name = make_name(psg_fp)
    save_path = out_path / save_name
    np.savez(
        save_path, x=x, y=y,
        fs=TARGET_FS,           # fs sau resample — nhất quán với trainer.py
        fs_orig=fs_orig,        # fs gốc để tham khảo
        ch_label=cfg.select_ch,
        dataset=cfg.dataset,
        epoch_duration=epoch_duration,
        n_all_epochs=n_all_epochs,
        n_epochs=len(x),
        label_format="0indexed",  # Bug-3 fix: khai báo rõ để trainer không đoán
        filtered=True,
        filter_range=[cfg.bp_low, cfg.bp_high],
        normalization=f"clip{int(cfg.clip_uv)}_scale{int(cfg.scale_factor)}",
    )
    logger.info(f"  ✓ {save_name}  ({len(x)} epochs)")
    return save_path


# ══════════════════════════════════════════════════════════════════
# Tiện ích dùng bởi UI
# ══════════════════════════════════════════════════════════════════

def list_edf_channels(edf_path: str) -> List[str]:
    """Trả về danh sách channel trong một file EDF."""
    try:
        f = pyedflib.EdfReader(edf_path)
        ch = f.getSignalLabels()
        f.close()
        return ch
    except Exception:
        return []


def scan_subjects(data_dir: str, dataset: str = "sleepedf") -> List[str]:
    """Trả về danh sách subject ID (dùng cho UI multiselect / preview)."""
    if dataset == "sleepedf":
        files = sorted(glob.glob(os.path.join(data_dir, "*PSG.edf")))
        return [os.path.basename(f)[:7] for f in files]
    elif dataset == "shhs":
        edf_dir = os.path.join(data_dir, "edfs")
        files   = sorted(set(
            glob.glob(os.path.join(edf_dir, "shhs*.edf")) +
            glob.glob(os.path.join(edf_dir, "**", "shhs*.edf"), recursive=True)
        ))
        return [os.path.basename(f).replace(".edf", "").split("-")[-1] for f in files]
    return []


def _make_console_logger() -> logging.Logger:
    logger = logging.getLogger("preprocess")
    logger.propagate = False
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        ch  = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger
