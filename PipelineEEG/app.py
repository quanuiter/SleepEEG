"""
app.py — Sleep Pipeline Streamlit UI
════════════════════════════════════════════════════════════════════
Chạy bằng:  streamlit run app.py

Gồm 5 tab:
  1. Preprocess  — EDF → NPZ với cấu hình bộ lọc
  2. Train       — K-Fold ResNet + TCN với cấu hình siêu tham số
  3. Evaluate    — Load weights cũ → cross-val trên data mới
  4. Results     — Bảng metrics + biểu đồ per-fold
  5. Inference   — Upload raw EDF → Preprocess → Trích xuất → Phân lớp (end-to-end)
"""

import json
import sys
from pathlib import Path

# Đảm bảo thư mục hiện tại nằm trong sys.path để import các module sibling
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import numpy as np
import streamlit as st

# ─────────────────────────────────────────────────────────────────
# Config persistence helpers
# ─────────────────────────────────────────────────────────────────
CONFIG_KEYS = [
    # Preprocess
    "pp_dataset", "pp_data_dir", "pp_output_dir", "pp_channel",
    "pp_max_subjects", "pp_random_seed",
    "apply_bp", "bp_low", "bp_high", "bp_order",
    "apply_notch", "notch_freq", "notch_q",
    "clip_uv", "scale_factor", "w_edge",
    # Train
    "tr_npz", "tr_out", "tr_ckpt",
    "n_folds", "seed", "folds_input",
    "rn_lr", "rn_epochs", "rn_batch", "rn_patience", "n_feat", "rn_val",
    "tcn_lr", "tcn_epochs", "tcn_patience", "tcn_batch",
    "tcn_dim", "tcn_blocks", "tcn_kernel", "tcn_dropout", "tcn_val",
    # Evaluate / Fine-tune
    "ev_npz", "ev_out", "ev_resnet_dir", "ev_tcn_dir",
    "use_resnet_ckpt", "use_tcn_ckpt", "ev_n_folds", "ev_seed", "ev_folds_input",
    "ev_mode", "finetune_lr", "freeze_resnet",
]

DEFAULT_CONFIG_PATH = "./pipeline_config.json"

def _save_config(path: str):
    data = {k: st.session_state.get(k) for k in CONFIG_KEYS if k in st.session_state}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def _load_config(path: str) -> bool:
    try:
        p = Path(path)
        if not p.exists():
            return False
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            st.session_state[k] = v
        return True
    except (OSError, json.JSONDecodeError):
        return False

# ─────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sleep Stage Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# Custom CSS — dark scientific aesthetic
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.08em;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    border-bottom: 2px solid #00d4aa !important;
    color: #00d4aa !important;
}

.metric-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1rem 1.4rem;
    text-align: center;
}
.metric-card .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #8b949e;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.metric-card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #00d4aa;
}

.log-box {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #c9d1d9;
    max-height: 340px;
    overflow-y: auto;
    line-height: 1.6;
}

.badge {
    display: inline-block;
    background: #1f6feb22;
    border: 1px solid #1f6feb66;
    border-radius: 12px;
    padding: 0.15rem 0.7rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #58a6ff;
}
.badge-green {
    background: #00d4aa18;
    border-color: #00d4aa44;
    color: #00d4aa;
}
.badge-orange {
    background: #f7931a18;
    border-color: #f7931a44;
    color: #f7931a;
}

div[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "preprocess_logs":  [],
        "train_logs":       [],
        "eval_logs":        [],
        "fold_metrics":     [],
        "running_preproc":  False,
        "running_train":    False,
        "running_eval":     False,
        "preproc_done":     False,
        "train_done":       False,
        "eval_done":        False,
        # Inference tab
        "infer_logs":       [],
        "running_infer":    False,
        "infer_done":       False,
        "infer_result":     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# Auto-load config on first run
if "config_loaded" not in st.session_state:
    st.session_state.config_loaded = False
if not st.session_state.config_loaded:
    if _load_config(DEFAULT_CONFIG_PATH):
        st.session_state.config_loaded = True


# ─────────────────────────────────────────────────────────────────
# Sidebar — quick status + config save/load
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Sleep Pipeline")
    st.markdown("---")

    # ── Config persistence ───────────────────────────────────────
    st.markdown("**💾 Config**")
    with st.expander("Đường dẫn file config (Dành cho Dev)", expanded=False):
        cfg_path = st.text_input(
            "Đường dẫn file config JSON",
            value=DEFAULT_CONFIG_PATH,
            key="cfg_path",
            help="Lưu/Load toàn bộ tham số vào file này (nên dùng đường dẫn trong Drive)"
        )

    col_sv, col_ld = st.columns(2)
    with col_sv:
        if st.button("💾 Save", use_container_width=True, help="Lưu tất cả tham số hiện tại"):
            try:
                _save_config(cfg_path)
                st.success(f"✓ Đã lưu!")
            except Exception as e:
                st.error(f"Lỗi: {e}")
    with col_ld:
        if st.button("📂 Load", use_container_width=True, help="Load tham số từ file"):
            if _load_config(cfg_path):
                st.success("✓ Đã load!")
                st.rerun()
            else:
                st.warning("Không tìm thấy file.")

    # Download config as JSON
    current_cfg = {k: st.session_state.get(k) for k in CONFIG_KEYS if k in st.session_state}
    st.download_button(
        "⬇ Download config",
        data=json.dumps(current_cfg, indent=2, ensure_ascii=False),
        file_name="pipeline_config.json",
        mime="application/json",
        use_container_width=True,
    )

    # Upload config
    uploaded_cfg = st.file_uploader("⬆ Upload config JSON", type="json", key="cfg_upload")
    if uploaded_cfg:
        try:
            data = json.load(uploaded_cfg)
            for k, v in data.items():
                st.session_state[k] = v
            st.success("✓ Config đã được áp dụng!")
            st.rerun()
        except Exception as e:
            st.error(f"Lỗi đọc config: {e}")

    st.markdown("---")
    st.markdown("**Trạng thái**")

    def _badge(label, done):
        cls = "badge-green" if done else "badge-orange"
        icon = "✓" if done else "○"
        st.markdown(f'<span class="badge {cls}">{icon} {label}</span>', unsafe_allow_html=True)

    _badge("Preprocess", st.session_state.preproc_done)
    st.markdown("")
    _badge("Train", st.session_state.train_done)
    st.markdown("")
    _badge("Evaluate", st.session_state.eval_done)
    st.markdown("")
    _badge("Inference", st.session_state.infer_done)

    # Reset nếu bị kẹt
    any_stuck = (
        st.session_state.running_preproc or
        st.session_state.running_train   or
        st.session_state.running_eval    or
        st.session_state.running_infer
    )
    if any_stuck:
        st.markdown("")
        st.warning("⚠️ Có tác vụ đang bị kẹt")
        if st.button("🔄 Reset trạng thái", use_container_width=True):
            st.session_state.running_preproc = False
            st.session_state.running_train   = False
            st.session_state.running_eval    = False
            st.session_state.running_infer   = False
            st.rerun()

    st.markdown("---")
    st.markdown("**Hardware**")
    try:
        import torch
        dev_str = f"🟢 CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "🟡 CPU"
    except ImportError:
        dev_str = "⚠️ PyTorch chưa cài"
    st.caption(dev_str)

    st.markdown("---")
    st.caption("v2.0 · Sleep Stage Pipeline")


# ─────────────────────────────────────────────────────────────────
# Main title
# ─────────────────────────────────────────────────────────────────
st.markdown("# Sleep Stage Classification Pipeline")
st.markdown("Sleep-EDF / SHHS → Preprocess → ResNet-1D → TCN → Sleep Staging")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚙️  PREPROCESS",
    "🏋️  TRAIN",
    "🔍  EVALUATE",
    "📊  RESULTS",
    "🔬  INFERENCE",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — PREPROCESS
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Tiền xử lý EDF → NPZ")
    st.markdown("Hỗ trợ **Sleep-EDF** (PSG + Hypnogram EDF) và **SHHS** (EDF + XML).")

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        # ── Dataset selector ─────────────────────────────────────
        st.markdown("#### 🗄️ Dataset")
        _ds_options = ["sleepedf", "shhs"]
        _ds_saved   = st.session_state.get("pp_dataset", "sleepedf")
        _ds_idx     = _ds_options.index(_ds_saved) if _ds_saved in _ds_options else 0
        pp_dataset  = st.selectbox(
            "Loại dataset",
            _ds_options,
            index=_ds_idx,
            key="pp_dataset",
            format_func=lambda x: "Sleep-EDF" if x == "sleepedf" else "SHHS",
            help="Sleep-EDF: thư mục chứa *PSG.edf + *Hypnogram.edf\n"
                 "SHHS: thư mục gốc có edfs/ và annotations-events-nsrr/",
        )

        st.markdown("#### 📁 Đường dẫn")
        _default_dir = {
            "sleepedf": "./data/sleepedf/sleep-cassette",
            "shhs":     "./data/shhs",
        }
        pp_data_dir = st.text_input(
            "Thư mục data",
            value=st.session_state.get("pp_data_dir", _default_dir[pp_dataset]),
            key="pp_data_dir",
            help="Sleep-EDF: chứa *PSG.edf  |  SHHS: thư mục gốc có edfs/ và annotations-events-nsrr/",
        )
        pp_output_dir = st.text_input("Thư mục output NPZ", value="./preprocessed",
                                      key="pp_output_dir")

        # ── Channel EEG ──────────────────────────────────────────
        _ch_presets = {
            "sleepedf": ["EEG Fpz-Cz", "EEG Pz-Oz", "EEG C4-A1", "EEG F4-A1"],
            "shhs":     ["EEG", "EEG(sec)", "EEG1", "EEG2"],
        }
        _ch_options = _ch_presets[pp_dataset]
        _ch_saved   = st.session_state.get("pp_channel", _ch_options[0])
        _ch_idx     = _ch_options.index(_ch_saved) if _ch_saved in _ch_options else 0
        pp_channel  = st.selectbox("EEG Channel", _ch_options,
                                   index=_ch_idx, key="pp_channel")

        # ── Chọn subset subject ──────────────────────────────────
        st.markdown("#### 👥 Chọn subject")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            pp_max_subjects = st.number_input(
                "Tối đa N subjects (0 = tất cả)",
                value=int(st.session_state.get("pp_max_subjects", 0)),
                min_value=0, step=1, key="pp_max_subjects",
                help="Chọn ngẫu nhiên N subject từ toàn bộ dataset",
            )
        with col_s2:
            pp_random_seed = st.number_input(
                "Random seed", value=int(st.session_state.get("pp_random_seed", 42)),
                min_value=0, key="pp_random_seed",
                help="Seed để chọn ngẫu nhiên subject (tái lập được)",
            )

        # Preview: quét subject IDs có sẵn
        with st.expander("🔍 Xem danh sách subject có sẵn", expanded=False):
            if st.button("Quét thư mục", key="scan_subjects_btn"):
                try:
                    from pipeline.preprocess import scan_subjects
                    sids = scan_subjects(pp_data_dir, pp_dataset)
                    if sids:
                        st.success(f"Tìm thấy {len(sids)} subjects")
                        st.code(", ".join(sids[:50]) + ("..." if len(sids) > 50 else ""))
                    else:
                        st.warning("Không tìm thấy subject nào.")
                except Exception as e:
                    st.error(f"Lỗi: {e}")

        st.markdown("#### 🔬 Bộ lọc")
        col_bp, col_notch = st.columns(2)
        with col_bp:
            apply_bp = st.checkbox("Bandpass", value=True, key="apply_bp")
            bp_low   = st.number_input("Low (Hz)",  value=0.5,  step=0.1, format="%.1f",
                                       disabled=not apply_bp, key="bp_low")
            bp_high  = st.number_input("High (Hz)", value=30.0, step=1.0, format="%.1f",
                                       disabled=not apply_bp, key="bp_high")
            bp_order = st.number_input("Order", value=4, min_value=1, max_value=8,
                                       disabled=not apply_bp, key="bp_order")
        with col_notch:
            apply_notch = st.checkbox("Notch filter", value=True, key="apply_notch")
            notch_freq  = st.number_input("Freq (Hz)", value=50.0, step=10.0, format="%.1f",
                                          disabled=not apply_notch, key="notch_freq")
            notch_q     = st.number_input("Q factor",  value=30.0, step=5.0,  format="%.1f",
                                          disabled=not apply_notch, key="notch_q")

        st.markdown("#### 📐 Chuẩn hoá & Cắt")
        col_n1, col_n2, col_n3 = st.columns(3)
        with col_n1:
            clip_uv      = st.number_input("Clip (µV)", value=800.0, step=50.0, key="clip_uv")
        with col_n2:
            scale_factor = st.number_input("Scale ÷",   value=100.0, step=10.0, key="scale_factor")
        with col_n3:
            w_edge       = st.number_input("Edge (min)", value=30, min_value=0, step=5, key="w_edge")

    with col_r:
        st.markdown("#### 📋 Log")
        log_placeholder = st.empty()

        def _render_pp_log():
            logs = st.session_state.preprocess_logs[-60:]
            html = "<br>".join(logs) if logs else "<span style='color:#555'>Chưa có log...</span>"
            log_placeholder.markdown(f'<div class="log-box">{html}</div>', unsafe_allow_html=True)

        _render_pp_log()
        prog_placeholder = st.empty()

        # Info box tuỳ dataset
        if pp_dataset == "shhs":
            st.info(
                "**SHHS**: đảm bảo thư mục data có cấu trúc:\n"
                "```\n"
                "data_dir/\n"
                "  edfs/shhs1-XXXXXX.edf\n"
                "  annotations-events-nsrr/shhs1-XXXXXX-nsrr.xml\n"
                "```"
            )
        else:
            st.info(
                "**Sleep-EDF**: đảm bảo thư mục data chứa:\n"
                "`*PSG.edf` và `*Hypnogram.edf` cùng cấp."
            )

        st.markdown("")
        run_pp_btn = st.button("▶ Bắt đầu Preprocess", key="run_pp",
                               type="primary",
                               disabled=st.session_state.running_preproc,
                               use_container_width=True)

    if run_pp_btn and not st.session_state.running_preproc:
        st.session_state.running_preproc = True
        st.session_state.preprocess_logs = []
        st.session_state.preproc_done    = False

        from pipeline.preprocess import PreprocessConfig, run_preprocess

        cfg = PreprocessConfig(
            dataset=pp_dataset,
            data_dir=pp_data_dir, output_dir=pp_output_dir,
            select_ch=pp_channel,
            max_subjects=int(pp_max_subjects) if pp_max_subjects > 0 else None,
            random_seed=int(pp_random_seed),
            apply_bandpass=apply_bp, bp_low=bp_low, bp_high=bp_high, bp_order=bp_order,
            apply_notch=apply_notch, notch_freq=notch_freq, notch_q=notch_q,
            clip_uv=clip_uv, scale_factor=scale_factor, w_edge_mins=w_edge,
        )

        def _pp_log(msg):
            st.session_state.preprocess_logs.append(msg)
            # Cập nhật log box ngay lập tức
            logs = st.session_state.preprocess_logs[-60:]
            html = "<br>".join(logs)
            log_placeholder.markdown(f'<div class="log-box">{html}</div>', unsafe_allow_html=True)

        def _pp_progress(done, total, fname):
            if total > 0:
                prog_placeholder.progress(done / total, text=fname)

        cfg.progress_cb = _pp_progress

        import logging
        logger = logging.getLogger("pp_stream")
        logger.propagate = False
        logger.handlers.clear()

        class _StreamHandler(logging.Handler):
            def emit(self, record):
                _pp_log(self.format(record))
        h = _StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)

        try:
            saved = run_preprocess(cfg, logger=logger)
            _pp_log(f"<b style='color:#00d4aa'>✓ Hoàn thành: {len(saved)} files</b>")
            st.session_state.preproc_done = True
        except Exception as e:
            _pp_log(f"<span style='color:#f85149'>✗ Lỗi: {e}</span>")
        finally:
            st.session_state.running_preproc = False
            prog_placeholder.empty()

        _render_pp_log()
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# TAB 2 — TRAIN
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Train ResNet-1D + SleepTCN")
    st.markdown("K-Fold subject-wise cross-validation. Lưu checkpoint mỗi fold.")

    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown("#### 📁 Paths")
        tr_npz_dir  = st.text_input("Thư mục NPZ", value="./preprocessed", key="tr_npz")
        tr_out_dir  = st.text_input("Thư mục output", value="./results",    key="tr_out")
        tr_ckpt_dir = st.text_input("Thư mục checkpoint", value="./results/checkpoints", key="tr_ckpt")

        st.markdown("#### 🔢 Fold")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            n_folds = st.number_input("N folds", value=10, min_value=2, max_value=20, key="n_folds")
        with col_f2:
            seed    = st.number_input("Seed", value=123, min_value=0, key="seed")
        with col_f3:
            folds_input = st.text_input("Chỉ chạy fold (vd: 1,3,5 hoặc để trống = tất cả)",
                                        value="", key="folds_input")

        st.markdown("#### 🏗️ ResNet-1D")
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            rn_lr      = st.number_input("LR",      value=1e-3, format="%.0e", key="rn_lr")
            rn_epochs  = st.number_input("Epochs",  value=40,   min_value=1,   key="rn_epochs")
        with col_r2:
            rn_batch   = st.number_input("Batch",   value=64,   min_value=4,   key="rn_batch")
            rn_patience= st.number_input("Patience",value=8,    min_value=1,   key="rn_patience")
        with col_r3:
            n_feat     = st.number_input("Feat dim",value=128,  min_value=32,  key="n_feat")
            rn_val     = st.slider("Val ratio", 0.05, 0.3, 0.15, 0.05, key="rn_val")

        st.markdown("#### 🌀 SleepTCN")
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            tcn_lr     = st.number_input("LR",        value=5e-4, format="%.0e", key="tcn_lr")
            tcn_epochs = st.number_input("Max epochs",value=300,  min_value=10,  key="tcn_epochs")
            tcn_patience=st.number_input("Patience",  value=30,   min_value=5,   key="tcn_patience")
        with col_t2:
            tcn_batch  = st.number_input("Batch",     value=8,    min_value=1,   key="tcn_batch")
            tcn_dim    = st.number_input("Dim",       value=128,  min_value=32,  key="tcn_dim")
            tcn_blocks = st.number_input("Blocks",    value=6,    min_value=1,   key="tcn_blocks")
        with col_t3:
            tcn_kernel = st.number_input("Kernel",    value=3,    min_value=1,   step=2, key="tcn_kernel")
            tcn_dropout= st.slider("Dropout", 0.0, 0.5, 0.2, 0.05, key="tcn_dropout")
            tcn_val    = st.slider("Val ratio",0.05, 0.3, 0.10, 0.05, key="tcn_val")

    with col_b:
        st.markdown("#### 📋 Log")
        train_log_ph = st.empty()
        train_prog_ph = st.empty()
        train_metric_ph = st.empty()

        def _render_train_log():
            logs = st.session_state.train_logs[-80:]
            html = "<br>".join(logs) if logs else "<span style='color:#555'>Chưa có log...</span>"
            train_log_ph.markdown(f'<div class="log-box">{html}</div>', unsafe_allow_html=True)

        _render_train_log()

        st.markdown("")
        run_train_btn = st.button("▶ Bắt đầu Train", key="run_train",
                                  type="primary",
                                  disabled=st.session_state.running_train,
                                  use_container_width=True)

    if run_train_btn and not st.session_state.running_train:
        st.session_state.running_train = True
        st.session_state.train_logs    = []
        st.session_state.fold_metrics  = []
        st.session_state.train_done    = False

        # Parse folds_to_run
        folds_to_run = None
        if folds_input.strip():
            try:
                folds_to_run = [int(f.strip()) - 1 for f in folds_input.split(",")]
            except ValueError:
                st.warning("Fold input không hợp lệ, chạy tất cả.")

        from pipeline.trainer import PipelineConfig, run_full_pipeline

        cfg = PipelineConfig(
            npz_dir=tr_npz_dir, output_dir=tr_out_dir, ckpt_dir=tr_ckpt_dir,
            n_folds=int(n_folds), seed=int(seed),
            n_feat=int(n_feat),
            resnet_lr=float(rn_lr), resnet_batch_size=int(rn_batch),
            resnet_epochs=int(rn_epochs), resnet_patience=int(rn_patience),
            resnet_val_ratio=float(rn_val),
            tcn_dim=int(tcn_dim), tcn_kernel_size=int(tcn_kernel),
            tcn_blocks=int(tcn_blocks), tcn_dropout=float(tcn_dropout),
            seq_lr=float(tcn_lr), seq_batch_size=int(tcn_batch),
            seq_max_epochs=int(tcn_epochs), seq_patience=int(tcn_patience),
            seq_val_ratio=float(tcn_val),
        )

        def _tlog(msg):
            st.session_state.train_logs.append(msg)
            # Cập nhật log box ngay lập tức
            logs = st.session_state.train_logs[-80:]
            html = "<br>".join(logs)
            train_log_ph.markdown(f'<div class="log-box">{html}</div>', unsafe_allow_html=True)

        def _tprog(done, total, label):
            if total > 0:
                train_prog_ph.progress(done / total, text=label)

        def _tmetrics(fold_i, m):
            st.session_state.fold_metrics.append(m)
            lines = [f"<b>Fold {m['fold']}</b>  "
                     f"ACC={m['acc']*100:.2f}%  MF1={m['mf1']*100:.2f}%  κ={m['kappa']:.4f}"]
            train_metric_ph.markdown(
                f'<div class="log-box">{"<br>".join(lines)}</div>', unsafe_allow_html=True
            )

        cfg.log_cb      = _tlog
        cfg.progress_cb = _tprog
        cfg.metrics_cb  = _tmetrics

        try:
            results = run_full_pipeline(
                cfg, folds_to_run=folds_to_run,
                train_resnet=True, train_tcn=True,
            )
            st.session_state.fold_metrics = results
            _tlog(f"<b style='color:#00d4aa'>✓ Train hoàn thành: {len(results)} folds</b>")
            st.session_state.train_done = True
        except Exception as e:
            import traceback
            _tlog(f"<span style='color:#f85149'>✗ Lỗi: {e}</span>")
            _tlog(traceback.format_exc().replace("\n", "<br>"))
        finally:
            st.session_state.running_train = False
            train_prog_ph.empty()

        _render_train_log()
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# TAB 3 — EVALUATE / FINE-TUNE
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Evaluate & Fine-tune")
    st.markdown("Load checkpoint đã train → đánh giá chéo hoặc fine-tune trên dataset mới.")

    col_e1, col_e2 = st.columns([1, 1], gap="large")

    with col_e1:
        # ── Mode selector ────────────────────────────────────────
        st.markdown("#### ⚙️ Chế độ")
        _mode_options = ["eval_all", "cross_eval", "fine_tune"]
        _mode_saved   = st.session_state.get("ev_mode", "eval_all")
        _mode_idx     = _mode_options.index(_mode_saved) if _mode_saved in _mode_options else 0
        ev_mode = st.selectbox(
            "Chế độ",
            _mode_options,
            index=_mode_idx,
            key="ev_mode",
            format_func=lambda x: {
                "eval_all":   "⚡ Eval toàn bộ  (không fold, 1 checkpoint)",
                "cross_eval": "🔍 Đánh giá chéo  (K-Fold, nhiều checkpoint)",
                "fine_tune":  "🔧 Fine-tune TCN  (load ResNet → freeze → train lại TCN)",
            }[x],
        )

        st.markdown("#### 📁 Data mới")
        ev_npz_dir = st.text_input("Thư mục NPZ mới", value="./new_data", key="ev_npz")
        ev_out_dir = st.text_input("Output dir",       value="./results_eval", key="ev_out")

        st.markdown("#### 🏗️ ResNet weights")
        ev_resnet_dir = st.text_input(
            "Thư mục chứa resnet_fold*.pth",
            value="./weight", key="ev_resnet_dir",
        )
        use_resnet_ckpt = st.checkbox(
            "Load ResNet từ checkpoint (bỏ tick = train mới)",
            value=True, key="use_resnet_ckpt",
        )

        st.markdown("#### 🌀 TCN weights")
        ev_tcn_dir = st.text_input(
            "Thư mục chứa tcn_fold*.pth",
            value="./weight", key="ev_tcn_dir",
        )
        use_tcn_ckpt = st.checkbox(
            "Load TCN từ checkpoint",
            value=(ev_mode == "cross_eval"), key="use_tcn_ckpt",
            help="Đánh giá chéo: cần tick  |  Fine-tune: tick để load rồi train tiếp",
        )

        # Fine-tune options (chỉ hiện khi mode = fine_tune)
        if ev_mode == "fine_tune":
            st.markdown("#### 🔧 Fine-tune options")
            col_ft1, col_ft2 = st.columns(2)
            with col_ft1:
                finetune_lr = st.number_input(
                    "Fine-tune LR", value=float(st.session_state.get("finetune_lr", 1e-4)),
                    format="%.0e", key="finetune_lr",
                    help="Learning rate nhỏ hơn seq_lr để fine-tune TCN",
                )
            with col_ft2:
                freeze_resnet = st.checkbox(
                    "Freeze ResNet", value=bool(st.session_state.get("freeze_resnet", True)),
                    key="freeze_resnet",
                    help="Giữ nguyên ResNet weights, chỉ cập nhật TCN",
                )

        # ── eval_all: chọn checkpoint trực tiếp ─────────────────
        if ev_mode == "eval_all":
            st.markdown("#### ⚡ Chọn checkpoint (1 file duy nhất)")
            st.info("Mode này **không dùng fold** — load 1 cặp checkpoint, chạy inference trên toàn bộ NPZ.")

            # Tự động liệt kê các checkpoint có sẵn
            def _list_ckpts(folder, pattern):
                try:
                    return sorted(Path(folder).glob(pattern)) if Path(folder).exists() else []
                except OSError:
                    return []

            resnet_ckpts = _list_ckpts(ev_resnet_dir, "resnet_fold*.pth")
            tcn_ckpts    = _list_ckpts(ev_tcn_dir,    "tcn_fold*.pth") +                            _list_ckpts(ev_tcn_dir,    "tcn_finetune_fold*.pth")

            resnet_names = [f.name for f in resnet_ckpts]
            tcn_names    = [f.name for f in tcn_ckpts]

            col_ea1, col_ea2 = st.columns(2)
            with col_ea1:
                resnet_pick = st.selectbox(
                    "ResNet checkpoint",
                    options=resnet_names if resnet_names else ["(không tìm thấy)"],
                    key="ea_resnet_pick",
                )
            with col_ea2:
                tcn_pick = st.selectbox(
                    "TCN checkpoint",
                    options=tcn_names if tcn_names else ["(không tìm thấy)"],
                    key="ea_tcn_pick",
                )

            ev_n_folds, ev_seed, ev_folds_input = 1, 123, ""  # unused placeholders

        else:
            st.markdown("#### 🔢 Fold")
            col_ef1, col_ef2 = st.columns(2)
            with col_ef1:
                ev_n_folds = st.number_input("N folds", value=10, min_value=2, key="ev_n_folds")
            with col_ef2:
                ev_seed    = st.number_input("Seed",    value=123, min_value=0, key="ev_seed")
            ev_folds_input = st.text_input("Chỉ chạy fold (để trống = tất cả)",
                                           value="", key="ev_folds_input")

        # Preview checkpoints
        st.markdown("**Checkpoints có sẵn:**")

        def _safe_glob(folder: str, pattern: str) -> list:
            try:
                p = Path(folder)
                return sorted(p.glob(pattern)) if p.exists() else []
            except OSError:
                return None

        ckpt_resnet_files = _safe_glob(ev_resnet_dir, "resnet_fold*.pth")
        ckpt_tcn_files    = _safe_glob(ev_tcn_dir,    "tcn_fold*.pth")

        if ckpt_resnet_files is None or ckpt_tcn_files is None:
            st.warning("⚠️ Không thể đọc thư mục — Google Drive chưa được mount. "
                       "Hãy chạy: `drive.mount('/content/drive')` trong Colab trước.")
        elif ckpt_resnet_files or ckpt_tcn_files:
            for f in ckpt_resnet_files:
                st.markdown(f'<span class="badge badge-green">ResNet {f.name}</span>', unsafe_allow_html=True)
            for f in ckpt_tcn_files:
                st.markdown(f'<span class="badge">TCN {f.name}</span>', unsafe_allow_html=True)
        else:
            st.caption("Không tìm thấy checkpoint trong thư mục trên.")

    with col_e2:
        st.markdown("#### 📋 Log")
        eval_log_ph  = st.empty()
        eval_prog_ph = st.empty()

        def _render_eval_log():
            logs = st.session_state.eval_logs[-80:]
            html = "<br>".join(logs) if logs else "<span style='color:#555'>Chưa có log...</span>"
            eval_log_ph.markdown(f'<div class="log-box">{html}</div>', unsafe_allow_html=True)

        _render_eval_log()

        st.markdown("")
        _btn_label = {
            "eval_all":   "⚡ Eval toàn bộ",
            "cross_eval": "▶ Bắt đầu Evaluate",
            "fine_tune":  "▶ Bắt đầu Fine-tune",
        }.get(ev_mode, "▶ Chạy")
        run_eval_btn = st.button(_btn_label, key="run_eval",
                                 type="primary",
                                 disabled=st.session_state.running_eval,
                                 use_container_width=True)

    if run_eval_btn and not st.session_state.running_eval:
        st.session_state.running_eval = True
        st.session_state.eval_logs    = []
        st.session_state.eval_done    = False

        # ── Callbacks cập nhật UI realtime (không chờ rerun) ──────
        def _elog(msg):
            st.session_state.eval_logs.append(msg)
            # Cập nhật log box ngay lập tức
            logs = st.session_state.eval_logs[-80:]
            html = "<br>".join(logs)
            eval_log_ph.markdown(f'<div class="log-box">{html}</div>', unsafe_allow_html=True)

        def _eprog(done, total, label):
            if total > 0:
                eval_prog_ph.progress(done / total, text=label)

        if ev_mode == "eval_all":
            # ── Mode ⚡: Eval toàn bộ, không fold ─────────────────
            from trainer import PipelineConfig, evaluate_all

            ea_resnet_path = str(Path(ev_resnet_dir) / st.session_state.get("ea_resnet_pick", ""))
            ea_tcn_path    = str(Path(ev_tcn_dir)    / st.session_state.get("ea_tcn_pick",    ""))

            ea_cfg = PipelineConfig(
                npz_dir=ev_npz_dir, output_dir=ev_out_dir,
                # Bug-6 fix: truyền đầy đủ architecture params để load checkpoint
                # đúng shape. Các giá trị này phải khớp với lúc train.
                n_feat          = int(st.session_state.get("n_feat",      128)),
                tcn_dim         = int(st.session_state.get("tcn_dim",     128)),
                tcn_kernel_size = int(st.session_state.get("tcn_kernel",  3)),
                tcn_blocks      = int(st.session_state.get("tcn_blocks",  6)),
                tcn_dropout     = float(st.session_state.get("tcn_dropout", 0.2)),
            )
            ea_cfg.log_cb      = _elog
            ea_cfg.progress_cb = _eprog

            try:
                result = evaluate_all(ea_cfg, resnet_ckpt=ea_resnet_path, tcn_ckpt=ea_tcn_path)
                st.session_state.fold_metrics = [result]
                _elog(f"<b style='color:#00d4aa'>✓ Eval xong: {result['n_records']} records, "
                      f"{result['n_epochs']} epochs</b>")
                st.session_state.eval_done = True
            except Exception as e:
                import traceback
                _elog(f"<span style='color:#f85149'>✗ Lỗi: {e}</span>")
                _elog(traceback.format_exc().replace("\n", "<br>"))
            finally:
                st.session_state.running_eval = False
                eval_prog_ph.empty()

        else:
            # ── Mode 🔍/🔧: K-Fold cross_eval / fine_tune ─────────
            ev_folds_to_run = None
            if ev_folds_input.strip():
                try:
                    ev_folds_to_run = [int(f.strip()) - 1 for f in ev_folds_input.split(",")]
                except ValueError:
                    st.warning("Fold input không hợp lệ, chạy tất cả.")

            from trainer import PipelineConfig, run_full_pipeline

            ev_cfg = PipelineConfig(
                npz_dir=ev_npz_dir, output_dir=ev_out_dir,
                ckpt_dir=str(Path(ev_out_dir) / "checkpoints"),
                n_folds=int(ev_n_folds), seed=int(ev_seed),
                finetune_lr=float(st.session_state.get("finetune_lr", 1e-4)),
                freeze_resnet=bool(st.session_state.get("freeze_resnet", True)),
            )

            def _emetrics(fold_i, m):
                st.session_state.fold_metrics.append(m)

            ev_cfg.log_cb      = _elog
            ev_cfg.progress_cb = _eprog
            ev_cfg.metrics_cb  = _emetrics

            is_fine_tune = (ev_mode == "fine_tune")

            try:
                results = run_full_pipeline(
                    ev_cfg,
                    folds_to_run=ev_folds_to_run,
                    resnet_ckpt_dir=ev_resnet_dir if use_resnet_ckpt else None,
                    tcn_ckpt_dir=ev_tcn_dir       if use_tcn_ckpt    else None,
                    train_resnet=not use_resnet_ckpt,
                    train_tcn=(not use_tcn_ckpt) or is_fine_tune,
                    fine_tune=is_fine_tune,
                )
                st.session_state.fold_metrics = results
                label = "Fine-tune" if is_fine_tune else "Evaluate"
                _elog(f"<b style='color:#00d4aa'>✓ {label} xong: {len(results)} folds</b>")
                st.session_state.eval_done = True
            except Exception as e:
                import traceback
                _elog(f"<span style='color:#f85149'>✗ Lỗi: {e}</span>")
                _elog(traceback.format_exc().replace("\n", "<br>"))
            finally:
                st.session_state.running_eval = False
                eval_prog_ph.empty()

        st.rerun()


# ══════════════════════════════════════════════════════════════════
# TAB 4 — RESULTS
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Kết quả cross-validation")

    # Nạp từ file nếu có (thử cả results và results_eval)
    metrics = st.session_state.fold_metrics
    if not metrics:
        for _candidate in [
            Path(st.session_state.get("tr_out", "./results")) / "fold_metrics.json",
            Path(st.session_state.get("ev_out", "./results_eval")) / "fold_metrics.json",
            Path("./results/fold_metrics.json"),
        ]:
            try:
                if _candidate.exists():
                    with open(_candidate) as f:
                        metrics = json.load(f)
                    st.session_state.fold_metrics = metrics
                    break
            except OSError:
                continue

    # Cũng cho phép load file tùy chỉnh
    uploaded = st.file_uploader("Hoặc upload fold_metrics.json", type="json",
                                key="results_upload")
    if uploaded:
        metrics = json.load(uploaded)
        st.session_state.fold_metrics = metrics

    if not metrics:
        st.info("Chưa có kết quả. Hãy chạy tab Train hoặc Evaluate trước.")

    if metrics:  # guard: chỉ hiển thị khi có data
     # ── Summary metrics ─────────────────────────────────────────────
     SLEEP_STAGES = ["W", "N1", "N2", "N3", "REM"]
     accs   = [m["acc"]   for m in metrics]
     mf1s   = [m["mf1"]   for m in metrics]
     kappas = [m["kappa"] for m in metrics]

     st.markdown("#### 📊 Tổng hợp")
     c1, c2, c3 = st.columns(3)
     with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Accuracy</div>
            <div class="value">{np.mean(accs)*100:.2f}%</div>
            <div class="label">± {np.std(accs)*100:.2f}%</div>
        </div>""", unsafe_allow_html=True)
     with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Macro F1</div>
            <div class="value">{np.mean(mf1s)*100:.2f}%</div>
            <div class="label">± {np.std(mf1s)*100:.2f}%</div>
        </div>""", unsafe_allow_html=True)
     with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Cohen κ</div>
            <div class="value">{np.mean(kappas):.4f}</div>
            <div class="label">± {np.std(kappas):.4f}</div>
        </div>""", unsafe_allow_html=True)

     st.markdown("")

     # ── Per-stage F1 ────────────────────────────────────────────────
     st.markdown("#### Per-stage F1")
     stage_cols = st.columns(5)
     for i, stage in enumerate(SLEEP_STAGES):
        key = f"f1_{stage}"
        vals = [m[key] for m in metrics if key in m]
        if vals:
            with stage_cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">{stage}</div>
                    <div class="value" style="font-size:1.2rem">{np.mean(vals)*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

     st.markdown("")

     # ── Per-fold table ───────────────────────────────────────────────
     st.markdown("#### Per-fold breakdown")
     import pandas as pd
     rows = []
     for m in metrics:
        row = {
            "Fold":     m["fold"],
            "Subjects": str(m.get("test_subjects", [])),
            "N epochs": m.get("n_test_epochs", "?"),
            "ACC (%)":  f"{m['acc']*100:.2f}",
            "MF1 (%)":  f"{m['mf1']*100:.2f}",
            "κ":        f"{m['kappa']:.4f}",
            "Time (s)": f"{m.get('elapsed_s', 0):.0f}",
        }
        for s in SLEEP_STAGES:
            k = f"f1_{s}"
            row[f"F1-{s} (%)"] = f"{m[k]*100:.1f}" if k in m else "?"
        rows.append(row)

     df = pd.DataFrame(rows)
     st.dataframe(df, use_container_width=True, hide_index=True)

     # ── Per-fold bar chart ───────────────────────────────────────────
     st.markdown("#### Per-fold chart")
     try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        folds  = [m["fold"] for m in metrics]
        fig, ax = plt.subplots(figsize=(max(8, len(folds)), 4))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        x = np.arange(len(folds))
        w = 0.26
        ax.bar(x - w, [a*100 for a in accs],  w, label="Accuracy", color="#4e79a7", alpha=0.9)
        ax.bar(x,     [a*100 for a in mf1s],  w, label="Macro F1", color="#00d4aa", alpha=0.9)
        ax.bar(x + w, [a*100 for a in kappas],w, label="κ × 100",  color="#f7931a", alpha=0.9)

        for vals, c in [(accs,"#4e79a7"), (mf1s,"#00d4aa"), (kappas,"#f7931a")]:
            ax.axhline(np.mean(vals)*100, color=c, lw=1.2, ls="--", alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"F{f}" for f in folds], color="#c9d1d9")
        ax.set_ylabel("Score", color="#c9d1d9")
        ax.tick_params(colors="#c9d1d9")
        ax.set_ylim(0, 100)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color("#21262d")
        ax.legend(fontsize=9, facecolor="#0d1117", labelcolor="#c9d1d9", edgecolor="#21262d")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
     except ImportError:
        st.warning("Cần cài matplotlib để hiển thị biểu đồ.")

     # ── Download JSON ────────────────────────────────────────────────
     st.download_button(
        "⬇ Download fold_metrics.json",
        data=json.dumps(metrics, indent=2, default=str),
        file_name="fold_metrics.json",
        mime="application/json",
     )


# ══════════════════════════════════════════════════════════════════
# TAB 5 — INFERENCE (End-to-End Pipeline)
# ══════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🔬 Inference Pipeline — Raw EDF → Sleep Staging")
    st.markdown("Upload file EDF thô → tự động tiền xử lý → trích xuất đặc trưng (ResNet) "
                "→ phân lớp giấc ngủ (TCN).")

    col_inf_l, col_inf_r = st.columns([1, 1], gap="large")

    with col_inf_l:
        # ── Upload EDF ────────────────────────────────────────────
        st.markdown("#### 📂 File EDF")
        uploaded_edf = st.file_uploader(
            "Chọn file EDF (PSG)", type=["edf"],
            key="inf_edf_upload",
            help="File EDF chứa tín hiệu EEG (ví dụ: *PSG.edf từ Sleep-EDF)",
        )

        # ── Channel selection ─────────────────────────────────────
        st.markdown("#### 📡 EEG Channel")
        _inf_ch_options = [
            "EEG Fpz-Cz", "EEG Pz-Oz", "EEG C4-A1", "EEG F4-A1",
            "EEG", "EEG(sec)", "EEG1", "EEG2",
        ]

        # Auto-detect channels from uploaded file
        if uploaded_edf is not None and "inf_detected_chs" not in st.session_state:
            import tempfile, os
            tmp_dir = os.path.join(os.path.dirname(__file__), "_tmp_inf")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, uploaded_edf.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_edf.getbuffer())
            try:
                from preprocess import list_edf_channels
                detected = list_edf_channels(tmp_path)
                if detected:
                    st.session_state["inf_detected_chs"] = detected
                    _inf_ch_options = detected
            except Exception:
                pass

        if "inf_detected_chs" in st.session_state:
            _inf_ch_options = st.session_state["inf_detected_chs"]

        inf_channel = st.selectbox(
            "Channel", _inf_ch_options, key="inf_channel",
            help="Chọn kênh EEG để xử lý",
        )

        # ── Filter config ─────────────────────────────────────────
        st.markdown("#### 🔬 Bộ lọc")
        col_fb, col_fn = st.columns(2)
        with col_fb:
            inf_bp = st.checkbox("Bandpass", value=True, key="inf_bp")
            inf_bp_lo = st.number_input("Low Hz", value=0.5, step=0.1, format="%.1f",
                                        disabled=not inf_bp, key="inf_bp_lo")
            inf_bp_hi = st.number_input("High Hz", value=30.0, step=1.0, format="%.1f",
                                        disabled=not inf_bp, key="inf_bp_hi")
        with col_fn:
            inf_notch = st.checkbox("Notch", value=True, key="inf_notch")
            inf_notch_f = st.number_input("Freq Hz", value=50.0, step=10.0, format="%.1f",
                                          disabled=not inf_notch, key="inf_notch_f")
            inf_notch_q = st.number_input("Q", value=30.0, step=5.0, format="%.1f",
                                          disabled=not inf_notch, key="inf_notch_q")

        # ── Model checkpoints ─────────────────────────────────────
        st.markdown("#### 🏗️ Checkpoints")
        with st.expander("Đường dẫn Checkpoints (Dành cho Dev)", expanded=False):
            inf_ckpt_dir = st.text_input("Thư mục checkpoints", value="./weight",
                                         key="inf_ckpt_dir")

        def _inf_list_ckpts(folder, pattern):
            try:
                return sorted(Path(folder).glob(pattern)) if Path(folder).exists() else []
            except OSError:
                return []

        inf_resnet_files = _inf_list_ckpts(inf_ckpt_dir, "resnet_fold*.pth")
        inf_tcn_files    = (_inf_list_ckpts(inf_ckpt_dir, "tcn_fold*.pth") +
                           _inf_list_ckpts(inf_ckpt_dir, "tcn_finetune_fold*.pth"))

        col_ck1, col_ck2 = st.columns(2)
        with col_ck1:
            rn_names = [f.name for f in inf_resnet_files]
            inf_rn_pick = st.selectbox(
                "ResNet", options=rn_names if rn_names else ["(không tìm thấy)"],
                key="inf_rn_pick",
            )
        with col_ck2:
            tcn_names = [f.name for f in inf_tcn_files]
            inf_tcn_pick = st.selectbox(
                "TCN", options=tcn_names if tcn_names else ["(không tìm thấy)"],
                key="inf_tcn_pick",
            )

        # ── Architecture params ───────────────────────────────────
        with st.expander("⚙️ Kiến trúc mô hình (phải khớp checkpoint)", expanded=False):
            col_ar1, col_ar2 = st.columns(2)
            with col_ar1:
                inf_nfeat = st.number_input("Feature dim", value=128, min_value=32,
                                            key="inf_nfeat")
                inf_tcn_dim = st.number_input("TCN dim", value=128, min_value=32,
                                              key="inf_tcn_dim")
                inf_tcn_blocks = st.number_input("TCN blocks", value=6, min_value=1,
                                                 key="inf_tcn_blocks")
            with col_ar2:
                inf_tcn_kernel = st.number_input("TCN kernel", value=3, min_value=1, step=2,
                                                 key="inf_tcn_kernel")
                inf_tcn_drop = st.slider("TCN dropout", 0.0, 0.5, 0.2, 0.05,
                                         key="inf_tcn_drop")

    with col_inf_r:
        st.markdown("#### 📋 Log")
        inf_log_ph = st.empty()

        def _render_inf_log():
            logs = st.session_state.infer_logs[-60:]
            html = "<br>".join(logs) if logs else "<span style='color:#555'>Chưa có log...</span>"
            inf_log_ph.markdown(f'<div class="log-box">{html}</div>', unsafe_allow_html=True)

        _render_inf_log()
        inf_prog_ph = st.empty()

        st.markdown("")
        run_inf_btn = st.button("▶ Chạy Inference", key="run_inf",
                                type="primary",
                                disabled=(st.session_state.running_infer or uploaded_edf is None),
                                use_container_width=True)

        if uploaded_edf is None:
            st.info("⬆ Hãy upload file EDF bên trái để bắt đầu.")

    # ── Run inference ─────────────────────────────────────────────
    if run_inf_btn and uploaded_edf is not None and not st.session_state.running_infer:
        st.session_state.running_infer = True
        st.session_state.infer_logs    = []
        st.session_state.infer_done    = False
        st.session_state.infer_result  = None

        # Reset detected channels for next upload
        if "inf_detected_chs" in st.session_state:
            del st.session_state["inf_detected_chs"]

        import os, tempfile as _tf

        def _ilog(msg):
            st.session_state.infer_logs.append(msg)
            logs = st.session_state.infer_logs[-60:]
            html = "<br>".join(logs)
            inf_log_ph.markdown(f'<div class="log-box">{html}</div>', unsafe_allow_html=True)

        try:
            # Save uploaded file to temp
            tmp_dir = os.path.join(os.path.dirname(__file__), "_tmp_inf")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_edf = os.path.join(tmp_dir, uploaded_edf.name)
            with open(tmp_edf, "wb") as f:
                f.write(uploaded_edf.getbuffer())

            from inference import InferenceConfig, preprocess_raw_edf, run_inference

            inf_cfg = InferenceConfig(
                select_ch=inf_channel,
                apply_bandpass=inf_bp, bp_low=inf_bp_lo, bp_high=inf_bp_hi, bp_order=4,
                apply_notch=inf_notch, notch_freq=inf_notch_f, notch_q=inf_notch_q,
                n_feat=int(inf_nfeat), tcn_dim=int(inf_tcn_dim),
                tcn_kernel_size=int(inf_tcn_kernel), tcn_blocks=int(inf_tcn_blocks),
                tcn_dropout=float(inf_tcn_drop),
                log_cb=_ilog,
            )

            inf_prog_ph.progress(0.1, text="Tiền xử lý ...")
            signals, fs_orig, n_all_epochs, raw_epochs = preprocess_raw_edf(tmp_edf, inf_cfg)

            inf_prog_ph.progress(0.4, text="Trích xuất đặc trưng + Phân lớp ...")
            resnet_path = str(Path(inf_ckpt_dir) / inf_rn_pick)
            tcn_path    = str(Path(inf_ckpt_dir) / inf_tcn_pick)
            result = run_inference(signals, resnet_path, tcn_path, inf_cfg)

            # Store results
            result["raw_epochs"]   = raw_epochs
            result["signals"]      = signals
            result["fs_orig"]      = fs_orig
            result["n_all_epochs"] = n_all_epochs
            result["edf_name"]     = uploaded_edf.name
            st.session_state.infer_result = result
            st.session_state.infer_done = True

            _ilog(f"<b style='color:#00d4aa'>✓ Inference hoàn thành: {len(result['predictions'])} epochs</b>")
            inf_prog_ph.progress(1.0, text="Hoàn thành!")

        except Exception as e:
            import traceback
            _ilog(f"<span style='color:#f85149'>✗ Lỗi: {e}</span>")
            _ilog(traceback.format_exc().replace("\n", "<br>"))
        finally:
            st.session_state.running_infer = False

        _render_inf_log()
        st.rerun()

    # ── Display results ───────────────────────────────────────────
    result = st.session_state.infer_result
    if result is not None:
        st.markdown("---")
        st.markdown(f"### 📊 Kết quả — `{result.get('edf_name', 'N/A')}`")

        STAGE_NAMES  = ["W", "N1", "N2", "N3", "REM"]
        STAGE_COLORS = ["#e74c3c", "#f39c12", "#3498db", "#2c3e50", "#2ecc71"]

        preds = result["predictions"]
        probs = result["probabilities"]
        dist  = result["stage_dist"]

        # ── Summary cards ─────────────────────────────────────────
        st.markdown("#### 📈 Tổng quan")
        n_epochs = len(preds)
        total_hours = n_epochs * 30 / 3600

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        cards = [
            ("TỔNG EPOCHS", str(n_epochs)),
            ("THỜI GIAN", f"{total_hours:.1f}h"),
            ("WAKE", f"{dist.get('W', 0)} ep"),
            ("NREM", f"{dist.get('N1',0)+dist.get('N2',0)+dist.get('N3',0)} ep"),
            ("REM", f"{dist.get('REM', 0)} ep"),
        ]
        for col, (lbl, val) in zip([mc1, mc2, mc3, mc4, mc5], cards):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">{lbl}</div>
                    <div class="value" style="font-size:1.2rem">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
            HAS_MPL = True
        except ImportError:
            HAS_MPL = False

        if HAS_MPL:
            # ── Raw signal visualization ──────────────────────────
            st.markdown("#### 🌊 Tín hiệu EEG thô")
            raw_epochs = result.get("raw_epochs")
            if raw_epochs is not None:
                n_ep = len(raw_epochs)
                # Epoch range selector
                col_rs1, col_rs2 = st.columns(2)
                with col_rs1:
                    raw_start = st.number_input("Epoch bắt đầu", 1, n_ep, 1, key="raw_ep_s")
                with col_rs2:
                    raw_end = st.number_input("Epoch kết thúc", 1, n_ep,
                                             min(raw_start + 19, n_ep), key="raw_ep_e")

                raw_start = max(1, raw_start) - 1
                raw_end   = min(n_ep, max(raw_end, raw_start + 1))
                n_show    = raw_end - raw_start

                fig_raw, axes_raw = plt.subplots(min(n_show, 20), 1,
                    figsize=(14, min(n_show, 20) * 1.2), squeeze=False)
                fig_raw.patch.set_facecolor("#0d1117")

                for idx in range(min(n_show, 20)):
                    ep_i = raw_start + idx
                    ax = axes_raw[idx, 0]
                    ax.set_facecolor("#0d1117")
                    t = np.arange(len(raw_epochs[ep_i])) / 100.0  # 100 Hz
                    ax.plot(t, raw_epochs[ep_i], color=STAGE_COLORS[preds[ep_i]],
                            lw=0.5, alpha=0.9)
                    stage = STAGE_NAMES[preds[ep_i]]
                    ax.set_ylabel(f"Ep {ep_i+1}\n{stage}", fontsize=7, color="#c9d1d9",
                                 rotation=0, labelpad=35)
                    ax.tick_params(colors="#555", labelsize=6)
                    for sp in ax.spines.values():
                        sp.set_color("#21262d")
                    if idx < min(n_show, 20) - 1:
                        ax.set_xticks([])

                axes_raw[-1, 0].set_xlabel("Time (s)", color="#c9d1d9", fontsize=8)
                fig_raw.suptitle(f"Raw EEG — Epochs {raw_start+1}–{raw_end}",
                                color="#c9d1d9", fontsize=11, y=1.01)
                fig_raw.tight_layout()
                st.pyplot(fig_raw, use_container_width=True)
                plt.close(fig_raw)

            # ── Hypnogram ─────────────────────────────────────────
            st.markdown("#### 🛏️ Hypnogram (dự đoán)")
            fig_hyp, ax_hyp = plt.subplots(figsize=(14, 3))
            fig_hyp.patch.set_facecolor("#0d1117")
            ax_hyp.set_facecolor("#0d1117")

            # Plot as step function (inverted: W on top)
            stage_map = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}  # W=top, REM=bottom
            mapped = np.array([stage_map[p] for p in preds])
            t_epochs = np.arange(len(preds)) * 30 / 3600  # hours

            for i in range(len(preds) - 1):
                ax_hyp.fill_between([t_epochs[i], t_epochs[i+1]],
                                    mapped[i], mapped[i] + 0.9,
                                    color=STAGE_COLORS[preds[i]], alpha=0.7)

            ax_hyp.step(t_epochs, mapped, where="post", color="#c9d1d9", lw=0.8)
            ax_hyp.set_yticks([0, 1, 2, 3, 4])
            ax_hyp.set_yticklabels(["REM", "N3", "N2", "N1", "W"], color="#c9d1d9", fontsize=9)
            ax_hyp.set_xlabel("Thời gian (giờ)", color="#c9d1d9", fontsize=9)
            ax_hyp.set_xlim(0, t_epochs[-1])
            ax_hyp.set_ylim(-0.3, 5)
            ax_hyp.tick_params(colors="#c9d1d9")
            for sp in ax_hyp.spines.values():
                sp.set_color("#21262d")
            fig_hyp.tight_layout()
            st.pyplot(fig_hyp, use_container_width=True)
            plt.close(fig_hyp)

            # ── Stage distribution ────────────────────────────────
            st.markdown("#### 📊 Phân bố giai đoạn ngủ")
            col_pie, col_bar = st.columns(2)

            with col_pie:
                fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
                fig_pie.patch.set_facecolor("#0d1117")
                sizes  = [dist.get(s, 0) for s in STAGE_NAMES]
                colors = STAGE_COLORS
                wedges, texts, autotexts = ax_pie.pie(
                    sizes, labels=STAGE_NAMES, colors=colors, autopct="%1.1f%%",
                    startangle=90, textprops={"color": "#c9d1d9", "fontsize": 10},
                )
                for at in autotexts:
                    at.set_fontsize(9)
                    at.set_color("#ffffff")
                ax_pie.set_title("Tỉ lệ %", color="#c9d1d9", fontsize=11)
                fig_pie.tight_layout()
                st.pyplot(fig_pie, use_container_width=True)
                plt.close(fig_pie)

            with col_bar:
                fig_bar, ax_bar = plt.subplots(figsize=(5, 5))
                fig_bar.patch.set_facecolor("#0d1117")
                ax_bar.set_facecolor("#0d1117")
                bars = ax_bar.bar(STAGE_NAMES,
                                  [dist.get(s, 0) for s in STAGE_NAMES],
                                  color=STAGE_COLORS, alpha=0.85)
                for bar, s in zip(bars, STAGE_NAMES):
                    h = bar.get_height()
                    ax_bar.text(bar.get_x() + bar.get_width()/2., h + 1,
                               f"{int(h)}", ha="center", va="bottom",
                               color="#c9d1d9", fontsize=9)
                ax_bar.set_ylabel("Số epochs", color="#c9d1d9")
                ax_bar.set_title("Số lượng epoch", color="#c9d1d9", fontsize=11)
                ax_bar.tick_params(colors="#c9d1d9")
                for sp in ax_bar.spines.values():
                    sp.set_color("#21262d")
                fig_bar.tight_layout()
                st.pyplot(fig_bar, use_container_width=True)
                plt.close(fig_bar)

            # ── Confidence plot ───────────────────────────────────
            st.markdown("#### 🎯 Độ tin cậy theo epoch")
            max_conf = np.max(probs, axis=1)
            fig_conf, ax_conf = plt.subplots(figsize=(14, 2.5))
            fig_conf.patch.set_facecolor("#0d1117")
            ax_conf.set_facecolor("#0d1117")

            scatter_colors = [STAGE_COLORS[p] for p in preds]
            ax_conf.scatter(t_epochs, max_conf * 100, c=scatter_colors,
                           s=3, alpha=0.6, edgecolors="none")
            ax_conf.axhline(y=50, color="#f85149", lw=0.8, ls="--", alpha=0.5)
            ax_conf.set_ylabel("Confidence %", color="#c9d1d9", fontsize=9)
            ax_conf.set_xlabel("Thời gian (giờ)", color="#c9d1d9", fontsize=9)
            ax_conf.set_xlim(0, t_epochs[-1])
            ax_conf.set_ylim(0, 105)
            ax_conf.tick_params(colors="#c9d1d9")
            for sp in ax_conf.spines.values():
                sp.set_color("#21262d")
            fig_conf.tight_layout()
            st.pyplot(fig_conf, use_container_width=True)
            plt.close(fig_conf)

            # Mean confidence
            st.markdown(f"""
            <div class="metric-card" style="max-width:300px">
                <div class="label">ĐỘ TIN CẬY TRUNG BÌNH</div>
                <div class="value">{np.mean(max_conf)*100:.1f}%</div>
                <div class="label">Min: {np.min(max_conf)*100:.1f}% — Max: {np.max(max_conf)*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        else:
            st.warning("Cần cài matplotlib để hiển thị biểu đồ.")

        # ── Epoch-level detail table ──────────────────────────────
        st.markdown("#### 📋 Chi tiết epoch")
        import pandas as pd
        n_show_tbl = min(len(preds), 500)
        rows_tbl = []
        for i in range(n_show_tbl):
            rows_tbl.append({
                "Epoch": i + 1,
                "Stage": STAGE_NAMES[preds[i]],
                "Confidence": f"{np.max(probs[i])*100:.1f}%",
                **{f"P({s})": f"{probs[i][j]*100:.1f}%" for j, s in enumerate(STAGE_NAMES)},
            })
        df_inf = pd.DataFrame(rows_tbl)
        st.dataframe(df_inf, use_container_width=True, hide_index=True, height=400)

        if len(preds) > 500:
            st.caption(f"Hiển thị 500/{len(preds)} epochs đầu tiên.")

        # ── Download predictions ──────────────────────────────────
        st.markdown("#### ⬇ Xuất kết quả")
        pred_data = {
            "edf_file": result.get("edf_name", ""),
            "n_epochs": int(len(preds)),
            "total_hours": float(len(preds) * 30 / 3600),
            "stage_distribution": dist,
            "mean_confidence": float(np.mean(np.max(probs, axis=1))),
            "predictions": [
                {"epoch": i+1, "stage": STAGE_NAMES[preds[i]],
                 "confidence": float(np.max(probs[i]))}
                for i in range(len(preds))
            ],
        }
        st.download_button(
            "⬇ Download predictions JSON",
            data=json.dumps(pred_data, indent=2, ensure_ascii=False),
            file_name=f"inference_{result.get('edf_name','result')}.json",
            mime="application/json",
            use_container_width=True,
        )