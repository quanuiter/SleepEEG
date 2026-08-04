"""
pipeline/postprocess.py
───────────────────────
Hypnogram post-processing cho pipeline SleepEEG.

Cung cấp 3 kỹ thuật smoothing (đều pure NumPy, không phụ thuộc torch):

  1. mode_filter_stages  : majority/mode filter trong cửa sổ w (tie-break
                           bằng tổng probability của mỗi lớp trong cửa sổ).
                           Tốt hơn median() trên integer labels vì stage
                           ID 0..4 không ordinal (W↔REM không có trung gian).

  2. remove_short_segments: AASM minimum-duration rule — quét run-length
                           encoding, segment < min_epochs bị merge hoặc gán
                           lại theo lớp có tổng probability lớn hơn.

  3. viterbi_smooth      : 5-state HMM với transition matrix đã ước lượng
                           từ train subjects. Emission = log(prob + eps).
                           KHÔNG ước lượng từ test để tránh leakage.

Mặc định `enabled=False` để giữ backward-compatible với pipeline cũ.

Lưu ý: Post-processing chỉ áp lên predictions có sẵn, không thay probabilities.
Output: predictions mới cùng shape với input (n_epochs,).

Usage:
    cfg = PostprocessConfig(enabled=True, methods=("min_duration",), min_duration_epochs=3)
    smoothed = postprocess_hypnogram(preds, probs, cfg)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


SLEEP_STAGES = ("W", "N1", "N2", "N3", "REM")
N_CLASSES    = 5


@dataclass
class PostprocessConfig:
    """Cấu hình post-processing. Mặc định disabled để backward-compatible."""

    enabled: bool = False

    # Thứ tự áp dụng trong postprocess_hypnogram()
    # Các giá trị hợp lệ: "mode", "min_duration", "hmm"
    methods: Tuple[str, ...] = ("min_duration",)

    # mode filter
    mode_window: int = 5

    # AASM minimum-duration rule
    min_duration_epochs: int = 3

    # HMM
    use_hmm: bool = False
    transition_matrix: Optional[np.ndarray] = field(default=None)
    initial_probabilities: Optional[np.ndarray] = field(default=None)
    hmm_emission_floor: float = 1e-8


# ══════════════════════════════════════════════════════════════════
# 1. Mode filter (majority vote với tie-break bằng probability sum)
# ══════════════════════════════════════════════════════════════════


def mode_filter_stages(
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    window_size: int = 5,
) -> np.ndarray:
    """
    Mode filter trên chuỗi stage ID. Với cửa sổ lẻ, kết quả tại i là mode
    của predictions[i - w//2 : i + w//2 + 1].

    Tie-break: nếu nhiều lớp có cùng số vote, dùng tổng probability của
    lớp đó trong cửa sổ (cần truyền `probabilities`). Không có tie-break
    nếu `probabilities` is None — chọn lớp nhỏ nhất (numpy argmax default).

    Biên: dùng reflect padding để không làm "ảo" Wake ở đầu/cuối.
    """
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(f"window_size phải là số lẻ ≥1, nhận {window_size}")
    if predictions.ndim != 1:
        predictions = predictions.ravel()

    n = len(predictions)
    if n == 0:
        return predictions.copy()

    half = window_size // 2
    # Reflect pad: phản chiếu các giá trị ở biên
    padded = np.pad(predictions, (half, half), mode="edge")
    out = np.empty(n, dtype=predictions.dtype)

    if probabilities is not None:
        if probabilities.shape != (n, N_CLASSES):
            raise ValueError(
                f"probabilities shape {probabilities.shape} != ({n}, {N_CLASSES})"
            )
        prob_padded = np.pad(probabilities, ((half, half), (0, 0)), mode="edge")
    else:
        prob_padded = None

    for i in range(n):
        window = padded[i : i + window_size]
        # Count votes cho mỗi class
        counts = np.bincount(window, minlength=N_CLASSES)
        max_count = counts.max()
        winners = np.where(counts == max_count)[0]
        if len(winners) == 1:
            out[i] = winners[0]
        elif prob_padded is not None:
            # Tie-break: dùng tổng probability của các lớp đang dẫn đầu
            win_probs = prob_padded[i : i + window_size, winners].sum(axis=0)
            out[i] = winners[win_probs.argmax()]
        else:
            # Không có tie-break info → chọn lớp nhỏ nhất (deterministic)
            out[i] = winners.min()
    return out


# ══════════════════════════════════════════════════════════════════
# 2. AASM minimum-duration rule
# ══════════════════════════════════════════════════════════════════


def _run_length_encode(arr: np.ndarray) -> List[Tuple[int, int, int]]:
    """Trả về list (stage, start, end_exclusive)."""
    if len(arr) == 0:
        return []
    out = []
    start = 0
    cur = arr[0]
    for i in range(1, len(arr)):
        if arr[i] != cur:
            out.append((int(cur), start, i))
            start = i
            cur = arr[i]
    out.append((int(cur), start, len(arr)))
    return out


def remove_short_segments(
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    min_epochs: int = 3,
) -> np.ndarray:
    """
    AASM rule: bất kỳ stage segment nào < min_epochs đều bị sửa.

    Quy tắc gán lại cho segment ngắn:
      - Nếu 2 bên cùng lớp → merge vào lớp đó.
      - Nếu 2 bên khác lớp:
          * Có probabilities: chọn lớp có tổng probability lớn hơn trên
            đoạn ngắn.
          * Không có probabilities: chọn lớp có segment dài hơn bên cạnh.
      - Nếu ở đầu/cuối (chỉ 1 bên): gán bằng lớp bên cạnh duy nhất.
    Lặp đến khi không còn segment vi phạm.
    """
    if min_epochs < 1:
        raise ValueError(f"min_epochs phải ≥1, nhận {min_epochs}")
    if predictions.ndim != 1:
        predictions = predictions.ravel()
    out = predictions.copy()
    if len(out) < min_epochs:
        return out

    n_iter = 0
    while n_iter < 100:  # safety cap
        n_iter += 1
        rle = _run_length_encode(out)
        # Tìm segment ngắn đầu tiên
        changed = False
        new_rle = list(rle)
        for idx, (stage, s, e) in enumerate(rle):
            if e - s >= min_epochs:
                continue
            # Xác định lớp lân cận
            left_stage  = new_rle[idx - 1][0] if idx > 0 else None
            right_stage = new_rle[idx + 1][0] if idx + 1 < len(new_rle) else None

            if left_stage is not None and right_stage is not None \
                    and left_stage == right_stage:
                # Hai bên giống nhau → merge vào lớp đó
                new_stage = left_stage
            elif probabilities is not None:
                # Tính tổng probability của left và right trên đoạn ngắn
                seg_probs = probabilities[s:e].sum(axis=0)
                if left_stage is None:
                    new_stage = right_stage
                elif right_stage is None:
                    new_stage = left_stage
                else:
                    # Chọn lớp có tổng probability lớn hơn
                    if seg_probs[left_stage] >= seg_probs[right_stage]:
                        new_stage = left_stage
                    else:
                        new_stage = right_stage
            else:
                # Không có probabilities: ưu tiên bên nào segment dài hơn
                if left_stage is None:
                    new_stage = right_stage
                elif right_stage is None:
                    new_stage = left_stage
                else:
                    left_len  = new_rle[idx - 1][2] - new_rle[idx - 1][1]
                    right_len = new_rle[idx + 1][2] - new_rle[idx + 1][1]
                    new_stage = left_stage if left_len >= right_len else right_stage

            # Áp dụng
            for i in range(s, e):
                out[i] = new_stage
            changed = True
            break  # quét lại từ đầu để xử lý hiệu ứng cascade

        if not changed:
            break
    return out


# ══════════════════════════════════════════════════════════════════
# 3. HMM/Viterbi smoothing
# ══════════════════════════════════════════════════════════════════


def viterbi_smooth(
    probabilities: np.ndarray,
    transition_matrix: np.ndarray,
    initial_probabilities: Optional[np.ndarray] = None,
    emission_floor: float = 1e-8,
) -> np.ndarray:
    """
    5-state HMM Viterbi decoder.

    probabilities        : (T, 5) softmax output của model, mỗi hàng ≥0, sum=1
    transition_matrix    : (5, 5) — log hoặc raw. Nếu raw (rows sum=1), sẽ
                           log hóa internally. KHÔNG ước lượng từ test.
    initial_probabilities: (5,) — phân phối khởi đầu. Nếu None, dùng uniform.
    emission_floor       : cộng vào probability trước khi log để tránh log(0).

    Returns: predictions (T,) với giá trị trong [0, 4].
    """
    if probabilities.ndim != 2 or probabilities.shape[1] != N_CLASSES:
        raise ValueError(
            f"probabilities shape {probabilities.shape} phải là (T, {N_CLASSES})"
        )
    if transition_matrix.shape != (N_CLASSES, N_CLASSES):
        raise ValueError(
            f"transition_matrix shape {transition_matrix.shape} phải là "
            f"({N_CLASSES}, {N_CLASSES})"
        )

    T = probabilities.shape[0]
    if T == 0:
        return np.array([], dtype=np.int64)

    # Log hoá emission và transition
    emission = np.log(np.maximum(probabilities, emission_floor))   # (T, 5)

    trans = transition_matrix
    if np.all(trans >= 0) and np.allclose(trans.sum(axis=1), 1.0, atol=1e-3):
        # Raw transition matrix → log
        trans_log = np.log(np.maximum(trans, emission_floor))      # (5, 5)
    else:
        # Đã là log transition
        trans_log = trans

    if initial_probabilities is None:
        init_log = np.zeros(N_CLASSES)
    else:
        if initial_probabilities.shape != (N_CLASSES,):
            raise ValueError(
                f"initial_probabilities shape phải là ({N_CLASSES},)"
            )
        init_log = np.log(np.maximum(initial_probabilities, emission_floor))

    # Viterbi
    V = np.empty((T, N_CLASSES), dtype=np.float64)
    bp = np.empty((T, N_CLASSES), dtype=np.int64)

    V[0] = init_log + emission[0]
    for t in range(1, T):
        # scores[j] = max over i of V[t-1, i] + trans_log[i, j]
        scores = V[t - 1][:, None] + trans_log                    # (5, 5)
        bp[t] = scores.argmax(axis=0)
        V[t] = scores[bp[t], np.arange(N_CLASSES)] + emission[t]

    # Backtrack
    states = np.empty(T, dtype=np.int64)
    states[T - 1] = int(V[T - 1].argmax())
    for t in range(T - 2, -1, -1):
        states[t] = bp[t + 1, states[t + 1]]
    return states


# ══════════════════════════════════════════════════════════════════
# Top-level entry point
# ══════════════════════════════════════════════════════════════════


def postprocess_hypnogram(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    cfg: PostprocessConfig,
) -> np.ndarray:
    """
    Áp dụng chuỗi phương pháp theo `cfg.methods`.

    Args:
        predictions:  (T,) argmax output từ model
        probabilities: (T, 5) softmax probabilities
        cfg: PostprocessConfig

    Returns:
        (T,) predictions mới. Raw `predictions` không bị thay đổi.
    """
    if not cfg.enabled:
        return predictions.copy()
    if predictions.ndim != 1:
        predictions = predictions.ravel()

    out = predictions.astype(np.int64, copy=True)

    for method in cfg.methods:
        m = method.lower()
        if m == "mode":
            out = mode_filter_stages(out, probabilities, window_size=cfg.mode_window)
        elif m == "min_duration":
            out = remove_short_segments(out, probabilities, min_epochs=cfg.min_duration_epochs)
        elif m == "hmm":
            if cfg.transition_matrix is None:
                raise ValueError(
                    "HMM post-process yêu cầu cfg.transition_matrix != None"
                )
            out = viterbi_smooth(
                probabilities,
                cfg.transition_matrix,
                cfg.initial_probabilities,
                cfg.hmm_emission_floor,
            )
        else:
            raise ValueError(f"Unknown post-process method: {method}")

    # Áp thêm HMM nếu flag riêng (overrides methods nếu khác)
    if cfg.use_hmm and "hmm" not in cfg.methods and cfg.transition_matrix is not None:
        out = viterbi_smooth(
            probabilities,
            cfg.transition_matrix,
            cfg.initial_probabilities,
            cfg.hmm_emission_floor,
        )

    return out


def estimate_transition_matrix(
    sequences: Sequence[np.ndarray],
    smoothing: float = 1.0,
) -> np.ndarray:
    """
    Ước lượng transition matrix từ danh sách các chuỗi ground-truth labels.

    sequences: iterable of 1D int arrays (giá trị trong [0, N_CLASSES-1]).
    smoothing: additive Laplace smoothing (đơn vị giả).

    Trả về: (N_CLASSES, N_CLASSES) raw transition matrix, mỗi hàng sum=1.

    Cảnh báo leakage: CHỉ dùng train subjects, không bao giờ test.
    """
    counts = np.full((N_CLASSES, N_CLASSES), smoothing, dtype=np.float64)
    for seq in sequences:
        if len(seq) < 2:
            continue
        for a, b in zip(seq[:-1], seq[1:]):
            if 0 <= a < N_CLASSES and 0 <= b < N_CLASSES:
                counts[a, b] += 1
    counts /= counts.sum(axis=1, keepdims=True)
    return counts


__all__ = [
    "PostprocessConfig",
    "SLEEP_STAGES",
    "N_CLASSES",
    "mode_filter_stages",
    "remove_short_segments",
    "viterbi_smooth",
    "postprocess_hypnogram",
    "estimate_transition_matrix",
]