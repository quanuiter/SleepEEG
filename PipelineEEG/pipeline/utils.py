"""
pipeline/utils.py
─────────────────
Tiện ích reproducibility cho SleepEEG pipeline:
  - set_seed(seed, deterministic): seed mọi RNG + CUDA deterministic flags
  - derive_seed(base, fold, stage): sinh seed con deterministic theo fold/stage
  - seed_worker(worker_id): init_fn cho DataLoader workers
  - make_torch_generator(seed): tạo torch.Generator reproducible
  - hash_file(path): SHA-256 của file trên disk
  - config_hash(cfg_dict): hash deterministic của config (để đặt tên run)

Mục đích: khi hai run cùng seed + cùng environment → cùng split, cùng
sampling order, lý tưởng cùng checkpoint SHA-256.
"""

import hashlib
import json
import os
import random
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import torch


# ══════════════════════════════════════════════════════════════════
# Seed
# ══════════════════════════════════════════════════════════════════


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Seed tất cả RNG phổ biến + set CUDA deterministic flags.

    Args:
        seed: integer seed
        deterministic: nếu True, ép cuDNN/CUDA hoạt động deterministic
                       (chậm hơn nhưng reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Một số CUDA op (atomicAdd trên float) chưa có deterministic kernel.
        # warn_only=True để pipeline không chết, chỉ warning.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # PyTorch < 1.12 chưa có warn_only
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def derive_seed(base_seed: int, fold: int, stage: str) -> int:
    """
    Sinh seed con deterministic từ (base_seed, fold, stage).

    Công thức: SHA-256 của chuỗi "{base_seed}|{fold}|{stage}" → 4 byte đầu
    dưới dạng unsigned int.

    Lý do: tránh collision khi nhiều fold/stage dùng seed cố định (42 + fold
    hoặc 99 + fold) và có thể trùng nhau giữa các run khác nhau.
    """
    payload = f"{int(base_seed)}|{int(fold)}|{stage}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def seed_worker(worker_id: int) -> None:
    """
    worker_init_fn cho DataLoader (khi num_workers > 0).
    Mỗi worker được seed riêng, deterministic từ base_seed + worker_id.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed: int) -> torch.Generator:
    """Tạo torch.Generator với seed thủ công (dùng cho DataLoader)."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


# ══════════════════════════════════════════════════════════════════
# Hash
# ══════════════════════════════════════════════════════════════════


def hash_file(path: str, algo: str = "sha256", chunk: int = 1 << 16) -> str:
    """
    SHA-256 (hoặc thuật toán khác) của nội dung file trên disk.

    Hash file bytes (sau khi đã ghi xong) — không hash Python object trước
    serialization vì không phản ánh file thực tế trên disk.
    """
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def config_hash(cfg: Dict[str, Any], length: int = 8) -> str:
    """
    Hash deterministic của config dict (JSON canonical).

    Dùng để đặt tên run_id và phát hiện config drift giữa các run.
    """
    canonical = json.dumps(cfg, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


# ══════════════════════════════════════════════════════════════════
# Run ID và metadata helpers
# ══════════════════════════════════════════════════════════════════


def make_run_id(seed: int, cfg_hash: str, prefix: str = "") -> str:
    """
    Tạo run_id có format: {prefix}{timestamp}_seed{seed}_{cfghash}

    Ví dụ: "20260804-103015_seed123_a1b2c3d4"
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if prefix:
        prefix = f"{prefix}_"
    return f"{prefix}{ts}_seed{int(seed)}_{cfg_hash}"


def utc_now_iso() -> str:
    """ISO 8601 UTC timestamp cho metadata."""
    return datetime.now(timezone.utc).isoformat()


def git_commit(repo_dir: Optional[str] = None) -> Optional[str]:
    """Trả về git commit SHA ngắn (7 ký tự) nếu repo là git, None nếu không."""
    try:
        import subprocess
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        if repo_dir:
            cmd = ["git", "-C", repo_dir] + cmd[1:]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return out or None
    except Exception:
        return None


def safe_yaml_dump(obj: Any) -> str:
    """
    YAML dump an toàn — nếu không có PyYAML thì fallback JSON.
    """
    try:
        import yaml  # type: ignore
        return yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)
    except ImportError:
        # JSON là subset hợp lệ cho hầu hết metadata; chỉ fallback khi thiếu PyYAML
        return json.dumps(obj, indent=2, default=str, ensure_ascii=False)


__all__ = [
    "set_seed",
    "derive_seed",
    "seed_worker",
    "make_torch_generator",
    "hash_file",
    "config_hash",
    "make_run_id",
    "utc_now_iso",
    "git_commit",
    "safe_yaml_dump",
]