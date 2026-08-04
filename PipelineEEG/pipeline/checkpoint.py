"""
pipeline/checkpoint.py
──────────────────────
Checkpoint với metadata sidecar cho reproducibility.

Mỗi .pth checkpoint được lưu kèm 1 file .meta.yaml chứa:
  - run_id, fold, config_hash
  - metrics: acc, mf1, kappa, f1_W, f1_N1, ...
  - sha256 của file .pth
  - timestamp ISO + git commit
  - đường dẫn tới config.yaml gốc

Hỗ trợ:
  - Atomic write (ghi tạm .tmp → rename) — tránh corrupt khi crash
  - SHA-256 hash tự động
  - Re-load metadata để verify integrity sau này
"""

import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from pipeline.utils import (
    git_commit,
    hash_file,
    safe_yaml_dump,
    utc_now_iso,
)


@dataclass
class CheckpointMeta:
    """Metadata cho 1 checkpoint. Sidecar file = <ckpt>.meta.yaml"""

    run_id:    str
    fold:      int
    stage:     str                 # "resnet" | "tcn" | "joint_resnet" | "joint_tcn"
    config_hash: str
    config_path: Optional[str]    # relative path to config.yaml inside run dir

    metrics: Dict[str, float] = field(default_factory=dict)

    timestamp: str = ""           # ISO 8601 UTC
    git_commit: Optional[str] = None
    sha256:     Optional[str] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _atomic_torch_save(obj: Any, target_path: Path) -> None:
    """
    torch.save atomic:
      1. Ghi vào <target>.tmp
      2. fsync (optional)
      3. os.replace(.tmp, target) — atomic trên POSIX và Windows.
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=target_path.stem + ".",
        suffix=".tmp",
        dir=target_path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        # torch.save cần binary file object
        torch.save(obj, tmp_path)
        os.replace(tmp_path, target_path)
    except Exception:
        # Dọn .tmp nếu lỗi
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def save_checkpoint_with_meta(
    state_dict: Dict[str, torch.Tensor],
    target_path: str,
    meta: CheckpointMeta,
    verbose: bool = True,
) -> Path:
    """
    Lưu checkpoint + metadata sidecar.

    Args:
        state_dict  : model state_dict (torch.Tensor values)
        target_path : đường dẫn tuyệt đối file .pth
        meta        : CheckpointMeta dataclass
        verbose     : in log nếu True

    Returns:
        Path tới file .meta.yaml
    """
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    # 1. Atomic torch.save
    _atomic_torch_save(state_dict, target)

    # 2. Hash file vừa ghi
    sha = hash_file(target)
    meta.sha256 = sha
    if not meta.timestamp:
        meta.timestamp = utc_now_iso()
    if meta.git_commit is None:
        meta.git_commit = git_commit()

    # 3. Sidecar YAML
    sidecar = target.with_suffix(target.suffix + ".meta.yaml")
    sidecar.write_text(safe_yaml_dump(meta.to_dict()), encoding="utf-8")

    if verbose:
        print(f"  ✓ Saved {target.name}  sha256={sha[:12]}…  meta={sidecar.name}")
    return sidecar


def load_checkpoint_meta(checkpoint_path: str) -> CheckpointMeta:
    """Load sidecar metadata. Raises nếu thiếu."""
    sidecar = Path(checkpoint_path).with_suffix(Path(checkpoint_path).suffix + ".meta.yaml")
    if not sidecar.exists():
        raise FileNotFoundError(f"Không tìm thấy sidecar: {sidecar}")

    import yaml  # type: ignore
    data = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
    return CheckpointMeta(**data)


def verify_checkpoint(checkpoint_path: str) -> bool:
    """
    Verify SHA-256 của file .pth khớp với metadata.
    Trả về True nếu khớp, False nếu không.
    """
    meta = load_checkpoint_meta(checkpoint_path)
    if not meta.sha256:
        return False
    actual = hash_file(checkpoint_path)
    return actual == meta.sha256


__all__ = [
    "CheckpointMeta",
    "save_checkpoint_with_meta",
    "load_checkpoint_meta",
    "verify_checkpoint",
]
