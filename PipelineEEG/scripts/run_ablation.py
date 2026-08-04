#!/usr/bin/env python3
"""
scripts/run_ablation.py
───────────────────────
CLI runner cho ablation matrix. Merge YAML configs (base + override),
chạy `run_full_pipeline` cho mỗi config, gom kết quả vào
`summary/ablation_comparison.csv`.

Usage:
    # Một config
    python scripts/run_ablation.py --configs configs/ablations/00_baseline.yaml

    # Nhiều config liên tiếp
    python scripts/run_ablation.py --configs configs/ablations/*.yaml

    # Custom folds + seed
    python scripts/run_ablation.py --configs c1.yaml c2.yaml \\
        --folds 0 1 2 --seed 42

    # Post-process only (dùng checkpoint có sẵn, không train)
    python scripts/run_ablation.py --configs c1.yaml \\
        --resnet-ckpt ./results/.../resnet_fold1.pth \\
        --tcn-ckpt    ./results/.../tcn_fold1.pth \\
        --postprocess-only

    # Resume
    python scripts/run_ablation.py --configs c1.yaml --resume

YAML extends:
    File có thể có `extends: base.yaml` ở đầu → sẽ merge với base.
"""

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Cho phép gọi từ thư mục gốc project
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from pipeline.trainer import PipelineConfig, run_full_pipeline
from pipeline.utils import config_hash, set_seed


# ══════════════════════════════════════════════════════════════════
# YAML merge với extends
# ══════════════════════════════════════════════════════════════════


def _deep_merge(base: Any, over: Any) -> Any:
    """Deep merge: dict được merge field-by-field, các kiểu khác bị override."""
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            if k in out:
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
        return out
    return over  # override


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_extends(cfg: Dict[str, Any], config_dir: Path,
                     seen: Optional[set] = None) -> Dict[str, Any]:
    """
    Resolve 'extends' chain. Trả về dict đã merge hoàn toàn.

    Path resolution: 'extends' được tìm theo thứ tự:
      1. <config_dir>/<ext_name>  (cùng thư mục)
      2. <config_dir>/../<ext_name>  (lên 1 cấp — thư mục configs/)
      3. <config_dir>/../../<ext_name>  (lên 2 cấp — project root)
    """
    seen = seen or set()
    if "extends" in cfg:
        ext_name = cfg["extends"]
        ext_path = None
        for cand_dir in (config_dir, config_dir.parent, config_dir.parent.parent):
            cand = (cand_dir / ext_name).resolve()
            if cand.exists():
                ext_path = cand
                break
        if ext_path is None:
            # Fallback: thử coi như tên không có 'yaml' → thêm .yaml
            if not ext_name.endswith((".yaml", ".yml")):
                ext_name_with_ext = ext_name + ".yaml"
                for cand_dir in (config_dir, config_dir.parent, config_dir.parent.parent):
                    cand = (cand_dir / ext_name_with_ext).resolve()
                    if cand.exists():
                        ext_path = cand
                        break
        if ext_path is None:
            raise FileNotFoundError(
                f"Không tìm thấy 'extends' file '{ext_name}' từ '{config_dir}'"
            )
        if ext_path in seen:
            raise ValueError(f"extends cycle: {ext_name} in {seen}")
        seen = seen | {ext_path}
        base = _load_yaml(ext_path)
        base = _resolve_extends(base, ext_path.parent, seen)
        cfg = _deep_merge(base, {k: v for k, v in cfg.items() if k != "extends"})
    return cfg


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML, resolve extends, trả về dict đã merge."""
    p = Path(path).resolve()
    cfg = _load_yaml(p)
    return _resolve_extends(cfg, p.parent)


# ══════════════════════════════════════════════════════════════════
# Dict → PipelineConfig
# ══════════════════════════════════════════════════════════════════


def _coerce_pipeline_config(cfg_dict: Dict[str, Any]) -> PipelineConfig:
    """
    Chuyển dict (sau merge) → PipelineConfig. Nested dict (loss, augment,
    postprocess) được convert thành dataclass tương ứng.
    """
    from pipeline.losses import LossConfig
    from pipeline.augment import AugmentConfig
    from pipeline.postprocess import PostprocessConfig

    d = dict(cfg_dict)

    # Nested dataclass
    if isinstance(d.get("loss"), dict):
        d["loss"] = LossConfig(**d["loss"])
    if isinstance(d.get("augment"), dict):
        d["augment"] = AugmentConfig(**d["augment"])
    if isinstance(d.get("postprocess"), dict):
        d["postprocess"] = PostprocessConfig(**d["postprocess"])

    return PipelineConfig(**d)


# ══════════════════════════════════════════════════════════════════
# CSV writer cho ablation comparison
# ══════════════════════════════════════════════════════════════════


def _safe_metric(metrics: Optional[Dict[str, Any]], key: str) -> str:
    if not metrics or key not in metrics:
        return ""
    v = metrics[key]
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _format_fold_row(cfg_name: str, fold_i: int, fold_result: Dict[str, Any]) -> Dict[str, str]:
    raw = fold_result.get("raw_metrics", {})
    pp  = fold_result.get("postprocess_metrics", None)
    return {
        "config":     cfg_name,
        "fold":       str(fold_i + 1),
        "n_epochs":   str(fold_result.get("n_test_epochs", "")),
        "elapsed_s":  f"{fold_result.get('elapsed_s', 0):.0f}",
        # Raw
        "raw_acc":    _safe_metric(raw, "acc"),
        "raw_mf1":    _safe_metric(raw, "mf1"),
        "raw_kappa":  _safe_metric(raw, "kappa"),
        "raw_f1_W":   _safe_metric(raw, "f1_W"),
        "raw_f1_N1":  _safe_metric(raw, "f1_N1"),
        "raw_f1_N2":  _safe_metric(raw, "f1_N2"),
        "raw_f1_N3":  _safe_metric(raw, "f1_N3"),
        "raw_f1_REM": _safe_metric(raw, "f1_REM"),
        # Post-processed (nếu có)
        "pp_acc":     _safe_metric(pp, "acc")  if pp else "",
        "pp_mf1":     _safe_metric(pp, "mf1")  if pp else "",
        "pp_kappa":   _safe_metric(pp, "kappa") if pp else "",
        "pp_f1_W":    _safe_metric(pp, "f1_W")  if pp else "",
        "pp_f1_N1":   _safe_metric(pp, "f1_N1") if pp else "",
        "pp_f1_N2":   _safe_metric(pp, "f1_N2") if pp else "",
        "pp_f1_N3":   _safe_metric(pp, "f1_N3") if pp else "",
        "pp_f1_REM":  _safe_metric(pp, "f1_REM") if pp else "",
    }


def _write_csv(rows: List[Dict[str, str]], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════


def _run_one(
    cfg_path: str,
    args: argparse.Namespace,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Load 1 config, chạy run_full_pipeline, trả về (config_name, fold_results)."""
    cfg_name = Path(cfg_path).stem
    print(f"\n{'═' * 70}\n  ▶ Config: {cfg_name}  ({cfg_path})\n{'═' * 70}")

    cfg_dict = load_config(cfg_path)
    cfg = _coerce_pipeline_config(cfg_dict)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.folds is not None:
        cfg.folds_to_run = args.folds  # sẽ được pick_up bởi run_full_pipeline

    print(f"  run_id hash: {config_hash(asdict(cfg))[:12]}")
    print(f"  loss_type:   {cfg.loss.loss_type}  γ={cfg.loss.focal_gamma}")
    print(f"  augment:     enabled={cfg.augment.enabled}")
    print(f"  postprocess: enabled={cfg.postprocess.enabled}  methods={cfg.postprocess.methods}")
    print(f"  joint:       enabled={cfg.joint_finetune}")

    # Run
    t0 = time.time()
    try:
        fold_metrics = run_full_pipeline(
            cfg,
            folds_to_run=args.folds,
            deterministic=not args.no_deterministic,
        )
    except Exception as e:
        print(f"\n  ✗ FAILED: {e}")
        traceback.print_exc()
        return cfg_name, [{"error": str(e)}]
    elapsed = time.time() - t0
    print(f"\n  ✓ Done in {elapsed:.0f}s  ({len(fold_metrics)} folds)")
    return cfg_name, fold_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ablation matrix.")
    parser.add_argument("--configs", nargs="+", required=True,
                        help="Paths to YAML configs (supports 'extends').")
    parser.add_argument("--folds", nargs="+", type=int, default=None,
                        help="Specific folds to run (e.g. 0 1 2).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed.")
    parser.add_argument("--output", type=str, default="./results/ablation",
                        help="Output directory.")
    parser.add_argument("--no-deterministic", action="store_true",
                        help="Disable CUDA deterministic flags (faster but non-reproducible).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip configs đã có summary trong output dir.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(exist_ok=True)

    all_rows: List[Dict[str, str]] = []
    config_summaries: Dict[str, Dict[str, Any]] = {}

    for cfg_path in args.configs:
        cfg_name = Path(cfg_path).stem
        summary_file = summary_dir / f"{cfg_name}.json"

        if args.resume and summary_file.exists():
            print(f"\n  ↻ Resume: skip {cfg_name} (summary exists)")
            with open(summary_file, encoding="utf-8") as f:
                fold_metrics = json.load(f)
        else:
            cfg_name_actual, fold_metrics = _run_one(cfg_path, args)
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(fold_metrics, f, indent=2, default=str)

        # CSV rows
        for fr in fold_metrics:
            if "error" in fr:
                all_rows.append({"config": cfg_name, "fold": "ERROR",
                                 "raw_acc": fr["error"][:80]})
            else:
                all_rows.append(_format_fold_row(cfg_name, fr.get("fold", 0) - 1, fr))

        # Summary
        config_summaries[cfg_name] = {
            "n_folds":   len(fold_metrics),
            "fold_results": fold_metrics,
        }

    # Write CSV
    csv_path = summary_dir / "ablation_comparison.csv"
    _write_csv(all_rows, csv_path)
    print(f"\n✓ Wrote {csv_path}  ({len(all_rows)} rows)")

    # Aggregate summary per config (mean ± std across folds)
    print("\n" + "═" * 70)
    print("  Ablation summary (mean ± std across folds)")
    print("═" * 70)
    print(f"  {'Config':<32s}  {'Acc':>7s}  {'MF1':>7s}  {'κ':>7s}  {'F1-N1':>7s}")
    print(f"  {'-'*32}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    for cfg_name, info in config_summaries.items():
        # Lấy metrics theo post-process nếu có, raw nếu không
        accs, mf1s, kappas, f1_n1s = [], [], [], []
        for fr in info["fold_results"]:
            if "error" in fr:
                continue
            m = fr.get("postprocess_metrics") or fr.get("raw_metrics") or fr
            if "acc" in m:  accs.append(m["acc"])
            if "mf1" in m:  mf1s.append(m["mf1"])
            if "kappa" in m: kappas.append(m["kappa"])
            if "f1_N1" in m: f1_n1s.append(m["f1_N1"])
        if not accs:
            print(f"  {cfg_name:<32s}  (no data)")
            continue
        def mean_std(xs):
            import numpy as np
            a = np.array(xs)
            return f"{a.mean():.4f}±{a.std():.4f}"
        print(f"  {cfg_name:<32s}  "
              f"{mean_std(accs):>9s}  {mean_std(mf1s):>9s}  "
              f"{mean_std(kappas):>9s}  {mean_std(f1_n1s):>9s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
