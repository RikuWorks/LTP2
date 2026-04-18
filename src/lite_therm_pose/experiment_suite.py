from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from .config import ExperimentConfig, load_config
from .evaluate import evaluate_detector_model, evaluate_joint_pipeline, evaluate_pose_model, write_summary_report
from .trainers import train_detector, train_pose
from .utils import ensure_dir


def clone_cfg(cfg: ExperimentConfig) -> ExperimentConfig:
    return deepcopy(cfg)


def run_suite(
    pretrain_config_path: str,
    finetune_config_path: str,
    model_names: list[str],
    output_dir: str | Path,
) -> list[dict[str, float | str]]:
    pretrain_cfg_base = load_config(pretrain_config_path)
    finetune_cfg_base = load_config(finetune_config_path)
    root = ensure_dir(output_dir)
    summary_rows: list[dict[str, float | str]] = []

    for model_name in model_names:
        model_root = ensure_dir(root / model_name)
        pretrain_cfg = clone_cfg(pretrain_cfg_base)
        finetune_cfg = clone_cfg(finetune_cfg_base)
        pretrain_cfg.model.name = model_name
        finetune_cfg.model.name = model_name

        det_pre_dir = model_root / "pretrain_detector"
        pose_pre_dir = model_root / "pretrain_pose"
        det_fine_dir = model_root / "finetune_detector"
        pose_fine_dir = model_root / "finetune_pose"

        detector_pre, det_pre_hist = train_detector(pretrain_cfg, det_pre_dir)
        pose_pre, pose_pre_hist = train_pose(pretrain_cfg, pose_pre_dir)
        detector_fine, det_fine_hist = train_detector(finetune_cfg, det_fine_dir, weights=str(det_pre_dir / "detector_last.pt"))
        pose_fine, pose_fine_hist = train_pose(finetune_cfg, pose_fine_dir, weights=str(pose_pre_dir / "pose_last.pt"))

        det_metrics = evaluate_detector_model(detector_fine, finetune_cfg)
        pose_metrics = evaluate_pose_model(pose_fine, finetune_cfg)
        joint_metrics = evaluate_joint_pipeline(detector_fine, pose_fine, finetune_cfg)

        row: dict[str, float | str] = {
            "model": model_name,
            "det_pre_loss": round(det_pre_hist[-1], 6) if det_pre_hist else 0.0,
            "pose_pre_loss": round(pose_pre_hist[-1], 6) if pose_pre_hist else 0.0,
            "det_fine_loss": round(det_fine_hist[-1], 6) if det_fine_hist else 0.0,
            "pose_fine_loss": round(pose_fine_hist[-1], 6) if pose_fine_hist else 0.0,
            **{key: round(value, 6) for key, value in det_metrics.items()},
            **{key: round(value, 6) for key, value in pose_metrics.items()},
            **{key: round(value, 6) for key, value in joint_metrics.items()},
        }
        summary_rows.append(row)
        with (model_root / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(row, handle, ensure_ascii=False, indent=2)

    write_summary_report(summary_rows, root)
    return summary_rows
