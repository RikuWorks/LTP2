from __future__ import annotations

import argparse
import json
from pathlib import Path

from lite_therm_pose import load_config
from lite_therm_pose.evaluate import evaluate_detector_model, evaluate_joint_pipeline, evaluate_pose_model, write_summary_report
from lite_therm_pose.paper_models import PAPER_YOLO_DIRECT_MODELS
from lite_therm_pose.profile import checkpoint_size_mb, parameter_stats
from lite_therm_pose.trainers import load_trained_detector, load_trained_pose, train_detector, train_pose
from lite_therm_pose.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the paper comparison direct-pose suite on another dataset from OTP2 YOLO checkpoints.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--source-suite", type=str, default="outputs/pose_direct_yolov5n_suite")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--models", type=str, default=",".join(PAPER_YOLO_DIRECT_MODELS))
    return parser.parse_args()


def selected_models(raw: str) -> list[str]:
    models = [item.strip() for item in raw.split(",") if item.strip()]
    if not models:
        raise ValueError("At least one model must be provided.")
    return models


def source_paths(source_suite: str, model_name: str) -> tuple[Path, Path]:
    root = Path(source_suite) / model_name
    detector_ckpt = root / "finetune_detector" / "detector_last.pt"
    pose_ckpt = root / "finetune_pose" / "pose_last.pt"
    if not detector_ckpt.exists():
        raise FileNotFoundError(f"Missing detector checkpoint: {detector_ckpt}")
    if not pose_ckpt.exists():
        raise FileNotFoundError(f"Missing pose checkpoint: {pose_ckpt}")
    return detector_ckpt, pose_ckpt


def main() -> None:
    args = parse_args()
    rows: list[dict[str, float | str]] = []
    output_root = ensure_dir(args.output)
    for model_name in selected_models(args.models):
        print("transfer fine-tune:", model_name)
        detector_weights, pose_weights = source_paths(args.source_suite, model_name)
        cfg = load_config(args.config)
        cfg.model.name = model_name
        cfg.detector.backend = "yolov5n"
        model_root = ensure_dir(output_root / model_name)
        det_dir = model_root / "finetune_detector"
        pose_dir = model_root / "finetune_pose"

        detector_model, det_stats = train_detector(cfg, det_dir, weights=str(detector_weights))
        pose_model, pose_stats = train_pose(cfg, pose_dir, weights=str(pose_weights))

        detector_model = load_trained_detector(cfg, det_dir / "detector_last.pt")
        pose_model = load_trained_pose(cfg, pose_dir / "pose_last.pt")
        det_metrics = evaluate_detector_model(detector_model, cfg)
        pose_metrics = evaluate_pose_model(pose_model, cfg)
        joint_metrics = evaluate_joint_pipeline(detector_model, pose_model, cfg)
        detector_profile = parameter_stats(detector_model, "detector")
        pose_profile = parameter_stats(pose_model, "pose")

        row: dict[str, float | str] = {
            "model": model_name,
            "model_family": "pose",
            "det_pre_loss": 0.0,
            "pose_pre_loss": 0.0,
            "det_fine_loss": round(float(det_stats["loss_history"][-1]), 6) if det_stats["loss_history"] else 0.0,
            "pose_fine_loss": round(float(pose_stats["loss_history"][-1]), 6) if pose_stats["loss_history"] else 0.0,
            "det_pre_train_seconds": 0.0,
            "pose_pre_train_seconds": 0.0,
            "det_fine_train_seconds": round(float(det_stats["train_seconds_total"]), 6),
            "pose_fine_train_seconds": round(float(pose_stats["train_seconds_total"]), 6),
            "det_epoch_seconds_mean": round(float(det_stats["epoch_seconds_mean"]), 6),
            "pose_epoch_seconds_mean": round(float(pose_stats["epoch_seconds_mean"]), 6),
            "det_train_peak_memory_mb": round(float(det_stats["train_peak_memory_mb"]), 6),
            "pose_train_peak_memory_mb": round(float(pose_stats["train_peak_memory_mb"]), 6),
            **{key: round(value, 6) for key, value in detector_profile.items()},
            **{key: round(value, 6) for key, value in pose_profile.items()},
            "total_params_m": round(detector_profile["detector_params_m"] + pose_profile["pose_params_m"], 6),
            "detector_checkpoint_mb": round(checkpoint_size_mb(det_dir / "detector_last.pt"), 6),
            "pose_checkpoint_mb": round(checkpoint_size_mb(pose_dir / "pose_last.pt"), 6),
            **{key: round(value, 6) for key, value in det_metrics.items()},
            **{key: round(value, 6) for key, value in pose_metrics.items()},
            **{key: round(value, 6) for key, value in joint_metrics.items()},
        }
        rows.append(row)
        with (model_root / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(row, handle, ensure_ascii=False, indent=2)

    rows.sort(key=lambda row: float(row["joint_score"]), reverse=True)
    write_summary_report(rows, output_root)


if __name__ == "__main__":
    main()
