from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from lite_therm_pose import load_config
from lite_therm_pose.evaluate import evaluate_detector_model, evaluate_joint_pipeline, evaluate_pose_model, write_summary_report
from lite_therm_pose.profile import checkpoint_size_mb, parameter_stats
from lite_therm_pose.trainers import load_trained_detector, load_trained_pose, train_detector, train_pose
from lite_therm_pose.utils import ensure_dir

from run_paper_existing_weights_suite import ExistingRun, _allowed_models, discover_existing_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune existing OTP2 weights on LWIRPOSE or UCH only.")
    parser.add_argument("--source-root", type=str, default="paper/outputs")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["lwir", "uch"], required=True)
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--lwir-tiny-config", type=str, default="configs/lwirpose_finetune_pose_direct.yaml")
    parser.add_argument("--lwir-yolo-config", type=str, default="configs/lwirpose_finetune_pose_direct_yolov5n.yaml")
    parser.add_argument("--uch-tiny-config", type=str, default="configs/uch_thermal_pose_direct.yaml")
    parser.add_argument("--uch-yolo-config", type=str, default="configs/uch_thermal_pose_direct_yolov5n.yaml")
    return parser.parse_args()


def _config_path(args: argparse.Namespace, run: ExistingRun) -> str:
    if args.dataset == "lwir":
        return args.lwir_yolo_config if run.backend == "yolov5n" else args.lwir_tiny_config
    return args.uch_yolo_config if run.backend == "yolov5n" else args.uch_tiny_config


def main() -> None:
    args = parse_args()
    runs = discover_existing_runs(Path(args.source_root), _allowed_models(args.models))
    if not runs:
        raise FileNotFoundError("No existing finetune checkpoint pairs were found under the source root.")
    output_root = ensure_dir(args.output)
    rows: list[dict[str, Any]] = []
    for run in runs:
        print("transfer fine-tune:", run.run_id, "->", args.dataset)
        cfg = load_config(_config_path(args, run))
        cfg.model.name = run.model_name
        cfg.detector.backend = run.backend
        model_root = ensure_dir(output_root / run.run_id)
        det_dir = model_root / "finetune_detector"
        pose_dir = model_root / "finetune_pose"

        detector_model, det_stats = train_detector(cfg, det_dir, weights=str(run.detector_ckpt))
        pose_model, pose_stats = train_pose(cfg, pose_dir, weights=str(run.pose_ckpt))

        detector_model = load_trained_detector(cfg, det_dir / "detector_last.pt")
        pose_model = load_trained_pose(cfg, pose_dir / "pose_last.pt")
        det_metrics = evaluate_detector_model(detector_model, cfg)
        pose_metrics = evaluate_pose_model(pose_model, cfg)
        joint_metrics = evaluate_joint_pipeline(detector_model, pose_model, cfg)
        detector_profile = parameter_stats(detector_model, "detector")
        pose_profile = parameter_stats(pose_model, "pose")

        row = {
            "run_id": run.run_id,
            "suite_name": run.suite_name,
            "model_name": run.model_name,
            "backend": run.backend,
            "dataset": args.dataset,
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
            "source_dir": str(run.source_dir),
        }
        rows.append(row)
        with (model_root / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(row, handle, ensure_ascii=False, indent=2)

    rows.sort(key=lambda row: float(row["joint_score"]), reverse=True)
    write_summary_report(rows, output_root)
    with (output_root / "transfer_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
