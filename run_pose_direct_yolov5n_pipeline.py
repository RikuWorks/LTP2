from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

import benchmark_otp2_testset as otp2_bench
from lite_therm_pose import load_config
from lite_therm_pose.evaluate import evaluate_detector_model, evaluate_joint_pipeline, evaluate_pose_model, write_summary_report
from lite_therm_pose.profile import checkpoint_size_mb, parameter_stats
from lite_therm_pose.runtime import resolve_model_device
from lite_therm_pose.trainers import load_trained_detector, load_trained_pose
from lite_therm_pose.utils import dataclass_to_dict, ensure_dir


POSE_MODEL = "pose_resnet50_se_direct_refine"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one improved direct pose model with a YOLOv5n detector and benchmark it on OTP2 test.")
    parser.add_argument("--pretrain-config", type=str, default="configs/coco_pretrain_pose_direct_yolov5n.yaml")
    parser.add_argument("--finetune-config", type=str, default="configs/openthermalpose2_finetune_pose_direct_yolov5n.yaml")
    parser.add_argument("--train-output", type=str, default="outputs/pose_direct_yolov5n")
    parser.add_argument("--benchmark-output", type=str, default="outputs/otp2_test_benchmark_pose_direct_yolov5n")
    parser.add_argument("--yolov5-repo", type=str, default="")
    parser.add_argument("--yolov5-weights", type=str, default="yolov5n.pt")
    parser.add_argument("--test-images-dir", type=str, default="")
    parser.add_argument("--test-labels-dir", type=str, default="")
    parser.add_argument("--search-root", type=str, action="append", default=[])
    parser.add_argument("--gpu-device", type=str, default="cuda:0")
    parser.add_argument("--cpu-device", type=str, default="cpu")
    parser.add_argument("--visual-samples", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def prepare_cfg(path: str, args: argparse.Namespace):
    cfg = load_config(path)
    cfg.model.name = POSE_MODEL
    cfg.detector.backend = "yolov5n"
    cfg.detector.yolov5_repo = args.yolov5_repo
    cfg.detector.yolov5_weights = args.yolov5_weights
    return cfg


def save_cfg(cfg, path: Path) -> None:
    payload = dataclass_to_dict(cfg)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _supports_parallel(pretrain_cfg, finetune_cfg) -> bool:
    pose_device = resolve_model_device(pretrain_cfg.runtime.pose_device, pretrain_cfg.runtime.device, "pose")
    detector_device = resolve_model_device(finetune_cfg.runtime.detector_device, finetune_cfg.runtime.device, "detector")
    return pose_device.type == "cuda" and detector_device.type == "cuda" and pose_device != detector_device


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _python_executable() -> str:
    return sys.executable


def _pose_train_command(config_path: Path, output_dir: Path, weights: str = "") -> list[str]:
    command = [
        _python_executable(),
        str(_repo_root() / "train_pose.py"),
        "--config",
        str(config_path),
        "--model",
        POSE_MODEL,
        "--output",
        str(output_dir),
    ]
    if weights:
        command.extend(["--weights", weights])
    return command


def _detector_train_command(config_path: Path, output_dir: Path, weights: str) -> list[str]:
    command = [
        _python_executable(),
        str(_repo_root() / "train_detector.py"),
        "--config",
        str(config_path),
        "--model",
        POSE_MODEL,
        "--output",
        str(output_dir),
    ]
    if weights:
        command.extend(["--weights", weights])
    return command


def _run_command(command: list[str], label: str) -> None:
    process = subprocess.run(command, cwd=_repo_root(), check=False)
    if process.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {process.returncode}")


def _read_training_stats(output_dir: Path, checkpoint_name: str) -> dict[str, float | list[float]]:
    checkpoint_path = output_dir / checkpoint_name
    results_csv = output_dir / "yolov5_runs" / "train" / "results.csv"

    if results_csv.exists():
        import csv

        with results_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        loss_history: list[float] = []
        for row in rows:
            parts: list[float] = []
            for key in ("train/box_loss", "train/obj_loss", "val/box_loss", "val/obj_loss"):
                value = row.get(key, "")
                if not value:
                    continue
                try:
                    parts.append(float(value))
                except ValueError:
                    continue
            if parts:
                loss_history.append(sum(parts) / len(parts))

        return {
            "loss_history": loss_history,
            "train_seconds_total": 0.0,
            "epoch_seconds_mean": 0.0,
            "train_peak_memory_mb": 0.0,
            "completed_epochs": len(rows),
        }

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Expected checkpoint was not found: {checkpoint_path}")
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    epoch = int(payload.get("epoch", 0)) if isinstance(payload, dict) else 0
    return {
        "loss_history": [],
        "train_seconds_total": 0.0,
        "epoch_seconds_mean": 0.0,
        "train_peak_memory_mb": 0.0,
        "completed_epochs": epoch,
    }


def run_training(args: argparse.Namespace) -> Path:
    pretrain_cfg = prepare_cfg(args.pretrain_config, args)
    finetune_cfg = prepare_cfg(args.finetune_config, args)
    root = ensure_dir(Path(args.train_output) / POSE_MODEL)
    effective_pretrain_cfg = root / "effective_pretrain_config.yaml"
    effective_finetune_cfg = root / "effective_finetune_config.yaml"
    save_cfg(pretrain_cfg, effective_pretrain_cfg)
    save_cfg(finetune_cfg, effective_finetune_cfg)
    pose_pre_dir = root / "pretrain_pose"
    det_fine_dir = root / "finetune_detector"
    pose_fine_dir = root / "finetune_pose"

    if _supports_parallel(pretrain_cfg, finetune_cfg):
        pose_pre_proc = subprocess.Popen(
            _pose_train_command(effective_pretrain_cfg, pose_pre_dir),
            cwd=_repo_root(),
        )
        det_fine_proc = subprocess.Popen(
            _detector_train_command(effective_finetune_cfg, det_fine_dir, args.yolov5_weights),
            cwd=_repo_root(),
        )
        pose_pre_code = pose_pre_proc.wait()
        det_fine_code = det_fine_proc.wait()
        if pose_pre_code != 0:
            raise RuntimeError(f"pose_pre failed with exit code {pose_pre_code}")
        if det_fine_code != 0:
            raise RuntimeError(f"det_fine failed with exit code {det_fine_code}")
        _run_command(
            _pose_train_command(effective_finetune_cfg, pose_fine_dir, str(pose_pre_dir / "pose_last.pt")),
            "pose_fine",
        )
    else:
        _run_command(_pose_train_command(effective_pretrain_cfg, pose_pre_dir), "pose_pre")
        _run_command(_detector_train_command(effective_finetune_cfg, det_fine_dir, args.yolov5_weights), "det_fine")
        _run_command(
            _pose_train_command(effective_finetune_cfg, pose_fine_dir, str(pose_pre_dir / "pose_last.pt")),
            "pose_fine",
        )

    pose_pre_stats = _read_training_stats(pose_pre_dir, "pose_last.pt")
    det_fine_stats = _read_training_stats(det_fine_dir, "detector_last.pt")
    pose_fine_stats = _read_training_stats(pose_fine_dir, "pose_last.pt")

    detector = load_trained_detector(finetune_cfg, det_fine_dir / "detector_last.pt")
    pose = load_trained_pose(finetune_cfg, pose_fine_dir / "pose_last.pt")
    det_metrics = evaluate_detector_model(detector, finetune_cfg)
    pose_metrics = evaluate_pose_model(pose, finetune_cfg)
    joint_metrics = evaluate_joint_pipeline(detector, pose, finetune_cfg)
    detector_profile = parameter_stats(detector, "detector")
    pose_profile = parameter_stats(pose, "pose")
    row = {
        "model": POSE_MODEL,
        "model_family": "pose",
        "det_pre_loss": 0.0,
        "pose_pre_loss": round(float(pose_pre_stats["loss_history"][-1]), 6) if pose_pre_stats["loss_history"] else 0.0,
        "det_fine_loss": round(float(det_fine_stats["loss_history"][-1]), 6) if det_fine_stats["loss_history"] else 0.0,
        "pose_fine_loss": round(float(pose_fine_stats["loss_history"][-1]), 6) if pose_fine_stats["loss_history"] else 0.0,
        "det_pre_train_seconds": 0.0,
        "pose_pre_train_seconds": round(float(pose_pre_stats["train_seconds_total"]), 6),
        "det_fine_train_seconds": round(float(det_fine_stats["train_seconds_total"]), 6),
        "pose_fine_train_seconds": round(float(pose_fine_stats["train_seconds_total"]), 6),
        "det_epoch_seconds_mean": round(float(det_fine_stats["epoch_seconds_mean"]), 6),
        "pose_epoch_seconds_mean": round(float(pose_fine_stats["epoch_seconds_mean"]), 6),
        "det_train_peak_memory_mb": round(float(det_fine_stats["train_peak_memory_mb"]), 6),
        "pose_train_peak_memory_mb": round(float(pose_fine_stats["train_peak_memory_mb"]), 6),
        **{key: round(value, 6) for key, value in detector_profile.items()},
        **{key: round(value, 6) for key, value in pose_profile.items()},
        "total_params_m": round(detector_profile["detector_params_m"] + pose_profile["pose_params_m"], 6),
        "detector_checkpoint_mb": round(checkpoint_size_mb(det_fine_dir / "detector_last.pt"), 6),
        "pose_checkpoint_mb": round(checkpoint_size_mb(pose_fine_dir / "pose_last.pt"), 6),
        **{key: round(value, 6) for key, value in det_metrics.items()},
        **{key: round(value, 6) for key, value in pose_metrics.items()},
        **{key: round(value, 6) for key, value in joint_metrics.items()},
    }
    write_summary_report([row], root)
    with (root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(row, handle, ensure_ascii=False, indent=2)
    return effective_finetune_cfg


def run_benchmark(args: argparse.Namespace, config_path: Path) -> None:
    bench_args = argparse.Namespace(
        config=str(config_path),
        suite_dir=[args.train_output],
        output=args.benchmark_output,
        models=POSE_MODEL,
        test_images_dir=args.test_images_dir,
        test_labels_dir=args.test_labels_dir,
        search_root=args.search_root,
        gpu_device=args.gpu_device,
        cpu_device=args.cpu_device,
        visual_samples=args.visual_samples,
        warmup=args.warmup,
    )
    output_dir = otp2_bench.ensure_dir(bench_args.output)
    runs = otp2_bench.discover_runs(bench_args.suite_dir, {POSE_MODEL})
    if not runs:
        raise FileNotFoundError("No runnable model checkpoints were found after training.")
    test_images = otp2_bench.load_otp2_test_images(
        images_dir=bench_args.test_images_dir or None,
        labels_dir=bench_args.test_labels_dir or None,
        search_roots=bench_args.search_root or None,
    )
    coco_gt = otp2_bench.build_coco_gt(test_images)
    rows: list[dict[str, float | str]] = []
    gpu_available = otp2_bench.torch.cuda.is_available()
    visual_device = bench_args.gpu_device if gpu_available else bench_args.cpu_device
    visual_bundle = otp2_bench.build_bundle(bench_args.config, runs[0], visual_device, visual_device, draw_parts=False)
    for run in runs:
        eval_device = bench_args.gpu_device if gpu_available else bench_args.cpu_device
        eval_bundle = otp2_bench.build_bundle(bench_args.config, run, eval_device, eval_device, draw_parts=False)
        detector_stats = otp2_bench.evaluate_detector(eval_bundle, test_images)
        pose_stats = otp2_bench.evaluate_pose_gt(eval_bundle, test_images)
        joint_stats = otp2_bench.evaluate_joint(eval_bundle, test_images)
        bbox_results, keypoint_results, joint_pose_predictions = otp2_bench.collect_coco_predictions(eval_bundle, test_images)
        _, det_coco_stats = otp2_bench._run_cocoeval(coco_gt, bbox_results, "bbox")
        _, pose_coco_stats = otp2_bench._run_cocoeval(coco_gt, keypoint_results, "keypoints")
        pose_mean_oks = otp2_bench.compute_mean_oks(test_images, joint_pose_predictions)
        cpu_bundle = otp2_bench.build_bundle(bench_args.config, run, bench_args.cpu_device, bench_args.cpu_device, draw_parts=False)
        cpu_speed = otp2_bench.benchmark_runtime(cpu_bundle, test_images, warmup=bench_args.warmup)
        gpu_speed = {}
        if gpu_available:
            gpu_bundle = otp2_bench.build_bundle(bench_args.config, run, bench_args.gpu_device, bench_args.gpu_device, draw_parts=False)
            gpu_speed = otp2_bench.benchmark_runtime(gpu_bundle, test_images, warmup=bench_args.warmup)
        rows.append(
            {
                "run_id": run.run_id,
                "suite_name": run.suite_name,
                "model_name": run.model_name,
                **otp2_bench.parameter_stats(eval_bundle.detector, "detector"),
                **otp2_bench.parameter_stats(eval_bundle.pose_model, "pose"),
                "total_params_m": otp2_bench.parameter_stats(eval_bundle.detector, "detector")["detector_params_m"] + otp2_bench.parameter_stats(eval_bundle.pose_model, "pose")["pose_params_m"],
                "detector_checkpoint_mb": otp2_bench.checkpoint_size_mb(run.detector_ckpt),
                "pose_checkpoint_mb": otp2_bench.checkpoint_size_mb(run.pose_ckpt),
                **detector_stats,
                **pose_stats,
                **joint_stats,
                **det_coco_stats,
                **pose_coco_stats,
                "pose_mean_oks_test": pose_mean_oks,
                "cpu_det_fps_images": cpu_speed["det_fps_images"],
                "cpu_det_latency_ms_test": cpu_speed["det_latency_ms_test"],
                "cpu_pose_fps_images": cpu_speed["pose_fps_images"],
                "cpu_pose_fps_persons": cpu_speed["pose_fps_persons"],
                "cpu_pose_latency_ms_test": cpu_speed["pose_latency_ms_test"],
                "cpu_joint_fps_images": cpu_speed["joint_fps_images"],
                "cpu_joint_latency_ms_test": cpu_speed["joint_latency_ms_test"],
                "gpu_det_fps_images": gpu_speed.get("det_fps_images", 0.0),
                "gpu_det_latency_ms_test": gpu_speed.get("det_latency_ms_test", 0.0),
                "gpu_det_peak_memory_mb_test": gpu_speed.get("det_peak_memory_mb_test", 0.0),
                "gpu_pose_fps_images": gpu_speed.get("pose_fps_images", 0.0),
                "gpu_pose_fps_persons": gpu_speed.get("pose_fps_persons", 0.0),
                "gpu_pose_latency_ms_test": gpu_speed.get("pose_latency_ms_test", 0.0),
                "gpu_pose_peak_memory_mb_test": gpu_speed.get("pose_peak_memory_mb_test", 0.0),
                "gpu_joint_fps_images": gpu_speed.get("joint_fps_images", 0.0),
                "gpu_joint_latency_ms_test": gpu_speed.get("joint_latency_ms_test", 0.0),
                "gpu_joint_peak_memory_mb_test": gpu_speed.get("joint_peak_memory_mb_test", 0.0),
                "test_images_count": len(test_images),
                "test_persons_count": sum(len(item.persons) for item in test_images),
            }
        )
    rows.sort(key=lambda row: float(row["joint_score_test"]), reverse=True)
    otp2_bench.write_rows(rows, output_dir)
    otp2_bench.plot_charts(rows, output_dir)
    otp2_bench.save_qualitative_examples(output_dir, bench_args.config, visual_bundle, runs, test_images, num_samples=bench_args.visual_samples)


def main() -> None:
    args = parse_args()
    print("pose model:", POSE_MODEL)
    config_path = run_training(args)
    run_benchmark(args, config_path)


if __name__ == "__main__":
    main()
