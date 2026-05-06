from __future__ import annotations

import argparse
import json
from pathlib import Path

import benchmark_otp2_testset as otp2_bench

import run_pose_direct_yolov5n_pipeline as single_pipeline
from lite_therm_pose.paper_models import PAPER_YOLO_DIRECT_MODELS
from lite_therm_pose.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and benchmark the paper comparison direct-pose suite with a YOLOv5n detector.")
    parser.add_argument("--models", type=str, default=",".join(PAPER_YOLO_DIRECT_MODELS))
    parser.add_argument("--pretrain-config", type=str, default="configs/coco_pretrain_pose_direct_yolov5n.yaml")
    parser.add_argument("--finetune-config", type=str, default="configs/openthermalpose2_finetune_pose_direct_yolov5n.yaml")
    parser.add_argument("--train-output", type=str, default="outputs/pose_direct_yolov5n_suite")
    parser.add_argument("--benchmark-output", type=str, default="outputs/otp2_test_benchmark_pose_direct_yolov5n_suite")
    parser.add_argument("--yolov5-repo", type=str, default="")
    parser.add_argument("--yolov5-weights", type=str, default="yolov5n.pt")
    parser.add_argument("--test-images-dir", type=str, default="")
    parser.add_argument("--test-labels-dir", type=str, default="")
    parser.add_argument("--search-root", type=str, action="append", default=[])
    parser.add_argument("--gpu-device", type=str, default="cuda:0")
    parser.add_argument("--cpu-device", type=str, default="cpu")
    parser.add_argument("--visual-samples", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--gpu-only", action="store_true", help="Skip CPU runtime benchmarking.")
    return parser.parse_args()


def selected_models(raw: str) -> list[str]:
    models = [item.strip() for item in raw.split(",") if item.strip()]
    if not models:
        raise ValueError("At least one model must be provided.")
    return models


def train_models(args: argparse.Namespace, models: list[str]) -> list[Path]:
    config_paths: list[Path] = []
    for model_name in models:
        model_args = argparse.Namespace(**vars(args))
        model_args.model = model_name
        print("training model:", model_name)
        config_paths.append(single_pipeline.run_training(model_args))
    return config_paths


def benchmark_models(args: argparse.Namespace, models: list[str]) -> None:
    output_dir = ensure_dir(args.benchmark_output)
    runs = otp2_bench.discover_runs([args.train_output], set(models))
    if not runs:
        raise FileNotFoundError("No runnable model checkpoints were found after training.")
    test_images = otp2_bench.load_otp2_test_images(
        images_dir=args.test_images_dir or None,
        labels_dir=args.test_labels_dir or None,
        search_roots=args.search_root or None,
    )
    coco_gt = otp2_bench.build_coco_gt(test_images)
    rows: list[dict[str, float | str]] = []
    gpu_available = otp2_bench.torch.cuda.is_available()
    visual_device = args.gpu_device if gpu_available else args.cpu_device
    visual_bundle = otp2_bench.build_bundle(args.finetune_config, runs[0], visual_device, visual_device, draw_parts=False)

    for run in runs:
        eval_device = args.gpu_device if gpu_available else args.cpu_device
        eval_bundle = otp2_bench.build_bundle(args.finetune_config, run, eval_device, eval_device, draw_parts=False)
        detector_stats = otp2_bench.evaluate_detector(eval_bundle, test_images)
        pose_stats = otp2_bench.evaluate_pose_gt(eval_bundle, test_images)
        joint_stats = otp2_bench.evaluate_joint(eval_bundle, test_images)
        bbox_results, keypoint_results, joint_pose_predictions = otp2_bench.collect_coco_predictions(eval_bundle, test_images)
        _, det_coco_stats = otp2_bench._run_cocoeval(coco_gt, bbox_results, "bbox")
        _, pose_coco_stats = otp2_bench._run_cocoeval(coco_gt, keypoint_results, "keypoints")
        pose_mean_oks = otp2_bench.compute_mean_oks(test_images, joint_pose_predictions)

        cpu_speed = {}
        if not args.gpu_only:
            cpu_bundle = otp2_bench.build_bundle(args.finetune_config, run, args.cpu_device, args.cpu_device, draw_parts=False)
            cpu_speed = otp2_bench.benchmark_runtime(cpu_bundle, test_images, warmup=args.warmup)
        gpu_speed = {}
        if gpu_available:
            gpu_bundle = otp2_bench.build_bundle(args.finetune_config, run, args.gpu_device, args.gpu_device, draw_parts=False)
            gpu_speed = otp2_bench.benchmark_runtime(gpu_bundle, test_images, warmup=args.warmup)

        detector_profile = otp2_bench.parameter_stats(eval_bundle.detector, "detector")
        pose_profile = otp2_bench.parameter_stats(eval_bundle.pose_model, "pose")
        stats = {
            "run_id": run.run_id,
            "suite_name": run.suite_name,
            "model_name": run.model_name,
            **detector_profile,
            **pose_profile,
            "total_params_m": detector_profile["detector_params_m"] + pose_profile["pose_params_m"],
            "detector_checkpoint_mb": otp2_bench.checkpoint_size_mb(run.detector_ckpt),
            "pose_checkpoint_mb": otp2_bench.checkpoint_size_mb(run.pose_ckpt),
            **detector_stats,
            **pose_stats,
            **joint_stats,
            **det_coco_stats,
            **pose_coco_stats,
            "pose_mean_oks_test": pose_mean_oks,
            "cpu_det_fps_images": cpu_speed.get("det_fps_images", 0.0),
            "cpu_det_latency_ms_test": cpu_speed.get("det_latency_ms_test", 0.0),
            "cpu_pose_fps_images": cpu_speed.get("pose_fps_images", 0.0),
            "cpu_pose_fps_persons": cpu_speed.get("pose_fps_persons", 0.0),
            "cpu_pose_latency_ms_test": cpu_speed.get("pose_latency_ms_test", 0.0),
            "cpu_joint_fps_images": cpu_speed.get("joint_fps_images", 0.0),
            "cpu_joint_latency_ms_test": cpu_speed.get("joint_latency_ms_test", 0.0),
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
        rows.append(stats)
        model_dir = ensure_dir(output_dir / run.run_id)
        with (model_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, ensure_ascii=False, indent=2)

    rows.sort(key=lambda row: float(row["joint_score_test"]), reverse=True)
    otp2_bench.write_rows(rows, output_dir)
    otp2_bench.plot_charts(rows, output_dir)
    otp2_bench.save_qualitative_examples(output_dir, args.finetune_config, visual_bundle, runs, test_images, num_samples=args.visual_samples)


def main() -> None:
    args = parse_args()
    models = selected_models(args.models)
    print("paper comparison models:", ",".join(models))
    train_models(args, models)
    benchmark_models(args, models)


if __name__ == "__main__":
    main()
