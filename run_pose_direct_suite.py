from __future__ import annotations

import argparse

import benchmark_otp2_testset as otp2_bench
from lite_therm_pose.experiment_suite import run_suite
from lite_therm_pose.models.specs import DIRECT_POSE_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and benchmark the direct-regression thermal pose model suite."
    )
    parser.add_argument("--pretrain-config", type=str, default="configs/coco_pretrain_pose_direct.yaml")
    parser.add_argument("--finetune-config", type=str, default="configs/openthermalpose2_finetune_pose_direct.yaml")
    parser.add_argument("--train-output", type=str, default="outputs/pose_direct_suite")
    parser.add_argument("--benchmark-output", type=str, default="outputs/otp2_test_benchmark_pose_direct_suite")
    parser.add_argument("--test-images-dir", type=str, default="")
    parser.add_argument("--test-labels-dir", type=str, default="")
    parser.add_argument("--search-root", type=str, action="append", default=[])
    parser.add_argument("--gpu-device", type=str, default="cuda:0")
    parser.add_argument("--cpu-device", type=str, default="cpu")
    parser.add_argument("--visual-samples", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def run_benchmark(args: argparse.Namespace) -> None:
    bench_args = argparse.Namespace(
        config=args.finetune_config,
        suite_dir=[args.train_output],
        output=args.benchmark_output,
        models=",".join(DIRECT_POSE_MODELS),
        test_images_dir=args.test_images_dir,
        test_labels_dir=args.test_labels_dir,
        search_root=args.search_root,
        gpu_device=args.gpu_device,
        cpu_device=args.cpu_device,
        visual_samples=args.visual_samples,
        warmup=args.warmup,
    )
    output_dir = otp2_bench.ensure_dir(bench_args.output)
    models_filter = {item.strip() for item in bench_args.models.split(",") if item.strip()}
    runs = otp2_bench.discover_runs(bench_args.suite_dir, models_filter)
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
        if gpu_available:
            gpu_bundle = otp2_bench.build_bundle(bench_args.config, run, bench_args.gpu_device, bench_args.gpu_device, draw_parts=False)
            gpu_speed = otp2_bench.benchmark_runtime(gpu_bundle, test_images, warmup=bench_args.warmup)
        else:
            gpu_speed = {}

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
        rows.append(stats)

    rows.sort(key=lambda row: float(row["joint_score_test"]), reverse=True)
    otp2_bench.write_rows(rows, output_dir)
    otp2_bench.plot_charts(rows, output_dir)
    otp2_bench.save_qualitative_examples(output_dir, bench_args.config, visual_bundle, runs, test_images, num_samples=bench_args.visual_samples)


def main() -> None:
    args = parse_args()
    print("direct pose models:", ",".join(DIRECT_POSE_MODELS))
    run_suite(args.pretrain_config, args.finetune_config, DIRECT_POSE_MODELS, args.train_output)
    run_benchmark(args)


if __name__ == "__main__":
    main()
