from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import benchmark_otp2_testset as otp2_bench
import torch

from lite_therm_pose import load_config
from lite_therm_pose.checkpoints import load_flexible_state_dict
from lite_therm_pose.detector_backend import build_detector_runtime
from lite_therm_pose.models.pose_direct import create_pose_model
from lite_therm_pose.profile import checkpoint_size_mb, parameter_stats
from lite_therm_pose.runtime import resolve_model_device
from lite_therm_pose.utils import ensure_dir


@dataclass
class ExistingRun:
    run_id: str
    suite_name: str
    model_name: str
    detector_ckpt: Path
    pose_ckpt: Path
    backend: str
    source_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark all existing checkpoint pairs under paper/outputs on OTP2 with GPU-only speed by default.")
    parser.add_argument("--source-root", type=str, default="paper/outputs")
    parser.add_argument("--output", type=str, default="paper/outputs/paper_existing_weights_suite")
    parser.add_argument("--models", type=str, default="", help="Optional comma-separated model names or run ids.")
    parser.add_argument("--otp2-heatmap-config", type=str, default="configs/openthermalpose2_finetune_paper_grade_fast.yaml")
    parser.add_argument("--otp2-direct-config", type=str, default="configs/openthermalpose2_finetune_pose_direct.yaml")
    parser.add_argument("--otp2-direct-yolo-config", type=str, default="configs/openthermalpose2_finetune_pose_direct_yolov5n.yaml")
    parser.add_argument("--test-images-dir", type=str, default="")
    parser.add_argument("--test-labels-dir", type=str, default="")
    parser.add_argument("--search-root", type=str, action="append", default=[])
    parser.add_argument("--gpu-device", type=str, default="cuda:0")
    parser.add_argument("--cpu-device", type=str, default="cpu")
    parser.add_argument("--visual-samples", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--gpu-only", action="store_true", default=True)
    return parser.parse_args()


def _allowed_models(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_model_name(run_dir_name: str, payload: dict[str, Any]) -> str:
    if payload.get("model_name"):
        return str(payload["model_name"])
    if payload.get("model"):
        return str(payload["model"])
    suffixes = [
        "_pckh_hires",
        "_pckh_refine",
        "_from_ppm",
    ]
    model_name = run_dir_name
    for suffix in suffixes:
        if model_name.endswith(suffix):
            model_name = model_name[: -len(suffix)]
    return model_name


def _infer_backend(source_dir: Path, model_name: str) -> str:
    source_text = str(source_dir).lower()
    if "yolov5" in source_text or "yolo" in source_text:
        return "yolov5n"
    return "tiny"


def _config_template(args: argparse.Namespace, run: ExistingRun) -> str:
    if run.backend == "yolov5n":
        return args.otp2_direct_yolo_config
    if "direct" in run.model_name:
        return args.otp2_direct_config
    return args.otp2_heatmap_config


def discover_existing_runs(source_root: Path, allow: set[str]) -> list[ExistingRun]:
    runs: list[ExistingRun] = []
    for det_ckpt in source_root.rglob("finetune_detector/detector_last.pt"):
        run_dir = det_ckpt.parent.parent
        pose_ckpt = run_dir / "finetune_pose" / "pose_last.pt"
        if not pose_ckpt.exists():
            continue
        suite_dir = run_dir.parent
        metrics_payload = _read_json(run_dir / "metrics.json")
        model_name = _normalize_model_name(run_dir.name, metrics_payload)
        run_id = run_dir.name
        if allow and run_id not in allow and model_name not in allow:
            continue
        runs.append(
            ExistingRun(
                run_id=run_id,
                suite_name=suite_dir.name,
                model_name=model_name,
                detector_ckpt=det_ckpt,
                pose_ckpt=pose_ckpt,
                backend=_infer_backend(run_dir, model_name),
                source_dir=run_dir,
            )
        )
    runs.sort(key=lambda item: (item.model_name, item.suite_name, item.run_id))
    return runs


def build_bundle_from_run(args: argparse.Namespace, run: ExistingRun, detector_device_name: str, pose_device_name: str, draw_parts: bool = False):
    cfg = load_config(_config_template(args, run))
    cfg.model.name = run.model_name
    cfg.detector.backend = run.backend
    cfg.runtime.detector_device = detector_device_name
    cfg.runtime.pose_device = pose_device_name
    cfg.runtime.device = detector_device_name if detector_device_name == pose_device_name else "auto"
    cfg.runtime.draw_parts = draw_parts
    detector_device = resolve_model_device(cfg.runtime.detector_device, cfg.runtime.device, "detector")
    pose_device = resolve_model_device(cfg.runtime.pose_device, cfg.runtime.device, "pose")
    detector, _ = build_detector_runtime(cfg, run.detector_ckpt, explicit_device=detector_device_name)
    pose_model = create_pose_model(
        num_keypoints=cfg.dataset.num_keypoints,
        num_parts=len(cfg.dataset.body_parts),
        in_channels=1 if cfg.dataset.grayscale else 3,
        model_name=run.model_name,
    ).to(pose_device).eval()
    load_flexible_state_dict(pose_model, str(run.pose_ckpt))
    return otp2_bench.RuntimeBundle(
        detector=detector,
        pose_model=pose_model,
        detector_device=detector_device,
        pose_device=pose_device,
        dataset_cfg=cfg.dataset,
        detector_cfg=cfg.detector,
        draw_parts=draw_parts,
    )


def write_manifest(runs: list[ExistingRun], output_dir: Path) -> None:
    rows = [
        {
            "run_id": run.run_id,
            "suite_name": run.suite_name,
            "model_name": run.model_name,
            "backend": run.backend,
            "detector_ckpt": str(run.detector_ckpt),
            "pose_ckpt": str(run.pose_ckpt),
            "source_dir": str(run.source_dir),
        }
        for run in runs
    ]
    with (output_dir / "existing_runs_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    if rows:
        with (output_dir / "existing_runs_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output)
    allow = _allowed_models(args.models)
    runs = discover_existing_runs(Path(args.source_root), allow)
    if not runs:
        raise FileNotFoundError("No existing finetune checkpoint pairs were found under the source root.")
    write_manifest(runs, output_dir)

    test_images = otp2_bench.load_otp2_test_images(
        images_dir=args.test_images_dir or None,
        labels_dir=args.test_labels_dir or None,
        search_roots=args.search_root or None,
    )
    coco_gt = otp2_bench.build_coco_gt(test_images)
    rows: list[dict[str, Any]] = []
    gpu_available = torch.cuda.is_available()
    visual_device = args.gpu_device if gpu_available else args.cpu_device
    visual_bundle = build_bundle_from_run(args, runs[0], visual_device, visual_device, draw_parts=False)

    for run in runs:
        eval_device = args.gpu_device if gpu_available else args.cpu_device
        eval_bundle = build_bundle_from_run(args, run, eval_device, eval_device, draw_parts=False)
        detector_stats = otp2_bench.evaluate_detector(eval_bundle, test_images)
        pose_stats = otp2_bench.evaluate_pose_gt(eval_bundle, test_images)
        joint_stats = otp2_bench.evaluate_joint(eval_bundle, test_images)
        bbox_results, keypoint_results, joint_pose_predictions = otp2_bench.collect_coco_predictions(eval_bundle, test_images)
        _, det_coco_stats = otp2_bench._run_cocoeval(coco_gt, bbox_results, "bbox")
        _, pose_coco_stats = otp2_bench._run_cocoeval(coco_gt, keypoint_results, "keypoints")
        pose_mean_oks = otp2_bench.compute_mean_oks(test_images, joint_pose_predictions)

        cpu_speed: dict[str, float] = {}
        if not args.gpu_only:
            cpu_bundle = build_bundle_from_run(args, run, args.cpu_device, args.cpu_device, draw_parts=False)
            cpu_speed = otp2_bench.benchmark_runtime(cpu_bundle, test_images, warmup=args.warmup)
        gpu_speed: dict[str, float] = {}
        if gpu_available:
            gpu_bundle = build_bundle_from_run(args, run, args.gpu_device, args.gpu_device, draw_parts=False)
            gpu_speed = otp2_bench.benchmark_runtime(gpu_bundle, test_images, warmup=args.warmup)

        detector_profile = parameter_stats(eval_bundle.detector, "detector")
        pose_profile = parameter_stats(eval_bundle.pose_model, "pose")
        row = {
            "run_id": run.run_id,
            "suite_name": run.suite_name,
            "model_name": run.model_name,
            "backend": run.backend,
            **detector_profile,
            **pose_profile,
            "total_params_m": detector_profile["detector_params_m"] + pose_profile["pose_params_m"],
            "detector_checkpoint_mb": checkpoint_size_mb(run.detector_ckpt),
            "pose_checkpoint_mb": checkpoint_size_mb(run.pose_ckpt),
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
            "source_dir": str(run.source_dir),
        }
        rows.append(row)
        model_dir = ensure_dir(output_dir / run.run_id)
        with (model_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(row, handle, ensure_ascii=False, indent=2)

    rows.sort(key=lambda row: float(row["joint_score_test"]), reverse=True)
    otp2_bench.write_rows(rows, output_dir)
    otp2_bench.plot_charts(rows, output_dir)
    otp2_bench.save_qualitative_examples(output_dir, _config_template(args, runs[0]), visual_bundle, [
        otp2_bench.BenchmarkRun(
            run_id=run.run_id,
            model_name=run.model_name,
            suite_name=run.suite_name,
            detector_ckpt=run.detector_ckpt,
            pose_ckpt=run.pose_ckpt,
        )
        for run in runs
    ], test_images, num_samples=args.visual_samples)


if __name__ == "__main__":
    main()
