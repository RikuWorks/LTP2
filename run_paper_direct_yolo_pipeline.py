from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

from lite_therm_pose.paper_models import PAPER_YOLO_DIRECT_MODELS
from lite_therm_pose.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full paper comparison pipeline for YOLO-backed direct pose models.")
    parser.add_argument("--models", type=str, default=",".join(PAPER_YOLO_DIRECT_MODELS))
    parser.add_argument("--otp2-pretrain-config", type=str, default="configs/coco_pretrain_pose_direct_yolov5n.yaml")
    parser.add_argument("--otp2-finetune-config", type=str, default="configs/openthermalpose2_finetune_pose_direct_yolov5n.yaml")
    parser.add_argument("--lwir-config", type=str, default="configs/lwirpose_finetune_pose_direct_yolov5n.yaml")
    parser.add_argument("--uch-config", type=str, default="configs/uch_thermal_pose_direct_yolov5n.yaml")
    parser.add_argument("--otp2-train-output", type=str, default="outputs/pose_direct_yolov5n_suite")
    parser.add_argument("--otp2-benchmark-output", type=str, default="outputs/otp2_test_benchmark_pose_direct_yolov5n_suite")
    parser.add_argument("--lwir-output", type=str, default="outputs/lwirpose_pose_direct_yolov5n_suite")
    parser.add_argument("--uch-output", type=str, default="outputs/uch_pose_direct_yolov5n_suite")
    parser.add_argument("--summary-output", type=str, default="outputs/paper_direct_yolo_comparison")
    parser.add_argument("--pose-pretrain-source", type=str, action="append", default=["outputs/pose_direct_suite", "outputs/pose_direct"], help="Existing suite directories containing COCO pose pretrain checkpoints.")
    parser.add_argument("--yolov5-repo", type=str, default="")
    parser.add_argument("--yolov5-weights", type=str, default="yolov5n.pt")
    parser.add_argument("--test-images-dir", type=str, default="")
    parser.add_argument("--test-labels-dir", type=str, default="")
    parser.add_argument("--search-root", type=str, action="append", default=[])
    parser.add_argument("--gpu-device", type=str, default="cuda:0")
    parser.add_argument("--cpu-device", type=str, default="cpu")
    parser.add_argument("--visual-samples", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--skip-otp2", action="store_true")
    parser.add_argument("--skip-lwir", action="store_true")
    parser.add_argument("--skip-uch", action="store_true")
    return parser.parse_args()


def _run(command: list[str]) -> None:
    env = os.environ.copy()
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    env.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    process = subprocess.run(command, cwd=Path(__file__).resolve().parent, check=False, env=env)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_table(rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        return
    columns = ["model"]
    seen = {"model"}
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            columns.append(key)
    with (output_dir / "merged_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{column: row.get(column, "") for column in columns} for row in rows])
    with (output_dir / "merged_comparison.md").open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(row.get(col, "") for col in columns) + " |\n")


def build_merged_summary(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.summary_output)
    otp_rows = _read_csv_rows(Path(args.otp2_benchmark_output) / "summary.csv") if not args.skip_otp2 else []
    lwir_rows = _read_csv_rows(Path(args.lwir_output) / "summary.csv") if not args.skip_lwir else []
    uch_rows = _read_csv_rows(Path(args.uch_output) / "summary.csv") if not args.skip_uch else []

    merged: dict[str, dict[str, str]] = {}
    for row in otp_rows:
        model = row["model_name"]
        merged.setdefault(model, {"model": model})
        merged[model]["otp2_pose_pck20"] = row.get("pose_pck20_test", "")
        merged[model]["otp2_pose_pckh50"] = row.get("pose_pckh50_test", "")
        merged[model]["otp2_joint_score"] = row.get("joint_score_test", "")
        merged[model]["otp2_joint_pckh50"] = row.get("joint_pckh50_test", "")
        merged[model]["otp2_pose_map50_95"] = row.get("pose_map50_95_test", "")
        merged[model]["otp2_gpu_joint_fps"] = row.get("gpu_joint_fps_images", "")
        merged[model]["otp2_gpu_joint_latency_ms"] = row.get("gpu_joint_latency_ms_test", "")
    for row in lwir_rows:
        model = row["model"]
        merged.setdefault(model, {"model": model})
        merged[model]["lwir_pose_pck20"] = row.get("pose_pck20", "")
        merged[model]["lwir_pose_pckh50"] = row.get("pose_pckh50", "")
        merged[model]["lwir_joint_score"] = row.get("joint_score", "")
        merged[model]["lwir_joint_pckh50"] = row.get("joint_pckh50", "")
        merged[model]["lwir_joint_fps"] = row.get("joint_fps", "")
        merged[model]["lwir_joint_latency_ms"] = row.get("joint_latency_ms", "")
    for row in uch_rows:
        model = row["model"]
        merged.setdefault(model, {"model": model})
        merged[model]["uch_pose_pck20"] = row.get("pose_pck20", "")
        merged[model]["uch_pose_pckh50"] = row.get("pose_pckh50", "")
        merged[model]["uch_joint_score"] = row.get("joint_score", "")
        merged[model]["uch_joint_pckh50"] = row.get("joint_pckh50", "")
        merged[model]["uch_joint_fps"] = row.get("joint_fps", "")
        merged[model]["uch_joint_latency_ms"] = row.get("joint_latency_ms", "")

    rows = list(merged.values())
    rows.sort(key=lambda row: float(row.get("otp2_joint_score", "0") or "0"), reverse=True)
    _write_table(rows, output_dir)


def main() -> None:
    args = parse_args()
    models = args.models
    root = Path(__file__).resolve().parent

    if not args.skip_otp2:
        _run(
            [
                sys.executable,
                str(root / "run_pose_direct_yolov5n_suite.py"),
                "--models",
                models,
                "--pretrain-config",
                args.otp2_pretrain_config,
                "--finetune-config",
                args.otp2_finetune_config,
                "--train-output",
                args.otp2_train_output,
                "--benchmark-output",
                args.otp2_benchmark_output,
            ]
            + sum([["--pose-pretrain-source", item] for item in args.pose_pretrain_source], [])
            + [
                "--yolov5-repo",
                args.yolov5_repo,
                "--yolov5-weights",
                args.yolov5_weights,
                "--test-images-dir",
                args.test_images_dir,
                "--test-labels-dir",
                args.test_labels_dir,
                "--gpu-device",
                args.gpu_device,
                "--cpu-device",
                args.cpu_device,
                "--visual-samples",
                str(args.visual_samples),
                "--warmup",
                str(args.warmup),
                "--gpu-only",
            ]
            + sum([["--search-root", item] for item in args.search_root], [])
        )
    if not args.skip_lwir:
        _run(
            [
                sys.executable,
                str(root / "run_pose_direct_yolov5n_transfer_suite.py"),
                "--config",
                args.lwir_config,
                "--source-suite",
                args.otp2_train_output,
                "--output",
                args.lwir_output,
                "--models",
                models,
            ]
        )
    if not args.skip_uch:
        _run(
            [
                sys.executable,
                str(root / "run_pose_direct_yolov5n_transfer_suite.py"),
                "--config",
                args.uch_config,
                "--source-suite",
                args.otp2_train_output,
                "--output",
                args.uch_output,
                "--models",
                models,
            ]
        )
    build_merged_summary(args)


if __name__ == "__main__":
    main()
