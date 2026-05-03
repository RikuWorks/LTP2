from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from lite_therm_pose import load_config
from lite_therm_pose.inference import RuntimeBundle, load_weights, run_topdown_inference
from lite_therm_pose.models.detector import TinyPersonDetector
from lite_therm_pose.models.pose_direct import create_pose_model
from lite_therm_pose.runtime import resolve_model_device
from lite_therm_pose.utils import ensure_dir


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run top-down thermal pose inference on all images in a directory.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--detector", type=str, required=True)
    parser.add_argument("--pose", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--draw-parts", dest="draw_parts", action="store_true")
    parser.add_argument("--no-draw-parts", dest="draw_parts", action="store_false")
    parser.set_defaults(draw_parts=None)
    return parser.parse_args()


def build_bundle(args: argparse.Namespace) -> RuntimeBundle:
    cfg = load_config(args.config)
    if args.model:
        cfg.model.name = args.model
    if args.draw_parts is not None:
        cfg.runtime.draw_parts = args.draw_parts
    detector_device = resolve_model_device(cfg.runtime.detector_device, cfg.runtime.device, "detector")
    pose_device = resolve_model_device(cfg.runtime.pose_device, cfg.runtime.device, "pose")
    in_channels = 1 if cfg.dataset.grayscale else 3
    detector_model_name = cfg.model.detector_name or cfg.model.name
    pose_model_name = cfg.model.pose_name or cfg.model.name
    detector = TinyPersonDetector(in_channels=in_channels, model_name=detector_model_name).to(detector_device).eval()
    pose_model = create_pose_model(
        num_keypoints=cfg.dataset.num_keypoints,
        num_parts=len(cfg.dataset.body_parts),
        in_channels=in_channels,
        model_name=pose_model_name,
    ).to(pose_device).eval()
    load_weights(detector, args.detector)
    load_weights(pose_model, args.pose)
    return RuntimeBundle(
        detector=detector,
        pose_model=pose_model,
        detector_device=detector_device,
        pose_device=pose_device,
        dataset_cfg=cfg.dataset,
        detector_cfg=cfg.detector,
        draw_parts=cfg.runtime.draw_parts,
    )


def iter_images(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    output_dir = ensure_dir(args.output_dir)
    bundle = build_bundle(args)
    images = iter_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No image files found in {input_dir}")
    for image_path in images:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"skip: {image_path}")
            continue
        if bundle.dataset_cfg.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, :, None]
        visual = run_topdown_inference(bundle, frame)
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), visual)
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
