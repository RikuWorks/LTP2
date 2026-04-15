from __future__ import annotations

import argparse

import cv2
import torch

from lite_therm_pose import load_config
from lite_therm_pose.inference import RuntimeBundle, load_weights, run_topdown_inference
from lite_therm_pose.models.detector import TinyPersonDetector
from lite_therm_pose.models.pose_topdown import TopDownPoseCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time CPU thermal top-down pose inference.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--detector", type=str, required=True)
    parser.add_argument("--pose", type=str, required=True)
    parser.add_argument("--source", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg.runtime.device)
    in_channels = 1 if cfg.dataset.grayscale else 3
    detector = TinyPersonDetector(in_channels=in_channels).to(device).eval()
    pose_model = TopDownPoseCNN(
        num_keypoints=cfg.dataset.num_keypoints,
        num_parts=len(cfg.dataset.body_parts),
        in_channels=in_channels,
    ).to(device).eval()
    load_weights(detector, args.detector)
    load_weights(pose_model, args.pose)
    stream_source = int(args.source) if args.source.isdigit() else args.source or cfg.runtime.camera_id
    cap = cv2.VideoCapture(stream_source)
    bundle = RuntimeBundle(
        detector=detector,
        pose_model=pose_model,
        device=device,
        dataset_cfg=cfg.dataset,
        detector_cfg=cfg.detector,
    )
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if cfg.dataset.grayscale and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, :, None]
        visual = run_topdown_inference(bundle, frame)
        cv2.imshow("LiteThermPose", visual)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
