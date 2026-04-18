from __future__ import annotations

import argparse

import torch

from lite_therm_pose import load_config
from lite_therm_pose.evaluate import evaluate_detector_model, evaluate_joint_pipeline, evaluate_pose_model
from lite_therm_pose.trainers import build_detector, build_pose_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detector-only, pose-only, and joint pipeline.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--detector-weights", type=str, required=True)
    parser.add_argument("--pose-weights", type=str, required=True)
    parser.add_argument("--model", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.model:
        cfg.model.name = args.model
    detector, _ = build_detector(cfg)
    pose_model, _ = build_pose_model(cfg)
    detector.load_state_dict(torch.load(args.detector_weights, map_location="cpu")["model"])
    pose_model.load_state_dict(torch.load(args.pose_weights, map_location="cpu")["model"])
    print(evaluate_detector_model(detector, cfg))
    print(evaluate_pose_model(pose_model, cfg))
    print(evaluate_joint_pipeline(detector, pose_model, cfg))


if __name__ == "__main__":
    main()
