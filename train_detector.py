from __future__ import annotations

import argparse

from lite_therm_pose import load_config
from lite_therm_pose.trainers import train_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight thermal person detector.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/detector")
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.model:
        cfg.model.name = args.model
    train_detector(cfg, args.output, weights=args.weights)


if __name__ == "__main__":
    main()
