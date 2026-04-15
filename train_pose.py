from __future__ import annotations

import argparse

import torch

from lite_therm_pose import load_config
from lite_therm_pose.data import PoseDataset
from lite_therm_pose.engine import make_loader, train_loop
from lite_therm_pose.models.pose_topdown import TopDownPoseCNN, pose_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a top-down thermal pose estimator.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/pose")
    parser.add_argument("--weights", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg.runtime.device)
    dataset = PoseDataset(cfg.dataset, cfg.augmentation, train=True)
    loader = make_loader(dataset, batch_size=cfg.optim.batch_size, workers=cfg.optim.workers)
    in_channels = 1 if cfg.dataset.grayscale else 3
    model = TopDownPoseCNN(
        num_keypoints=cfg.dataset.num_keypoints,
        num_parts=len(cfg.dataset.body_parts),
        in_channels=in_channels,
    ).to(device)
    if args.weights:
        payload = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(payload["model"] if "model" in payload else payload, strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    train_loop(model, loader, optimizer, pose_loss, device, cfg.optim.epochs, args.output, "pose_last.pt")


if __name__ == "__main__":
    main()
