from __future__ import annotations

import argparse

import torch

from lite_therm_pose import load_config
from lite_therm_pose.data import DetectorDataset
from lite_therm_pose.engine import make_loader, train_loop
from lite_therm_pose.models.detector import TinyPersonDetector, detector_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight thermal person detector.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/detector")
    parser.add_argument("--weights", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg.runtime.device)
    dataset = DetectorDataset(cfg.dataset, cfg.detector, cfg.augmentation)
    loader = make_loader(dataset, batch_size=cfg.optim.batch_size, workers=cfg.optim.workers)
    in_channels = 1 if cfg.dataset.grayscale else 3
    model = TinyPersonDetector(in_channels=in_channels).to(device)
    if args.weights:
        payload = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(payload["model"] if "model" in payload else payload, strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    train_loop(model, loader, optimizer, detector_loss, device, cfg.optim.epochs, args.output, "detector_last.pt")


if __name__ == "__main__":
    main()
