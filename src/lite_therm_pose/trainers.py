from __future__ import annotations

from pathlib import Path

import torch

from .config import ExperimentConfig
from .data import DetectorDataset, PoseDataset
from .engine import make_loader, train_loop
from .models.detector import TinyPersonDetector, detector_loss
from .models.pose_topdown import TopDownPoseCNN, pose_loss
from .runtime import resolve_model_device


def build_detector(cfg: ExperimentConfig) -> tuple[TinyPersonDetector, torch.device]:
    device = resolve_model_device(cfg.runtime.detector_device, cfg.runtime.device, "detector")
    in_channels = 1 if cfg.dataset.grayscale else 3
    return TinyPersonDetector(in_channels=in_channels, model_name=cfg.model.name).to(device), device


def build_pose_model(cfg: ExperimentConfig) -> tuple[TopDownPoseCNN, torch.device]:
    device = resolve_model_device(cfg.runtime.pose_device, cfg.runtime.device, "pose")
    in_channels = 1 if cfg.dataset.grayscale else 3
    return TopDownPoseCNN(
        num_keypoints=cfg.dataset.num_keypoints,
        num_parts=len(cfg.dataset.body_parts),
        in_channels=in_channels,
        model_name=cfg.model.name,
    ).to(device), device


def train_detector(cfg: ExperimentConfig, output_dir: str | Path, weights: str = "") -> tuple[TinyPersonDetector, dict[str, float | list[float]]]:
    model, device = build_detector(cfg)
    if weights:
        payload = torch.load(weights, map_location="cpu")
        model.load_state_dict(payload["model"] if "model" in payload else payload, strict=False)
    dataset = DetectorDataset(cfg.dataset, cfg.detector, cfg.augmentation)
    loader = make_loader(dataset, batch_size=cfg.optim.batch_size, workers=cfg.optim.workers, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    stats = train_loop(model, loader, optimizer, detector_loss, device, cfg.optim.epochs, output_dir, "detector_last.pt")
    return model, stats


def train_pose(cfg: ExperimentConfig, output_dir: str | Path, weights: str = "") -> tuple[TopDownPoseCNN, dict[str, float | list[float]]]:
    model, device = build_pose_model(cfg)
    if weights:
        payload = torch.load(weights, map_location="cpu")
        model.load_state_dict(payload["model"] if "model" in payload else payload, strict=False)
    dataset = PoseDataset(cfg.dataset, cfg.augmentation, train=True)
    loader = make_loader(dataset, batch_size=cfg.optim.batch_size, workers=cfg.optim.workers, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    stats = train_loop(model, loader, optimizer, pose_loss, device, cfg.optim.epochs, output_dir, "pose_last.pt")
    return model, stats
