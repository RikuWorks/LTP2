from __future__ import annotations

from pathlib import Path

import torch

from .checkpoints import load_flexible_state_dict
from .config import ExperimentConfig
from .data import DetectorDataset, PoseDataset
from .engine import make_loader, train_loop
from .models.detector import TinyPersonDetector, detector_loss
from .models.pose_direct import create_pose_model, pose_objective
from .models.pose_topdown import TopDownPoseCNN
from .resume import load_initial_weights_if_needed, try_resume_training
from .runtime import resolve_model_device


def build_detector(cfg: ExperimentConfig) -> tuple[TinyPersonDetector, torch.device]:
    device = resolve_model_device(cfg.runtime.detector_device, cfg.runtime.device, "detector")
    in_channels = 1 if cfg.dataset.grayscale else 3
    model_name = cfg.model.detector_name or cfg.model.name
    return TinyPersonDetector(in_channels=in_channels, model_name=model_name).to(device), device


def build_pose_model(cfg: ExperimentConfig) -> tuple[TopDownPoseCNN, torch.device]:
    device = resolve_model_device(cfg.runtime.pose_device, cfg.runtime.device, "pose")
    in_channels = 1 if cfg.dataset.grayscale else 3
    model_name = cfg.model.pose_name or cfg.model.name
    return create_pose_model(
        num_keypoints=cfg.dataset.num_keypoints,
        num_parts=len(cfg.dataset.body_parts),
        in_channels=in_channels,
        model_name=model_name,
    ).to(device), device


def train_detector(cfg: ExperimentConfig, output_dir: str | Path, weights: str = "") -> tuple[TinyPersonDetector, dict[str, float | list[float]]]:
    model, device = build_detector(cfg)
    dataset = DetectorDataset(cfg.dataset, cfg.detector, cfg.augmentation)
    loader = make_loader(dataset, batch_size=cfg.optim.batch_size, workers=cfg.optim.workers, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    resume_info = try_resume_training(model, optimizer, output_dir, "detector_last.pt")
    warm_started = False
    if not resume_info["resumed"]:
        warm_started = load_initial_weights_if_needed(model, str(weights))
    stats = train_loop(
        model,
        loader,
        optimizer,
        detector_loss,
        device,
        cfg.optim.epochs,
        output_dir,
        "detector_last.pt",
        start_epoch=int(resume_info["start_epoch"]),
    )
    stats["auto_resumed"] = bool(resume_info["resumed"])
    stats["warm_started"] = warm_started
    return model, stats


def train_pose(cfg: ExperimentConfig, output_dir: str | Path, weights: str = "") -> tuple[TopDownPoseCNN, dict[str, float | list[float]]]:
    model, device = build_pose_model(cfg)
    dataset = PoseDataset(cfg.dataset, cfg.augmentation, train=True)
    loader = make_loader(dataset, batch_size=cfg.optim.batch_size, workers=cfg.optim.workers, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    resume_info = try_resume_training(model, optimizer, output_dir, "pose_last.pt")
    warm_started = False
    if not resume_info["resumed"]:
        warm_started = load_initial_weights_if_needed(model, str(weights))
    stats = train_loop(
        model,
        loader,
        optimizer,
        pose_objective(model),
        device,
        cfg.optim.epochs,
        output_dir,
        "pose_last.pt",
        start_epoch=int(resume_info["start_epoch"]),
    )
    stats["auto_resumed"] = bool(resume_info["resumed"])
    stats["warm_started"] = warm_started
    return model, stats


def load_trained_detector(cfg: ExperimentConfig, checkpoint_path: str | Path) -> TinyPersonDetector:
    model, _ = build_detector(cfg)
    load_flexible_state_dict(model, str(checkpoint_path))
    return model


def load_trained_pose(cfg: ExperimentConfig, checkpoint_path: str | Path) -> TopDownPoseCNN:
    model, _ = build_pose_model(cfg)
    load_flexible_state_dict(model, str(checkpoint_path))
    return model
