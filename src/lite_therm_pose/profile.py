from __future__ import annotations

from pathlib import Path

import torch


def parameter_stats(model: torch.nn.Module, prefix: str) -> dict[str, float]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    size_mb = sum(param.numel() * param.element_size() for param in model.parameters()) / (1024 ** 2)
    return {
        f"{prefix}_params_m": total / 1_000_000.0,
        f"{prefix}_trainable_params_m": trainable / 1_000_000.0,
        f"{prefix}_model_size_mb": size_mb,
    }


def checkpoint_size_mb(path: str | Path) -> float:
    target = Path(path)
    if not target.exists():
        return 0.0
    return target.stat().st_size / (1024 ** 2)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
