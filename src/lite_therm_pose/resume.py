from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .checkpoints import load_flexible_state_dict


def checkpoint_path(output_dir: str | Path, checkpoint_name: str) -> Path:
    return Path(output_dir) / checkpoint_name


def try_resume_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: str | Path,
    checkpoint_name: str,
) -> dict[str, Any]:
    path = checkpoint_path(output_dir, checkpoint_name)
    if not path.exists():
        return {"resumed": False, "start_epoch": 0, "path": str(path)}
    payload = torch.load(path, map_location="cpu")
    state = payload["model"] if "model" in payload else payload
    model.load_state_dict(state, strict=False)
    if isinstance(payload, dict) and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    start_epoch = int(payload.get("epoch", 0)) if isinstance(payload, dict) else 0
    return {"resumed": True, "start_epoch": start_epoch, "path": str(path)}


def load_initial_weights_if_needed(model: torch.nn.Module, weights: str) -> bool:
    if not weights:
        return False
    load_flexible_state_dict(model, weights)
    return True
