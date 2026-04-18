from __future__ import annotations

import torch


def resolve_device(device_name: str) -> torch.device:
    normalized = device_name.lower().strip()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device is set to 'cuda' but CUDA is not available.")
    return torch.device(normalized)


def resolve_model_device(explicit_device: str, fallback_device: str, role: str) -> torch.device:
    chosen = explicit_device.strip() or fallback_device
    device = resolve_device(chosen)
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"{role} device is set to '{chosen}' but only {torch.cuda.device_count()} CUDA device(s) are available."
        )
    return device


def loader_pin_memory(device: torch.device) -> bool:
    return device.type == "cuda"
