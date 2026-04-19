from __future__ import annotations

from typing import Any

import torch


def _extract_state_dict(payload: dict[str, Any] | torch.Tensor) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def _adapt_input_conv(source: torch.Tensor, target_shape: torch.Size) -> torch.Tensor | None:
    if source.ndim != 4 or len(target_shape) != 4:
        return None
    if source.shape[0] != target_shape[0] or source.shape[2:] != target_shape[2:]:
        return None
    if source.shape[1] == target_shape[1]:
        return source
    if source.shape[1] == 3 and target_shape[1] == 1:
        # Fallback path for older RGB checkpoints; current training is grayscale-first.
        return source.mean(dim=1, keepdim=True)
    if source.shape[1] == 1 and target_shape[1] == 3:
        return source.repeat(1, 3, 1, 1) / 3.0
    return None


def load_flexible_state_dict(model: torch.nn.Module, checkpoint_path: str) -> dict[str, list[str]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(payload)
    model_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    adapted: list[str] = []

    for name, tensor in state_dict.items():
        if name not in model_state:
            skipped.append(name)
            continue
        target = model_state[name]
        if tensor.shape == target.shape:
            compatible[name] = tensor
            continue
        adapted_tensor = _adapt_input_conv(tensor, target.shape)
        if adapted_tensor is not None and adapted_tensor.shape == target.shape:
            compatible[name] = adapted_tensor
            adapted.append(name)
            continue
        skipped.append(name)

    model.load_state_dict(compatible, strict=False)
    return {"adapted": adapted, "skipped": skipped}
