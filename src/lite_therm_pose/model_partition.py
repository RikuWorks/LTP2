from __future__ import annotations

from math import ceil


def partition_models(model_names: list[str], num_machines: int) -> list[list[str]]:
    if num_machines <= 0:
        raise ValueError("num_machines must be positive")
    chunk = ceil(len(model_names) / num_machines)
    return [model_names[i * chunk : (i + 1) * chunk] for i in range(num_machines)]


def assigned_models(model_names: list[str], machine_index: int, num_machines: int) -> list[str]:
    parts = partition_models(model_names, num_machines)
    if machine_index < 0 or machine_index >= len(parts):
        raise ValueError(f"machine_index must be in [0, {len(parts) - 1}]")
    return parts[machine_index]
