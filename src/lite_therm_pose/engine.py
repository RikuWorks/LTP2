from __future__ import annotations

from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .profile import peak_memory_mb, reset_peak_memory
from .runtime import loader_pin_memory
from .utils import ensure_dir, to_device


def make_loader(dataset, batch_size: int, workers: int, device: torch.device, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=loader_pin_memory(device),
        drop_last=False,
    )


def save_checkpoint(output_dir: str | Path, name: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    target_dir = ensure_dir(output_dir)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
        target_dir / name,
    )


def train_loop(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    epochs: int,
    output_dir: str | Path,
    checkpoint_name: str,
) -> dict[str, float | list[float]]:
    model.train()
    history: list[float] = []
    epoch_times: list[float] = []
    peak_memory = 0.0
    total_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        reset_peak_memory(device)
        epoch_start = time.perf_counter()
        running_loss = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch["image"])
            loss = loss_fn(preds, batch)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            progress.set_postfix(loss=f"{running_loss / max(progress.n, 1):.4f}")
        history.append(running_loss / max(len(loader), 1))
        epoch_times.append(time.perf_counter() - epoch_start)
        peak_memory = max(peak_memory, peak_memory_mb(device))
        save_checkpoint(output_dir, checkpoint_name, model, optimizer, epoch)
    total_seconds = time.perf_counter() - total_start
    return {
        "loss_history": history,
        "epoch_seconds_mean": sum(epoch_times) / max(len(epoch_times), 1),
        "train_seconds_total": total_seconds,
        "train_peak_memory_mb": peak_memory,
    }
