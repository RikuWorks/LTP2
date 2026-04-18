from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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
) -> list[float]:
    model.train()
    history: list[float] = []
    for epoch in range(1, epochs + 1):
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
        save_checkpoint(output_dir, checkpoint_name, model, optimizer, epoch)
    return history
