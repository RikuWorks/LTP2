from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def dataclass_to_dict(obj):
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def gaussian2d(shape: tuple[int, int], sigma: float) -> np.ndarray:
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap: np.ndarray, center: tuple[int, int], radius: int) -> None:
    diameter = 2 * radius + 1
    gaussian = gaussian2d((diameter, diameter), sigma=diameter / 6.0)
    x, y = center
    height, width = heatmap.shape[:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    tl = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    br = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    return inter / (area1[:, None] + area2[None, :] - inter + 1.0e-6)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        current = int(order[0])
        keep.append(current)
        if order.numel() == 1:
            break
        iou = box_iou_xyxy(boxes[current].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def resize_and_normalize(image: np.ndarray, size: tuple[int, int], grayscale: bool) -> tuple[np.ndarray, float, float]:
    target_h, target_w = size
    src_h, src_w = image.shape[:2]
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    if grayscale and resized.ndim == 2:
        resized = resized[:, :, None]
    if grayscale and resized.shape[2] == 1:
        normalized = resized.astype(np.float32) / 255.0
    else:
        normalized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return normalized, target_w / src_w, target_h / src_h


def skeleton_edges() -> list[tuple[int, int]]:
    return [
        (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 6), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (0, 1), (0, 2), (1, 3), (2, 4),
    ]


def draw_pose(
    image: np.ndarray,
    keypoints: Iterable[tuple[float, float, float]],
    parts: dict[str, tuple[int, int]] | None = None,
) -> np.ndarray:
    canvas = image.copy()
    pts = list(keypoints)
    for idx1, idx2 in skeleton_edges():
        if idx1 >= len(pts) or idx2 >= len(pts):
            continue
        x1, y1, c1 = pts[idx1]
        x2, y2, c2 = pts[idx2]
        if min(c1, c2) > 0.15:
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
    for x, y, conf in pts:
        if conf > 0.15:
            cv2.circle(canvas, (int(x), int(y)), 3, (0, 128, 255), -1)
    if parts:
        for name, (x, y) in parts.items():
            cv2.putText(canvas, name, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
    return canvas
