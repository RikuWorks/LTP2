from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import DepthwiseSeparableConv, TinyBackbone


@dataclass
class PosePrediction:
    keypoints: torch.Tensor
    keypoint_scores: torch.Tensor
    body_parts: dict[str, tuple[int, int]]


class TopDownPoseCNN(nn.Module):
    def __init__(self, num_keypoints: int, num_parts: int, in_channels: int = 1, width_mult: float = 1.0) -> None:
        super().__init__()
        self.backbone = TinyBackbone(in_channels=in_channels, width_mult=width_mult)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.backbone.out_channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DepthwiseSeparableConv(128, 96),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DepthwiseSeparableConv(96, 64),
        )
        self.keypoint_head = nn.Conv2d(64, num_keypoints, kernel_size=1)
        self.part_head = nn.Conv2d(64, num_parts, kernel_size=1)
        self.visibility_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_keypoints),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features, _ = self.backbone(x)
        decoded = self.decoder(features)
        return {
            "keypoint_heatmaps": self.keypoint_head(decoded),
            "part_heatmaps": self.part_head(decoded),
            "visibility": self.visibility_head(decoded),
        }


def pose_loss(preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    heatmap_loss = F.mse_loss(preds["keypoint_heatmaps"], batch["keypoint_heatmaps"])
    part_loss = F.mse_loss(preds["part_heatmaps"], batch["part_heatmaps"])
    visibility_loss = F.binary_cross_entropy_with_logits(preds["visibility"], batch["keypoint_visible"])
    return heatmap_loss + 0.5 * part_loss + 0.1 * visibility_loss


@torch.no_grad()
def decode_pose(
    preds: dict[str, torch.Tensor],
    crop_boxes: torch.Tensor,
    image_size: tuple[int, int],
    body_parts: list[str],
) -> list[PosePrediction]:
    del image_size
    keypoint_maps = preds["keypoint_heatmaps"].sigmoid()
    part_maps = preds["part_heatmaps"].sigmoid()
    visibility = preds["visibility"].sigmoid()
    out: list[PosePrediction] = []
    for kp_map, part_map, vis, box in zip(keypoint_maps, part_maps, visibility, crop_boxes):
        heat_h, heat_w = kp_map.shape[-2:]
        x1, y1, x2, y2 = box.tolist()
        box_w = max(x2 - x1, 1.0)
        box_h = max(y2 - y1, 1.0)
        coords = []
        scores = []
        for idx in range(kp_map.shape[0]):
            flat_idx = kp_map[idx].argmax()
            py = int(flat_idx // heat_w)
            px = int(flat_idx % heat_w)
            score = float(kp_map[idx, py, px] * vis[idx])
            x = x1 + (px / max(heat_w - 1, 1)) * box_w
            y = y1 + (py / max(heat_h - 1, 1)) * box_h
            coords.append([x, y])
            scores.append(score)
        part_centers = {}
        for idx, name in enumerate(body_parts):
            flat_idx = part_map[idx].argmax()
            py = int(flat_idx // heat_w)
            px = int(flat_idx % heat_w)
            x = int(x1 + (px / max(heat_w - 1, 1)) * box_w)
            y = int(y1 + (py / max(heat_h - 1, 1)) * box_h)
            part_centers[name] = (x, y)
        out.append(
            PosePrediction(
                keypoints=torch.tensor(coords),
                keypoint_scores=torch.tensor(scores),
                body_parts=part_centers,
            )
        )
    return out
