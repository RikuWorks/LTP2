from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import DepthwiseSeparableConv, build_backbone
from .specs import get_model_spec


@dataclass
class PosePrediction:
    keypoints: torch.Tensor
    keypoint_scores: torch.Tensor
    keypoint_visibility: torch.Tensor
    body_parts: dict[str, tuple[int, int]]


class TopDownPoseCNN(nn.Module):
    def __init__(self, num_keypoints: int, num_parts: int, in_channels: int = 1, model_name: str = "dsconv_s") -> None:
        super().__init__()
        spec = get_model_spec(model_name)
        c0, c1, c2 = spec.decoder_channels
        self.use_low_level_fusion = "dualpath" in model_name
        self.backbone = build_backbone(model_name, in_channels=in_channels)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.backbone.out_channels, c0, kernel_size=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DepthwiseSeparableConv(c0, c1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DepthwiseSeparableConv(c1, c2),
        )
        if self.use_low_level_fusion:
            self.low_level_proj = nn.Sequential(
                nn.Conv2d(self.backbone.low_level_channels, c2, kernel_size=1, bias=False),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
            )
            self.fusion_refine = nn.Sequential(
                DepthwiseSeparableConv(c2 * 2, c2),
                DepthwiseSeparableConv(c2, c2),
            )
        self.keypoint_head = nn.Conv2d(c2, num_keypoints, kernel_size=1)
        self.part_head = nn.Conv2d(c2, num_parts, kernel_size=1)
        self.visibility_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c2, num_keypoints),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features, low_level = self.backbone(x)
        decoded = self.decoder(features)
        target_h = max(x.shape[-2] // 4, 1)
        target_w = max(x.shape[-1] // 4, 1)
        if decoded.shape[-2:] != (target_h, target_w):
            decoded = F.interpolate(decoded, size=(target_h, target_w), mode="bilinear", align_corners=False)
        if self.use_low_level_fusion:
            low = self.low_level_proj(low_level)
            if low.shape[-2:] != decoded.shape[-2:]:
                low = F.interpolate(low, size=decoded.shape[-2:], mode="bilinear", align_corners=False)
            decoded = self.fusion_refine(torch.cat([decoded, low], dim=1))
        return {
            "keypoint_heatmaps": self.keypoint_head(decoded),
            "part_heatmaps": self.part_head(decoded),
            "visibility": self.visibility_head(decoded),
        }


def pose_loss(preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    heatmap_weights = batch["keypoint_weights"].unsqueeze(-1).unsqueeze(-1)
    heatmap_loss = ((preds["keypoint_heatmaps"] - batch["keypoint_heatmaps"]) ** 2 * heatmap_weights).mean()
    part_loss = F.mse_loss(preds["part_heatmaps"], batch["part_heatmaps"])
    visibility_loss = F.binary_cross_entropy_with_logits(
        preds["visibility"],
        batch["keypoint_visible"],
        weight=batch["keypoint_weights"].clamp(min=0.25),
    )
    return heatmap_loss + 0.5 * part_loss + 0.1 * visibility_loss


def _refine_peak(heatmap: torch.Tensor, px: int, py: int) -> tuple[float, float]:
    height, width = heatmap.shape
    refined_x = float(px)
    refined_y = float(py)
    if 1 <= px < width - 1:
        refined_x += float(torch.sign(heatmap[py, px + 1] - heatmap[py, px - 1]) * 0.25)
    if 1 <= py < height - 1:
        refined_y += float(torch.sign(heatmap[py + 1, px] - heatmap[py - 1, px]) * 0.25)
    return refined_x, refined_y


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
            refined_px, refined_py = _refine_peak(kp_map[idx], px, py)
            score = float(kp_map[idx, py, px] * vis[idx])
            x = x1 + (refined_px / max(heat_w - 1, 1)) * box_w
            y = y1 + (refined_py / max(heat_h - 1, 1)) * box_h
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
                keypoint_visibility=vis.detach().cpu(),
                body_parts=part_centers,
            )
        )
    return out
