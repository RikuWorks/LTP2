from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import ChannelAttention, build_backbone
from .pose_topdown import PosePrediction, TopDownPoseCNN, decode_pose as decode_heatmap_pose, pose_loss as heatmap_pose_loss


BODY_PART_GROUPS: dict[str, tuple[int, ...]] = {
    "head": (0, 1, 2, 3, 4),
    "torso": (5, 6, 11, 12),
    "left_arm": (5, 7, 9),
    "right_arm": (6, 8, 10),
    "left_leg": (11, 13, 15),
    "right_leg": (12, 14, 16),
}


def is_direct_pose_model_name(model_name: str) -> bool:
    return model_name.endswith("_direct") or "_direct_" in model_name


def is_direct_pose_model(model: nn.Module) -> bool:
    return bool(getattr(model, "is_direct_regression", False))


def _init_linear(module: nn.Module) -> None:
    for item in module.modules():
        if isinstance(item, nn.Linear):
            nn.init.trunc_normal_(item.weight, std=0.02)
            if item.bias is not None:
                nn.init.zeros_(item.bias)


class DirectPoseRegressor(nn.Module):
    def __init__(self, num_keypoints: int, num_parts: int, in_channels: int = 1, model_name: str = "pose_resnet101_se_thermal_direct") -> None:
        del num_parts
        super().__init__()
        self.is_direct_regression = True
        self.num_keypoints = num_keypoints
        self.backbone = build_backbone(model_name, in_channels=in_channels)
        hidden = max(self.backbone.out_channels // 2, 256)
        low_hidden = max(self.backbone.low_level_channels, 64)
        fused_dim = self.backbone.out_channels + low_hidden

        self.high_pool = nn.AdaptiveAvgPool2d(1)
        self.low_pool = nn.AdaptiveAvgPool2d(1)
        self.high_attn = ChannelAttention(self.backbone.out_channels)
        self.low_proj = nn.Sequential(
            nn.Conv2d(self.backbone.low_level_channels, low_hidden, kernel_size=1, bias=False),
            nn.GroupNorm(8 if low_hidden >= 8 else 1, low_hidden),
            nn.GELU(),
        )
        self.pre_norm = nn.LayerNorm(fused_dim)
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.xy_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden // 2, num_keypoints * 2),
        )
        self.visibility_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden // 2, num_keypoints),
        )
        _init_linear(self.trunk)
        _init_linear(self.xy_head)
        _init_linear(self.visibility_head)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        high, low = self.backbone(x)
        high = self.high_attn(high)
        pooled_high = self.high_pool(high).flatten(1)
        pooled_low = self.low_pool(self.low_proj(low)).flatten(1)
        fused = self.pre_norm(torch.cat([pooled_high, pooled_low], dim=1))
        feat = self.trunk(fused)
        keypoint_xy = torch.sigmoid(self.xy_head(feat)).view(batch_size, self.num_keypoints, 2)
        visibility = self.visibility_head(feat)
        return {
            "keypoint_xy": keypoint_xy,
            "visibility": visibility,
        }


def _normalized_targets(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    crop_boxes = batch["crop_box"].float()
    keypoint_xy = batch["keypoint_xy"].float()
    x1 = crop_boxes[:, 0].unsqueeze(1)
    y1 = crop_boxes[:, 1].unsqueeze(1)
    widths = (crop_boxes[:, 2] - crop_boxes[:, 0]).clamp(min=1.0).unsqueeze(1)
    heights = (crop_boxes[:, 3] - crop_boxes[:, 1]).clamp(min=1.0).unsqueeze(1)
    norm_x = ((keypoint_xy[:, :, 0] - x1) / widths).clamp(0.0, 1.0)
    norm_y = ((keypoint_xy[:, :, 1] - y1) / heights).clamp(0.0, 1.0)
    return torch.stack([norm_x, norm_y], dim=-1)


def direct_pose_loss(preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    targets = _normalized_targets(batch)
    visible = batch["keypoint_visible"].float()
    weights = batch["keypoint_weights"].float()
    coord_err = F.smooth_l1_loss(preds["keypoint_xy"], targets, reduction="none").sum(dim=-1)
    coord_weight = (weights * visible).clamp(min=0.0)
    coord_loss = (coord_err * coord_weight).sum() / coord_weight.sum().clamp(min=1.0)
    visibility_loss = F.binary_cross_entropy_with_logits(
        preds["visibility"],
        visible,
        weight=weights.clamp(min=0.25),
    )
    return coord_loss + 0.15 * visibility_loss


def _part_centers_from_keypoints(
    keypoints_xy: torch.Tensor,
    visibility: torch.Tensor,
    body_parts: list[str],
) -> dict[str, tuple[int, int]]:
    part_centers: dict[str, tuple[int, int]] = {}
    for name in body_parts:
        indices = BODY_PART_GROUPS.get(name, ())
        if not indices:
            continue
        coords = keypoints_xy[list(indices)]
        vis = visibility[list(indices)] >= 0.5
        if vis.any():
            coords = coords[vis]
        center = coords.mean(dim=0)
        part_centers[name] = (int(center[0].item()), int(center[1].item()))
    return part_centers


@torch.no_grad()
def decode_direct_pose(
    preds: dict[str, torch.Tensor],
    crop_boxes: torch.Tensor,
    image_size: tuple[int, int],
    body_parts: list[str],
) -> list[PosePrediction]:
    del image_size
    keypoint_xy = preds["keypoint_xy"].clamp(0.0, 1.0)
    visibility = preds["visibility"].sigmoid()
    out: list[PosePrediction] = []
    for xy_norm, vis, box in zip(keypoint_xy, visibility, crop_boxes):
        x1, y1, x2, y2 = box.tolist()
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
        xy = torch.empty_like(xy_norm)
        xy[:, 0] = x1 + xy_norm[:, 0] * width
        xy[:, 1] = y1 + xy_norm[:, 1] * height
        out.append(
            PosePrediction(
                keypoints=xy.detach().cpu(),
                keypoint_scores=vis.detach().cpu(),
                keypoint_visibility=vis.detach().cpu(),
                body_parts=_part_centers_from_keypoints(xy.detach().cpu(), vis.detach().cpu(), body_parts),
            )
        )
    return out


def create_pose_model(num_keypoints: int, num_parts: int, in_channels: int, model_name: str) -> nn.Module:
    if is_direct_pose_model_name(model_name):
        return DirectPoseRegressor(
            num_keypoints=num_keypoints,
            num_parts=num_parts,
            in_channels=in_channels,
            model_name=model_name,
        )
    return TopDownPoseCNN(
        num_keypoints=num_keypoints,
        num_parts=num_parts,
        in_channels=in_channels,
        model_name=model_name,
    )


def pose_objective(model: nn.Module):
    return direct_pose_loss if is_direct_pose_model(model) else heatmap_pose_loss


def decode_pose_outputs(
    model: nn.Module,
    preds: dict[str, torch.Tensor],
    crop_boxes: torch.Tensor,
    image_size: tuple[int, int],
    body_parts: list[str],
) -> list[PosePrediction]:
    if is_direct_pose_model(model):
        return decode_direct_pose(preds, crop_boxes, image_size, body_parts)
    return decode_heatmap_pose(preds, crop_boxes, image_size, body_parts)
