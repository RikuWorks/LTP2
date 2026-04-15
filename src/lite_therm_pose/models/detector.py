from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..utils import nms
from .backbone import TinyBackbone


@dataclass
class DetectorPrediction:
    boxes: torch.Tensor
    scores: torch.Tensor


class TinyPersonDetector(nn.Module):
    def __init__(self, in_channels: int = 1, width_mult: float = 1.0) -> None:
        super().__init__()
        self.backbone = TinyBackbone(in_channels=in_channels, width_mult=width_mult)
        channels = self.backbone.out_channels
        self.neck = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Conv2d(96, 1, kernel_size=1)
        self.size_head = nn.Conv2d(96, 2, kernel_size=1)
        self.offset_head = nn.Conv2d(96, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features, _ = self.backbone(x)
        fused = self.neck(features)
        return {
            "heatmap": self.heatmap_head(fused),
            "size": F.relu(self.size_head(fused)),
            "offset": self.offset_head(fused),
        }


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    pred = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    neg_weights = torch.pow(1 - target, beta)
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds
    num_pos = pos_inds.sum()
    if num_pos < 1:
        return neg_loss.sum()
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def regression_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(1).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
    denom = mask.sum().clamp(min=1.0)
    return loss / denom


def detector_loss(preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    hm_loss = focal_loss(preds["heatmap"], batch["det_heatmap"])
    size_loss = regression_l1_loss(preds["size"], batch["det_size"], batch["det_mask"])
    offset_loss = regression_l1_loss(preds["offset"], batch["det_offset"], batch["det_mask"])
    return hm_loss + 0.1 * size_loss + offset_loss


@torch.no_grad()
def decode_detections(
    preds: dict[str, torch.Tensor],
    stride: int,
    score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
) -> list[DetectorPrediction]:
    heatmap = preds["heatmap"].sigmoid()
    size = preds["size"]
    offset = preds["offset"]
    batch_predictions: list[DetectorPrediction] = []
    for hm, sz, off in zip(heatmap, size, offset):
        scores, indices = hm.view(-1).topk(k=min(max_detections * 8, hm.numel()))
        keep = scores > score_threshold
        scores = scores[keep]
        indices = indices[keep]
        if scores.numel() == 0:
            batch_predictions.append(DetectorPrediction(boxes=torch.zeros((0, 4)), scores=torch.zeros(0)))
            continue
        ys = (indices // hm.shape[-1]).float()
        xs = (indices % hm.shape[-1]).float()
        off_x = off[0].view(-1)[indices]
        off_y = off[1].view(-1)[indices]
        widths = sz[0].view(-1)[indices] * stride
        heights = sz[1].view(-1)[indices] * stride
        centers_x = (xs + off_x) * stride
        centers_y = (ys + off_y) * stride
        boxes = torch.stack(
            [
                centers_x - widths / 2.0,
                centers_y - heights / 2.0,
                centers_x + widths / 2.0,
                centers_y + heights / 2.0,
            ],
            dim=1,
        )
        keep_idx = nms(boxes, scores, nms_iou_threshold)[:max_detections]
        batch_predictions.append(DetectorPrediction(boxes=boxes[keep_idx], scores=scores[keep_idx]))
    return batch_predictions
