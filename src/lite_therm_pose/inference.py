from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .checkpoints import load_flexible_state_dict
from .config import DatasetConfig, DetectorConfig
from .models.detector import TinyPersonDetector, decode_detections
from .models.pose_topdown import TopDownPoseCNN, decode_pose
from .utils import draw_pose, resize_and_normalize


@dataclass
class RuntimeBundle:
    detector: TinyPersonDetector
    pose_model: TopDownPoseCNN
    detector_device: torch.device
    pose_device: torch.device
    dataset_cfg: DatasetConfig
    detector_cfg: DetectorConfig
    draw_parts: bool = True


def load_weights(model: torch.nn.Module, checkpoint_path: str) -> None:
    load_flexible_state_dict(model, checkpoint_path)


@torch.no_grad()
def run_topdown_inference(bundle: RuntimeBundle, frame: np.ndarray) -> np.ndarray:
    detector_input, scale_x, scale_y = resize_and_normalize(frame, bundle.detector_cfg.image_size, bundle.dataset_cfg.grayscale)
    det_tensor = torch.from_numpy(detector_input.transpose(2, 0, 1)).unsqueeze(0).to(bundle.detector_device)
    det_preds = bundle.detector(det_tensor)
    dets = decode_detections(
        det_preds,
        stride=bundle.detector_cfg.stride,
        score_threshold=bundle.detector_cfg.score_threshold,
        nms_iou_threshold=bundle.detector_cfg.nms_iou_threshold,
        max_detections=bundle.detector_cfg.max_detections,
    )[0]
    visual = frame.copy()
    if visual.ndim == 3 and visual.shape[2] == 1:
        visual = cv2.cvtColor(visual, cv2.COLOR_GRAY2BGR)
    elif visual.ndim == 2:
        visual = cv2.cvtColor(visual, cv2.COLOR_GRAY2BGR)
    if dets.boxes.numel() == 0:
        return visual
    crops = []
    crop_boxes = []
    for box in dets.boxes.cpu().tolist():
        x1, y1, x2, y2 = box
        x1 = int(round(x1 / max(scale_x, 1.0e-6)))
        y1 = int(round(y1 / max(scale_y, 1.0e-6)))
        x2 = int(round(x2 / max(scale_x, 1.0e-6)))
        y2 = int(round(y2 / max(scale_y, 1.0e-6)))
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(max(x2, x1 + 1), frame.shape[1])
        y2 = min(max(y2, y1 + 1), frame.shape[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_norm, _, _ = resize_and_normalize(crop, bundle.dataset_cfg.image_size, bundle.dataset_cfg.grayscale)
        crops.append(crop_norm.transpose(2, 0, 1))
        crop_boxes.append([x1, y1, x2, y2])
    if not crops:
        return visual
    crop_tensor = torch.from_numpy(np.stack(crops)).to(bundle.pose_device)
    pose_preds = bundle.pose_model(crop_tensor)
    poses = decode_pose(
        pose_preds,
        crop_boxes=torch.tensor(crop_boxes, device=bundle.pose_device, dtype=torch.float32),
        image_size=bundle.dataset_cfg.image_size,
        body_parts=bundle.dataset_cfg.body_parts,
    )
    for det_box, det_score, pose in zip(crop_boxes, dets.scores.cpu().tolist(), poses):
        x1, y1, x2, y2 = det_box
        cv2.rectangle(visual, (x1, y1), (x2 - 1, y2 - 1), (255, 120, 0), 2)
        cv2.putText(visual, f"person {det_score:.2f}", (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 1)
        keypoints = [
            (float(x), float(y), float(score))
            for (x, y), score in zip(pose.keypoints.tolist(), pose.keypoint_scores.tolist())
        ]
        visual = draw_pose(visual, keypoints, pose.body_parts if bundle.draw_parts else None)
    return visual
