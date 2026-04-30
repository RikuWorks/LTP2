from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import AugmentationConfig, DatasetConfig, DetectorConfig
from ..utils import draw_gaussian, load_image, resize_and_normalize


@dataclass
class AnnotationRecord:
    image_path: Path
    image_id: int
    bbox: list[float]
    keypoints: list[float]
    category_id: int
    body_parts: dict[str, list[float]]


def load_coco_records(cfg: DatasetConfig) -> list[AnnotationRecord]:
    image_root = Path(cfg.image_root)
    with Path(cfg.annotation_file).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    images = {item["id"]: item for item in payload["images"]}
    records: list[AnnotationRecord] = []
    for ann in payload["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        if ann.get("category_id", cfg.person_category_id) != cfg.person_category_id:
            continue
        bbox = ann["bbox"]
        if min(bbox[2], bbox[3]) < cfg.min_box_size:
            continue
        keypoints = ann.get("keypoints", [0.0] * (cfg.num_keypoints * 3))
        body_parts = ann.get("body_parts", {})
        image_info = images[ann["image_id"]]
        records.append(
            AnnotationRecord(
                image_path=image_root / image_info["file_name"],
                image_id=ann["image_id"],
                bbox=bbox,
                keypoints=keypoints,
                category_id=ann["category_id"],
                body_parts=body_parts,
            )
        )
    return records


class DetectorDataset(Dataset):
    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        detector_cfg: DetectorConfig,
        aug_cfg: AugmentationConfig,
    ) -> None:
        del aug_cfg
        self.dataset_cfg = dataset_cfg
        self.detector_cfg = detector_cfg
        self.records = load_coco_records(dataset_cfg)
        self.output_h = detector_cfg.image_size[0] // detector_cfg.stride
        self.output_w = detector_cfg.image_size[1] // detector_cfg.stride

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        image = load_image(record.image_path, grayscale=self.dataset_cfg.grayscale)
        normalized, scale_x, scale_y = resize_and_normalize(image, self.detector_cfg.image_size, self.dataset_cfg.grayscale)
        bbox = np.array(record.bbox, dtype=np.float32)
        bbox[[0, 2]] *= scale_x
        bbox[[1, 3]] *= scale_y
        center_x = (bbox[0] + bbox[2] / 2.0) / self.detector_cfg.stride
        center_y = (bbox[1] + bbox[3] / 2.0) / self.detector_cfg.stride
        heatmap = np.zeros((1, self.output_h, self.output_w), dtype=np.float32)
        size = np.zeros((2, self.output_h, self.output_w), dtype=np.float32)
        offset = np.zeros((2, self.output_h, self.output_w), dtype=np.float32)
        mask = np.zeros((self.output_h, self.output_w), dtype=np.float32)
        cx_int, cy_int = int(center_x), int(center_y)
        radius = max(1, int(math.sqrt((bbox[2] * bbox[3]) / max(self.detector_cfg.stride ** 2, 1)) / 3))
        draw_gaussian(heatmap[0], (cx_int, cy_int), radius)
        if 0 <= cx_int < self.output_w and 0 <= cy_int < self.output_h:
            size[:, cy_int, cx_int] = np.array([bbox[2], bbox[3]], dtype=np.float32) / self.detector_cfg.stride
            offset[:, cy_int, cx_int] = [center_x - cx_int, center_y - cy_int]
            mask[cy_int, cx_int] = 1.0
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1))
        return {
            "image": tensor,
            "det_heatmap": torch.from_numpy(heatmap),
            "det_size": torch.from_numpy(size),
            "det_offset": torch.from_numpy(offset),
            "det_mask": torch.from_numpy(mask),
            "gt_bbox": torch.tensor([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=torch.float32),
        }


class PoseDataset(Dataset):
    def __init__(self, dataset_cfg: DatasetConfig, aug_cfg: AugmentationConfig, train: bool = True) -> None:
        self.dataset_cfg = dataset_cfg
        self.aug_cfg = aug_cfg
        self.train = train
        self.records = load_coco_records(dataset_cfg)
        self.part_map = {name: idx for idx, name in enumerate(dataset_cfg.body_parts)}
        self.head_indices = {0, 1, 2, 3, 4}
        self.extremity_indices = {9, 10, 15, 16}
        self.limb_indices = {5, 6, 7, 8, 11, 12, 13, 14}

    def _joint_weight_multiplier(self, joint_idx: int) -> float:
        if joint_idx in self.head_indices:
            return self.dataset_cfg.head_weight_boost
        if joint_idx in self.extremity_indices:
            return self.dataset_cfg.extremity_weight_boost
        if joint_idx in self.limb_indices:
            return self.dataset_cfg.limb_weight_boost
        return 1.0

    def __len__(self) -> int:
        return len(self.records)

    def _augment_crop(self, bbox: np.ndarray) -> np.ndarray:
        if not self.train:
            return bbox
        bbox = bbox.copy()
        if random.random() < self.aug_cfg.partial_crop_prob:
            side = random.choice(["left", "right", "top", "bottom"])
            if side == "left":
                bbox[0] += bbox[2] * 0.12
            elif side == "right":
                bbox[2] *= 0.88
            elif side == "top":
                bbox[1] += bbox[3] * 0.12
            else:
                bbox[3] *= 0.88
        return bbox

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        image = load_image(record.image_path, grayscale=self.dataset_cfg.grayscale)
        bbox = self._augment_crop(np.array(record.bbox, dtype=np.float32))
        keypoints = np.array(record.keypoints, dtype=np.float32).reshape(-1, 3)
        x, y, w, h = bbox.tolist()
        x1, y1 = max(int(x), 0), max(int(y), 0)
        x2, y2 = min(int(x + w), image.shape[1] - 1), min(int(y + h), image.shape[0] - 1)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = image
            x1, y1, x2, y2 = 0, 0, image.shape[1], image.shape[0]
        normalized, scale_x, scale_y = resize_and_normalize(crop, self.dataset_cfg.image_size, self.dataset_cfg.grayscale)
        heat_h, heat_w = self.dataset_cfg.heatmap_size
        kp_heatmaps = np.zeros((self.dataset_cfg.num_keypoints, heat_h, heat_w), dtype=np.float32)
        part_heatmaps = np.zeros((len(self.dataset_cfg.body_parts), heat_h, heat_w), dtype=np.float32)
        visible = np.zeros((self.dataset_cfg.num_keypoints,), dtype=np.float32)
        weights = np.zeros((self.dataset_cfg.num_keypoints,), dtype=np.float32)
        keypoint_xy = np.zeros((self.dataset_cfg.num_keypoints, 2), dtype=np.float32)
        for joint_idx, (px, py, vis) in enumerate(keypoints):
            keypoint_xy[joint_idx] = [px, py]
            if vis <= 0:
                continue
            local_x = (px - x1) * scale_x
            local_y = (py - y1) * scale_y
            map_x = int(local_x * heat_w / self.dataset_cfg.image_size[1])
            map_y = int(local_y * heat_h / self.dataset_cfg.image_size[0])
            if 0 <= map_x < heat_w and 0 <= map_y < heat_h:
                draw_gaussian(kp_heatmaps[joint_idx], (map_x, map_y), radius=2)
                visible[joint_idx] = 1.0
                base_weight = 1.0 if vis > 1 else 0.5
                weights[joint_idx] = base_weight * self._joint_weight_multiplier(joint_idx)
        for name, center in record.body_parts.items():
            if name not in self.part_map:
                continue
            px, py = center[:2]
            local_x = (px - x1) * scale_x
            local_y = (py - y1) * scale_y
            map_x = int(local_x * heat_w / self.dataset_cfg.image_size[1])
            map_y = int(local_y * heat_h / self.dataset_cfg.image_size[0])
            if 0 <= map_x < heat_w and 0 <= map_y < heat_h:
                draw_gaussian(part_heatmaps[self.part_map[name]], (map_x, map_y), radius=3)
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1))
        crop_box = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
        return {
            "image": tensor,
            "crop_box": crop_box,
            "keypoint_heatmaps": torch.from_numpy(kp_heatmaps),
            "part_heatmaps": torch.from_numpy(part_heatmaps),
            "keypoint_visible": torch.from_numpy(visible),
            "keypoint_weights": torch.from_numpy(weights),
            "keypoint_xy": torch.from_numpy(keypoint_xy),
            "bbox_size": torch.tensor([max(x2 - x1, 1), max(y2 - y1, 1)], dtype=torch.float32),
        }
