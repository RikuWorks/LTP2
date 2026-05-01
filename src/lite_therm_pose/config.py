from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    image_root: str = ""
    annotation_file: str = ""
    val_image_root: str = ""
    val_annotation_file: str = ""
    image_size: tuple[int, int] = (256, 192)
    heatmap_size: tuple[int, int] = (64, 48)
    num_keypoints: int = 17
    body_parts: list[str] = field(
        default_factory=lambda: ["head", "torso", "left_arm", "right_arm", "left_leg", "right_leg"]
    )
    person_category_id: int = 1
    grayscale: bool = False
    single_person: bool = False
    min_box_size: float = 16.0
    head_weight_boost: float = 1.35
    extremity_weight_boost: float = 1.2
    limb_weight_boost: float = 1.1


@dataclass
class AugmentationConfig:
    flip_prob: float = 0.5
    scale_jitter: float = 0.25
    rotate_deg: float = 25.0
    partial_crop_prob: float = 0.3
    intensity_jitter: float = 0.15


@dataclass
class OptimConfig:
    batch_size: int = 16
    epochs: int = 60
    lr: float = 2.5e-3
    weight_decay: float = 1.0e-4
    workers: int = 4


@dataclass
class DetectorConfig:
    image_size: tuple[int, int] = (320, 320)
    stride: int = 8
    max_detections: int = 16
    score_threshold: float = 0.25
    nms_iou_threshold: float = 0.45


@dataclass
class ModelConfig:
    name: str = "dsconv_s"
    detector_name: str = ""
    pose_name: str = ""


@dataclass
class RuntimeConfig:
    device: str = "auto"
    detector_device: str = ""
    pose_device: str = ""
    compile: bool = False
    camera_id: int = 0
    draw_parts: bool = True


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if path is None:
        return cfg
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return _merge_dataclass(cfg, payload)
