from __future__ import annotations


PAPER_YOLO_DIRECT_MODELS = [
    "pose_resnet50_se_direct",
    "pose_resnet50_se_direct_refine",
    "pose_resnet101_se_direct",
    "pose_resnet50_se_thermal_direct",
    "pose_resnet101_se_thermal_direct",
]


PRETRAIN_SOURCE_CANDIDATES = {
    "pose_resnet50_se_direct": ["pose_resnet50_se_direct"],
    "pose_resnet50_se_direct_refine": ["pose_resnet50_se_direct_refine", "pose_resnet50_se_direct"],
    "pose_resnet101_se_direct": ["pose_resnet101_se_direct"],
    "pose_resnet50_se_thermal_direct": ["pose_resnet50_se_thermal_direct"],
    "pose_resnet101_se_thermal_direct": ["pose_resnet101_se_thermal_direct"],
}


def pretrain_source_candidates(model_name: str) -> list[str]:
    return PRETRAIN_SOURCE_CANDIDATES.get(model_name, [model_name])
