from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    family: str
    width_mult: float
    block_type: str
    stem_channels: int
    neck_channels: int
    decoder_channels: tuple[int, int, int]


MODEL_SPECS: dict[str, BackboneSpec] = {
    "dsconv_nano": BackboneSpec("dsconv_nano", "flex", 0.25, "dsconv", 8, 40, (56, 40, 24)),
    "dsconv_pico": BackboneSpec("dsconv_pico", "flex", 0.35, "dsconv", 12, 48, (64, 48, 32)),
    "dsconv_xs": BackboneSpec("dsconv_xs", "flex", 0.50, "dsconv", 16, 64, (96, 72, 48)),
    "dsconv_s": BackboneSpec("dsconv_s", "flex", 0.75, "dsconv", 20, 80, (128, 96, 64)),
    "dsconv_m": BackboneSpec("dsconv_m", "flex", 1.00, "dsconv", 24, 96, (160, 128, 80)),
    "dsconv_l": BackboneSpec("dsconv_l", "flex", 1.25, "dsconv", 28, 112, (192, 144, 96)),
    "dsconv_xl": BackboneSpec("dsconv_xl", "flex", 1.50, "dsconv", 32, 128, (224, 160, 112)),
    "resds_xs": BackboneSpec("resds_xs", "flex", 0.50, "residual", 16, 72, (112, 80, 56)),
    "resds_s": BackboneSpec("resds_s", "flex", 0.75, "residual", 20, 88, (144, 104, 72)),
    "resds_m": BackboneSpec("resds_m", "flex", 1.00, "residual", 24, 104, (176, 128, 88)),
    "resds_l": BackboneSpec("resds_l", "flex", 1.25, "residual", 28, 120, (208, 152, 104)),
    "resds_xl": BackboneSpec("resds_xl", "flex", 1.50, "residual", 32, 136, (240, 176, 120)),
    "hrnet_w18_small": BackboneSpec("hrnet_w18_small", "hrnet", 0.75, "hrnet", 18, 96, (160, 128, 96)),
    "hrnet_w18": BackboneSpec("hrnet_w18", "hrnet", 1.00, "hrnet", 18, 112, (192, 144, 112)),
    "hrnet_w32": BackboneSpec("hrnet_w32", "hrnet", 1.50, "hrnet", 32, 144, (224, 176, 128)),
    "unet_tiny": BackboneSpec("unet_tiny", "unet", 0.75, "unet", 16, 88, (144, 104, 72)),
    "unet_small": BackboneSpec("unet_small", "unet", 1.00, "unet", 24, 104, (176, 128, 88)),
    "hourglass_tiny": BackboneSpec("hourglass_tiny", "hourglass", 0.75, "hourglass", 16, 88, (144, 104, 72)),
    "hourglass_small": BackboneSpec("hourglass_small", "hourglass", 1.00, "hourglass", 24, 104, (176, 128, 88)),
    "fpn_tiny": BackboneSpec("fpn_tiny", "fpn", 0.75, "fpn", 16, 96, (144, 104, 72)),
    "fpn_small": BackboneSpec("fpn_small", "fpn", 1.00, "fpn", 24, 112, (176, 128, 88)),
    "csp_pico": BackboneSpec("csp_pico", "csp", 0.50, "csp", 12, 64, (88, 64, 40)),
    "csp_tiny": BackboneSpec("csp_tiny", "csp", 0.75, "csp", 16, 96, (144, 104, 72)),
    "pose_resnet50": BackboneSpec("pose_resnet50", "resnet", 1.25, "bottleneck", 32, 144, (224, 176, 128)),
    "pose_resnet101": BackboneSpec("pose_resnet101", "resnet", 1.50, "bottleneck", 40, 160, (256, 192, 144)),
    "pose_resnet50_se": BackboneSpec("pose_resnet50_se", "seresnet", 1.25, "se_bottleneck", 32, 152, (232, 184, 136)),
    "pose_resnet101_se": BackboneSpec("pose_resnet101_se", "seresnet", 1.50, "se_bottleneck", 40, 168, (264, 200, 152)),
    "pose_resnet152_se": BackboneSpec("pose_resnet152_se", "seresnet", 1.72, "se_bottleneck", 44, 184, (288, 216, 168)),
    "pose_resnet101_se_fuse": BackboneSpec("pose_resnet101_se_fuse", "seresnet", 1.60, "se_bottleneck", 40, 176, (272, 208, 160)),
    "pose_resnet50_se_thermal": BackboneSpec("pose_resnet50_se_thermal", "thermresnet", 1.30, "thermal_se", 32, 160, (240, 192, 144)),
    "pose_resnet101_se_thermal": BackboneSpec("pose_resnet101_se_thermal", "thermresnet", 1.55, "thermal_se", 40, 176, (272, 208, 160)),
    "pose_resnet152_se_thermal": BackboneSpec("pose_resnet152_se_thermal", "thermresnet", 1.80, "thermal_se", 44, 192, (296, 224, 176)),
    "pose_resnet101_se_thermal_ppm": BackboneSpec("pose_resnet101_se_thermal_ppm", "thermresnet", 1.65, "thermal_se", 40, 184, (280, 216, 168)),
    "pose_resnet50_se_direct": BackboneSpec("pose_resnet50_se_direct", "seresnet", 1.25, "se_bottleneck", 32, 152, (232, 184, 136)),
    "pose_resnet50_se_direct_refine": BackboneSpec("pose_resnet50_se_direct_refine", "seresnet", 1.35, "se_bottleneck", 36, 160, (248, 192, 144)),
    "pose_resnet101_se_direct": BackboneSpec("pose_resnet101_se_direct", "seresnet", 1.50, "se_bottleneck", 40, 168, (264, 200, 152)),
    "pose_resnet152_se_direct": BackboneSpec("pose_resnet152_se_direct", "seresnet", 1.72, "se_bottleneck", 44, 184, (288, 216, 168)),
    "pose_resnet50_se_thermal_direct": BackboneSpec("pose_resnet50_se_thermal_direct", "thermresnet", 1.30, "thermal_se", 32, 160, (240, 192, 144)),
    "pose_resnet101_se_thermal_direct": BackboneSpec("pose_resnet101_se_thermal_direct", "thermresnet", 1.55, "thermal_se", 40, 176, (272, 208, 160)),
    "pose_resnet152_se_thermal_direct": BackboneSpec("pose_resnet152_se_thermal_direct", "thermresnet", 1.80, "thermal_se", 44, 192, (296, 224, 176)),
    "pose_resnet101_se_thermal_elite_ppm": BackboneSpec("pose_resnet101_se_thermal_elite_ppm", "thermresnet", 1.74, "thermal_se", 44, 198, (304, 232, 184)),
    "pose_resnet101_se_thermal_dualpath_ppm": BackboneSpec("pose_resnet101_se_thermal_dualpath_ppm", "thermresnet", 1.72, "thermal_se", 44, 192, (296, 224, 176)),
    "pose_resnet101_se_thermal_dualpath_refine_ppm": BackboneSpec("pose_resnet101_se_thermal_dualpath_refine_ppm", "thermresnet", 1.78, "thermal_se", 44, 200, (304, 232, 184)),
    "pose_resnet101_se_thermal_dualpath_modulated_ppm": BackboneSpec("pose_resnet101_se_thermal_dualpath_modulated_ppm", "thermresnet", 1.80, "thermal_se", 44, 204, (312, 240, 184)),
    "pose_resnet101_se_thermal_ppm_xh": BackboneSpec("pose_resnet101_se_thermal_ppm_xh", "thermresnet", 1.85, "thermal_se", 48, 208, (320, 256, 192)),
    "hrnet_w48": BackboneSpec("hrnet_w48", "hrnet", 2.00, "hrnet", 48, 176, (256, 208, 160)),
    "higherhrnet_w32": BackboneSpec("higherhrnet_w32", "higherhrnet", 1.50, "higherhrnet", 32, 160, (240, 192, 144)),
    "thermhr_csp_bifpn": BackboneSpec("thermhr_csp_bifpn", "thermhybrid", 1.10, "thermhybrid", 24, 136, (208, 152, 104)),
    "stacked_hourglass_large": BackboneSpec("stacked_hourglass_large", "hourglass", 1.50, "hourglass", 32, 144, (224, 176, 128)),
}

PAPER_GRADE_MODELS: list[str] = [
    "pose_resnet50",
    "pose_resnet101",
    "hrnet_w48",
    "higherhrnet_w32",
    "thermhr_csp_bifpn",
    "stacked_hourglass_large",
]

RESNET_PLUS_MODELS: list[str] = [
    "pose_resnet50",
    "pose_resnet101",
    "pose_resnet50_se",
    "pose_resnet101_se",
    "pose_resnet101_se_fuse",
]

RESNET_NOVEL_MODELS: list[str] = [
    "pose_resnet101_se",
    "pose_resnet50_se_thermal",
    "pose_resnet101_se_thermal",
    "pose_resnet101_se_thermal_ppm",
    "pose_resnet50_se_thermal_direct",
    "pose_resnet101_se_thermal_direct",
    "pose_resnet101_se_thermal_elite_ppm",
    "pose_resnet101_se_thermal_dualpath_ppm",
    "pose_resnet101_se_thermal_dualpath_refine_ppm",
    "pose_resnet101_se_thermal_dualpath_modulated_ppm",
    "pose_resnet101_se_thermal_ppm_xh",
]

DIRECT_POSE_MODELS: list[str] = [
    "pose_resnet50_se_direct",
    "pose_resnet101_se_direct",
    "pose_resnet152_se_direct",
    "pose_resnet50_se_thermal_direct",
    "pose_resnet101_se_thermal_direct",
]

ULTRALIGHT_MODELS: list[str] = [
    "dsconv_nano",
    "dsconv_pico",
    "csp_pico",
]


def get_model_spec(name: str) -> BackboneSpec:
    if name not in MODEL_SPECS:
        valid = ", ".join(MODEL_SPECS)
        raise KeyError(f"Unknown model '{name}'. Available models: {valid}")
    return MODEL_SPECS[name]
