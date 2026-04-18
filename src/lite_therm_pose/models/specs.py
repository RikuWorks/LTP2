from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    width_mult: float
    block_type: str
    stem_channels: int
    neck_channels: int
    decoder_channels: tuple[int, int, int]


MODEL_SPECS: dict[str, BackboneSpec] = {
    "dsconv_xs": BackboneSpec("dsconv_xs", 0.50, "dsconv", 16, 64, (96, 72, 48)),
    "dsconv_s": BackboneSpec("dsconv_s", 0.75, "dsconv", 20, 80, (128, 96, 64)),
    "dsconv_m": BackboneSpec("dsconv_m", 1.00, "dsconv", 24, 96, (160, 128, 80)),
    "dsconv_l": BackboneSpec("dsconv_l", 1.25, "dsconv", 28, 112, (192, 144, 96)),
    "dsconv_xl": BackboneSpec("dsconv_xl", 1.50, "dsconv", 32, 128, (224, 160, 112)),
    "resds_xs": BackboneSpec("resds_xs", 0.50, "residual", 16, 72, (112, 80, 56)),
    "resds_s": BackboneSpec("resds_s", 0.75, "residual", 20, 88, (144, 104, 72)),
    "resds_m": BackboneSpec("resds_m", 1.00, "residual", 24, 104, (176, 128, 88)),
    "resds_l": BackboneSpec("resds_l", 1.25, "residual", 28, 120, (208, 152, 104)),
    "resds_xl": BackboneSpec("resds_xl", 1.50, "residual", 32, 136, (240, 176, 120)),
}


def get_model_spec(name: str) -> BackboneSpec:
    if name not in MODEL_SPECS:
        valid = ", ".join(sorted(MODEL_SPECS))
        raise KeyError(f"Unknown model '{name}'. Available models: {valid}")
    return MODEL_SPECS[name]
