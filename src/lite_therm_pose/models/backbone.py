from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualDSConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + x)


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


class FlexibleBackbone(nn.Module):
    def __init__(self, spec: BackboneSpec, in_channels: int = 1) -> None:
        super().__init__()
        ch = [max(spec.stem_channels, int(c * spec.width_mult)) for c in [24, 48, 96, 160]]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, ch[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(ch[0], ch[1], spec.block_type)
        self.stage2 = self._make_stage(ch[1], ch[2], spec.block_type)
        self.stage3 = self._make_stage(ch[2], ch[3], spec.block_type)
        self.out_channels = ch[3]
        self.low_level_channels = ch[1]

    def _make_stage(self, in_channels: int, out_channels: int, block_type: str) -> nn.Sequential:
        layers: list[nn.Module] = [
            DepthwiseSeparableConv(in_channels, in_channels),
            DepthwiseSeparableConv(in_channels, out_channels, stride=2),
        ]
        if block_type == "residual":
            layers.append(ResidualDSConv(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        low_level = self.stage1(x)
        x = self.stage2(low_level)
        x = self.stage3(x)
        return x, low_level


def get_model_spec(name: str) -> BackboneSpec:
    if name not in MODEL_SPECS:
        valid = ", ".join(sorted(MODEL_SPECS))
        raise KeyError(f"Unknown model '{name}'. Available models: {valid}")
    return MODEL_SPECS[name]
