from __future__ import annotations

import torch
from torch import nn

from .specs import BackboneSpec, MODEL_SPECS, get_model_spec


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
