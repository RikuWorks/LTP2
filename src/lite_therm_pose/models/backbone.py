from __future__ import annotations

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


class TinyBackbone(nn.Module):
    def __init__(self, in_channels: int = 1, width_mult: float = 1.0) -> None:
        super().__init__()
        ch = [max(16, int(c * width_mult)) for c in [24, 48, 96, 160]]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, ch[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(ch[0], ch[0]),
            DepthwiseSeparableConv(ch[0], ch[1], stride=2),
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(ch[1], ch[1]),
            DepthwiseSeparableConv(ch[1], ch[2], stride=2),
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(ch[2], ch[2]),
            DepthwiseSeparableConv(ch[2], ch[3], stride=2),
        )
        self.out_channels = ch[3]
        self.low_level_channels = ch[1]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        low_level = self.stage1(x)
        x = self.stage2(low_level)
        x = self.stage3(x)
        return x, low_level
