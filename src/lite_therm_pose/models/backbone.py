from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .specs import BackboneSpec, get_model_spec


def conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, groups: int = 1) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            conv_bn_relu(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels),
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


class CSPBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 8)
        self.left = conv_bn_relu(channels, hidden, kernel_size=1)
        self.right = nn.Sequential(
            conv_bn_relu(channels, hidden, kernel_size=1),
            DepthwiseSeparableConv(hidden, hidden),
            DepthwiseSeparableConv(hidden, hidden),
        )
        self.fuse = conv_bn_relu(hidden * 2, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.left(x), self.right(x)], dim=1))


class ResidualBottleneck(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 4, 16)
        self.block = nn.Sequential(
            conv_bn_relu(channels, hidden, kernel_size=1),
            conv_bn_relu(hidden, hidden, kernel_size=3),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + x)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class SEResidualBottleneck(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // 4, 16)
        self.block = nn.Sequential(
            conv_bn_relu(channels, hidden, kernel_size=1),
            conv_bn_relu(hidden, hidden, kernel_size=3),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.se = ChannelAttention(channels, reduction=reduction)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.se(self.block(x)) + x)


class SpatialContrastGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=5, padding=2, bias=False),
            nn.Sigmoid(),
        )
        self.refine = conv_bn_relu(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        local_mean = F.avg_pool2d(avg_map, kernel_size=5, stride=1, padding=2)
        contrast_map = torch.abs(avg_map - local_mean)
        gate = self.project(torch.cat([avg_map, max_map, contrast_map], dim=1))
        return self.refine(x * gate + x)


class ThermalFusionGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            conv_bn_relu(channels * 2, channels, kernel_size=1),
            nn.Conv2d(channels, 2, kernel_size=1),
        )
        self.out = nn.Sequential(
            conv_bn_relu(channels, channels, kernel_size=1),
            ChannelAttention(channels),
        )

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        if high.shape[-2:] != low.shape[-2:]:
            high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.gate(torch.cat([low, high], dim=1))
        weights = torch.softmax(logits, dim=1)
        fused = low * weights[:, 0:1] + high * weights[:, 1:2]
        return self.out(fused)


class PyramidContext(nn.Module):
    def __init__(self, channels: int, bins: tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        hidden = max(channels // len(bins), 16)
        self.paths = nn.ModuleList([conv_bn_relu(channels, hidden, kernel_size=1) for _ in bins])
        self.bins = bins
        self.merge = conv_bn_relu(channels + hidden * len(bins), channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for bin_size, path in zip(self.bins, self.paths):
            pooled = F.adaptive_avg_pool2d(x, output_size=bin_size)
            pooled = path(pooled)
            features.append(F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False))
        return self.merge(torch.cat(features, dim=1))


class FlexibleBackbone(nn.Module):
    def __init__(self, spec: BackboneSpec, in_channels: int = 1) -> None:
        super().__init__()
        self.spec = spec
        builder = {
            "flex": self._build_flex,
            "hrnet": self._build_hrnet,
            "higherhrnet": self._build_higherhrnet,
            "thermhybrid": self._build_thermhybrid,
            "unet": self._build_unet,
            "hourglass": self._build_hourglass,
            "fpn": self._build_fpn,
            "csp": self._build_csp,
            "resnet": self._build_resnet,
            "seresnet": self._build_seresnet,
            "thermresnet": self._build_thermresnet,
        }[spec.family]
        builder(in_channels)

    def _channels(self) -> list[int]:
        return [max(self.spec.stem_channels, int(c * self.spec.width_mult)) for c in [24, 48, 96, 160]]

    def _build_flex(self, in_channels: int) -> None:
        ch = self._channels()
        self.mode = "flex"
        self.stem = conv_bn_relu(in_channels, ch[0], stride=2)
        self.stage1 = self._make_stage(ch[0], ch[1], self.spec.block_type)
        self.stage2 = self._make_stage(ch[1], ch[2], self.spec.block_type)
        self.stage3 = self._make_stage(ch[2], ch[3], self.spec.block_type)
        self.out_channels = ch[3]
        self.low_level_channels = ch[1]

    def _build_hrnet(self, in_channels: int) -> None:
        ch = self._channels()
        self.mode = "hrnet"
        self.stem = nn.Sequential(
            conv_bn_relu(in_channels, ch[0], stride=2),
            conv_bn_relu(ch[0], ch[0]),
        )
        self.high = nn.Sequential(conv_bn_relu(ch[0], ch[1]), ResidualDSConv(ch[1]))
        self.low = nn.Sequential(conv_bn_relu(ch[0], ch[1], stride=2), ResidualDSConv(ch[1]))
        self.exchange1 = conv_bn_relu(ch[1] * 2, ch[2])
        self.high2 = nn.Sequential(conv_bn_relu(ch[2], ch[2]), ResidualDSConv(ch[2]))
        self.low2 = nn.Sequential(conv_bn_relu(ch[2], ch[3], stride=2), ResidualDSConv(ch[3]))
        self.fuse = conv_bn_relu(ch[2] + ch[3], ch[3], kernel_size=1)
        self.out_channels = ch[3]
        self.low_level_channels = ch[2]

    def _build_higherhrnet(self, in_channels: int) -> None:
        ch = self._channels()
        self.mode = "higherhrnet"
        self.stem = nn.Sequential(
            conv_bn_relu(in_channels, ch[0], stride=2),
            conv_bn_relu(ch[0], ch[1]),
        )
        self.branch1 = nn.Sequential(conv_bn_relu(ch[1], ch[1]), ResidualBottleneck(ch[1]))
        self.branch2 = nn.Sequential(conv_bn_relu(ch[1], ch[2], stride=2), ResidualBottleneck(ch[2]))
        self.branch3 = nn.Sequential(conv_bn_relu(ch[2], ch[3], stride=2), ResidualBottleneck(ch[3]))
        self.fuse1 = conv_bn_relu(ch[1] + ch[2], ch[2], kernel_size=1)
        self.fuse2 = conv_bn_relu(ch[2] + ch[3], ch[3], kernel_size=1)
        self.out_channels = ch[3]
        self.low_level_channels = ch[2]

    def _build_thermhybrid(self, in_channels: int) -> None:
        ch = self._channels()
        self.mode = "thermhybrid"
        self.stem = nn.Sequential(
            conv_bn_relu(in_channels, ch[0], stride=2),
            DepthwiseSeparableConv(ch[0], ch[1]),
        )
        self.high = nn.Sequential(
            DepthwiseSeparableConv(ch[1], ch[1]),
            CSPBlock(ch[1]),
            ChannelAttention(ch[1]),
        )
        self.mid = nn.Sequential(
            conv_bn_relu(ch[1], ch[2], stride=2),
            CSPBlock(ch[2]),
            ChannelAttention(ch[2]),
        )
        self.low = nn.Sequential(
            conv_bn_relu(ch[2], ch[3], stride=2),
            ResidualBottleneck(ch[3]),
            ChannelAttention(ch[3]),
        )
        self.top_down = conv_bn_relu(ch[3] + ch[2], ch[2], kernel_size=1)
        self.bottom_up = conv_bn_relu(ch[2] + ch[1], ch[3], kernel_size=1)
        self.out_fuse = nn.Sequential(
            conv_bn_relu(ch[3] + ch[2], ch[3], kernel_size=1),
            ChannelAttention(ch[3]),
        )
        self.out_channels = ch[3]
        self.low_level_channels = ch[1]

    def _build_unet(self, in_channels: int) -> None:
        ch = self._channels()
        self.mode = "unet"
        self.enc1 = nn.Sequential(conv_bn_relu(in_channels, ch[0]), conv_bn_relu(ch[0], ch[0]))
        self.enc2 = nn.Sequential(conv_bn_relu(ch[0], ch[1], stride=2), conv_bn_relu(ch[1], ch[1]))
        self.enc3 = nn.Sequential(conv_bn_relu(ch[1], ch[2], stride=2), conv_bn_relu(ch[2], ch[2]))
        self.bottleneck = nn.Sequential(conv_bn_relu(ch[2], ch[3], stride=2), conv_bn_relu(ch[3], ch[3]))
        self.dec3 = conv_bn_relu(ch[3] + ch[2], ch[2])
        self.dec2 = conv_bn_relu(ch[2] + ch[1], ch[2])
        self.out_channels = ch[2]
        self.low_level_channels = ch[1]

    def _build_hourglass(self, in_channels: int) -> None:
        ch = self._channels()
        self.mode = "hourglass"
        self.stem = conv_bn_relu(in_channels, ch[0], stride=2)
        self.down1 = conv_bn_relu(ch[0], ch[1], stride=2)
        self.down2 = conv_bn_relu(ch[1], ch[2], stride=2)
        self.bottleneck = nn.Sequential(ResidualDSConv(ch[2]), ResidualDSConv(ch[2]))
        self.up1 = conv_bn_relu(ch[2] + ch[1], ch[2])
        self.up2 = conv_bn_relu(ch[2] + ch[0], ch[3])
        self.out_channels = ch[3]
        self.low_level_channels = ch[1]

    def _build_fpn(self, in_channels: int) -> None:
        ch = self._channels()
        self.mode = "fpn"
        self.c1 = conv_bn_relu(in_channels, ch[0], stride=2)
        self.c2 = conv_bn_relu(ch[0], ch[1], stride=2)
        self.c3 = conv_bn_relu(ch[1], ch[2], stride=2)
        self.c4 = conv_bn_relu(ch[2], ch[3], stride=2)
        self.l3 = conv_bn_relu(ch[2], ch[2], kernel_size=1)
        self.l4 = conv_bn_relu(ch[3], ch[2], kernel_size=1)
        self.out_proj = conv_bn_relu(ch[2], ch[3])
        self.out_channels = ch[3]
        self.low_level_channels = ch[2]

    def _build_csp(self, in_channels: int) -> None:
        ch = self._channels()
        self.mode = "csp"
        self.stem = conv_bn_relu(in_channels, ch[0], stride=2)
        self.stage1 = nn.Sequential(conv_bn_relu(ch[0], ch[1], stride=2), CSPBlock(ch[1]))
        self.stage2 = nn.Sequential(conv_bn_relu(ch[1], ch[2], stride=2), CSPBlock(ch[2]))
        self.stage3 = nn.Sequential(conv_bn_relu(ch[2], ch[3], stride=2), CSPBlock(ch[3]))
        self.out_channels = ch[3]
        self.low_level_channels = ch[1]

    def _build_resnet(self, in_channels: int) -> None:
        ch = self._channels()
        self.mode = "resnet"
        self.stem = nn.Sequential(
            conv_bn_relu(in_channels, ch[0], stride=2),
            conv_bn_relu(ch[0], ch[0]),
        )
        self.layer1 = nn.Sequential(conv_bn_relu(ch[0], ch[1], stride=2), ResidualBottleneck(ch[1]), ResidualBottleneck(ch[1]))
        self.layer2 = nn.Sequential(conv_bn_relu(ch[1], ch[2], stride=2), ResidualBottleneck(ch[2]), ResidualBottleneck(ch[2]))
        self.layer3 = nn.Sequential(conv_bn_relu(ch[2], ch[3], stride=2), ResidualBottleneck(ch[3]), ResidualBottleneck(ch[3]))
        self.out_channels = ch[3]
        self.low_level_channels = ch[1]

    def _build_seresnet(self, in_channels: int) -> None:
        ch = self._channels()
        self.mode = "seresnet"
        self.stem = nn.Sequential(
            conv_bn_relu(in_channels, ch[0] // 2, stride=2),
            conv_bn_relu(ch[0] // 2, ch[0]),
            conv_bn_relu(ch[0], ch[0]),
        )
        self.layer1 = nn.Sequential(
            conv_bn_relu(ch[0], ch[1], stride=2),
            SEResidualBottleneck(ch[1]),
            SEResidualBottleneck(ch[1]),
        )
        self.layer2 = nn.Sequential(
            conv_bn_relu(ch[1], ch[2], stride=2),
            SEResidualBottleneck(ch[2]),
            SEResidualBottleneck(ch[2]),
        )
        self.layer3 = nn.Sequential(
            conv_bn_relu(ch[2], ch[3], stride=2),
            SEResidualBottleneck(ch[3]),
            SEResidualBottleneck(ch[3]),
            SEResidualBottleneck(ch[3]),
        )
        self.low_reduce = conv_bn_relu(ch[1], ch[1], kernel_size=1)
        self.deep_reduce = conv_bn_relu(ch[3], ch[1], kernel_size=1)
        self.low_refine = nn.Sequential(
            conv_bn_relu(ch[1] * 2, ch[1], kernel_size=1),
            SEResidualBottleneck(ch[1]),
        )
        self.out_refine = nn.Sequential(
            conv_bn_relu(ch[3] + ch[2], ch[3], kernel_size=1),
            SEResidualBottleneck(ch[3]),
        )
        self.out_channels = ch[3]
        self.low_level_channels = ch[1]

    def _build_thermresnet(self, in_channels: int) -> None:
        ch = self._channels()
        self.mode = "thermresnet"
        self.stem = nn.Sequential(
            conv_bn_relu(in_channels, ch[0] // 2, stride=2),
            conv_bn_relu(ch[0] // 2, ch[0]),
            SpatialContrastGate(ch[0]),
        )
        self.layer1 = nn.Sequential(
            conv_bn_relu(ch[0], ch[1], stride=2),
            SEResidualBottleneck(ch[1]),
            SpatialContrastGate(ch[1]),
            SEResidualBottleneck(ch[1]),
        )
        self.layer2 = nn.Sequential(
            conv_bn_relu(ch[1], ch[2], stride=2),
            SEResidualBottleneck(ch[2]),
            SpatialContrastGate(ch[2]),
            SEResidualBottleneck(ch[2]),
        )
        self.layer3 = nn.Sequential(
            conv_bn_relu(ch[2], ch[3], stride=2),
            SEResidualBottleneck(ch[3]),
            SEResidualBottleneck(ch[3]),
            SpatialContrastGate(ch[3]),
            SEResidualBottleneck(ch[3]),
        )
        self.low_reduce = conv_bn_relu(ch[1], ch[1], kernel_size=1)
        self.mid_reduce = conv_bn_relu(ch[2], ch[1], kernel_size=1)
        self.deep_reduce = conv_bn_relu(ch[3], ch[1], kernel_size=1)
        self.thermal_low_fuse = ThermalFusionGate(ch[1])
        self.thermal_mid_fuse = ThermalFusionGate(ch[1])
        self.out_reduce = conv_bn_relu(ch[2], ch[3], kernel_size=1)
        self.out_refine = nn.Sequential(
            conv_bn_relu(ch[3] * 2, ch[3], kernel_size=1),
            SEResidualBottleneck(ch[3]),
            SpatialContrastGate(ch[3]),
        )
        self.context = PyramidContext(ch[3]) if "ppm" in self.spec.name else nn.Identity()
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
        if self.mode == "flex":
            x = self.stem(x)
            low = self.stage1(x)
            x = self.stage2(low)
            x = self.stage3(x)
            return x, low
        if self.mode == "hrnet":
            x = self.stem(x)
            high = self.high(x)
            low = self.low(x)
            low_up = F.interpolate(low, size=high.shape[-2:], mode="bilinear", align_corners=False)
            fused = self.exchange1(torch.cat([high, low_up], dim=1))
            high2 = self.high2(fused)
            low2 = self.low2(fused)
            low2_up = F.interpolate(low2, size=high2.shape[-2:], mode="bilinear", align_corners=False)
            out = self.fuse(torch.cat([high2, low2_up], dim=1))
            return out, high2
        if self.mode == "higherhrnet":
            x = self.stem(x)
            b1 = self.branch1(x)
            b2 = self.branch2(x)
            b2_up = F.interpolate(b2, size=b1.shape[-2:], mode="bilinear", align_corners=False)
            f1 = self.fuse1(torch.cat([b1, b2_up], dim=1))
            b3 = self.branch3(b2)
            b3_up = F.interpolate(b3, size=f1.shape[-2:], mode="bilinear", align_corners=False)
            out = self.fuse2(torch.cat([f1, b3_up], dim=1))
            return out, f1
        if self.mode == "thermhybrid":
            x = self.stem(x)
            high = self.high(x)
            mid = self.mid(high)
            low = self.low(mid)
            td = self.top_down(torch.cat([F.interpolate(low, size=mid.shape[-2:], mode="bilinear", align_corners=False), mid], dim=1))
            bu = self.bottom_up(torch.cat([F.interpolate(td, size=high.shape[-2:], mode="bilinear", align_corners=False), high], dim=1))
            out = self.out_fuse(torch.cat([F.interpolate(bu, size=td.shape[-2:], mode="bilinear", align_corners=False), td], dim=1))
            return out, high
        if self.mode == "unet":
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            bottleneck = self.bottleneck(e3)
            d3 = self.dec3(torch.cat([F.interpolate(bottleneck, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], dim=1))
            out = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1))
            return out, e2
        if self.mode == "hourglass":
            s = self.stem(x)
            d1 = self.down1(s)
            d2 = self.down2(d1)
            b = self.bottleneck(d2)
            u1 = self.up1(torch.cat([F.interpolate(b, size=d1.shape[-2:], mode="bilinear", align_corners=False), d1], dim=1))
            out = self.up2(torch.cat([F.interpolate(u1, size=s.shape[-2:], mode="bilinear", align_corners=False), s], dim=1))
            return out, d1
        if self.mode == "fpn":
            c1 = self.c1(x)
            c2 = self.c2(c1)
            c3 = self.c3(c2)
            c4 = self.c4(c3)
            p4 = self.l4(c4)
            p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
            out = self.out_proj(p3)
            return out, p3
        if self.mode == "resnet":
            x = self.stem(x)
            low = self.layer1(x)
            x = self.layer2(low)
            x = self.layer3(x)
            return x, low
        if self.mode == "seresnet":
            x = self.stem(x)
            low = self.layer1(x)
            mid = self.layer2(low)
            high = self.layer3(mid)
            low_fused = self.low_refine(
                torch.cat(
                    [
                        self.low_reduce(low),
                        F.interpolate(self.deep_reduce(high), size=low.shape[-2:], mode="bilinear", align_corners=False),
                    ],
                    dim=1,
                )
            )
            out = self.out_refine(
                torch.cat([high, F.interpolate(mid, size=high.shape[-2:], mode="bilinear", align_corners=False)], dim=1)
            )
            return out, low_fused
        if self.mode == "thermresnet":
            x = self.stem(x)
            low = self.layer1(x)
            mid = self.layer2(low)
            high = self.layer3(mid)
            low_fused = self.thermal_low_fuse(
                self.low_reduce(low),
                self.deep_reduce(high),
            )
            low_fused = self.thermal_mid_fuse(
                low_fused,
                self.mid_reduce(mid),
            )
            high_context = self.context(high)
            out = self.out_refine(
                torch.cat(
                    [
                        high_context,
                        F.interpolate(self.out_reduce(mid), size=high_context.shape[-2:], mode="bilinear", align_corners=False),
                    ],
                    dim=1,
                )
            )
            return out, low_fused
        s = self.stem(x)
        low = self.stage1(s)
        x = self.stage2(low)
        x = self.stage3(x)
        return x, low


def build_backbone(model_name: str, in_channels: int = 1) -> FlexibleBackbone:
    return FlexibleBackbone(get_model_spec(model_name), in_channels=in_channels)
