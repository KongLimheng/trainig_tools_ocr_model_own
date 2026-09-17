"""MobileNetV3 Backbone tailored for lightweight Khmer OCR."""

import torch
import torch.nn as nn


class ConvBNAct(nn.Sequential):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, act=nn.Hardswish):
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_c),
            act(inplace=True),
        )


class MobileBlock(nn.Module):
    def __init__(self, in_c, exp_c, out_c, kernel_size=3, stride=1):
        super().__init__()
        self.use_res = stride == 1 and in_c == out_c
        layers = [
            ConvBNAct(in_c, exp_c, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(exp_c, exp_c, kernel_size, stride, kernel_size // 2, groups=exp_c, bias=False),
            nn.BatchNorm2d(exp_c),
            nn.Hardswish(inplace=True),
            nn.Conv2d(exp_c, out_c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_c),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x) if self.use_res else self.block(x)


class MobileNetOCR(nn.Module):
    """Lightweight MobileNetV3 for edge deployment."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.stem = ConvBNAct(in_channels, 16, kernel_size=3, stride=(2, 2), padding=1)
        self.stage1 = nn.Sequential(
            MobileBlock(16, 16, 16, stride=(2, 1)),
            MobileBlock(16, 64, 24, stride=(2, 1)),
        )
        self.stage2 = nn.Sequential(
            MobileBlock(24, 72, 24, stride=1),
            MobileBlock(24, 96, 40, stride=(2, 1)),
        )
        self.stage3 = nn.Sequential(
            MobileBlock(40, 120, 80, stride=1),
            MobileBlock(80, 240, 128, stride=(3, 1)),
        )
        self.out_channels = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        return out
