"""ResNet Backbones adapted for OCR text line feature extraction."""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNetOCR(nn.Module):
    """ResNet modified for text line OCR with asymmetric pooling (preserving width resolution)."""

    def __init__(self, in_channels: int = 1, base_channels: int = 64):
        super().__init__()
        self.in_planes = base_channels

        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        # Downsample 1: 48 -> 24
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(BasicBlock, base_channels, 2, stride=1)

        # Downsample 2: 24 -> 12
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = self._make_layer(BasicBlock, base_channels * 2, 2, stride=1)

        # Downsample 3 (Asymmetric: pool height 2, stride width 1 to preserve text sequence length): 12 -> 6
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(BasicBlock, base_channels * 4, 2, stride=1)

        # Downsample 4: 6 -> 3
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))
        self.layer4 = self._make_layer(BasicBlock, base_channels * 8, 2, stride=1)

        # Final Conv: 3 -> 1
        self.conv_final = nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=(3, 1), stride=(3, 1), bias=False)
        self.bn_final = nn.BatchNorm2d(base_channels * 8)
        self.out_channels = base_channels * 8

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool1(out)
        out = self.layer1(out)
        out = self.maxpool2(out)
        out = self.layer2(out)
        out = self.maxpool3(out)
        out = self.layer3(out)
        out = self.maxpool4(out)
        out = self.layer4(out)
        out = self.relu(self.bn_final(self.conv_final(out)))
        return out
