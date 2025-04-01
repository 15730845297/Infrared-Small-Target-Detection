import torch.nn as nn
import torch

class AsymBiChaFuse(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AsymBiChaFuse, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // r)

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.channels, out_channels=self.bottleneck_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.bottleneck_channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.bottleneck_channels, out_channels=self.channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels, momentum=0.1),
            nn.Sigmoid()
        )

        self.bottomup = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.bottleneck_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.bottleneck_channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.bottleneck_channels, out_channels=self.channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels, momentum=0.1),
            nn.Sigmoid()
        )

        self.post = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(channels, momentum=0.1),
            nn.ReLU(inplace=True)
        )

    def forward(self, xh, xl):
        # 检查输入类型
        if not isinstance(xh, torch.Tensor) or not isinstance(xl, torch.Tensor):
            raise TypeError("Inputs xh and xl must be of type torch.Tensor")

        # 检查输入形状并调整
        if xh.shape != xl.shape:
            xl = nn.functional.interpolate(xl, size=xh.shape[2:], mode='bilinear', align_corners=False)

        # 计算权重
        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)

        # 逐元素相乘并加权
        xs = 2 * xl * topdown_wei + 2 * xh * bottomup_wei

        # 后处理
        xs = self.post(xs)
        return xs