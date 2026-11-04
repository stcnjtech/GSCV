import numpy as np
import random as rd
import torch
from torch import nn

def subpel_conv1x1(in_ch, out_ch, r=1):
    """1x1 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), nn.PixelShuffle(r)
    )


class DepthConv(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernel=3, stride=1, slope=0.01, inplace=False):
        super().__init__()
        dw_ch = in_ch * 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, dw_ch, 1, stride=stride),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
        )
        self.depth_conv = nn.Conv2d(dw_ch, dw_ch, depth_kernel, padding=depth_kernel // 2,
                                    groups=dw_ch)
        self.conv2 = nn.Conv2d(dw_ch, out_ch, 1)

        self.adaptor = None
        if stride != 1:
            assert stride == 2
            self.adaptor = nn.Conv2d(in_ch, out_ch, 2, stride=2)
        elif in_ch != out_ch:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.depth_conv(out)
        out = self.conv2(out)

        return out + identity


class ConvFFN(nn.Module):
    def __init__(self, in_ch, slope=0.1, inplace=False):
        super().__init__()
        internal_ch = max(min(in_ch * 4, 1024), in_ch * 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, internal_ch, 1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
            nn.Conv2d(internal_ch, in_ch, 1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
        )

    def forward(self, x):
        identity = x
        return identity + self.conv(x)


class DepthConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernel=3, stride=1,
                 slope_depth_conv=0.01, slope_ffn=0.1, inplace=False):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv(in_ch, out_ch, depth_kernel, stride, slope=slope_depth_conv, inplace=inplace),
            ConvFFN(out_ch, slope=slope_ffn, inplace=inplace),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch=128, out_ch=3, inplace=False):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = DepthConvBlock(in_ch, 128, inplace=inplace)
        self.conv2 = DepthConvBlock(128, 128, inplace=inplace)
        self.conv3 = DepthConvBlock(128, 128, inplace=inplace)

        self.context_refine = nn.Sequential(
            DepthConvBlock(128, 128, inplace=inplace),
            DepthConvBlock(128, 128, inplace=inplace),
            DepthConvBlock(128, 128, inplace=inplace),
            DepthConvBlock(128, 128, inplace=inplace),
        )

        self.up3 = subpel_conv1x1(128, 128, 2)
        self.up_conv3 = DepthConvBlock(256, 128, inplace=inplace)

        self.up2 = subpel_conv1x1(128, 128, 2)
        self.up_conv2 = DepthConvBlock(256, 128, inplace=inplace)
        self.conv_out = nn.Conv2d(128, out_ch, kernel_size=1)

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)
        x2 = self.max_pool(x1)

        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)

        x3 = self.conv3(x3)
        x3 = self.context_refine(x3)

        # decoding + concat path
        d3 = self.up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        return self.conv_out(d2)