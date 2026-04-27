from torch import nn
import torch

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, stride: int=1, padding: int=1):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise: nn.Conv2d = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=in_channels
        )

        self.pointwise: nn.Conv2d  = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0, groups=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class GatedDSC(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size = 3, padding = 1, stride: int=1, normalize: bool=True, activation: bool=True):
        super(GatedDSC, self).__init__()

        self.feature_conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

        self.gate_conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

        self.sigmoid = nn.Sigmoid()

        self.norm = nn.InstanceNorm2d(out_channels) if normalize else nn.Identity()
        self.activation = nn.LeakyReLU(0.2) if activation else nn.Identity()

    def forward(self, x):
        features = self.feature_conv(x)
        gate = self.gate_conv(x)

        mask = self.sigmoid(gate)

        gated_output = features * mask

        out = self.norm(gated_output)
        return self.activation(out)