import torch
from torch import nn
from .unet_no_gru import UNetNoGRU

class ViperNoGRU(nn.Module):
    def __init__(self, in_channels, base_channels, num_layers, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.unet = UNetNoGRU(in_channels, base_channels, num_layers, kernel_size, stride, padding)

    def forward(self, x, h_prev=None):
        # h_prev is ignored for compatibility with trainer
        return self.unet(x), None