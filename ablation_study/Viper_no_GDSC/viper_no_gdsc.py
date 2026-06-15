import torch
import torch.nn as nn
from .unet_cell import UNetCell

class ViperNoGDSC(nn.Module):
    def __init__(self, in_channels, base_channels, num_layers, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.unet_cell = UNetCell(in_channels, base_channels, num_layers, kernel_size, stride, padding)
        self.num_layers = num_layers
        self.base_channels = base_channels

    def _make_zero_hidden(self, x):
        B = x.size(0)
        H, W = x.size(-2), x.size(-1)
        hidden_dim = self.base_channels * (2 ** (self.num_layers - 1))
        spatial_h = H // (2 ** self.num_layers)
        spatial_w = W // (2 ** self.num_layers)
        return torch.zeros(B, hidden_dim, spatial_h, spatial_w, device=x.device)

    def forward(self, x, h_prev=None):
        if h_prev is None:
            h_prev = self._make_zero_hidden(x)
        return self.unet_cell(x, h_prev)