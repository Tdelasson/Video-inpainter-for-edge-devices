from .unet_cell import UNetCell
from torch import nn
import torch

class VideoInpainter(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, num_layers: int,
                 kernel_size: int=3, stride: int=1, padding: int=1):
        super(VideoInpainter, self).__init__()
        self.unet_cell = UNetCell(in_channels, base_channels, num_layers,
                                  kernel_size, stride, padding)
        self.num_layers = num_layers
        self.base_channels = base_channels

    def _make_zero_hidden(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        H, W = x.size(-2), x.size(-1)
        hidden_dim = self.base_channels * (2 ** (self.num_layers - 1))
        spatial_h = H // (2 ** self.num_layers)
        spatial_w = W // (2 ** self.num_layers)
        return torch.zeros(B, hidden_dim, spatial_h, spatial_w, device=x.device)

    def forward(self, x: torch.Tensor,
                h_prev: torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor]:
        if h_prev is None:
            h_prev = self._make_zero_hidden(x)
        return self.unet_cell(x, h_prev)