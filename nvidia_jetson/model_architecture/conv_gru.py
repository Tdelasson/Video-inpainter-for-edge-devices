from torch import nn
from gated_dsc import DepthwiseSeparableConv
import torch

class ConvolutionalGatedRecurrentUnits(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int=3, stride: int=1, padding: int=1):
        super(ConvolutionalGatedRecurrentUnits, self).__init__()

        self.hidden_dim = hidden_dim

        self.conv_zr = DepthwiseSeparableConv(in_channels + hidden_dim, 2 * hidden_dim,
                                 kernel_size, stride=stride, padding=padding)
        self.conv_h = DepthwiseSeparableConv(in_channels + hidden_dim, hidden_dim,
                                kernel_size, stride=stride, padding=padding)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor= None) -> torch.Tensor:
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3)).to(x.device)

        combined_zr_conv: torch.Tensor = self.conv_zr(torch.cat([x, h_prev], dim=1))
        z, r = torch.split(combined_zr_conv, combined_zr_conv.shape[1] // 2, dim=1)

        r = self.sigmoid(r)
        z = self.sigmoid(z)

        combined_h_conv = self.conv_h(torch.cat([x, r * h_prev], dim=1))
        h_tilde = self.tanh(combined_h_conv)

        h_next: torch.Tensor = (1 - z) * h_prev + z * h_tilde
        return h_next