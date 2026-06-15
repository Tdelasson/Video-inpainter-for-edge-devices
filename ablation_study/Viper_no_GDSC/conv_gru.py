import torch
import torch.nn as nn

class ConvGRU(nn.Module):
    def __init__(self, in_channels, hidden_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv_zr = nn.Conv2d(in_channels + hidden_dim, 2 * hidden_dim, kernel_size, stride=stride, padding=padding)
        self.conv_h = nn.Conv2d(in_channels + hidden_dim, hidden_dim, kernel_size, stride=stride, padding=padding)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h_prev=None):
        if h_prev is None:
            h_prev = torch.full(
                (x.size(0), self.hidden_dim, x.size(2), x.size(3)),
                0.1, device=x.device
            )
        combined = torch.cat([x, h_prev], dim=1)
        zr = self.conv_zr(combined)
        z, r = torch.split(zr, self.hidden_dim, dim=1)
        z = self.sigmoid(z)
        r = self.sigmoid(r)
        h_tilde = self.tanh(self.conv_h(torch.cat([x, r * h_prev], dim=1)))
        h_next = (1 - z) * h_prev + z * h_tilde
        return h_next