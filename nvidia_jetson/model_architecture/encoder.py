from torch import nn
import torch
from .gated_dsc import GatedDSC

class Encoder (nn.Module):
    def __init__(self, in_channels: int, base_channels: int, num_layers: int, kernel_size: int=3, padding: int=1):
        super(Encoder, self).__init__()
        self.num_layers = num_layers

        layers = []
        current_in: int = in_channels

        for i in range(num_layers):
            out_channels: int = base_channels * (2 ** i)

            should_normalize: bool = (i != 0)

            layers.append(GatedDSC(
                in_channels=current_in,
                out_channels=out_channels,
                stride=2,
                kernel_size=kernel_size,
                padding=padding,
                normalize=should_normalize,
                activation=True
            ))

            current_in = out_channels

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        skips = []

        for layer in self.layers:
            x = layer(x)
            skips.append(x)

        return skips