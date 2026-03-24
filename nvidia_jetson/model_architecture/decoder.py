from torch import nn
import torch
from gated_dsc import GatedDSC

class Decoder (nn.Module):
    def __init__(self, in_channels: int, base_channels: int, num_layers: int, raw_channels=10):
        super(Decoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        layers = []
        current_channels = in_channels

        for i in range(num_layers):
            if i == num_layers - 1:
                skip_channels = raw_channels
            else:
                skip_channels = base_channels * (2 ** (num_layers - 2 - i))

            input_to_layer: int = current_channels + skip_channels

            out_channels: int = base_channels

            layers.append(GatedDSC(
                in_channels=input_to_layer,
                out_channels=out_channels,
                stride=1,
                normalize=True,
                activation=True
            ))

            current_channels = out_channels

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, skips: torch.Tensor) -> torch.Tensor:
        # reverse skips list such that the smallest skip connection is first
        skips = list(reversed(skips))

        for i, layer in enumerate(self.layers):
            x = self.upsample(x)

            skip = skips[i]

            combined = torch.cat([x, skip], dim=1)

            x = layer(combined)

        return x

