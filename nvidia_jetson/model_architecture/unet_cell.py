from torch import nn
import torch
from .encoder import Encoder
from .decoder import Decoder
from .conv_gru import ConvolutionalGatedRecurrentUnits

class UNetCell(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, num_layers: int, kernel_size: int=3, stride: int=1, padding: int=1):
        super(UNetCell,self).__init__()

        channels_in_deepest_layer: int = base_channels * (2 ** (num_layers - 1))

        self.encoder = Encoder(in_channels, base_channels, num_layers)
        self.decoder = Decoder(channels_in_deepest_layer, base_channels, num_layers, raw_channels=in_channels)

        self.conv_gru = ConvolutionalGatedRecurrentUnits(channels_in_deepest_layer, channels_in_deepest_layer, kernel_size, stride, padding)

        self.head = nn.Sequential(
            nn.Conv2d(base_channels, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor]:
        skips: list[torch.Tensor] = self.encoder(x)
        skips.insert(0,x)

        feature_in_deepest_layer: torch.Tensor = skips[-1]
        h_next: torch.Tensor = self.conv_gru(feature_in_deepest_layer, h_prev)

        decoded_features: torch.Tensor = self.decoder(h_next, skips[:-1])

        output = self.head(decoded_features)

        return output, h_next
