import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .conv_gru import ConvGRU

class UNetCell(nn.Module):
    def __init__(self, in_channels, base_channels, num_layers, kernel_size=3, stride=1, padding=1):
        super().__init__()
        channels_in_deepest_layer = base_channels * (2 ** (num_layers - 1))
        self.encoder = Encoder(in_channels, base_channels, num_layers)
        self.decoder = Decoder(channels_in_deepest_layer, base_channels, num_layers, raw_channels=in_channels)
        self.conv_gru = ConvGRU(channels_in_deepest_layer, channels_in_deepest_layer, kernel_size, stride, padding)
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, h_prev=None):
        skips = self.encoder(x)
        skips.insert(0, x)
        feature_in_deepest_layer = skips[-1]
        h_next = self.conv_gru(feature_in_deepest_layer, h_prev)
        decoded_features = self.decoder(h_next, skips[:-1])
        output = self.head(decoded_features)
        return output, h_next