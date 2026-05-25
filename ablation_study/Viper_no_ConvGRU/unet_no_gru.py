# ablation_study/Unet_no_GRU/unet_no_gru.py
import torch
from torch import nn
from model_architecture.encoder import Encoder
from model_architecture.decoder import Decoder

class UNetNoGRU(nn.Module):
    def __init__(self, in_channels, base_channels, num_layers, kernel_size=3, stride=1, padding=1):
        super().__init__()
        channels_in_deepest_layer = base_channels * (2 ** (num_layers - 1))
        self.encoder = Encoder(in_channels, base_channels, num_layers, kernel_size, padding)
        self.decoder = Decoder(channels_in_deepest_layer, base_channels, num_layers, raw_channels=in_channels)
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        skips = self.encoder(x)
        skips.insert(0, x)
        feature_in_deepest_layer = skips[-1]
        decoded_features = self.decoder(feature_in_deepest_layer, skips[:-1])
        output = self.head(decoded_features)
        return output