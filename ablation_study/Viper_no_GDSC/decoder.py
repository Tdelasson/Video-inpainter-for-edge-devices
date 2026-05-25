import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self, in_channels, base_channels, num_layers, raw_channels=10, kernel_size=3, padding=1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        layers = []
        current_channels = in_channels
        for i in range(num_layers):
            if i == num_layers - 1:
                skip_channels = raw_channels
                out_channels = base_channels
            else:
                skip_channels = base_channels * (2 ** (num_layers - 2 - i))
                out_channels = base_channels * (2 ** (num_layers - 2 - i))
            input_to_layer = current_channels + skip_channels
            layers.append(nn.Conv2d(input_to_layer, out_channels, kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            current_channels = out_channels
        self.layers = nn.ModuleList(layers)

    def forward(self, x, skips):
        skips = list(reversed(skips))
        for i in range(0, len(self.layers), 3):
            x = self.upsample(x)
            skip = skips[i // 3]
            x = torch.cat([x, skip], dim=1)
            x = self.layers[i](x)
            x = self.layers[i+1](x)
            x = self.layers[i+2](x)
        return x