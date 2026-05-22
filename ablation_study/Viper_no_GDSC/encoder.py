import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels, base_channels, num_layers, kernel_size=3, padding=1):
        super().__init__()
        layers = []
        current_in = in_channels
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            layers.append(nn.Conv2d(current_in, out_channels, kernel_size, stride=2, padding=padding))
            layers.append(nn.BatchNorm2d(out_channels) if i != 0 else nn.Identity())
            layers.append(nn.ReLU(inplace=True))
            current_in = out_channels
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        skips = []
        for i in range(0, len(self.layers), 3):
            x = self.layers[i](x)
            if not isinstance(self.layers[i+1], nn.Identity):
                x = self.layers[i+1](x)
            x = self.layers[i+2](x)
            skips.append(x)
        return skips