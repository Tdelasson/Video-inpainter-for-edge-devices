import torch

class SpatioTemporalDiscriminator(torch.nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        def conv_block(in_f, out_f, stride=2):
            return torch.nn.Sequential(
                torch.nn.Conv3d(in_f, out_f, kernel_size=3, stride=stride, padding=1),
                torch.nn.LeakyReLU(0.2, inplace=True)
            )

        # Input shape: [B, Channels, Time, H, W]
        self.model = torch.nn.Sequential(
            conv_block(in_channels, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            torch.nn.Conv3d(256, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.model(x)