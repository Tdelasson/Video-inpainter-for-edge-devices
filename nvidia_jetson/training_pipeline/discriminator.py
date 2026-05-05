import torch

class NoisyDiscriminator(torch.nn.Module):
    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator
        self.current_std = 0.0

    def set_std(self, std):
        self.current_std = std

    def forward(self, x):
        if self.current_std > 0:
            noise = torch.randn_like(x) * self.current_std
            x = torch.clamp(x + noise, 0.0, 1.0)
        return self.discriminator(x)

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