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

        def conv_block(in_f, out_f, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), normalize=True):
            layers = [
                torch.nn.utils.spectral_norm(
                    torch.nn.Conv3d(in_f, out_f, kernel_size=kernel_size,
                                    stride=stride, padding=padding)
                )
            ]
            if normalize:
                layers.append(torch.nn.InstanceNorm3d(out_f, affine=True))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return torch.nn.Sequential(*layers)

        self.model = torch.nn.Sequential(
            conv_block(in_channels, 64, normalize=False),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512, stride=1),
            torch.nn.utils.spectral_norm(
                torch.nn.Conv3d(512, 1, kernel_size=(3, 4, 4), stride=1, padding=(1, 1, 1))
            )
        )

    def forward(self, x):
        return self.model(x)