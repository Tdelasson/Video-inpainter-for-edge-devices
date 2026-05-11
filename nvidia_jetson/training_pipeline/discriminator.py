import torch
import torch.nn as nn

class NoisyDiscriminator(nn.Module):
    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator
        self.current_std = 0.0

    def set_std(self, std):
        self.current_std = std

    def forward(self, x):
        if self.current_std > 0:
            x = x + torch.randn_like(x) * self.current_std
        return self.discriminator(x)


class VideoPatchGAN(nn.Module):
    def __init__(self, in_channels=4, ndf=64, n_layers=3):
        super().__init__()
        # Use temporal kernel 3 and padding 1 to perfectly preserve sequence length
        kw = (3, 4, 4)
        padw = (1, 1, 1)

        # Strided purely spatially to maintain temporal resolution
        stride = (1, 2, 2)

        sequence = [
            nn.Conv3d(in_channels, ndf, kernel_size=kw, stride=stride, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.utils.spectral_norm(
                    nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=stride, padding=padw)
                ),
                nn.BatchNorm3d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.utils.spectral_norm(
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=(1, 1, 1), padding=padw)
            ),
            nn.BatchNorm3d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Output 1 channel prediction map
        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=(1, 1, 1), padding=padw)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        # output shape: (B, 1, T, H', W')
        return self.model(x)