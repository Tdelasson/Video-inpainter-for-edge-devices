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


class PretrainedPatchDiscriminator(torch.nn.Module):
    """
    Discriminator with pretrained VGG backbone.
    Backbone frozen — immediately sensitive to texture quality.
    Scoring head trained — learns real/fake distinction.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

        # Freeze pretrained features — they already know what real looks like
        self.backbone = vgg[:16]  # up to pool3, 256 channels
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Trainable scoring head — learns to score real vs fake
        self.head = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(256, 128, kernel_size=3, padding=1)
            ),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(128, 1, kernel_size=3, padding=1)
            )
        )

    def forward(self, x):
        # x: (B, C, T, H, W) — process each frame independently
        B, C, T, H, W = x.shape

        # Take RGB only (drop mask channel if present)
        x_rgb = x[:, :3, :, :, :]

        # Flatten temporal into batch
        x_flat = x_rgb.permute(0, 2, 1, 3, 4).reshape(B * T, 3, H, W)

        # ImageNet normalization for pretrained backbone
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                           device=x.device).view(1, 3, 1, 1)
        x_norm = (x_flat - mean) / std

        with torch.no_grad():
            features = self.backbone(x_norm)  # (B*T, 256, H/8, W/8)

        scores = self.head(features)  # (B*T, 1, H/8, W/8)

        # Restore temporal dimension
        _, _, fH, fW = scores.shape
        return scores.reshape(B, 1, T, fH, fW)