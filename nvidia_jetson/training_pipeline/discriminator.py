import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# NOISE REGULARIZATION
# =========================================================
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

class TemporalAttentionBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, x):
        # x shape: (Seq=T, Batch=B*H*W, Emit=C)
        x_norm = self.norm(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        return x + x_attn  # Add residual connection to stabilize gradients


# =========================================================
# ATTENTION MODULE
# =========================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t, h, w = x.shape
        y = x.mean(dim=(2,3,4))  # global pooling
        y = self.mlp(y).view(b, c, 1, 1, 1)
        return x * y


# =========================================================
# RESIDUAL 3D BLOCK
# =========================================================
class ResBlock3D(nn.Module):
    def __init__(self, in_f, out_f, stride=(1,2,2)):
        super().__init__()

        self.conv1 = nn.utils.spectral_norm(
            nn.Conv3d(in_f, out_f, 3, stride=stride, padding=1)
        )
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv3d(out_f, out_f, 3, padding=1)
        )

        self.skip = nn.Conv3d(in_f, out_f, 1, stride=stride) if in_f != out_f else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + residual)


# =========================================================
# SHARED BACKBONE (multi-branch feature extractor)
# =========================================================
class SharedBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = ResBlock3D(3, 64)
        self.block2 = ResBlock3D(64, 128)
        self.block3 = ResBlock3D(128, 256)

        self.attn = ChannelAttention(256)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.attn(x)
        return x


# =========================================================
# FULL MULTI-SCALE VIDEO DISCRIMINATOR
# =========================================================
class SpatioTemporalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = SharedBackbone()

        self.mask_branch = nn.Sequential(
            ResBlock3D(1, 16),
            ResBlock3D(16, 32),
            ResBlock3D(32, 64)
        )

        self.motion_branch = nn.Sequential(
            ResBlock3D(3, 32),
            ResBlock3D(32, 64),
            ResBlock3D(64, 128)
        )

        in_channels_fusion = 256 + 64 + 128

        self.fusion = nn.Sequential(
            ResBlock3D(in_channels_fusion, 512),  # Default stride is (1,2,2)
            ResBlock3D(512, 512, stride=(1, 1, 1))
        )

        self.skip_proj = nn.Conv3d(in_channels_fusion, 512, kernel_size=1, stride=(1, 2, 2))

        self.temporal_attn = TemporalAttentionBlock(embed_dim=512, num_heads=4)

        self.head_global = nn.utils.spectral_norm(nn.Conv3d(512, 1, 3, padding=1))
        self.head_local = nn.utils.spectral_norm(nn.Conv3d(512, 1, 1))
        self.head_temporal = nn.utils.spectral_norm(nn.Conv3d(512, 1, (3, 1, 1), padding=(1, 0, 0)))

        self.features = {}

    def forward(self, x):
        rgb = x[:, :3]
        mask = x[:, 3:4]

        # Branches
        feat_rgb = self.backbone(rgb)
        feat_mask = self.mask_branch(mask)

        motion = rgb[:, :, 1:] - rgb[:, :, :-1]
        motion = F.pad(motion, (0, 0, 0, 0, 1, 0))
        feat_motion = self.motion_branch(motion)

        # Fusion & Skip
        x_concat = torch.cat([feat_rgb, feat_mask, feat_motion], dim=1)
        x_fused = self.fusion(x_concat)
        x = x_fused + self.skip_proj(x_concat)

        b, c, t, h, w = x.shape
        x_temp = x.permute(2, 0, 3, 4, 1).reshape(t, b * h * w, c)  # (Seq=T, Batch=B*H*W, Emit=C)
        x_attn = self.temporal_attn(x_temp)
        x = x_attn.reshape(t, b, h, w, c).permute(1, 4, 0, 2, 3)

        self.features["fusion"] = x

        return {
            "global": self.head_global(x),
            "local": self.head_local(x),
            "temporal": self.head_temporal(x)
        }

# =========================================================
# FEATURE MATCHING LOSS HELPER
# =========================================================
def feature_matching_loss(real_features, fake_features):
    loss = 0.0
    for k in real_features:
        loss += F.l1_loss(real_features[k], fake_features[k])
    return loss