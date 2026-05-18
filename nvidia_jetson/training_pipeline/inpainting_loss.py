import torch
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

class InpaintingLoss(torch.nn.Module):
    def __init__(self, pixel_m_w, pixel_f_w, perceptual_w, style_w, temporal_w, adv_w):
        super(InpaintingLoss, self).__init__()
        # Load VGG for Perceptual/Style
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False

        self.slice1 = vgg[:5]  # pool1
        self.slice2 = vgg[5:10]  # pool2
        self.slice3 = vgg[10:17]  # pool3

        self.l1 = torch.nn.L1Loss()

        # Weights
        self.pixel_m_w = pixel_m_w
        self.pixel_f_w = pixel_f_w
        self.perceptual_w = perceptual_w
        self.style_w = style_w
        self.temporal_w = temporal_w
        self.adv_w = adv_w

        # ImageNet normalization due to vgg training being trained this normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def gram_matrix(self, features):
        B, C, H, W = features.shape
        f = features.view(B, C, H * W)
        gram = torch.bmm(f, f.transpose(1, 2))
        return gram / (C * H * W)

    def compute_temporal_loss(self, output, prev_output_model, mask, w_temp):
        """
        Feature-level temporal consistency.
        Penalises change in VGG features between consecutive outputs
        in the masked region only.
        No optical flow required.
        """
        if prev_output_model is None or w_temp == 0:
            return torch.tensor(0.0, device=output.device)

        # VGG features for current and previous output
        norm_curr = self.normalize(output)
        norm_prev = self.normalize(prev_output_model)

        # Only need first VGG slice — captures texture without being too abstract
        feat_curr = self.slice1(norm_curr)
        with torch.no_grad():
            feat_prev = self.slice1(norm_prev)

        # Ensure the mask is exactly 1 channel so it broadcasts to 64 channels
        if mask.shape[1] > 1:
            mask = mask[:, 0:1, :, :]

        # Downsample mask to feature resolution
        mask_ds = F.interpolate(
            mask, size=feat_curr.shape[2:], mode='bilinear', align_corners=False
        )
        mask_ds = (mask_ds > 0.5).float()

        # Penalise feature change in masked region
        diff = (feat_curr - feat_prev).abs() * mask_ds
        temp_loss = diff.sum() / (mask_ds.sum() * feat_curr.shape[1] + 1e-8)

        return temp_loss * w_temp

    def forward(self, output, target, mask, prev_output_gt=None,
                prev_output_model=None, discriminator=None, fake_seq=None, weight_overrides=None):
        if mask.shape[1] != output.shape[1]:
            mask = mask.expand_as(output)

        w = weight_overrides or {
            "w_pixel_m": self.pixel_m_w,
            "w_pixel_f": self.pixel_f_w,
            "w_perc": self.perceptual_w,
            "w_style": self.style_w,
            "w_temp": self.temporal_w,
            "w_adv": self.adv_w,
        }

        # Pixel Losses
        l1_mask = self.l1(output * mask, target * mask) * w["w_pixel_m"] * 10.0
        l1_frame = self.l1(output * (1 - mask), target * (1 - mask)) * w["w_pixel_f"] * 1.0

        # Perceptual and Style (VGG)
        norm_output = self.normalize(output)
        norm_target = self.normalize(target)
        out1 = self.slice1(norm_output)
        out2 = self.slice2(out1)
        out3 = self.slice3(out2)
        with torch.no_grad():
            tgt1 = self.slice1(norm_target)
            tgt2 = self.slice2(tgt1)
            tgt3 = self.slice3(tgt2)

        perceptual_loss = (self.l1(out1, tgt1) + self.l1(out2, tgt2) + self.l1(out3, tgt3)) * w["w_perc"]
        style_loss = (self.l1(self.gram_matrix(out1), self.gram_matrix(tgt1)) +
                      self.l1(self.gram_matrix(out2), self.gram_matrix(tgt2)) +
                      self.l1(self.gram_matrix(out3), self.gram_matrix(tgt3))) * w["w_style"]

        # Temporal Loss
        temp_loss = self.compute_temporal_loss(output, prev_output_model, mask, w["w_temp"])

        # Adversarial Loss (Masked Hinge)

        adv_loss = torch.tensor(0.0, device=output.device)
        if discriminator is not None and fake_seq is not None:
            # PatchGAN directly outputs the prediction grid
            g_fake_pred = discriminator(fake_seq)

            mask_seq = fake_seq[:, 3:4, ...]  # Shape: (B, 1, T, H, W)

            # Interpolate mask to match the discriminator's output patch grid mathematically
            downsampled_mask = F.interpolate(
                mask_seq,
                size=g_fake_pred.shape[2:],
                mode='trilinear',
                align_corners=False
            )

            # Generator wants fake to be evaluated as real (> 0)
            masked_pred = -g_fake_pred * downsampled_mask
            adv_loss = masked_pred.sum() / (downsampled_mask.sum() + 1e-8)
            adv_loss = adv_loss * w["w_adv"]

        total_loss = l1_mask + l1_frame + perceptual_loss + style_loss + temp_loss + adv_loss

        return total_loss, l1_mask, l1_frame, perceptual_loss, style_loss, temp_loss, adv_loss
