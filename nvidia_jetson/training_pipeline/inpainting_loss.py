import torch
from torchvision.models import vgg16, VGG16_Weights
from training_pipeline.warp import warp

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

        self.adv_criterion = torch.nn.MSELoss()


    def normalize(self, x):
        return (x - self.mean) / self.std

    def gram_matrix(self, features):
        B, C, H, W = features.shape
        f = features.view(B, C, H * W)
        gram = torch.bmm(f, f.transpose(1, 2))
        return gram / (C * H * W)

    def forward(self, output, target, mask, prev_output_gt=None,
            prev_output_model=None, flow=None, discriminator=None, fake_seq=None):
        if mask.shape[1] != output.shape[1]:
            mask = mask.expand_as(output)

        # Pixel Losses
        l1_mask = self.l1(output * mask, target * mask) * self.pixel_m_w * 10.0
        l1_frame = self.l1(output * (1 - mask), target * (1 - mask)) * self.pixel_f_w * 1.0

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

        perceptual_loss = (self.l1(out1, tgt1) + self.l1(out2, tgt2) + self.l1(out3, tgt3)) * self.perceptual_w
        style_loss = (self.l1(self.gram_matrix(out1), self.gram_matrix(tgt1)) +
                      self.l1(self.gram_matrix(out2), self.gram_matrix(tgt2)) +
                      self.l1(self.gram_matrix(out3), self.gram_matrix(tgt3))) * self.style_w

        # Warped Temporal Loss
        temp_loss = torch.tensor(0.0, device=output.device)
        if prev_output_gt is not None and prev_output_model is not None and flow is not None:
            warped_gt = warp(prev_output_gt, flow)
            warped_model = warp(prev_output_model, flow)

            unmasked = (1 - mask)

            # Known pixels: enforce consistency with warped GT
            temp_loss_unmasked = self.l1(
                output * unmasked,
                warped_gt * unmasked
            )

            # Masked pixels: enforce consistency with warped previous model output
            temp_loss_masked = self.l1(
                output * mask,
                warped_model * mask
            )

            temp_loss = (temp_loss_unmasked + temp_loss_masked) * self.temporal_w

        # Adversarial Loss
        adv_loss = torch.tensor(0.0, device=output.device)
        if discriminator is not None and fake_seq is not None:
            g_fake_pred = discriminator(fake_seq)
            adv_loss = self.adv_criterion(g_fake_pred, torch.ones_like(g_fake_pred)) * self.adv_w

        total_loss = l1_mask + l1_frame + perceptual_loss + style_loss + temp_loss + adv_loss

        return total_loss, l1_mask, l1_frame, perceptual_loss, style_loss, temp_loss, adv_loss