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

        self.bce = torch.nn.BCEWithLogitsLoss()

    def normalize(self, x):
        return (x - self.mean) / self.std

    def gram_matrix(self, features):
        B, C, H, W = features.shape
        f = features.view(B, C, H * W)
        gram = torch.bmm(f, f.transpose(1, 2))
        return gram / (C * H * W)

    def forward(self, output, target, mask, prev_output=None, flow=None, discriminator=None, fake_seq=None):
        if mask.shape[1] != output.shape[1]:
            mask = mask.expand_as(output)

        # Pixel Losses
        l1_mask = self.l1(output * mask, target * mask) * self.pixel_m_w * 10.0
        l1_frame = self.l1(output * (1 - mask), target * (1 - mask)) * self.pixel_f_w * 1.0

        # Perceptual and Style (VGG)
        norm_output = self.normalize(output)
        norm_target = self.normalize(target)
        out1, out2, out3 = self.slice1(norm_output), self.slice2(self.slice1(norm_output)), self.slice3(
            self.slice2(self.slice1(norm_output)))
        with torch.no_grad():
            tgt1, tgt2, tgt3 = self.slice1(norm_target), self.slice2(self.slice1(norm_target)), self.slice3(
                self.slice2(self.slice1(norm_target)))

        perceptual_loss = (self.l1(out1, tgt1) + self.l1(out2, tgt2) + self.l1(out3, tgt3)) * self.perceptual_w
        style_loss = (self.l1(self.gram_matrix(out1), self.gram_matrix(tgt1)) +
                      self.l1(self.gram_matrix(out2), self.gram_matrix(tgt2)) +
                      self.l1(self.gram_matrix(out3), self.gram_matrix(tgt3))) * self.style_w

        # Warped Temporal Loss
        temp_loss = torch.tensor(0.0, device=output.device)
        if prev_output is not None and flow is not None:
            warped_prev = warp(prev_output.detach(), flow)
            temp_loss = self.l1(output, warped_prev) * self.temporal_w

        # Adversarial Loss
        adv_loss = torch.tensor(0.0, device=output.device)
        if discriminator is not None and fake_seq is not None:
            g_fake_pred = discriminator(fake_seq)
            adv_loss = self.bce(g_fake_pred, torch.ones_like(g_fake_pred)) * self.adv_w

        total_loss = l1_mask + l1_frame + perceptual_loss + style_loss + temp_loss + adv_loss

        return total_loss, l1_mask, l1_frame, perceptual_loss, style_loss, temp_loss, adv_loss