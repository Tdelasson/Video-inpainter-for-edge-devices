import torch
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
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

        # 1. Initialize and freeze pre-trained RAFT
        self.flow_model = raft_small(weights=Raft_Small_Weights.DEFAULT).eval()
        for param in self.flow_model.parameters():
            param.requires_grad = False

        self.l1 = torch.nn.L1Loss()

        # Weights
        self.pixel_m_w = pixel_m_w
        self.pixel_f_w = pixel_f_w
        self.perceptual_w = perceptual_w
        self.style_w = style_w
        self.temporal_w = temporal_w
        self.adv_w = adv_w

        # ImageNet normalization for VGG
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def gram_matrix(self, features):
        B, C, H, W = features.shape
        f = features.view(B, C, H * W).to(torch.float32)
        gram = torch.bmm(f, f.transpose(1, 2))
        return gram / (C * H * W)

    def compute_temporal_loss(self, output, prev_output_model, mask, w_temp, prev_frame=None, curr_frame=None):
        """
        Optical flow-based temporal consistency using RAFT.
        """
        if prev_output_model is None or w_temp == 0 or prev_frame is None or curr_frame is None:
            return torch.tensor(0.0, device=output.device)

        # 2. RAFT expects float images scaled to [-1, 1]
        img1 = prev_frame * 2.0 - 1.0
        img2 = curr_frame * 2.0 - 1.0

        # Defensive padding: RAFT requires dimensions to be divisible by 8
        H, W = img1.shape[-2:]
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            img1 = F.pad(img1, (0, pad_w, 0, pad_h), mode='replicate')
            img2 = F.pad(img2, (0, pad_w, 0, pad_h), mode='replicate')

        # 3. Predict optical flow from prev_frame to curr_frame
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            list_of_flows = self.flow_model(img1, img2)
            flow = list_of_flows[-1]  # Extract final update iteration flow

        # Remove padding if it was added
        if pad_h > 0 or pad_w > 0:
            flow = flow[..., :H, :W]

        # 4. Warp previous composited model output to match current frame spacing
        warped_prev = warp(prev_output_model, flow)

        # VGG feature penalty on the warped result
        with torch.autocast("cuda", enabled=False):
            norm_curr = self.normalize(output.float())
            norm_prev = self.normalize(warped_prev.float())
            feat_curr = self.slice1(norm_curr)
            feat_prev = self.slice1(norm_prev)

        # Downsample mask to feature resolution
        if mask.shape[1] > 1:
            mask = mask[:, 0:1, :, :]
        mask_ds = F.interpolate(mask, size=feat_curr.shape[2:], mode='bilinear', align_corners=False)
        mask_ds = (mask_ds > 0.5).float()

        diff = (feat_curr - feat_prev).abs() * mask_ds
        temp_loss = diff.sum() / (mask_ds.sum() * feat_curr.shape[1] + 1e-8)
        return temp_loss * w_temp

    def forward(self, output, target, mask, prev_output_gt=None,
                prev_output_model=None, discriminator=None, fake_seq=None, weight_overrides=None,
                prev_frame=None, curr_frame=None):  # 5. Add frame inputs here
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
        with torch.autocast("cuda", enabled=False):
            norm_output = self.normalize(output.float())
            norm_target = self.normalize(target.float())
            out1 = self.slice1(norm_output)
            out2 = self.slice2(out1)
            out3 = self.slice3(out2)
            with torch.no_grad():
                tgt1 = self.slice1(norm_target)
                tgt2 = self.slice2(tgt1)
                tgt3 = self.slice3(tgt2)

            perceptual_loss = (self.l1(out1, tgt1) + self.l1(out2, tgt2) + self.l1(out3, tgt3)) * float(w["w_perc"])
            style_loss = (self.l1(self.gram_matrix(out1), self.gram_matrix(tgt1)) +
                          self.l1(self.gram_matrix(out2), self.gram_matrix(tgt2)) +
                          self.l1(self.gram_matrix(out3), self.gram_matrix(tgt3))) * float(w["w_style"])

        # 6. Pass frames into temporal loss computation
        temp_loss = self.compute_temporal_loss(
            output, prev_output_model, mask, w["w_temp"],
            prev_frame=prev_frame, curr_frame=curr_frame
        )

        # Adversarial Loss (Masked Hinge)
        adv_loss = torch.tensor(0.0, device=output.device)
        if discriminator is not None and fake_seq is not None:
            g_fake_pred = discriminator(fake_seq)
            mask_seq = fake_seq[:, 3:4, ...]
            downsampled_mask = F.interpolate(mask_seq, size=g_fake_pred.shape[2:], mode='trilinear',
                                             align_corners=False)
            masked_pred = -g_fake_pred * downsampled_mask
            adv_loss = masked_pred.sum() / (downsampled_mask.sum() + 1e-8)
            adv_loss = adv_loss * w["w_adv"]

        total_loss = l1_mask + l1_frame + perceptual_loss + style_loss + temp_loss + adv_loss

        return total_loss, l1_mask, l1_frame, perceptual_loss, style_loss, temp_loss, adv_loss