import torch
from torchvision.models import vgg16, VGG16_Weights

class InpaintingLoss(torch.nn.Module):
    def __init__(self, pixel_w, perceptual_w, style_w, temporal_w):
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
        self.pixel_w = pixel_w
        self.perceptual_w = perceptual_w
        self.style_w = style_w
        self.temporal_w = temporal_w

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

    def forward(self, output, target, prev_output=None):
        # Pixel (L1) Loss
        l1_loss = self.l1(output, target) * self.pixel_w

        # Perceptual and Style Loss
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

        style_loss = (
                             self.l1(self.gram_matrix(out1), self.gram_matrix(tgt1)) +
                             self.l1(self.gram_matrix(out2), self.gram_matrix(tgt2)) +
                             self.l1(self.gram_matrix(out3), self.gram_matrix(tgt3))
                     ) * self.style_w

        # Temporal Loss
        temporal_loss = torch.tensor(0.0, device=output.device)
        if prev_output is not None:
            temporal_loss = self.l1(output, prev_output.detach()) * self.temporal_w

        total_loss = l1_loss + perceptual_loss + style_loss + temporal_loss

        return total_loss, l1_loss, perceptual_loss, style_loss, temporal_loss