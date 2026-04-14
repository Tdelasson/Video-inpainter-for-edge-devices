import os
import cv2
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import vgg16, VGG16_Weights
from training_pipeline.config import *

from training_pipeline.dataset import YouTubeVOSDataset
from model_architecture.video_inpainter import VideoInpainter
from training_pipeline.mask_generator import *


class PerceptualAndStyleLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualAndStyleLoss, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False

        # Extract features at multiple layers
        # pool1=4, pool2=9, pool3=16
        self.slice1 = vgg[:5]  # up to pool1
        self.slice2 = vgg[5:10]  # up to pool2
        self.slice3 = vgg[10:17]  # up to pool3

        self.l1 = torch.nn.L1Loss()

        # ImageNet normalization — VGG was trained with these
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def gram_matrix(self, features):
        B, C, H, W = features.shape
        # Reshape to (B, C, H*W)
        f = features.view(B, C, H * W)
        # Gram matrix: (B, C, C)
        gram = torch.bmm(f, f.transpose(1, 2))
        # Normalize by number of elements
        return gram / (C * H * W)

    def forward(self, output, target):
        # Normalize before passing to VGG
        output = self.normalize(output)
        target = self.normalize(target)

        # Get features at each layer for both output and target
        out1 = self.slice1(output)
        out2 = self.slice2(out1)
        out3 = self.slice3(out2)

        with torch.no_grad():
            tgt1 = self.slice1(target)
            tgt2 = self.slice2(tgt1)
            tgt3 = self.slice3(tgt2)

        # Perceptual loss: feature map distances
        perceptual_loss = (
                self.l1(out1, tgt1) +
                self.l1(out2, tgt2) +
                self.l1(out3, tgt3)
        )

        # Style loss: Gram matrix distances
        style_loss = (
                self.l1(self.gram_matrix(out1), self.gram_matrix(tgt1)) +
                self.l1(self.gram_matrix(out2), self.gram_matrix(tgt2)) +
                self.l1(self.gram_matrix(out3), self.gram_matrix(tgt3))
        )

        return perceptual_loss, style_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET & DATALOADER ---
root_dir = os.getcwd()
train_path = os.path.join(root_dir, "training_data", "train")
dataset = YouTubeVOSDataset(root_dir=train_path)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers = 0,  # multiple cpu cores loads data, 0 on windows, 4 or more on linux
    drop_last=True,
)

# --- MODEL SETUP ---
IN_CHANNELS = SEQ_LEN * 3 + SEQ_LEN # per frame 3 RGBs + 1 mask channels
model = VideoInpainter(in_channels=IN_CHANNELS, base_channels=BASE_CHANNELS, num_layers=NUM_LAYERS).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_ITERATIONS, eta_min=1e-6)

perceptual_criterion = PerceptualAndStyleLoss().to(device)
l1_criterion = torch.nn.L1Loss()

save_dir = "pictures"
os.makedirs(save_dir, exist_ok=True)

print(f"Starting training on {device}...")

def train():
    current_iter = 0
    while current_iter < NUM_ITERATIONS:
        for video_data in train_loader:
            if current_iter >= NUM_ITERATIONS: break

            # Preprocessing
            video_data = video_data.float() / 255.0
            video_data = video_data.permute(0, 1, 4, 2, 3).to(device)
            B, T, C, H, W = video_data.shape

            # Reset ConvGRU hidden state and prev output for each new video
            hidden_state = None
            prev_output = None

            # Generate mask for full video once based on current iteration phase
            if current_iter < ITERATIONS_RANDOM_SQUARE:
                masked_video, masks = generate_random_square_mask(video_data)
            elif current_iter < ITERATIONS_RANDOM_SQUARE + ITERATIONS_FLYING_SQUARE:
                masked_video, masks = generate_flying_square_mask(video_data)
            elif current_iter < ITERATIONS_RANDOM_SQUARE + ITERATIONS_FLYING_SQUARE + ITERATIONS_ARBITRARY_MASK:
                masked_video, masks = generate_arbitrary_shape_mask(video_data)
            else:
                masked_video, masks = generate_video_object_mask(video_data)

            # Slide SEQ_LEN window through the full video
            for t in range(0, T - SEQ_LEN + 1):
                optimizer.zero_grad()

                # Slice current window
                window = video_data[:, t:t + SEQ_LEN]
                masked_window = masked_video[:, t:t + SEQ_LEN]
                masks_window = masks[:, t:t + SEQ_LEN]

                # Stack frames and masks as channels
                pixel_input = masked_window.reshape(B, SEQ_LEN * C, H, W)
                mask_input = masks_window.reshape(B, SEQ_LEN, H, W)
                full_input = torch.cat([pixel_input, mask_input], dim=1)

                # Target is the last frame of the window unmasked
                target = window[:, -1]

                # Forward pass, carrying hidden state across windows
                output, _ = model(full_input)
                # hidden_state = hidden_state.detach()

                # Composite: use model output only where masked, original pixels elsewhere
                current_mask = masks_window[:, -1]  # (B, 1, H, W) — mask for target frame
                composited = output * current_mask + target * (1 - current_mask)

                # Losses
                l1_loss = l1_criterion(composited, target) * PIXEL_LOSS_WEIGHT
                perceptual_loss, style_loss = perceptual_criterion(output, target)
                perceptual_loss = perceptual_loss * PERCEPTUAL_LOSS_WEIGHT
                style_loss = style_loss * STYLE_LOSS_WEIGHT

                if prev_output is not None:
                    temporal_loss = l1_criterion(composited, prev_output.detach()) * TEMPORAL_LOSS_WEIGHT
                else:
                    temporal_loss = torch.tensor(0.0, device=device)

                prev_output = output
                total_loss = l1_loss + perceptual_loss + style_loss + temporal_loss

                total_loss.backward()
                optimizer.step()
                scheduler.step()

                # Logging
                if current_iter % 10 == 0:
                    print(
                        f"Iter {current_iter} | "
                        f"Total: {total_loss.item():.4f} | "
                        f"L1: {l1_loss.item():.4f} | "
                        f"Perceptual: {perceptual_loss.item():.4f} | "
                        f"Style: {style_loss.item():.4f} | "
                        f"Temporal: {temporal_loss.item():.4f}",
                        flush=True
                    )

                if current_iter % 500 == 0:
                    out_img = (output[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    out_img_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

                    target_img = (target[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    target_img_bgr = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)

                    input_img = (masked_window[0, -1].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                    cv2.imwrite(os.path.join(save_dir, f"iter_{current_iter}_output.png"), out_img_bgr)
                    cv2.imwrite(os.path.join(save_dir, f"iter_{current_iter}_target.png"), target_img_bgr)
                    cv2.imwrite(os.path.join(save_dir, f"iter_{current_iter}_input.png"),
                                cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))

                current_iter += 1

    print("Training Done!")

if __name__ == '__main__':
    train()