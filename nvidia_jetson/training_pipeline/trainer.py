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


class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:VGG_FEATURE_LAYER].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.mse = torch.nn.MSELoss()

    def forward(self, x, y):
        return self.mse(self.vgg(x), self.vgg(y))

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

perceptual_criterion = PerceptualLoss().to(device)
l1_criterion = torch.nn.L1Loss()

save_dir = "pictures"
os.makedirs(save_dir, exist_ok=True)

print(f"Starting training on {device}...")

def train():
    current_iter = 0
    while current_iter < NUM_ITERATIONS:
        for video_data in train_loader:  # video_data: (1, T, H, W, C) - one full video
            if current_iter >= NUM_ITERATIONS: break

            # Preprocessing
            video_data = video_data.float() / 255.0
            video_data = video_data.permute(0, 1, 4, 2, 3).to(device)
            B, T, C, H, W = video_data.shape  # B=1, T = total video length

            # Reset ConvGRU hidden state for each new video
            hidden_state = None
            prev_output = None
            total_loss = torch.tensor(0.0, device=device)

            # Generate mask position once per video (so mask moves naturally across frames)
            mask_size = np.random.randint(MASK_SIZE_RANGE[0], MASK_SIZE_RANGE[1])
            y1 = np.random.randint(0, H - mask_size)
            x1 = np.random.randint(0, W - mask_size)

            # Slide SEQ_LEN window through the full video
            for t in range(0, T - SEQ_LEN + 1):
                optimizer.zero_grad()

                # Get current window of SEQ_LEN frames
                window = video_data[:, t:t + SEQ_LEN]  # (1, SEQ_LEN, C, H, W)

                # Build mask for this window
                masks = torch.zeros((B, SEQ_LEN, 1, H, W), device=device)
                masks[:, :, :, y1:y1 + mask_size, x1:x1 + mask_size] = 1.0

                # Remove pixels in mask
                masked_window = window * (1.0 - masks)

                # Stack frames and masks as channels
                pixel_input = masked_window.reshape(B, SEQ_LEN * C, H, W)
                mask_input = masks.reshape(B, SEQ_LEN, H, W)
                full_input = torch.cat([pixel_input, mask_input], dim=1)

                # Target is the last frame of the window unmasked
                target = window[:, -1]  # (1, C, H, W)

                # Forward pass, carrying hidden state across windows
                output, hidden_state = model(full_input, hidden_state)
                hidden_state = hidden_state.detach()  # detach to prevent backprop through full video history

                # Losses
                l1_loss = l1_criterion(output, target) * PIXEL_LOSS_WEIGHT
                vgg_loss = perceptual_criterion(output, target) * PERCEPTUAL_LOSS_WEIGHT

                # Temporal consistency loss (only from second window onwards)
                if prev_output is not None:
                    temporal_loss = l1_criterion(output, prev_output.detach()) * TEMPORAL_LOSS_WEIGHT
                else:
                    temporal_loss = torch.tensor(0.0, device=device)

                prev_output = output

                total_loss = l1_loss + vgg_loss + temporal_loss
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                # Logging
                if current_iter % 10 == 0:
                    print(f"Iter {current_iter} | Total: {total_loss.item():.4f} | L1: {l1_loss.item():.4f} | Temporal: {temporal_loss.item():.4f}")

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