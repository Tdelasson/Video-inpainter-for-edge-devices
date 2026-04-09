import os
import cv2
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import vgg16, VGG16_Weights

from training_pipeline.dataset import YouTubeVOSDataset
from model_architecture.video_inpainter import VideoInpainter


class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.mse = torch.nn.MSELoss()

    def forward(self, x, y):
        return self.mse(self.vgg(x), self.vgg(y))


# --- CONFIG ---
TARGET_RES = (256, 256)
BATCH_SIZE = 1
NUM_ITERATIONS = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET & DATALOADER ---
root_dir = os.getcwd()
train_path = os.path.join(root_dir, "training_data", "train")
dataset = YouTubeVOSDataset(root_dir=train_path, seq_len=5)

# DataLoaderen sørger for at loade data i baggrunden
train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,  # Bruger 4 CPU kerner til at pre-loade data
    drop_last=True
)

# --- MODEL SETUP ---
IN_CHANNELS = 5 * 3 + 5 # seq_len * RGB + mask channels
model = VideoInpainter(in_channels=IN_CHANNELS, base_channels=128, num_layers=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_ITERATIONS, eta_min=1e-6)

perceptual_criterion = PerceptualLoss().to(device)
l1_criterion = torch.nn.L1Loss()

save_dir = "pictures"
os.makedirs(save_dir, exist_ok=True)

print(f"Starting training on {device}...")

def train():
    current_iter = 0
    while current_iter < NUM_ITERATIONS:
        for batch_data in train_loader:
            if current_iter >= NUM_ITERATIONS: break

            optimizer.zero_grad()

            # Preprocessing af hele batchen (B, Seq, H, W, C) -> (B, Seq, C, H, W)
            batch_data = batch_data.float() / 255.0
            batch_data = batch_data.permute(0, 1, 4, 2, 3).to(device)
            B, S, C, H, W = batch_data.shape

            # 1. Lav masken (B, S, 1, H, W)
            # Lad os sige vi laver en 80x80 boks med tilfældig placering i hvert billede
            mask_size = 80
            y1 = np.random.randint(0, H - mask_size)
            x1 = np.random.randint(0, W - mask_size)

            masks = torch.zeros((B, S, 1, 256, 256)).to(device)
            masks[:, :, :, y1:y1 + mask_size, x1:x1 + mask_size] = 1.0

            # 2. FJERN pixels fra billedet (Gør dem sorte)
            masked_batch = batch_data * (1.0 - masks)

            # 3. Forbered model input (Merge frames til kanaler)
            # Vi reshaper så vi har (B, 5*3, 256, 256) pixels
            pixel_input = masked_batch.reshape(B, -1, 256, 256)
            mask_input = masks.reshape(B, -1, 256, 256)

            # 4. Sæt dem sammen så modellen ser både de sorte billeder og hvor hullerne er
            # Dette giver en tensor med 20 kanaler (15 RGB + 5 Mask)
            full_input = torch.cat([pixel_input, mask_input], dim=1)

            target = batch_data[:, -1]  # (B, C, 256, 256)

            # Model forward
            output, _ = model(full_input.unsqueeze(1))
            output = output.squeeze(1)  # Fjern seq dim igen

            # Loss beregning
            l1_loss = l1_criterion(output, target)
            vgg_loss = perceptual_criterion(output, target)
            total_loss = l1_loss + (0.1 * vgg_loss)

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if current_iter % 10 == 0:
                print(f"Iter {current_iter} | Total: {total_loss.item():.4f} | L1: {l1_loss.item():.4f}")

            if current_iter % 500 == 0:
                out_img = (output[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                out_img_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

                target_img = (target[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                target_img_bgr = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)

                input_img = (masked_batch[0, -1].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                out_path = os.path.join(save_dir, f"iter_{current_iter}_output.png")
                target_path = os.path.join(save_dir, f"iter_{current_iter}_target.png")

                cv2.imwrite(out_path, out_img_bgr)
                cv2.imwrite(target_path, target_img_bgr)
                cv2.imwrite(os.path.join(save_dir, f"iter_{current_iter}_input.png"),
                            cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))

            current_iter += 1

    print("Training Done!")

if __name__ == '__main__':
    train()