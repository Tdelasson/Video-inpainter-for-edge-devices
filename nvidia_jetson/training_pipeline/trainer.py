import os
import cv2
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from training_pipeline.dataset import YouTubeVOSDataset
from model_architecture.video_inpainter import VideoInpainter
from training_pipeline.mask_generator import *
from training_pipeline.inpainting_loss import InpaintingLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET & DATALOADER ---
root_dir = os.getcwd()
train_path = os.path.join(root_dir, "training_data", "train")
dataset = YouTubeVOSDataset(root_dir=train_path)
train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers = 4,  # multiple cpu cores loads data, 0 on windows, 4 or more on linux
    drop_last=True,
)

# --- MODEL SETUP ---
IN_CHANNELS = SEQ_LEN * 3 + SEQ_LEN # per frame 3 RGBs + 1 mask channels
model = VideoInpainter(in_channels=IN_CHANNELS, base_channels=BASE_CHANNELS, num_layers=NUM_LAYERS).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_ITERATIONS, eta_min=1e-6)
criterion = InpaintingLoss(
    pixel_w=PIXEL_LOSS_WEIGHT,
    perceptual_w=PERCEPTUAL_LOSS_WEIGHT,
    style_w=STYLE_LOSS_WEIGHT,
    temporal_w=TEMPORAL_LOSS_WEIGHT
).to(device)

# --- RESULT SAVING SETUP ---
folder_name = (
    f"BC{BASE_CHANNELS}_"
    f"L{NUM_LAYERS}_"
    f"SL{SEQ_LEN}_"
    f"LR{LEARNING_RATE}_"
    f"PL{PIXEL_LOSS_WEIGHT}_"
    f"PR{PERCEPTUAL_LOSS_WEIGHT}_"
    f"T{TEMPORAL_LOSS_WEIGHT}_"
    f"ST{STYLE_LOSS_WEIGHT}"
)
save_dir = os.path.join("results", folder_name)
os.makedirs(save_dir, exist_ok=True)
print(f"Results will be saved to: {save_dir}")

# --- TRAINING LOOP ---
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
                total_loss, l1_v, perc_v, style_v, temp_v = criterion(composited, target, prev_output)

                prev_output = composited

                total_loss.backward()
                optimizer.step()
                scheduler.step()

                # Logging
                if current_iter % 10 == 0:
                    print(
                        f"Iter {current_iter} | "
                        f"Total: {total_loss.item():.4f} | "
                        f"L1: {l1_v.item():.4f} | "
                        f"Perceptual: {perc_v.item():.4f} | "
                        f"Style: {style_v.item():.4f} | "
                        f"Temporal: {temp_v.item():.4f}",
                        flush=True
                    )

                if current_iter % 500 == 0:
                    out_img = (composited[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    out_img_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

                    target_img = (target[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    target_img_bgr = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)

                    input_img = (masked_window[0, -1].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                    cv2.imwrite(os.path.join(save_dir, f"iter_{current_iter}_output.png"), out_img_bgr)
                    cv2.imwrite(os.path.join(save_dir, f"iter_{current_iter}_target.png"), target_img_bgr)
                    cv2.imwrite(os.path.join(save_dir, f"iter_{current_iter}_input.png"),
                                cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))

                current_iter += 1

    torch.save(model.state_dict(), save_dir)
    print(f"Training Done! Model saved to: {save_dir}")


if __name__ == '__main__':
    train()