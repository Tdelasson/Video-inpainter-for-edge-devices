import os
import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from training_pipeline.config import *

# Pipeline Imports
from training_pipeline.dataset import YouTubeVOSDataset
from model_architecture.video_inpainter import VideoInpainter
from training_pipeline.mask_generator import (
    generate_random_square_mask,
    generate_flying_square_mask,
    generate_arbitrary_shape_mask,
    generate_video_object_mask
)
from training_pipeline.inpainting_loss import InpaintingLoss
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from training_pipeline.discriminator import SpatioTemporalDiscriminator
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train Video Inpainter Phase")
    parser.add_argument("--phase_name", type=str, required=True, help="Folder name for results")
    parser.add_argument("--iterations", type=int, required=True, default=20000)
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint .pth")

    # Architecture/Data Config
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0001)

    # Curriculum Parameters
    parser.add_argument("--mask_type", type=str, choices=["random", "flying", "arbitrary", "youtube"], required=True)
    parser.add_argument("--use_memory", action="store_true", help="Enable persistent ConvGRU state")

    # Loss Weights
    parser.add_argument("--w_pixel_m", type=float, default=MASK_PIXEL_LOSS_WEIGHT)
    parser.add_argument("--w_pixel_f", type=float, default=FRAME_PIXEL_LOSS_WEIGHT)
    parser.add_argument("--w_perc", type=float, default=PERCEPTUAL_LOSS_WEIGHT)
    parser.add_argument("--w_style", type=float, default=STYLE_LOSS_WEIGHT)
    parser.add_argument("--w_temp", type=float, default=TEMPORAL_LOSS_WEIGHT)
    parser.add_argument("--w_adv", type=float, default=ADVERSARIAL_LOSS_WEIGHT)

    return parser.parse_args()


def train(args, model, flow_model, discriminator, train_loader, optimizer_model, optimizer_disc, scheduler_model,
          scheduler_disc, criterion, adversarial_criterion, device, save_dir):
    mask_generators = {
        "random": generate_random_square_mask,
        "flying": generate_flying_square_mask,
        "arbitrary": generate_arbitrary_shape_mask,
        "youtube": generate_video_object_mask
    }
    generate_mask = mask_generators[args.mask_type]

    current_iter = 0
    print(f"Starting Phase: {args.phase_name} | Mask: {args.mask_type} | Memory: {args.use_memory}")

    while current_iter < args.iterations:
        for video_data in train_loader:
            if current_iter >= args.iterations: break

            # Preprocessing
            video_data = video_data.float().to(device) / 255.0
            video_data = video_data.permute(0, 1, 4, 2, 3)
            B, T, C, H, W = video_data.shape

            hidden_state = None
            prev_output = None
            masked_video, masks = generate_mask(video_data)

            for t in range(0, T - args.seq_len + 1):
                if not args.use_memory:
                    hidden_state = None

                window = video_data[:, t:t + args.seq_len]
                masked_window = masked_video[:, t:t + args.seq_len]
                masks_window = masks[:, t:t + args.seq_len]

                target = window[:, -1]
                target_mask = masks_window[:, -1]

                pixel_input = masked_window.reshape(B, args.seq_len * C, H, W)
                mask_input = masks_window.reshape(B, args.seq_len, H, W)
                full_input = torch.cat([pixel_input, mask_input], dim=1)

                # Optical Flow
                flow = None
                if t > 0 and args.w_temp > 0:
                    with torch.no_grad():
                        flow = flow_model(window[:, -2] * 255, window[:, -1] * 255)[-1]

                # 2. Train Discriminator
                optimizer_disc.zero_grad()
                output, hidden_state = model(full_input, hidden_state)

                if args.use_memory and hidden_state is not None:
                    hidden_state = hidden_state.detach()

                fake_frame = output * target_mask + target * (1 - target_mask)
                fake_seq = torch.stack([window[:, -3], window[:, -2], fake_frame], dim=2)

                if args.w_adv > 0:
                    # Real pass
                    real_seq = window[:, -3:].permute(0, 2, 1, 3, 4)
                    d_real_loss = adversarial_criterion(discriminator(real_seq),
                                                        torch.ones_like(discriminator(real_seq)))
                    # Fake pass (detach to avoid updating model)
                    d_fake_loss = adversarial_criterion(discriminator(fake_seq.detach()),
                                                        torch.zeros_like(discriminator(fake_seq)))
                    d_loss = (d_real_loss + d_fake_loss) * 0.5
                    d_loss.backward()
                    optimizer_disc.step()

                # 3. Train Model
                optimizer_model.zero_grad()
                total_loss, l1_m, l1_f, perc_v, style_v, temp_v, adv = criterion(
                    output=output, target=target, mask=target_mask,
                    prev_output=prev_output, flow=flow,
                    discriminator=discriminator, fake_seq=fake_seq
                )
                total_loss.backward()
                optimizer_model.step()

                # Step Schedulers
                scheduler_model.step()
                scheduler_disc.step()

                # Prep for next window
                with torch.no_grad():
                    composited = output * target_mask + target * (1 - target_mask)
                    prev_output = composited

                # Logging & Saving
                if current_iter % 10 == 0:
                    print(
                        f"[{args.phase_name}] Iter {current_iter} | "
                        f"Total: {total_loss.item():.4f} | "
                        f"Mask: {l1_m.item():.4f} | "
                        f"Frame: {l1_f.item():.4f} | "
                        f"Perc: {perc_v.item():.4f} | "
                        f"Style: {style_v.item():.4f} | "
                        f"Temp: {temp_v.item():.4f} | "
                        f"Adv: {adv.item():.4f}"
                    )

                if current_iter % 500 == 0:
                    save_previews(save_dir, current_iter, composited, target, masked_window)

                current_iter += 1

def save_previews(save_dir, it, comp, tgt, m_win):
    out_img = (comp[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    target_img = (tgt[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    input_img = (m_win[0, -1].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    cv2.imwrite(os.path.join(save_dir, f"iter_{it}_output.png"), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_dir, f"iter_{it}_target.png"), cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_dir, f"iter_{it}_input.png"), cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save Directory
    save_dir = os.path.join("results", args.phase_name)
    os.makedirs(save_dir, exist_ok=True)

    # Models
    in_channels = args.seq_len * 3 + args.seq_len
    model = VideoInpainter(in_channels=in_channels, base_channels=BASE_CHANNELS, num_layers=NUM_LAYERS).to(device)
    if args.resume_from:
        model.load_state_dict(torch.load(args.resume_from))
        print(f"Resumed from {args.resume_from}")

    flow_model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()
    discriminator = SpatioTemporalDiscriminator().to(device)

    # Optimizers
    opt_model = optim.Adam(model.parameters(), lr=args.lr)
    opt_disc = optim.Adam(discriminator.parameters(), lr=args.lr * 0.1)

    scheduler_model = CosineAnnealingLR(opt_model, T_max=args.iterations, eta_min=1e-6)
    scheduler_disc = CosineAnnealingLR(opt_disc, T_max=args.iterations, eta_min=1e-7)

    # Loss
    criterion = InpaintingLoss(
        pixel_m_w=args.w_pixel_m, pixel_f_w=args.w_pixel_f,
        perceptual_w=args.w_perc, style_w=args.w_style,
        temporal_w=args.w_temp, adv_w=args.w_adv
    ).to(device)
    adv_crit = torch.nn.BCEWithLogitsLoss()

    # Data
    dataset = YouTubeVOSDataset(root_dir=os.path.join(os.getcwd(), "training_data", "train"))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # Start Training
    train(args, model, flow_model, discriminator, loader, opt_model, opt_disc, scheduler_model, scheduler_disc, criterion,
          adv_crit, device, save_dir)

    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))


if __name__ == '__main__':
    main()