import os
import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from training_pipeline.config import *

# Pipeline Imports
from training_pipeline.dataset import *
from model_architecture.video_inpainter import VideoInpainter
from training_pipeline.mask_generator import (
    generate_random_square_mask,
    generate_flying_square_mask,
    generate_arbitrary_shape_mask,
    generate_video_object_mask,
    random_dilate_and_blur_mask
)
from training_pipeline.inpainting_loss import InpaintingLoss
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from training_pipeline.discriminator import SpatioTemporalDiscriminator
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Train Video Inpainter Phase")
    parser.add_argument("--model_name", type=str, required=True, help="Master folder name (e.g. MyModel_V1)")
    parser.add_argument("--phase_name", type=str, required=True, help="Folder name for results")
    parser.add_argument("--iterations", type=int, required=True, default=20000)
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint .pth")

    # Architecture/Data Config
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0001)

    # Curriculum Parameters
    parser.add_argument("--mask_type", type=str, choices=["random", "flying", "arbitrary", "human"], required=True)
    parser.add_argument("--use_memory", action="store_true", help="Enable persistent ConvGRU state")

    # Loss Weights
    parser.add_argument("--w_pixel_m", type=float, required=True)
    parser.add_argument("--w_pixel_f", type=float, required=True)
    parser.add_argument("--w_perc", type=float, required=True)
    parser.add_argument("--w_style", type=float, required=True)
    parser.add_argument("--w_temp", type=float, required=True)
    parser.add_argument("--w_adv", type=float, required=True)

    return parser.parse_args()

def validate(args, model, flow_model, val_loader, val_mask_dataset, criterion, device, save_dir, current_iter):
    model.eval()
    metrics = {"total": [], "mask": [], "frame": [], "psnr": [], "temporal_consistency": []}

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i >= 100:
                break

            video_data = data.float().to(device) / 255.0
            video_data = video_data.permute(0, 1, 4, 2, 3)
            B, T, C, H, W = video_data.shape

            # Always uses arbitrary - This gives a consistent benchmark across all phases
            masks = generate_arbitrary_shape_mask(video_data, val_mask_dataset)
            masks = random_dilate_and_blur_mask(masks)
            masked_video = video_data * (1.0 - masks)

            hidden_state = None
            prev_composited = None

            for t in range(0, T - args.seq_len + 1):
                window = video_data[:, t:t + args.seq_len]
                masked_window = masked_video[:, t:t + args.seq_len]
                masks_window = masks[:, t:t + args.seq_len]

                target = window[:, -1]
                target_mask = masks_window[:, -1]

                pixel_input = masked_window.reshape(B, args.seq_len * C, H, W)
                mask_input = masks_window.reshape(B, args.seq_len, H, W)
                full_input = torch.cat([pixel_input, mask_input], dim=1)

                output, hidden_state = model(full_input, hidden_state)
                if hidden_state is not None:
                    hidden_state = hidden_state.detach()

                composited = output * target_mask + target * (1 - target_mask)

                # Losses — no discriminator during validation
                total_loss, l1_m, l1_f, _, _, _, _ = criterion(
                    output=output, target=target, mask=target_mask,
                    prev_output=None, flow=None,
                    discriminator=None, fake_seq=None
                )

                # PSNR on masked region only
                mse = torch.mean((output * target_mask - target * target_mask) ** 2)
                psnr = 10 * torch.log10(1.0 / (mse + 1e-8))

                # Temporal consistency — how much does the output change between frames
                # A flickering model will score poorly here
                if prev_composited is not None:
                    temp_consistency = torch.mean(
                        torch.abs(composited - prev_composited)
                    ).item()
                    metrics["temporal_consistency"].append(temp_consistency)

                metrics["total"].append(total_loss.item())
                metrics["mask"].append(l1_m.item())
                metrics["frame"].append(l1_f.item())
                metrics["psnr"].append(psnr.item())

                prev_composited = composited

    # Save one validation preview so you can visually inspect
    save_previews(save_dir, f"val_{current_iter}", composited, target, masked_window)

    model.train()
    return {k: np.mean(v) for k, v in metrics.items() if len(v) > 0}


def train(args, model, flow_model, discriminator, train_loader, val_loader, mask_dataset, val_mask_dataset,optimizer_model, optimizer_disc, scheduler_model,
          scheduler_disc, criterion, adversarial_criterion, device, save_dir):

    metrics_history = {"total": [], "mask": [], "frame": [], "perc": [], "style": [], "temp": [], "adv": []}
    best_val_loss = float("inf")
    patience_counter = 0

    current_iter = 0
    print(f"Starting Phase: {args.phase_name} | Mask: {args.mask_type} | Memory: {args.use_memory}")

    while current_iter < args.iterations:
        for data in train_loader:
            if current_iter >= args.iterations: break

            if args.mask_type == "human":
                # Data is a dictionary from HumanInpaintingDataset
                video_data = data["video"].float().to(device) / 255.0
                masks = data["mask"].float().to(device)  # Masks are already 0.0-1.0
            else:
                # Data is a raw tensor from YouTubeVOSDataset
                video_data = data.float().to(device) / 255.0

            # Permute video to (B, T, C, H, W)
            video_data = video_data.permute(0, 1, 4, 2, 3)
            B, T, C, H, W = video_data.shape

            # --- MASK GENERATION ---
            if args.mask_type == "human":
                # masks are already (B, T, 1, H, W) from the HumanMaskDataset
                pass
            else:
                # Generate synthetic masks (ignore the initially returned masked_video)
                if args.mask_type == "random":
                    masks = generate_random_square_mask(video_data)
                elif args.mask_type == "flying":
                    masks = generate_flying_square_mask(video_data)
                elif args.mask_type == "arbitrary":
                    masks = generate_arbitrary_shape_mask(video_data, mask_dataset)

            masks = random_dilate_and_blur_mask(masks)

            masked_video = video_data * (1.0 - masks)

            hidden_state = None
            prev_output = None
            for t in range(0, T - args.seq_len + 1):
                if current_iter >= args.iterations:
                    break

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
                        img1 = (window[:, -2] * 2.0) - 1.0
                        img2 = (window[:, -1] * 2.0) - 1.0

                        flow = flow_model(img1, img2)[-1]

                # Train Discriminator
                optimizer_disc.zero_grad()
                output, hidden_state = model(full_input, hidden_state)

                if args.use_memory and hidden_state is not None:
                    hidden_state = hidden_state.detach()

                fake_frame = output * target_mask + target * (1 - target_mask)
                seq_frames = [window[:, i] for i in range(window.shape[1] - 1)] + [fake_frame]
                fake_seq = torch.stack(seq_frames, dim=2)

                if args.w_adv > 0:
                    # Match the shape for the real pass
                    real_seq = window.permute(0, 2, 1, 3, 4)
                    d_real_loss = adversarial_criterion(discriminator(real_seq),
                                                        torch.ones_like(discriminator(real_seq)))
                    # Fake pass (detach to avoid updating model)
                    d_fake_loss = adversarial_criterion(discriminator(fake_seq.detach()),
                                                        torch.zeros_like(discriminator(fake_seq)))
                    d_loss = (d_real_loss + d_fake_loss) * 0.5
                    d_loss.backward()
                    optimizer_disc.step()

                # Train Model
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

                #Add parameters to metrics_history for logging
                metrics_history["total"].append(total_loss.item())
                metrics_history["mask"].append(l1_m.item())
                metrics_history["frame"].append(l1_f.item())
                metrics_history["perc"].append(perc_v.item())
                metrics_history["style"].append(style_v.item())
                metrics_history["temp"].append(temp_v.item())
                metrics_history["adv"].append(adv.item())

                # Prep for next window
                with torch.no_grad():
                    composited = output * target_mask + target * (1 - target_mask)
                    prev_output = target

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

                if current_iter % 5000 == 0:
                    if args.mask_type != "human" and args.mask_type != "arbitrary":
                        val_metrics = validate(
                            args, model, flow_model, val_loader,
                            val_mask_dataset, criterion, device, save_dir, current_iter
                        )

                        print(
                            f"[VAL] Iter {current_iter} | "
                            f"Total: {val_metrics['total']:.4f} | "
                            f"Mask L1: {val_metrics['mask']:.4f} | "
                            f"PSNR: {val_metrics['psnr']:.2f}dB | "
                            f"Temp Consistency: {val_metrics.get('temporal_consistency', 0):.4f}"
                        )

                        # Save best model
                        if val_metrics["total"] < best_val_loss:
                            best_val_loss = val_metrics["total"]
                            patience_counter = 0
                            torch.save(model.state_dict(),
                                       os.path.join(save_dir, "best_model.pth"))
                            print(f"Model saved")
                        else:
                            patience_counter += 1
                            print(f"No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")
                            if patience_counter >= EARLY_STOP_PATIENCE:
                                print(f"Early stopping triggered at iter {current_iter}")
                                return {k: np.mean(v) for k, v in metrics_history.items()}

                        # Log to file
                        with open(os.path.join(save_dir, "val_log.jsonl"), "a") as f:
                            f.write(json.dumps({"iter": current_iter, **val_metrics}) + "\n")

                elif current_iter % 20000 == 0:
                    print(f"Skipping validation for human/irregular mask phase at iter {current_iter}")
                    torch.save(model.state_dict(),
                                os.path.join(save_dir, f"model_iter_{current_iter}.pth"))
                    torch.save(discriminator.state_dict(),
                               os.path.join(save_dir, f"disc_iter_{current_iter}.pth"))
                    print(f"Model checkpoint saved at iter {current_iter}")

            if current_iter >= args.iterations:
                break

    torch.save(discriminator.state_dict(), os.path.join(save_dir, "final_discriminator.pth"))
    return {k: np.mean(v) for k, v in metrics_history.items()}

def save_previews(save_dir, it, comp, tgt, m_win):
    out_img = (comp[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    target_img = (tgt[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    input_img = (m_win[0, -1].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    cv2.imwrite(os.path.join(save_dir, "image_results", f"iter_{it}_output.png"), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_dir, "image_results", f"iter_{it}_target.png"), cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_dir, "image_results", f"iter_{it}_input.png"), cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save Directory
    master_folder = args.model_name if args.model_name else f"Model_BC{BASE_CHANNELS}_L{NUM_LAYERS}"
    save_dir = os.path.join("results", master_folder, args.phase_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "image_results"), exist_ok=True)

    print(f"Saving this phase to: {save_dir}")

    # Models
    in_channels = args.seq_len * 3 + args.seq_len
    model = VideoInpainter(in_channels=in_channels, base_channels=BASE_CHANNELS, num_layers=NUM_LAYERS).to(device)
    discriminator = SpatioTemporalDiscriminator().to(device)

    if args.resume_from:
        # Load Generator
        model.load_state_dict(torch.load(args.resume_from))
        print(f"Resumed Generator from {args.resume_from}")

        # ADD THIS: Attempt to load Discriminator
        # We assume if the model is 'final_model.pth', the disc is 'final_discriminator.pth'
        disc_path = args.resume_from.replace("final_model.pth", "final_discriminator.pth")

        if os.path.exists(disc_path):
            discriminator.load_state_dict(torch.load(disc_path))
            print(f"Resumed Discriminator from {disc_path}")
        else:
            print("No discriminator checkpoint found. Starting with a fresh Discriminator.")

    flow_model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()

    # Optimizers
    opt_model = optim.Adam(model.parameters(), lr=args.lr)
    opt_disc = optim.Adam(discriminator.parameters(), lr=args.lr * 2.0)

    scheduler_model = CosineAnnealingLR(opt_model, T_max=args.iterations, eta_min=1e-6)
    scheduler_disc = CosineAnnealingLR(opt_disc, T_max=args.iterations, eta_min=1e-7)

    # Loss
    criterion = InpaintingLoss(
        pixel_m_w=args.w_pixel_m, pixel_f_w=args.w_pixel_f,
        perceptual_w=args.w_perc, style_w=args.w_style,
        temporal_w=args.w_temp, adv_w=args.w_adv
    ).to(device)
    adv_crit = torch.nn.MSELoss()

    # Data
    mask_dataset = None
    if args.mask_type == "human":
        print("Initializing Human-centric Inpainting Phase...")
        clean_ds = YouTubeVOSDatasetWithoutHumans(root_dir=os.path.join(os.getcwd(), "training_data", "train"))
        mask_ds = HumanMaskDataset(root_dir=os.path.join(os.getcwd(), "training_data", "train"))

        # This pairs them together automatically
        combined_dataset = HumanInpaintingDataset(clean_ds, mask_ds)
        loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    else:
        print(f"Initializing Synthetic Inpainting Phase: {args.mask_type}")
        dataset = YouTubeVOSDataset(root_dir=os.path.join(os.getcwd(), "training_data", "train"))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

        # Only load the irregular mask PNGs if we are in "arbitrary" mode
        if args.mask_type == "arbitrary":
            mask_dataset = IrregularMaskDataset(
                root_dir=os.path.join(os.getcwd(), "training_data", "irregular_mask", "disocclusion_img_mask"))

    val_dataset = YouTubeVOSDataset(root_dir=os.path.join(os.getcwd(), "training_data", "valid"))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=True)
    val_mask_dataset = IrregularMaskDataset(
        root_dir=os.path.join(os.getcwd(), "training_data", "irregular_mask", "disocclusion_img_mask")
    )

    # Start Training
    final_metrics = train(
        args, model, flow_model, discriminator, loader, val_loader,
        mask_dataset, val_mask_dataset, opt_model, opt_disc,
        scheduler_model, scheduler_disc, criterion, adv_crit, device, save_dir
    )

    log_data = {
        "hyperparameters": vars(args),
        "final_metrics": final_metrics,
        "config_constants": {
            "TARGET_RES": TARGET_RES,
            "MASK_SPEED": MASK_PIXEL_MOVEMENT_SPEED,
            "MASK_SIZE": MASK_SIZE_RANGE,
            "BASE_CHANNELS": BASE_CHANNELS,
            "NUM_LAYERS": NUM_LAYERS
        }
    }
    with open(os.path.join(save_dir, "phase_log.json"), "w") as f:
        json.dump(log_data, f, indent=4)

    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))


if __name__ == '__main__':
    main()