import os
import cv2
import numpy as np
import torch
import sys
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from training_pipeline.config import *
from training_pipeline.warp import warp
import torch.nn.functional as F

# Pipeline Imports
from training_pipeline.dataset import *
from model_architecture.viper import Viper
from training_pipeline.mask_generator import (
    generate_random_square_mask,
    generate_flying_square_mask,
    generate_arbitrary_shape_mask,
    generate_video_object_mask,
    random_dilate_and_blur_mask
)
from training_pipeline.inpainting_loss import InpaintingLoss
from training_pipeline.discriminator import PretrainedPatchDiscriminator, NoisyDiscriminator
import argparse
import json
from torch.amp import autocast, GradScaler



def get_loss_weights(current_iter, total_iters, args):
    """Continuously scheduled loss weights — no discrete phase jumps."""

    # Warmup fraction for each loss
    perc_warmup = 0.05  # perceptual active almost immediately
    style_warmup = 0.10  # style follows shortly after
    temp_warmup = 0.20  # temporal needs some spatial foundation first
    adv_warmup = 0.40  # adversarial introduced once model is competent

    def ramp(start_frac, end_frac=None, target=1.0):
        end_frac = end_frac or (start_frac + 0.20)
        progress = (current_iter / total_iters - start_frac) / (end_frac - start_frac)
        return float(np.clip(progress, 0.0, 1.0)) * target

    return {
        "w_pixel_m": args.w_pixel_m,  # always on
        "w_pixel_f": args.w_pixel_f,  # always on
        "w_perc": args.w_perc * ramp(perc_warmup),
        "w_style": args.w_style * ramp(style_warmup),
        "w_temp": args.w_temp * ramp(temp_warmup),
        "w_adv": args.w_adv * ramp(adv_warmup),
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Train Video Inpainter Phase")
    parser.add_argument("--model_name", type=str, required=True, help="Master folder name (e.g. MyModel_V1)")
    parser.add_argument("--phase_name", type=str, required=True, help="Folder name for results")
    parser.add_argument("--iterations", type=int, required=True, default=20000)
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Full training checkpoint to resume from (includes iter count)")

    # Architecture/Data Config
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0001)

    # Curriculum Parameters
    parser.add_argument("--use_memory", action="store_true", help="Enable persistent ConvGRU state")

    # Loss Weights
    parser.add_argument("--w_pixel_m", type=float, required=True)
    parser.add_argument("--w_pixel_f", type=float, required=True)
    parser.add_argument("--w_perc", type=float, required=True)
    parser.add_argument("--w_style", type=float, required=True)
    parser.add_argument("--w_temp", type=float, required=True)
    parser.add_argument("--w_adv", type=float, required=True)

    parser.add_argument("--save_dir", type=str, default="results", help="Base directory for results")

    return parser.parse_args()


def get_human_mask(video_data, human_mask_dataset):
    """Sample a human silhouette mask sequence from the HumanMaskDataset."""
    B, T, C, H, W = video_data.shape
    device = video_data.device
    all_masks = []

    for b in range(B):
        # Sample a random mask video sequence
        idx = np.random.randint(0, len(human_mask_dataset))
        mask_seq = human_mask_dataset[idx]  # Returns (Seq_Len, 1, H, W) tensor

        # Match temporal length
        t_len = min(mask_seq.shape[0], T)
        mask_seq = mask_seq[:t_len]

        # Resize the entire sequence at once to match video resolution
        mask_resized = F.interpolate(
            mask_seq.float(),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        all_masks.append(mask_resized)

    return torch.stack(all_masks).to(device)  # (B, T, 1, H, W)


def validate(args, model, val_loader, val_mask_dataset, human_mask_dataset, criterion, device, save_dir, current_iter):
    model.eval()
    metrics = {"total": [], "mask": [], "frame": [], "psnr": [], "temporal_consistency": []}

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i >= 10:
                break

            if isinstance(data, dict) and "video" in data:
                video_data = data["video"].float().to(device) / 255.0
            else:
                video_data = data.float().to(device) / 255.0

            video_data = video_data.permute(0, 1, 4, 2, 3)
            B, T, C, H, W = video_data.shape

            masks = get_mask_for_iter(current_iter, args.iterations, video_data,
                                      val_mask_dataset, human_mask_dataset)

            masks = random_dilate_and_blur_mask(masks)
            masked_video = video_data * (1.0 - masks)

            hidden_state = None
            prev_composited = None

            comps = []
            tgts = []
            ins = []

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

                if i == 0:
                    comps.append(composited)
                    tgts.append(target)
                    ins.append(masked_window)

                total_loss, l1_m, l1_f, _, _, _, _ = criterion(
                    output=output, target=target, mask=target_mask,
                    prev_output_gt=None,
                    prev_output_model=None,
                    discriminator=None, fake_seq=None
                )

                mse = torch.mean((output * target_mask - target * target_mask) ** 2)
                psnr = 10 * torch.log10(1.0 / (mse + 1e-8))

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

            if i == 0 and comps:
                save_video_previews(save_dir, f"val_{current_iter}", comps, tgts, ins)

    model.train()
    return {k: np.mean(v) for k, v in metrics.items() if len(v) > 0}

class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # Needed for Python 3 compatibility
        self.terminal.flush()
        self.log.flush()

def get_mask_for_iter(current_iter, total_iters, video_data,
                       mask_dataset, human_mask_dataset=None):
    progress = current_iter / total_iters
    H, W = video_data.shape[3], video_data.shape[4]

    # After 70% of training, mix in human masks
    if human_mask_dataset is not None and progress > 0.70:
        return get_human_mask(video_data, human_mask_dataset)

    # Arbitrary mask with growing size
    min_frac = 0.10
    max_frac = 0.40
    current_max_frac = min_frac + progress * (max_frac - min_frac)
    current_max_size = int(current_max_frac * min(H, W))
    current_min_size = max(20, int(current_max_size * 0.3))

    return generate_arbitrary_shape_mask(
        video_data, mask_dataset,
        size_range=(current_min_size, current_max_size)
    )

def train(args, model, discriminator, train_loader, val_loader, mask_dataset, val_mask_dataset, human_mask_dataset,
          optimizer_model, optimizer_disc, scheduler_model,
          scheduler_disc, criterion, adversarial_criterion, device, save_dir, start_iter=0):
    metrics_history = {"total": [], "mask": [], "frame": [], "perc": [], "style": [], "temp": [], "adv": []}
    best_val_loss = float("inf")
    patience_counter = 0

    current_iter = start_iter
    print(f"Starting Phase: {args.phase_name} | Memory: {args.use_memory}")

    # Initialize scalers (already created in main(), but ensure they're passed or created here)
    scaler_model = GradScaler()
    scaler_disc = GradScaler()

    while current_iter < args.iterations:
        for data in train_loader:
            if current_iter >= args.iterations: break

            if isinstance(data, dict) and "video" in data:
                video_data = data["video"].float().to(device) / 255.0
            else:
                video_data = data.float().to(device) / 255.0

            video_data = video_data.permute(0, 1, 4, 2, 3)
            masks = get_mask_for_iter(current_iter, args.iterations, video_data,
                                      mask_dataset, human_mask_dataset)

            B, T, C, H, W = video_data.shape
            masks = random_dilate_and_blur_mask(masks)
            masked_video = video_data * (1.0 - masks)

            hidden_state = None
            prev_output_gt = None
            prev_output_model = None

            all_real_seqs = []
            all_fake_seqs = []

            optimizer_model.zero_grad()
            optimizer_disc.zero_grad()

            accumulated_loss = 0
            total_t_steps = T - args.seq_len + 1

            disc_stepped = False
            gen_stepped = False

            noise_decay_iters = args.iterations * 0.75
            current_sigma = max(0.0, 0.00 * (1.0 - (current_iter / noise_decay_iters)))
            discriminator.set_std(current_sigma)

            weights = get_loss_weights(current_iter, args.iterations, args)

            for t in range(0, total_t_steps):
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

                # Wrap model forward pass in autocast
                with autocast("cuda"):
                    output, hidden_state = model(full_input, hidden_state)

                composited = output * target_mask + target * (1 - target_mask)

                # Generate Fake Sequence
                fake_frame = composited
                seq_frames = [window[:, i] for i in range(window.shape[1] - 1)] + [fake_frame]
                fake_pixel_seq = torch.stack(seq_frames, dim=2)
                mask_seq = masks_window.permute(0, 2, 1, 3, 4)

                fake_seq = torch.cat([fake_pixel_seq, mask_seq], dim=1)
                real_pixel_seq = window.permute(0, 2, 1, 3, 4)
                real_seq = torch.cat([real_pixel_seq, mask_seq], dim=1)

                all_real_seqs.append(real_seq.detach())
                all_fake_seqs.append(fake_seq.detach())

                if current_iter > 0:
                    weights = get_loss_weights(current_iter, args.iterations, args)

                    # Wrap criterion forward pass in autocast
                    with autocast("cuda"):
                        total_loss, l1_m, l1_f, perc_v, style_v, temp_v, adv = criterion(
                            output=output, target=target, mask=target_mask,
                            prev_output_gt=prev_output_gt,
                            prev_output_model=prev_output_model,
                            discriminator=discriminator,
                            fake_seq=fake_seq,
                            weight_overrides=weights
                        )

                    accumulated_loss = accumulated_loss + (total_loss / total_t_steps)
                    gen_stepped = True

                else:
                    with torch.no_grad(), autocast("cuda"):
                        total_loss, l1_m, l1_f, perc_v, style_v, temp_v, adv = criterion(
                            output=output,
                            target=target,
                            mask=target_mask,
                            prev_output_gt=prev_output_gt,
                            prev_output_model=prev_output_model,
                            discriminator=discriminator,
                            fake_seq=fake_seq,
                            weight_overrides=weights  # Added this line
                        )

                with torch.no_grad():
                    prev_output_gt = target
                    prev_output_model = composited

            # Generator step with scaler
            if gen_stepped:
                # Scale loss and backward
                scaler_model.scale(accumulated_loss).backward()

                # Unscale before gradient clipping
                scaler_model.unscale_(optimizer_model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                # Scaled step and update
                scaler_model.step(optimizer_model)
                scaler_model.update()
                scheduler_model.step()

            # Discriminator step with scaler
            if args.w_adv > 0 and all_real_seqs:
                # Iterate sequentially to prevent OOM
                for real_seq, fake_seq in zip(all_real_seqs, all_fake_seqs):
                    with autocast("cuda"):
                        real_pred = discriminator(real_seq)
                        fake_pred = discriminator(fake_seq)
                        # Average the loss across the temporal steps
                        step_d_loss = (F.relu(1.0 - real_pred).mean() + F.relu(1.0 + fake_pred).mean()) * 0.5
                        step_d_loss = step_d_loss / len(all_real_seqs)

                    # Accumulate gradients sequentially
                    scaler_disc.scale(step_d_loss).backward()

                # Unscale, clip, step
                scaler_disc.unscale_(optimizer_disc)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
                scaler_disc.step(optimizer_disc)
                scaler_disc.update()
                scheduler_disc.step()
                disc_stepped = True

            metrics_history["total"].append(total_loss.item())
            metrics_history["mask"].append(l1_m.item())
            metrics_history["frame"].append(l1_f.item())
            metrics_history["perc"].append(perc_v.item())
            metrics_history["style"].append(style_v.item())
            metrics_history["temp"].append(temp_v.item())
            metrics_history["adv"].append(adv.item())

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
                print(f"Running validation at iteration {current_iter}...")
                val_metrics = validate(args, model, val_loader, val_mask_dataset, human_mask_dataset, criterion, device,
                                       save_dir, current_iter)
                print(f"Validation Metrics: {val_metrics}")

            current_iter += 1

            if current_iter % 2500 == 0:
                checkpoint = {
                    "current_iter": current_iter,
                    "model_state": model.state_dict(),
                    "disc_state": discriminator.discriminator.state_dict(),
                    "opt_model_state": optimizer_model.state_dict(),
                    "opt_disc_state": optimizer_disc.state_dict(),
                    "scheduler_model_state": scheduler_model.state_dict(),
                    "scheduler_disc_state": scheduler_disc.state_dict(),
                }
                torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_{current_iter}.pth"))
                torch.save(checkpoint, os.path.join(save_dir, "checkpoint_latest.pth"))

            if current_iter >= args.iterations:
                break

    torch.save(discriminator.discriminator.state_dict(),
               os.path.join(save_dir, "final_discriminator.pth"))
    return {k: np.mean(v) for k, v in metrics_history.items()}

def save_video_previews(save_dir, it, comp_list, tgt_list, in_list, fps=10):
    if not comp_list:
        return

    frames = []
    for comp, tgt, m_win in zip(comp_list, tgt_list, in_list):
        out_img = (comp[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        target_img = (tgt[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        input_img = (m_win[0, -1].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        combined = np.concatenate((input_img, out_img, target_img), axis=1)
        frames.append(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_path = os.path.join(save_dir, "image_results", f"iter_{it}_video.mp4")
    video = cv2.VideoWriter(vid_path, fourcc, fps, (width, height))

    for frame in frames:
        video.write(frame)

    video.release()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    master_folder = args.model_name if args.model_name else f"Model_BC{BASE_CHANNELS}_L{NUM_LAYERS}"
    save_dir = os.path.join(args.save_dir, args.model_name, args.phase_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "image_results"), exist_ok=True)

    log_file_path = os.path.join(save_dir, "training_log.txt")
    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout

    print(f"Saving this phase to: {save_dir}")

    in_channels = args.seq_len * 3 + args.seq_len
    model = Viper(in_channels=in_channels, base_channels=BASE_CHANNELS, num_layers=NUM_LAYERS).to(device)

    base_discriminator = PretrainedPatchDiscriminator().to(device)
    discriminator = NoisyDiscriminator(base_discriminator)

    opt_model = optim.Adam(model.parameters(), lr=args.lr)
    opt_disc = optim.Adam(discriminator.parameters(), lr=args.lr * 1.0)

    scheduler_model = CosineAnnealingLR(opt_model, T_max=args.iterations, eta_min=1e-5)
    scheduler_disc = CosineAnnealingLR(opt_disc, T_max=args.iterations, eta_min=1e-6)

    criterion = InpaintingLoss(
        pixel_m_w=args.w_pixel_m, pixel_f_w=args.w_pixel_f,
        perceptual_w=args.w_perc, style_w=args.w_style,
        temporal_w=args.w_temp, adv_w=args.w_adv
    ).to(device)
    adv_crit = torch.nn.MSELoss()

    mask_dataset = None
    val_mask_dataset = None

    start_iter = 0
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint)
        model.load_state_dict(ckpt["model_state"])
        base_discriminator.load_state_dict(ckpt["disc_state"])
        opt_model.load_state_dict(ckpt["opt_model_state"])
        opt_disc.load_state_dict(ckpt["opt_disc_state"])
        scheduler_model.load_state_dict(ckpt["scheduler_model_state"])
        scheduler_disc.load_state_dict(ckpt["scheduler_disc_state"])
        start_iter = ckpt["current_iter"]
        print(f"Resumed full training state from iter {start_iter}")

    dataset = YouTubeVOSDataset(root_dir=os.path.join(os.getcwd(), "training_data", "train"))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    val_dataset = YouTubeVOSDataset(root_dir=os.path.join(os.getcwd(), "training_data", "valid"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    mask_dataset = IrregularMaskDataset(
        root_dir=os.path.join(os.getcwd(), "training_data", "irregular_mask", "disocclusion_img_mask"))
    val_mask_dataset = mask_dataset

    human_mask_dataset = HumanMaskDataset(
        root_dir=os.path.join(os.getcwd(), "training_data", "train"))

    final_metrics = train(
        args, model, discriminator, loader, val_loader,
        mask_dataset, val_mask_dataset, human_mask_dataset, opt_model, opt_disc,
        scheduler_model, scheduler_disc, criterion, adv_crit, device, save_dir,  start_iter=start_iter
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