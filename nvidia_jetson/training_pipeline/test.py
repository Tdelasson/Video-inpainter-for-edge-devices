import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from training_pipeline.config import *
from training_pipeline.dataset import YouTubeVOSDataset, IrregularMaskDataset
from training_pipeline.mask_generator import generate_arbitrary_shape_mask, random_dilate_and_blur_mask
from training_pipeline.inpainting_loss import InpaintingLoss
from model_architecture.video_inpainter import VideoInpainter
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Test Video Inpainter")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save results")
    parser.add_argument("--seq_len", type=int, default=5)

    # Use same loss weights as final training phase for comparable metrics
    parser.add_argument("--w_pixel_m", type=float, default=1.0)
    parser.add_argument("--w_pixel_f", type=float, default=0.5)
    parser.add_argument("--w_perc",    type=float, default=6.0)
    parser.add_argument("--w_style",   type=float, default=15.0)
    parser.add_argument("--w_temp",    type=float, default=0.0)  # no flow during test
    parser.add_argument("--w_adv",     type=float, default=0.0)  # no discriminator during test
    return parser.parse_args()


def save_test_preview(output_dir, video_idx, frame_idx, composited, target, masked):
    out_img    = (composited[0].detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
    target_img = (target[0].detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
    masked_img = (masked[0,-1].detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)

    prefix = os.path.join(output_dir, "previews", f"video{video_idx:04d}_frame{frame_idx:04d}")
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    cv2.imwrite(f"{prefix}_output.png",  cv2.cvtColor(out_img,    cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{prefix}_target.png",  cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{prefix}_masked.png",  cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Model ---
    in_channels = args.seq_len * 3 + args.seq_len
    model = VideoInpainter(
        in_channels=in_channels,
        base_channels=BASE_CHANNELS,
        num_layers=NUM_LAYERS
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {args.model_path}")

    # --- Loss (no adv, no temp) ---
    criterion = InpaintingLoss(
        pixel_m_w=args.w_pixel_m, pixel_f_w=args.w_pixel_f,
        perceptual_w=args.w_perc, style_w=args.w_style,
        temporal_w=0.0, adv_w=0.0
    ).to(device)

    # --- Data — held-out test split, never seen during training ---
    test_dataset = YouTubeVOSDataset(
        root_dir=os.path.join(os.getcwd(), "training_data", "test")
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=2, drop_last=True)
    mask_dataset = IrregularMaskDataset(
        root_dir=os.path.join(os.getcwd(), "training_data",
                              "irregular_mask", "disocclusion_img_mask")
    )
    print(f"Test set: {len(test_dataset)} videos")

    # --- Metrics ---
    # Tracked per difficulty tier so you can see where the model struggles
    tiers = {
        "small":   (0.02, 0.08),   # mask covers 2-8% of frame
        "medium":  (0.08, 0.20),   # 8-20%
        "large":   (0.20, 0.40),   # 20-40%
    }
    results = {tier: {"psnr": [], "mask_l1": [], "temp_consistency": []}
               for tier in tiers}
    results["overall"] = {"psnr": [], "mask_l1": [], "temp_consistency": []}

    NUM_PREVIEW_VIDEOS = 10  # save visual previews for first N videos

    with torch.no_grad():
        for video_idx, data in enumerate(test_loader):

            video_data = data.float().to(device) / 255.0
            video_data = video_data.permute(0, 1, 4, 2, 3)
            B, T, C, H, W = video_data.shape

            masks = generate_arbitrary_shape_mask(video_data, mask_dataset)
            masks = random_dilate_and_blur_mask(masks)
            masked_video = video_data * (1.0 - masks)

            # Determine mask difficulty tier for this video
            mask_coverage = masks.mean().item()
            tier = "large"
            for tier_name, (lo, hi) in tiers.items():
                if lo <= mask_coverage < hi:
                    tier = tier_name
                    break

            hidden_state = None
            prev_composited = None

            for t in range(0, T - args.seq_len + 1):
                window       = video_data[:, t:t + args.seq_len]
                masked_window = masked_video[:, t:t + args.seq_len]
                masks_window  = masks[:, t:t + args.seq_len]

                target      = window[:, -1]
                target_mask = masks_window[:, -1]

                pixel_input = masked_window.reshape(B, args.seq_len * C, H, W)
                mask_input  = masks_window.reshape(B, args.seq_len, H, W)
                full_input  = torch.cat([pixel_input, mask_input], dim=1)

                output, hidden_state = model(full_input, hidden_state)
                if hidden_state is not None:
                    hidden_state = hidden_state.detach()

                composited = output * target_mask + target * (1 - target_mask)

                # PSNR on masked region only
                mse  = torch.mean((output * target_mask - target * target_mask) ** 2)
                psnr = 10 * torch.log10(1.0 / (mse + 1e-8))

                # Mask L1
                mask_l1 = torch.mean(torch.abs(output * target_mask - target * target_mask))

                # Temporal consistency
                if prev_composited is not None:
                    temp_c = torch.mean(torch.abs(composited - prev_composited)).item()
                    results[tier]["temp_consistency"].append(temp_c)
                    results["overall"]["temp_consistency"].append(temp_c)

                results[tier]["psnr"].append(psnr.item())
                results[tier]["mask_l1"].append(mask_l1.item())
                results["overall"]["psnr"].append(psnr.item())
                results["overall"]["mask_l1"].append(mask_l1.item())

                # Save previews for first N videos
                if video_idx < NUM_PREVIEW_VIDEOS:
                    save_test_preview(args.output_dir, video_idx, t,
                                      composited, target, masked_window)

                prev_composited = composited

            if video_idx % 50 == 0:
                print(f"Tested {video_idx}/{len(test_dataset)} videos...")

    # --- Summarise ---
    summary = {}
    for tier_name, tier_metrics in results.items():
        if len(tier_metrics["psnr"]) == 0:
            continue
        summary[tier_name] = {
            "mean_psnr":             round(np.mean(tier_metrics["psnr"]), 3),
            "std_psnr":              round(np.std(tier_metrics["psnr"]), 3),
            "mean_mask_l1":          round(np.mean(tier_metrics["mask_l1"]), 4),
            "mean_temp_consistency": round(np.mean(tier_metrics["temp_consistency"]), 4)
                                     if tier_metrics["temp_consistency"] else None,
            "n_frames":              len(tier_metrics["psnr"]),
        }

    print("\n===== TEST RESULTS =====")
    for tier_name, s in summary.items():
        print(f"\n[{tier_name.upper()}]")
        print(f"  PSNR:              {s['mean_psnr']:.2f} ± {s['std_psnr']:.2f} dB")
        print(f"  Mask L1:           {s['mean_mask_l1']:.4f}")
        print(f"  Temp Consistency:  {s['mean_temp_consistency']}")
        print(f"  Frames evaluated:  {s['n_frames']}")

    out_path = os.path.join(args.output_dir, "test_results.json")
    with open(out_path, "w") as f:
        json.dump({"model": args.model_path, "results": summary}, f, indent=4)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    test(args)