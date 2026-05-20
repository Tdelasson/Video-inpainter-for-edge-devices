#!/usr/bin/env python3
import os
import sys
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

PROPAINTER_PATH = "/home/sw66/Projects/Video-inpainter-for-edge-devices/Baselines_Repos/ProPainter-main"
if PROPAINTER_PATH not in sys.path:
    sys.path.insert(0, PROPAINTER_PATH)

from RAFT import RAFT

# Explicitly pull the padder utility using an absolute module import
import importlib.util
import os
padder_spec = importlib.util.spec_from_file_location(
    "propainter_utils",
    os.path.join(PROPAINTER_PATH, "utils", "utils.py")
)
propainter_utils = importlib.util.module_from_spec(padder_spec)
padder_spec.loader.exec_module(propainter_utils)
InputPadder = propainter_utils.InputPadder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="DAVIS")
    parser.add_argument("--mask-type", type=str, default="synthetic")
    parser.add_argument("--frames-subdir", type=str, default="JPEGImages_432_240")
    parser.add_argument("--models", nargs="*", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def warp_frame(x, flow):
    """Warps a frame using native PyTorch grid sampling."""
    B, C, H, W = x.size()
    # Create meshgrid
    xx = torch.arange(0, W, device=x.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=x.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    vgrid = grid + flow
    # Scale grid to [-1, 1] for grid_sample
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    return F.grid_sample(x, vgrid, padding_mode='border', align_corners=True)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize RAFT flow model from ProPainter's assets
    # Using dummy args matching RAFT specifications
    class RAFTArgs:
        small = False
        mixed_precision = False

    flow_model = torch.nn.DataParallel(RAFT(RAFTArgs()))
    # Load ProPainter's bundled RAFT weights
    raft_weights = "/home/sw66/Projects/Video-inpainter-for-edge-devices/Baselines_Repos/ProPainter-main/weights/raft- things.pth"
    if os.path.exists(raft_weights):
        flow_model.load_state_dict(torch.load(raft_weights, map_location=device))
    flow_model = flow_model.module.to(device).eval()

    # --- Setup directories ---
    repo_root = Path(__file__).resolve().parents[1]
    results_root = repo_root / "nvidia_jetson" / "Results"
    dataset_root = repo_root / "nvidia_jetson" / "Test_Data" / args.dataset

    with open(dataset_root / "test.json", "r") as f:
        test_json = json.load(f)

    split_dir = "synthetic" if args.mask_type == "synthetic" else "RealObject"
    output_rows = []

    for model_name in args.models:
        pred_root = results_root / model_name / args.dataset / split_dir / "_official_eval_pred"
        if not pred_root.exists():
            continue

        video_errors = []

        for video_name, frame_count in test_json.items():
            accumulated_mse = 0.0
            count = 0

            for t in range(frame_count - 1):
                # Load GT frames to calculate target optical flow vector
                gt1_p = dataset_root / args.frames_subdir / video_name / f"{t:05d}.jpg"
                gt2_p = dataset_root / args.frames_subdir / video_name / f"{t + 1:05d}.jpg"

                # Load corresponding Predictions
                p1_p = pred_root / video_name / f"frame_{t:04d}_pred.png"
                p2_p = pred_root / video_name / f"frame_{t + 1:04d}_pred.png"

                if not (gt1_p.exists() and gt2_p.exists() and p1_p.exists() and p2_p.exists()):
                    continue

                # Read images into tensors
                gt1 = torch.from_numpy(np.array(Image.open(gt1_p))).permute(2, 0, 1).float().unsqueeze(0).to(device)
                gt2 = torch.from_numpy(np.array(Image.open(gt2_p))).permute(2, 0, 1).float().unsqueeze(0).to(device)
                p2 = torch.from_numpy(np.array(Image.open(p2_p))).permute(2, 0, 1).float().unsqueeze(0).to(device)

                # Calculate Flow via native RAFT
                padder = InputPadder(gt1.shape)
                gt1_pad, gt2_pad = padder.pad(gt1, gt2)
                with torch.no_grad():
                    _, flow = flow_model(gt1_pad, gt2_pad, iters=12, test_mode=True)
                    flow = padder.unpad(flow)

                    # Warp prediction 2 backward to match prediction 1 context
                    warped_p2 = warp_frame(p2, flow)

                    # Compute MSE Warp Error
                    mse = F.mse_loss(warped_p2, p2).item()
                    accumulated_mse += mse
                    count += 1

            if count > 0:
                video_errors.append(accumulated_mse / count)

        mean_ewarp = np.mean(video_errors) if video_errors else 0.0

        output_rows.append({
            "model": model_name,
            "dataset": args.dataset,
            "mask_type": args.mask_type,
            "ewarp": float(mean_ewarp),
            "ewarp_x1e2": float(mean_ewarp * 100.0)
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "ewarp_results.json", "w") as f:
        json.dump(output_rows, f, indent=2)
    print(f"✅ Successfully written metrics ledger payload to ewarp_results.json!")


if __name__ == "__main__":
    main()