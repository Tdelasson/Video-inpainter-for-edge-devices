import sys
import os
sys.path.insert(0, ".")

import cv2
import torch
from pathlib import Path
from Data.dataloader import TestDataset
from Baselines.fuseformer_om_adapter import FuseFormerOMAdapter

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
adapter = FuseFormerOMAdapter(
    weights_path="../Baselines_Repos/pthFiles/OnlineInpainting/fuseformer.pth",
    device=device,
    fp16=False,      # True on Jetson to save memory
)

# Load a video
dataset = TestDataset("Data/Test_Data", "DAVIS", "object")
video = dataset[0]

print(f"Inpainting '{video.name}': {len(video.frames)} frames at {video.frames[0].shape}")

# Inpaint
result = adapter.inpaint(video.frames, video.masks)

# Save results
out_dir = Path("Results") / adapter.name / video.dataset / video.name
out_dir.mkdir(parents=True, exist_ok=True)

for i, frame in enumerate(result):
    # Save as RGB -> BGR for cv2
    cv2.imwrite(str(out_dir / f"{i:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

print(f"Done! Saved {len(result)} frames to {out_dir}")
