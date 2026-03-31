import sys
import json
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")

from Test_Data.dataloader import TestDataset
from Baselines.fuseformer_om_adapter import FuseFormerOMAdapter, MODEL_H, MODEL_W
from Metrics.metrics import (
    compute_psnr,
    compute_ssim,
    compute_ewarp,
    compute_vfid,
    measure_video_run,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

adapter = FuseFormerOMAdapter(
    weights_path="../Baselines_Repos/pthFiles/OnlineInpainting/fuseformer.pth",
    device=device,
    fp16=False,
)

I3D_MODEL_PATH = "../Baselines_Repos/pthFiles/ProPainter/i3d_rgb_imagenet.pt"
OFFICIAL_EVAL_REPO = "../Baselines_Repos/video-inpainting-evaluation"
OFFICIAL_EVAL_FEATS = {
    ("DAVIS", "synthetic"): None,
    ("YouTube-VOS", "synthetic"): None,
}

EVAL_SPLITS = [
    ("DAVIS", "synthetic"),
    ("DAVIS", "RealObject"),
    ("YouTube-VOS", "synthetic"),
]


def save_video(result, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(result):
        cv2.imwrite(str(out_dir / f"{i:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def resize_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    return [cv2.resize(frame, (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR) for frame in frames]


def evaluate_video(video, result, perf):
    gt_resized = resize_frames(video.frames)
    psnr_vals = [compute_psnr(gt, pred) for gt, pred in zip(gt_resized, result)]
    ssim_vals = [compute_ssim(gt, pred) for gt, pred in zip(gt_resized, result)]

    return {
        "psnr": round(float(np.mean(psnr_vals)), 4),
        "ssim": round(float(np.mean(ssim_vals)), 4),
        "fps": perf["fps"],
        "latency_ms": perf["latency_ms"],
        "peak_memory_mb": perf["peak_memory_mb"],
        "num_frames": len(video.frames),
    }


for dataset_name, mask_type in EVAL_SPLITS:
    dataset = TestDataset("Test_Data", dataset_name, mask_type)

    print(f"\nRunning {dataset_name} / {mask_type} on {len(dataset)} videos")

    split_metrics = []
    gt_videos = []
    pred_videos = []
    synthetic_videos = []

    for video in dataset:
        print(f"Inpainting '{video.name}' ({len(video.frames)} frames)")

        result, perf = measure_video_run(
            lambda: adapter.inpaint(video.frames, video.masks, resize_to_original=False),
            num_frames=len(video.frames),
            use_cuda=(device == "cuda"),
        )

        out_dir = Path("Results2") / adapter.name / video.dataset / video.mask_type / video.name
        save_video(result, out_dir)

        video_metrics = {
            "video": video.name,
            "dataset": video.dataset,
            "mask_type": video.mask_type,
        }

        if mask_type == "synthetic":
            video_metrics.update(evaluate_video(video, result, perf))
            gt_videos.append(resize_frames(video.frames))
            pred_videos.append(result)
            synthetic_videos.append(video)
        else:
            video_metrics.update({
                "fps": perf["fps"],
                "latency_ms": perf["latency_ms"],
                "peak_memory_mb": perf["peak_memory_mb"],
                "num_frames": len(video.frames),
            })

        split_metrics.append(video_metrics)

        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(video_metrics, f, indent=2)

    summary = {
        "dataset": dataset_name,
        "mask_type": mask_type,
        "num_videos": len(split_metrics),
    }

    if split_metrics:
        for key in ("fps", "latency_ms", "peak_memory_mb"):
            vals = [m[key] for m in split_metrics if key in m]
            if vals:
                summary[key] = round(float(np.mean(vals)), 4)

    if mask_type == "synthetic" and split_metrics:
        for key in ("psnr", "ssim"):
            vals = [m[key] for m in split_metrics if key in m]
            if vals:
                summary[key] = round(float(np.mean(vals)), 4)

        if len(gt_videos) >= 2:
            summary["vfid"] = round(float(compute_vfid(gt_videos, pred_videos, I3D_MODEL_PATH)), 4)
        else:
            summary["vfid"] = None

        pred_root = Path("Results2") / adapter.name / dataset_name / mask_type
        try:
            ewarp_raw = compute_ewarp(
                videos=synthetic_videos,
                pred_root=pred_root,
                eval_repo_root=OFFICIAL_EVAL_REPO,
                eval_feats_root=OFFICIAL_EVAL_FEATS.get((dataset_name, mask_type)),
                output_size=(MODEL_W, MODEL_H),
            )
            summary["ewarp"] = round(float(ewarp_raw), 6)
            summary["ewarp_x1e2"] = round(float(ewarp_raw) * 100.0, 4)
        except Exception as exc:
            summary["ewarp"] = None
            summary["ewarp_x1e2"] = None
            summary["ewarp_error"] = str(exc)

    summary_dir = Path("Results2") / adapter.name / dataset_name / mask_type
    summary_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Finished {dataset_name} / {mask_type}")
    print(json.dumps(summary, indent=2))
