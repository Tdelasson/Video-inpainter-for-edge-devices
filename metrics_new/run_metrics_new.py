from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from skimage.metrics import structural_similarity as ssim_metric


@dataclass
class VideoMetrics:
    psnr_mean: float
    ssim_mean: float
    gt_clip_feat: np.ndarray
    pred_clip_feat: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate predictions with E2FGVI-style PSNR/SSIM/VFID and merge speed metrics"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="DAVIS",
        help="Dataset name under nvidia_jetson/Test_Data",
    )
    parser.add_argument(
        "--mask-type",
        type=str,
        default="synthetic",
        choices=["synthetic", "RealObject"],
        help="Mask type to evaluate",
    )
    parser.add_argument(
        "--frames-subdir",
        type=str,
        default="JPEGImages_432_240",
        help="Frame subfolder under Test_Data/<dataset>",
    )
    parser.add_argument(
        "--masks-subdir",
        type=str,
        default=None,
        help="Mask subfolder under Test_Data/<dataset>. Defaults from --mask-type.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model result folders under nvidia_jetson/Results. If omitted, auto-discover.",
    )
    parser.add_argument(
        "--pred-subdir",
        type=str,
        default="_official_eval_pred",
        help="Prediction subfolder under each model split folder",
    )
    parser.add_argument(
        "--eval-width",
        type=int,
        default=None,
        help="Canonical evaluation width for fair comparison",
    )
    parser.add_argument(
        "--eval-height",
        type=int,
        default=None,
        help="Canonical evaluation height for fair comparison",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for VFID feature extraction",
    )
    parser.add_argument(
        "--i3d-weights",
        type=Path,
        default=None,
        help="Path to i3d_rgb_imagenet.pt or rgb_imagenet.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Output folder for JSON summaries",
    )
    return parser.parse_args()


def _find_i3d_weights(repo_root: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        if explicit_path.exists():
            return explicit_path
        raise FileNotFoundError(f"I3D weights not found: {explicit_path}")

    candidates = [
        repo_root / "Baselines_Repos" / "E2FGVI-master" / "release_model" / "i3d_rgb_imagenet.pt",
        repo_root / "Baselines_Repos" / "video-inpainting-evaluation-public" / "rgb_imagenet.pt",
        repo_root / "Baselines_Repos" / "video-inpainting-evaluation-public" / "pretrained_models" / "rgb_imagenet.pt",
    ]
    for path in candidates:
        if path.exists():
            return path

    msg = "Could not find I3D weights. Provide --i3d-weights."
    raise FileNotFoundError(msg)


def _load_i3d_model(repo_root: Path, weights_path: Path, device: torch.device):
    eval_repo = repo_root / "Baselines_Repos" / "video-inpainting-evaluation-public"
    if not eval_repo.exists():
        raise FileNotFoundError(f"Missing repository: {eval_repo}")

    sys.path.insert(0, str(eval_repo))
    from src.models.i3d.pytorch_i3d import InceptionI3d  # pylint: disable=import-error

    model = InceptionI3d(400, in_channels=3)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


def _video_feature(model, frames_rgb: list[np.ndarray], device: torch.device) -> np.ndarray:
    arr = np.stack(frames_rgb).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.extract_features(tensor.transpose(1, 2), target_endpoints=["Logits"])
    # extract_features returns a list of tensors when target_endpoints is a list
    feat_tensor = feat[0] if isinstance(feat, list) else feat
    return feat_tensor.flatten().detach().cpu().numpy()


def _resize_if_needed(image: Image.Image, size: tuple[int, int], mode: str) -> Image.Image:
    if image.size == size:
        return image
    if mode == "rgb":
        return image.resize(size, Image.BICUBIC)
    return image.resize(size, Image.NEAREST)


def _mask_subdir(mask_type: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    if mask_type == "synthetic":
        return "SyntheticMasks"
    return "RealObjectMasks"


def _discover_models(results_root: Path, dataset: str, mask_type: str, pred_subdir: str) -> list[str]:
    split_dir = "synthetic" if mask_type == "synthetic" else "RealObject"
    found: list[str] = []
    for child in sorted(results_root.iterdir()):
        pred_root = child / dataset / split_dir / pred_subdir
        if pred_root.exists():
            found.append(child.name)
    return found


def _load_speed_summary(results_root: Path, model_name: str, dataset: str, mask_type: str) -> dict:
    split_dir = "synthetic" if mask_type == "synthetic" else "RealObject"
    summary_path = results_root / model_name / dataset / split_dir / "summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _frame_mask_paths(
    dataset_root: Path,
    video_name: str,
    frame_index: int,
    frames_subdir: str,
    masks_subdir: str,
) -> tuple[Path, Path]:
    frame_path = dataset_root / frames_subdir / video_name / f"{frame_index:05d}.jpg"
    mask_path = dataset_root / masks_subdir / video_name / f"{frame_index:04d}.png"
    return frame_path, mask_path


def evaluate_one_model(
    *,
    model_name: str,
    test_json: dict[str, int],
    dataset_root: Path,
    pred_root: Path,
    frames_subdir: str,
    masks_subdir: str,
    eval_size: tuple[int, int] | None,
    i3d_model,
    device: torch.device,
) -> tuple[list[dict], float, float, float]:
    per_video: list[dict] = []
    gt_feats: list[np.ndarray] = []
    pred_feats: list[np.ndarray] = []
    all_psnr: list[float] = []
    all_ssim: list[float] = []

    for video_name, frame_count in test_json.items():
        gt_eval_frames: list[np.ndarray] = []
        comp_eval_frames: list[np.ndarray] = []
        video_psnr: list[float] = []
        video_ssim: list[float] = []

        for t in range(frame_count):
            frame_path, mask_path = _frame_mask_paths(
                dataset_root,
                video_name,
                t,
                frames_subdir,
                masks_subdir,
            )
            pred_path = pred_root / video_name / f"frame_{t:04d}_pred.png"

            if not frame_path.exists():
                raise FileNotFoundError(f"Missing GT frame: {frame_path}")
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask frame: {mask_path}")
            if not pred_path.exists():
                raise FileNotFoundError(f"Missing prediction frame: {pred_path}")

            gt_img = Image.open(frame_path).convert("RGB")
            mask_img = Image.open(mask_path).convert("L")
            pred_img = Image.open(pred_path).convert("RGB")

            target_size = eval_size if eval_size is not None else gt_img.size

            gt_img = _resize_if_needed(gt_img, target_size, "rgb")
            mask_img = _resize_if_needed(mask_img, target_size, "mask")
            pred_img = _resize_if_needed(pred_img, target_size, "rgb")

            gt_np = np.asarray(gt_img, dtype=np.uint8)
            mask_np = np.asarray(mask_img, dtype=np.uint8)
            pred_np = np.asarray(pred_img, dtype=np.uint8)

            hole = (mask_np > 0).astype(np.uint8)[..., None]
            comp_np = pred_np * hole + gt_np * (1 - hole)

            psnr_val = 20.0 * np.log10(255.0 / np.sqrt(max(1e-12, np.mean((gt_np.astype(np.float64) - comp_np.astype(np.float64)) ** 2))))
            ssim_val = ssim_metric(gt_np, comp_np, data_range=255, channel_axis=2, win_size=65)

            video_psnr.append(float(psnr_val))
            video_ssim.append(float(ssim_val))
            all_psnr.append(float(psnr_val))
            all_ssim.append(float(ssim_val))

            gt_eval_frames.append(gt_np)
            comp_eval_frames.append(comp_np)

        gt_feat = _video_feature(i3d_model, gt_eval_frames, device)
        pred_feat = _video_feature(i3d_model, comp_eval_frames, device)
        gt_feats.append(gt_feat)
        pred_feats.append(pred_feat)

        per_video.append(
            {
                "video": video_name,
                "num_frames": frame_count,
                "psnr": round(float(np.mean(video_psnr)), 4),
                "ssim": round(float(np.mean(video_ssim)), 4),
            }
        )

    gt_feats_np = np.stack(gt_feats, axis=0)
    pred_feats_np = np.stack(pred_feats, axis=0)
    gt_mu, gt_sigma = np.mean(gt_feats_np, axis=0), np.cov(gt_feats_np, rowvar=False)
    pred_mu, pred_sigma = np.mean(pred_feats_np, axis=0), np.cov(pred_feats_np, rowvar=False)
    vfid = _calculate_frechet_distance(gt_mu, gt_sigma, pred_mu, pred_sigma)

    return per_video, float(np.mean(all_psnr)), float(np.mean(all_ssim)), float(vfid)


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.resolve()
    results_root = repo_root / "nvidia_jetson" / "Results"
    dataset_root = repo_root / "nvidia_jetson" / "Test_Data" / args.dataset

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    masks_subdir = _mask_subdir(args.mask_type, args.masks_subdir)
    test_json_path = dataset_root / "test.json"
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_json = json.load(f)

    models = args.models
    if not models:
        models = _discover_models(results_root, args.dataset, args.mask_type, args.pred_subdir)
    if not models:
        raise RuntimeError("No models found to evaluate")

    eval_size = None
    if args.eval_width is not None and args.eval_height is not None:
        eval_size = (args.eval_width, args.eval_height)
    elif args.eval_width is not None or args.eval_height is not None:
        raise ValueError("Provide both --eval-width and --eval-height")

    requested_device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    i3d_weights = _find_i3d_weights(repo_root, args.i3d_weights)
    i3d_model = _load_i3d_model(repo_root, i3d_weights, requested_device)

    split_dir = "synthetic" if args.mask_type == "synthetic" else "RealObject"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        pred_root = results_root / model_name / args.dataset / split_dir / args.pred_subdir
        if not pred_root.exists():
            print(f"Skipping {model_name}: missing {pred_root}")
            continue

        per_video, psnr_mean, ssim_mean, vfid = evaluate_one_model(
            model_name=model_name,
            test_json=test_json,
            dataset_root=dataset_root,
            pred_root=pred_root,
            frames_subdir=args.frames_subdir,
            masks_subdir=masks_subdir,
            eval_size=eval_size,
            i3d_model=i3d_model,
            device=requested_device,
        )

        speed_summary = _load_speed_summary(results_root, model_name, args.dataset, args.mask_type)
        payload = {
            "model": model_name,
            "dataset": args.dataset,
            "mask_type": args.mask_type,
            "frames_subdir": args.frames_subdir,
            "masks_subdir": masks_subdir,
            "eval_size": {"width": eval_size[0], "height": eval_size[1]} if eval_size else None,
            "num_videos": len(per_video),
            "psnr": round(psnr_mean, 4),
            "ssim": round(ssim_mean, 4),
            "vfid": round(vfid, 4),
            "ewarp": None,
            "ewarp_x1e2": None,
            "speed": {
                "fps": speed_summary.get("fps"),
                "latency_ms": speed_summary.get("latency_ms"),
                "peak_memory_mb": speed_summary.get("peak_memory_mb"),
            },
            "per_video": per_video,
        }

        out_path = args.output_dir / f"{model_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
