import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")

from Baselines.fuseformer_om_adapter import FuseFormerOMAdapter
from Baselines.propainter_adapter import ProPainterAdapter
from Metrics.metrics import compute_psnr, compute_ssim, measure_video_run
from Metrics.official_eval import run_official_synthetic_eval, save_prediction_video
from Test_Data.dataloader import TestDataset

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "Results2"
DEFAULT_OFFICIAL_EVAL_REPO = (REPO_ROOT / "../Baselines_Repos/video-inpainting-evaluation-public").resolve()
DEFAULT_FUSEFORMER_WEIGHTS_PATH = (REPO_ROOT / "../Baselines_Repos/pthFiles/OnlineInpainting/fuseformer.pth").resolve()
DEFAULT_PROPAINTER_WEIGHTS_PATH = (REPO_ROOT / "../Baselines_Repos/pthFiles/ProPainter/ProPainter.pth").resolve()
DEFAULT_PROPAINTER_RAFT_WEIGHTS_PATH = (REPO_ROOT / "../Baselines_Repos/pthFiles/ProPainter/raft-things.pth").resolve()
DEFAULT_PROPAINTER_FLOW_WEIGHTS_PATH = (
    REPO_ROOT / "../Baselines_Repos/pthFiles/ProPainter/recurrent_flow_completion.pth"
).resolve()
DEFAULT_SPLITS = [
    ("DAVIS", "synthetic"),
    ("DAVIS", "RealObject"),
    ("YouTube-VOS", "synthetic"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-style video inpainting evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default="fuseformer_om",
        choices=["fuseformer_om", "propainter"],
        help="Model adapter to run",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=[f"{dataset}:{mask_type}" for dataset, mask_type in DEFAULT_SPLITS],
        help="Split specifiers like DAVIS:synthetic or YouTube-VOS:synthetic",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of videos per split",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory where predictions and metrics are saved",
    )
    parser.add_argument(
        "--official-eval-repo",
        type=Path,
        default=DEFAULT_OFFICIAL_EVAL_REPO,
        help="Path to MichiganCOG/video-inpainting-evaluation-public",
    )
    parser.add_argument(
        "--official-eval-feats-root",
        type=Path,
        default=None,
        help="Optional precomputed eval feature root for the official evaluator",
    )
    parser.add_argument(
        "--official-eval-python",
        type=Path,
        default=None,
        help="Optional Python executable to use for the official evaluator environment",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=None,
        help="Path to model checkpoint. Defaults depend on --model.",
    )
    parser.add_argument(
        "--raft-weights-path",
        type=Path,
        default=DEFAULT_PROPAINTER_RAFT_WEIGHTS_PATH,
        help="Path to RAFT weights (used by ProPainter)",
    )
    parser.add_argument(
        "--flow-weights-path",
        type=Path,
        default=DEFAULT_PROPAINTER_FLOW_WEIGHTS_PATH,
        help="Path to recurrent flow completion weights (used by ProPainter)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Run the adapter in fp16 mode",
    )
    return parser.parse_args()


def parse_splits(raw_splits: list[str]) -> list[tuple[str, str]]:
    splits: list[tuple[str, str]] = []
    for item in raw_splits:
        if ":" not in item:
            raise ValueError(f"Invalid split '{item}'. Expected DATASET:MASK_TYPE")
        dataset_name, mask_type = item.split(":", 1)
        splits.append((dataset_name, mask_type))
    return splits


def _build_adapter(args: argparse.Namespace, device: str):
    model_key = args.model.lower()
    if model_key == "fuseformer_om":
        weights_path = args.weights_path or DEFAULT_FUSEFORMER_WEIGHTS_PATH
        adapter = FuseFormerOMAdapter(
            weights_path=str(weights_path),
            device=device,
            fp16=args.fp16,
        )
        return adapter, adapter.model_h, adapter.model_w

    if model_key == "propainter":
        weights_path = args.weights_path or DEFAULT_PROPAINTER_WEIGHTS_PATH
        adapter = ProPainterAdapter(
            weights_path=str(weights_path),
            raft_weights_path=str(args.raft_weights_path),
            flow_weights_path=str(args.flow_weights_path),
            device=device,
            fp16=args.fp16,
        )
        return adapter, adapter.model_h, adapter.model_w

    raise ValueError(f"Unsupported model: {args.model}")


def evaluate_video(video, result, perf, model_w: int, model_h: int) -> dict:
    gt_resized = [cv2.resize(frame, (model_w, model_h), interpolation=cv2.INTER_LINEAR) for frame in video.frames]
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


def main() -> None:
    args = parse_args()
    eval_splits = parse_splits(args.splits)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    adapter, model_h, model_w = _build_adapter(args, device)

    for dataset_name, mask_type in eval_splits:
        dataset = TestDataset("Test_Data", dataset_name, mask_type)
        if args.limit is not None:
            dataset.video_names = dataset.video_names[: args.limit]

        print(f"\nRunning {dataset_name} / {mask_type} on {len(dataset)} videos")

        split_metrics = []
        synthetic_videos = []
        split_root = args.results_dir / adapter.name / dataset_name / mask_type
        official_pred_root = split_root / "_official_eval_pred"

        for video in dataset:
            print(f"Inpainting '{video.name}' ({len(video.frames)} frames)")

            result, perf = measure_video_run(
                lambda: adapter.inpaint(video.frames, video.masks, resize_to_original=False),
                num_frames=len(video.frames),
                use_cuda=(device == "cuda"),
            )

            video_metrics = {
                "video": video.name,
                "dataset": video.dataset,
                "mask_type": video.mask_type,
            }

            if mask_type == "synthetic":
                video_metrics.update(evaluate_video(video, result, perf, model_w=model_w, model_h=model_h))
                synthetic_videos.append(video)
                save_prediction_video(video.name, result, official_pred_root)
            else:
                video_metrics.update(
                    {
                        "fps": perf["fps"],
                        "latency_ms": perf["latency_ms"],
                        "peak_memory_mb": perf["peak_memory_mb"],
                        "num_frames": len(video.frames),
                    }
                )

            split_metrics.append(video_metrics)

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

            try:
                official_vfid = run_official_synthetic_eval(
                    videos=synthetic_videos,
                    pred_root=official_pred_root,
                    repo_root=args.official_eval_repo,
                    eval_feats_root=args.official_eval_feats_root,
                    output_size=(model_w, model_h),
                    metrics=("vfid",),
                    python_executable=str(args.official_eval_python) if args.official_eval_python else sys.executable,
                )
                summary["vfid"] = round(float(official_vfid["vfid"]), 4)
            except Exception as exc:
                summary["vfid"] = None
                summary["official_eval_error_vfid"] = str(exc)

            try:
                official_ewarp = run_official_synthetic_eval(
                    videos=synthetic_videos,
                    pred_root=official_pred_root,
                    repo_root=args.official_eval_repo,
                    eval_feats_root=args.official_eval_feats_root,
                    output_size=(model_w, model_h),
                    metrics=("warp_error_mask",),
                    python_executable=str(args.official_eval_python) if args.official_eval_python else sys.executable,
                )
                summary["ewarp"] = round(float(official_ewarp["warp_error_mask"]), 6)
                summary["ewarp_x1e2"] = round(float(official_ewarp["warp_error_mask"]) * 100.0, 4)
            except Exception as exc:
                summary["ewarp"] = None
                summary["ewarp_x1e2"] = None
                summary["official_eval_error_ewarp"] = str(exc)

        summary_dir = split_root
        summary_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Finished {dataset_name} / {mask_type}")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
