from __future__ import annotations

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
from Baselines.vinet_adapter import ViNETAdapter
from Metrics.metrics import measure_video_run
from Test_Data.dataloader import TestDataset
from viper_adapter import ViperAdapter

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "Results2"
DEFAULT_FUSEFORMER_WEIGHTS_PATH = (REPO_ROOT / "../Baselines_Repos/pthFiles/OnlineInpainting/fuseformer.pth").resolve()
DEFAULT_PROPAINTER_WEIGHTS_PATH = (REPO_ROOT / "../Baselines_Repos/pthFiles/ProPainter/ProPainter.pth").resolve()
DEFAULT_PROPAINTER_RAFT_WEIGHTS_PATH = (REPO_ROOT / "../Baselines_Repos/pthFiles/ProPainter/raft-things.pth").resolve()
DEFAULT_PROPAINTER_FLOW_WEIGHTS_PATH = (
    REPO_ROOT / "../Baselines_Repos/pthFiles/ProPainter/recurrent_flow_completion.pth"
).resolve()
DEFAULT_VINET_WEIGHTS_PATH = (REPO_ROOT / "../Baselines_Repos/pthFiles/ViNETsave_agg_rec_512.pth").resolve()
DEFAULT_VIPER_WEIGHTS_PATH = (REPO_ROOT / "final_model.pth").resolve()
DEFAULT_SPLITS = [
    ("DAVIS", "synthetic"),
    ("DAVIS", "RealObject"),
    ("YouTube-VOS", "synthetic"),
]


def save_prediction_video(video_name: str, frames: list[np.ndarray], pred_root: Path) -> None:
    """Write predicted frames in evaluator-compatible filename format."""
    video_dir = pred_root / video_name
    video_dir.mkdir(parents=True, exist_ok=True)

    for idx, frame in enumerate(frames):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(video_dir / f"frame_{idx:04d}_pred.png"), frame_bgr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-style video inpainting evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default="fuseformer_om",
        choices=["fuseformer_om", "propainter", "vinet", "viper"],
        help="Model adapter to run",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=[f"{dataset}:{mask_type}" for dataset, mask_type in DEFAULT_SPLITS],
        help="Split specifiers like DAVIS:synthetic or YouTube-VOS:synthetic",
    )
    parser.add_argument(
        "--frames-subdir",
        type=str,
        default="JPEGImages",
        help="Frame folder under Test_Data/<dataset> (e.g. JPEGImages or JPEGImages_432_240)",
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
        "--propainter-ref-stride",
        type=int,
        default=10,
        help="Stride of global reference frames (higher uses less memory)",
    )
    parser.add_argument(
        "--propainter-neighbor-length",
        type=int,
        default=10,
        help="Length of local neighboring frames (lower uses less memory)",
    )
    parser.add_argument(
        "--propainter-subvideo-length",
        type=int,
        default=80,
        help="Sub-video length for long videos (lower uses less memory)",
    )
    parser.add_argument(
        "--propainter-raft-iters",
        type=int,
        default=20,
        help="RAFT iterations for ProPainter flow estimation",
    )
    parser.add_argument(
        "--viper-seq-len",
        type=int,
        default=5,
        help="Sequence length for Viper adapter",
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
            ref_stride=args.propainter_ref_stride,
            neighbor_length=args.propainter_neighbor_length,
            subvideo_length=args.propainter_subvideo_length,
            raft_iters=args.propainter_raft_iters,
        )
        return adapter, adapter.model_h, adapter.model_w

    if model_key == "vinet":
        weights_path = args.weights_path or DEFAULT_VINET_WEIGHTS_PATH
        adapter = ViNETAdapter(
            weights_path=str(weights_path),
            device=device,
            fp16=args.fp16,
        )
        return adapter, adapter.model_h, adapter.model_w

    if model_key == "viper":
        weights_path = args.weights_path or DEFAULT_VIPER_WEIGHTS_PATH
        adapter = ViperAdapter(
            model_path=str(weights_path),
            device=device,
            seq_len=args.viper_seq_len,
            fp16=args.fp16,
        )
        if not hasattr(adapter, "name"):
            adapter.name = "viper"
        return adapter, getattr(adapter, "model_h", None), getattr(adapter, "model_w", None)

    raise ValueError(f"Unsupported model: {args.model}")


def _run_video_inpaint(adapter, model_key: str, frames: list[np.ndarray], masks: list[np.ndarray]) -> list[np.ndarray]:
    if model_key != "viper":
        return adapter.inpaint(frames, masks, resize_to_original=False)

    # Viper requires exactly seq_len frames. Repeat-pad with the first real frame
    # so predictions are produced from frame 0 (evaluation-only; not applied in streaming).
    if hasattr(adapter, "hidden_state"):
        adapter.hidden_state = None

    seq_len = adapter.seq_len
    outputs: list[np.ndarray] = []
    for idx in range(len(frames)):
        pad = max(0, seq_len - (idx + 1))
        padded_frames = [frames[0]] * pad + frames[: idx + 1]
        padded_masks = [masks[0]] * pad + masks[: idx + 1]
        pred = adapter.inpaint(padded_frames, padded_masks, resize_to_original=False)
        outputs.append(pred[-1])

    return outputs


def main() -> None:
    args = parse_args()
    eval_splits = parse_splits(args.splits)
    model_key = args.model.lower()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    adapter, _, _ = _build_adapter(args, device)

    for dataset_name, mask_type in eval_splits:
        dataset = TestDataset(
            "Test_Data",
            dataset_name,
            mask_type,
            frames_subdir=args.frames_subdir,
        )
        if args.limit is not None:
            dataset.video_names = dataset.video_names[: args.limit]

        print(f"\nRunning {dataset_name} / {mask_type} on {len(dataset)} videos")

        split_metrics = []
        split_root = args.results_dir / adapter.name / dataset_name / mask_type
        official_pred_root = split_root / "_official_eval_pred"

        for video in dataset:
            print(f"Inpainting '{video.name}' ({len(video.frames)} frames)")

            result, perf = measure_video_run(
                lambda: _run_video_inpaint(adapter, model_key, video.frames, video.masks),
                num_frames=len(video.frames),
                use_cuda=(device == "cuda"),
            )

            video_metrics = {
                "video": video.name,
                "dataset": video.dataset,
                "mask_type": video.mask_type,
                "fps": perf["fps"],
                "latency_ms": perf["latency_ms"],
                "peak_memory_mb": perf["peak_memory_mb"],
                "num_frames": len(video.frames),
            }

            if mask_type == "synthetic":
                save_prediction_video(video.name, result, official_pred_root)

            split_metrics.append(video_metrics)

        summary = {
            "model": adapter.name,
            "fp16": bool(args.fp16),
            "dataset": dataset_name,
            "mask_type": mask_type,
            "num_videos": len(split_metrics),
        }

        if split_metrics:
            for key in ("fps", "latency_ms", "peak_memory_mb"):
                vals = [m[key] for m in split_metrics if key in m]
                if vals:
                    summary[key] = round(float(np.mean(vals)), 4)

        summary_dir = split_root
        summary_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Finished {dataset_name} / {mask_type}")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
