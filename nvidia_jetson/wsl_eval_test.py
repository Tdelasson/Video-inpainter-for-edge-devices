from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from Metrics.official_eval import run_official_synthetic_eval
from Test_Data.dataloader import TestDataset


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate existing prediction frames with official VFID/EWarp metrics")
    parser.add_argument("--dataset", default="DAVIS", choices=["DAVIS", "YouTube-VOS"])
    parser.add_argument("--mask-type", default="synthetic", choices=["synthetic", "RealObject"])
    parser.add_argument("--model-name", default="FuseFormer_OM", help="Results2 model folder name")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of videos")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "Results2",
        help="Root results directory containing model outputs",
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=None,
        help="Optional explicit prediction folder (expects frame_XXXX_pred.png files)",
    )
    parser.add_argument(
        "--official-eval-repo",
        type=Path,
        default=(repo_root / "../Baselines_Repos/video-inpainting-evaluation-public").resolve(),
        help="Path to MichiganCOG/video-inpainting-evaluation-public",
    )
    parser.add_argument(
        "--official-eval-feats-root",
        type=Path,
        default=None,
        help="Optional precomputed evaluator features root",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional output JSON path for metric summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    dataset = TestDataset(str(repo_root / "Test_Data"), args.dataset, args.mask_type)
    if args.limit is not None:
        dataset.video_names = dataset.video_names[: args.limit]
    videos = [video for video in dataset]

    pred_root = (
        args.pred_root
        if args.pred_root is not None
        else args.results_dir / args.model_name / args.dataset / args.mask_type / "_official_eval_pred"
    )

    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction root does not exist: {pred_root}")

    metrics = run_official_synthetic_eval(
        videos=videos,
        pred_root=pred_root,
        repo_root=args.official_eval_repo,
        output_size=(432, 240),
        eval_feats_root=args.official_eval_feats_root,
        metrics=("vfid", "warp_error_mask"),
        python_executable=sys.executable,
    )

    summary = {
        "dataset": args.dataset,
        "mask_type": args.mask_type,
        "model_name": args.model_name,
        "num_videos": len(videos),
        "pred_root": str(pred_root),
        "vfid": float(metrics["vfid"]),
        "ewarp": float(metrics["warp_error_mask"]),
        "ewarp_x1e2": float(metrics["warp_error_mask"]) * 100.0,
    }

    out_json = args.out_json
    if out_json is None:
        out_json = args.results_dir / args.model_name / args.dataset / args.mask_type / "official-eval-summary.json"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
