#!/usr/bin/env python3
"""
Automated Video Inpainting Benchmark Script.
Automatically extracts nested 'model_state' dictionaries from training
checkpoints to prevent PyTorch state_dict loading crashes.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import torch  # Added to unpack the state dictionaries on the fly


def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluate inpainting checkpoints.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing your .pth model checkpoints",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="viper",
        help="Adapter key type to pass down to inference execution engine",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="DAVIS",
        help="Dataset name used for tracking evaluation targets",
    )
    parser.add_argument(
        "--mask-type",
        type=str,
        default="synthetic",
        choices=["synthetic", "RealObject"],
        help="Mask execution split archetype",
    )
    parser.add_argument(
        "--frames-subdir",
        type=str,
        default="JPEGImages_432_240",
        help="Canonical subdirectory structural layout containing frames",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Hardware calculation accelerator context target",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Pass down half-precision context execution flags to inference adapter",
    )
    return parser.parse_args()


def run_command(cmd: list[str], cwd: Path, description: str) -> None:
    """Helper method to execute shell commands inside a specific folder context."""
    print(f"\n--- [Executing Task] {description} ---")
    print(f"Working Dir: {cwd}")
    print(f"Command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=str(cwd), check=True)
    except subprocess.CalledProcessError as e:
        print(f"CRITICAL ERROR during {description}: {e}", file=sys.stderr)
        raise e


def main():
    args = parse_args()

    # Define exact directories based on your workspace setup
    viper_metrics_dir = Path(__file__).resolve().parent

    # 2 folders up, then inside nvidia_jetson
    nvidia_jetson_dir = (viper_metrics_dir / ".." / ".." / "nvidia_jetson").resolve()

    if not (nvidia_jetson_dir / "run_test_inference.py").exists():
        print(f"❌ ERROR: Could not find run_test_inference.py inside: {nvidia_jetson_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"🎯 Verified nvidia_jetson directory at: {nvidia_jetson_dir}")

    # Establish benchmarking outputs cleanly within your current VIPER_metrics workspace
    # NOTE: run_metrics_new.py defaults to checking 'Results'. If your workspace metrics pipeline
    # relies on default structure, we match that folder name context directly.
    results_dir_name = "Results"
    custom_results_root = nvidia_jetson_dir / results_dir_name
    outputs_dir = viper_metrics_dir / "outputs"
    final_report_dir = viper_metrics_dir / "benchmark_reports"

    # Directory to place clean unpacked weights temporary files
    tmp_weights_dir = viper_metrics_dir / "tmp_unpacked_weights"

    final_report_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    tmp_weights_dir.mkdir(parents=True, exist_ok=True)

    # Get absolute path for checkpoints directory
    absolute_ckpt_dir = args.checkpoint_dir.resolve()
    if not absolute_ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint source folder path missing: {absolute_ckpt_dir}")

    checkpoints = sorted(list(absolute_ckpt_dir.glob("*.pth")))
    if not checkpoints:
        print(f"No target weights files (*.pth) located inside {absolute_ckpt_dir}.")
        return

    print(f"Found {len(checkpoints)} checkpoints for evaluation benchmarking pipeline.")

    # Calculate paths relative to where the commands will be running (nvidia_jetson_dir)
    rel_outputs_dir = os.path.relpath(outputs_dir, nvidia_jetson_dir)

    # Iterate over checkpoints
    for idx, ckpt_path in enumerate(checkpoints, 1):
        model_tag = ckpt_path.stem
        print(f"\n=======================================================")
        print(f" PROCESSING CHECKPOINT [{idx}/{len(checkpoints)}]: {model_tag}")
        print(f"=======================================================")

        # --- AUTO-EXTRACT NESTED MODEL STATE-DICTS ---
        print(f"🔧 Analyzing state dictionary structure for {ckpt_path.name}...")
        checkpoint_dict = torch.load(ckpt_path, map_location="cpu")

        # Check if it's a bundled checkpoint dictionary or raw weights
        if isinstance(checkpoint_dict, dict) and "model_state" in checkpoint_dict:
            print("💡 Found nested 'model_state' key. Extracting pure network weights layer keys...")
            clean_weights = checkpoint_dict["model_state"]
        else:
            print("✅ File structure appears to be direct raw weights already.")
            clean_weights = checkpoint_dict

        # Write clean pure file to our temporary location
        clean_weights_path = tmp_weights_dir / f"clean_{model_tag}.pth"
        torch.save(clean_weights, clean_weights_path)
        # ---------------------------------------------

        # Target directory where this specific checkpoint's frames go
        target_eval_pred_dir = custom_results_root / model_tag / args.dataset / args.mask_type / "_official_eval_pred"

        if target_eval_pred_dir.exists():
            shutil.rmtree(target_eval_pred_dir)

        # TASK A: Run Test Inference Framework
        inference_cmd = [
            sys.executable, "run_test_inference.py",
            "--model", model_tag,
            "--splits", f"{args.dataset}:{args.mask_type}",
            "--frames-subdir", args.frames_subdir,
            "--weights-path", str(clean_weights_path),
            "--results-dir", str(custom_results_root)
        ]

        if args.fp16:
            inference_cmd.append("--fp16")

        run_command(inference_cmd, nvidia_jetson_dir, f"Running Inference for {model_tag}")

        # TASK B: Run Spatial/Perceptual Metrics (PSNR, SSIM, VFID)
        metrics_cmd = [
            sys.executable, "run_metrics_new.py",
            "--dataset", args.dataset,
            "--mask-type", args.mask_type,
            "--frames-subdir", args.frames_subdir,
            "--device", args.device,
            "--models", model_tag,
            "--output-dir", f"{rel_outputs_dir}",
            "--i3d-weights",
            "/home/sw66/Projects/Video-inpainter-for-edge-devices/Baselines_Repos/pthFiles/ProPainter/i3d_rgb_imagenet.pt"
        ]
        metrics_dir = nvidia_jetson_dir.parent / "metrics_new"
        run_command(metrics_cmd, metrics_dir, f"Extracting PSNR, SSIM, & VFID for {model_tag}")

        # TASK C: Compute Modern Native Warping Error Matrix (FWE)
        ewarp_cmd = [
            sys.executable, "run_custom_ewarp.py",
            "--dataset", args.dataset,
            "--mask-type", args.mask_type,
            "--frames-subdir", args.frames_subdir,
            "--models", model_tag,
            "--output-dir", f"{rel_outputs_dir}"
        ]
        run_command(ewarp_cmd, metrics_dir, f"Evaluating Temporal Consistency (FWE via RAFT) for {model_tag}")

        # TASK D: Consolidate data logs
        metric_file = None
        for f in outputs_dir.glob("*.json"):
            if model_tag in f.name:
                metric_file = f
                break

        if metric_file and metric_file.exists():
            destination = final_report_dir / f"{model_tag}_metrics.json"
            shutil.copy2(metric_file, destination)
            print(f" Successfully tracked comprehensive logging ledger to: {destination}")

            with open(destination, "r", encoding="utf-8") as rf:
                summary_data = json.load(rf)
                print(
                    f"| Model: {model_tag} | PSNR: {summary_data.get('psnr')} | SSIM: {summary_data.get('ssim')} | VFID: {summary_data.get('vfid')} |")
        else:
            print(f" [Warning]: Target evaluation payload missing for {model_tag} in outputs/ folder.", file=sys.stderr)

    # Clean up temporary parsed weights folder when finished
    if tmp_weights_dir.exists():
        shutil.rmtree(tmp_weights_dir)
        print("\n🧹 Cleaned up temporary unpacked weight instances.")

    print("\n Benchmarking process complete. Final ledgers archived inside 'benchmark_reports/'.")


if __name__ == "__main__":
    main()