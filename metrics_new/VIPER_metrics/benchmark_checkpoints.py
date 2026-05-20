#!/usr/bin/env python3
"""
Automated Video Inpainting Benchmark Script.
Iterates over a directory of checkpoints (.pth), runs inference with FP16,
computes pixel/perceptual metrics, tracks temporal warping errors,
and generates a comparative evaluation ledger.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


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
        "--fast-blind-root",
        type=Path,
        required=True,
        help="Root path to the fast_blind repository for FWE calculations",
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
    return parser.parse_args()


def run_command(cmd: list[str], description: str) -> None:
    """Helper method to execute shell commands with verbose real-time logging."""
    print(f"\n--- [Executing Task] {description} ---")
    print(f"Command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"CRITICAL ERROR during {description}: {e}", file=sys.stderr)
        raise e


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent

    # Matching your preferred evaluation directory structure
    results_dir_name = "JetsonResults"
    custom_results_root = script_dir / "nvidia_jetson" / results_dir_name
    outputs_dir = script_dir / "outputs"
    final_report_dir = script_dir / "benchmark_reports"

    final_report_dir.mkdir(parents=True, exist_ok=True)

    # Locate and catalog all target checkpoint files (.pth)
    if not args.checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint source folder path missing: {args.checkpoint_dir}")

    checkpoints = sorted(list(args.checkpoint_dir.glob("*.pth")))
    if not checkpoints:
        print(f"No target weights files (*.pth) located inside {args.checkpoint_dir}.")
        return

    print(f"Found {len(checkpoints)} checkpoints for evaluation benchmarking pipeline.")

    # Iterate over each file checkpoint sequentially
    for idx, ckpt_path in enumerate(checkpoints, 1):
        model_tag = ckpt_path.stem
        print(f"\n=======================================================")
        print(f" PROCESSING CHECKPOINT [{idx}/{len(checkpoints)}]: {model_tag}")
        print(f"=======================================================")

        # Clean existing runtime subdirectories to ensure zero data-leak across metrics
        run_results_path = custom_results_root / model_tag
        if run_results_path.exists():
            shutil.rmtree(run_results_path)

        # TASK A: Run Test Inference Framework (Using requested options)
        inference_cmd = [
            sys.executable, "run_test_inference.py",
            "--model", args.model_type,
            "--splits", f"{args.dataset}:{args.mask_type}",
            "--frames-subdir", args.frames_subdir,
            "--fp16",
            "--weights-path", str(ckpt_path),
            "--results-dir", str(run_results_path)
        ]
        run_command(inference_cmd, f"Running FP16 Inference for {model_tag}")

        # TASK B: Run Spatial/Perceptual Metrics (PSNR, SSIM, VFID)
        # Because run_metrics_new.py hardcodes its search to:
        # repo_root / "nvidia_jetson" / "Results"
        # We temporarily update the args context parameter or adjust the model directory tracking flag.
        metrics_cmd = [
            sys.executable, "run_metrics_new.py",
            "--dataset", args.dataset,
            "--mask-type", args.mask_type,
            "--frames-subdir", args.frames_subdir,
            "--device", args.device,
            # We bypass the hardcoded default folder structure by routing through the path flag string trick
            "--models", f"../{results_dir_name}/{model_tag}",
            "--output-dir", str(outputs_dir)
        ]
        run_command(metrics_cmd, f"Extracting PSNR, SSIM, & VFID for {model_tag}")

        # TASK C: Compute Fast Blind Warping Error Matrix (FWE)
        ewarp_cmd = [
            sys.executable, "run_fast_blind_ewarp.py",
            "--fast-blind-root", str(args.fast_blind_root),
            "--dataset", args.dataset,
            "--mask-type", args.mask_type,
            "--frames-subdir", args.frames_subdir,
            # Match the relative routing directory sequence used in Task B
            "--models", f"../{results_dir_name}/{model_tag}",
            "--output-dir", str(outputs_dir),
            "--copy-input" if idx == 1 else ""
        ]
        ewarp_cmd = [item for item in ewarp_cmd if item]
        run_command(ewarp_cmd, f"Evaluating Temporal Consistency (FWE) for {model_tag}")

        # TASK D: Consolidate data payload records securely
        # Clean the tracking folder reference mapping before locating the JSON file
        metric_file = outputs_dir / f".._{results_dir_name}_{model_tag}.json"

        # Fallback check if your script cleans string dots dynamically in path processing
        if not metric_file.exists():
            metric_file = outputs_dir / f"{model_tag}.json"

        if metric_file.exists():
            destination = final_report_dir / f"{model_tag}_metrics.json"
            shutil.copy2(metric_file, destination)
            print(f" Successfully tracked comprehensive logging ledger to: {destination}")

            with open(destination, "r", encoding="utf-8") as rf:
                summary_data = json.load(rf)
                print(
                    f"| Model: {model_tag} | PSNR: {summary_data.get('psnr')} | SSIM: {summary_data.get('ssim')} | VFID: {summary_data.get('vfid')} |")
        else:
            print(f" [Warning]: Target evaluation payload missing for {model_tag}. Verify script outputs pathing.",
                  file=sys.stderr)

    print("\n Benchmarking process complete. Final ledgers archived inside 'benchmark_reports/'.")


if __name__ == "__main__":
    main()