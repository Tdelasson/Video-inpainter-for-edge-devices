Metrics New Workflow

This folder provides a paper-style evaluation pipeline for:

- PSNR, SSIM, VFID using E2FGVI-style definitions.
- E_warp using fast_blind_video_consistency.
- FPS, latency, and peak memory merged from existing summaries in nvidia_jetson/Results.

The goal is to make comparison across models fair even when model-native output resolutions differ.

Fair Comparison Protocol

1. Pick one canonical evaluation resolution for all models.
2. Run each model at its own native resolution if needed.
3. Resize prediction, GT, and mask to the same canonical resolution before metrics.
4. Use the exact same DAVIS split and frame counts for every model.
5. Keep one set of settings fixed when comparing models.

Recommended canonical setups:

- 432x240 if you want to match many original E2FGVI/FuseFormer style reports.
- 512x512 if you want square-model stress testing, but do not compare directly to 432x240 papers.

Required Files

VFID I3D weights (one of these):

- Baselines_Repos/E2FGVI-master/release_model/i3d_rgb_imagenet.pt
- Baselines_Repos/video-inpainting-evaluation-public/rgb_imagenet.pt

If missing, download from E2FGVI pretrained section and place in one of the paths above.

Script 1: PSNR SSIM VFID plus speed

File:

- metrics_new/run_metrics_new.py

What it does:

- Reads predictions from nvidia_jetson/Results/<MODEL>/<DATASET>/<MASK_SPLIT>/_official_eval_pred.
- Loads GT frames and masks from nvidia_jetson/Test_Data.
- Re-composites with mask like E2FGVI evaluate.py.
- Computes PSNR/SSIM per frame and VFID per video.
- Merges FPS/latency/memory from existing summary.json.
- Saves one JSON per model in metrics_new/outputs.

Example:

python metrics_new/run_metrics_new.py \
  --dataset DAVIS \
  --mask-type synthetic \
  --frames-subdir JPEGImages_432_240 \
  --masks-subdir SyntheticMasks \
  --eval-width 432 --eval-height 240 \
  --models ViNET ProPainter E2FGVI_HQ FuseFormer_OM

Notes:

- If --models is omitted, models are auto-discovered from nvidia_jetson/Results.
- If --eval-width and --eval-height are omitted, GT frame size is used.

Script 2: E_warp with fast_blind

File:

- metrics_new/run_fast_blind_ewarp.py

Prerequisites:

1. Clone fast_blind_video_consistency.
2. Install and compile its dependencies.
3. Download FlowNet2 weights expected by that repo.

The helper script:

- Exports your predictions to fast_blind naming and folder structure.
- Optionally exports input GT frames to fast_blind input layout.
- Runs compute_flow_occlusion.py (unless skipped).
- Runs evaluate_WarpError.py per model.
- Saves merged E_warp values in metrics_new/outputs/ewarp_results.json.

Example:

python metrics_new/run_fast_blind_ewarp.py \
  --fast-blind-root /path/to/fast_blind_video_consistency \
  --dataset DAVIS \
  --mask-type synthetic \
  --frames-subdir JPEGImages_432_240 \
  --eval-width 432 --eval-height 240 \
  --copy-input \
  --models ViNET ProPainter E2FGVI_HQ FuseFormer_OM

Then merge ewarp numbers into your JSON files in metrics_new/outputs.

Output Format

Each model JSON includes:

- model, dataset, mask_type
- psnr, ssim, vfid
- ewarp, ewarp_x1e2 (null unless merged)
- speed.fps, speed.latency_ms, speed.peak_memory_mb
- per_video breakdown

This mirrors your current Video-Evaluation schema while switching to E2FGVI-style quality metrics.
