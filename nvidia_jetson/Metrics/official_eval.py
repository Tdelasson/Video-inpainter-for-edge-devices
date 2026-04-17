from __future__ import annotations
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def _write_ground_truth_video(
    video_name: str,
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    gt_root: Path,
    output_size: tuple[int, int],
) -> None:
    video_dir = gt_root / video_name
    video_dir.mkdir(parents=True, exist_ok=True)

    for idx, (frame, mask) in enumerate(zip(frames, masks)):
        frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask.astype(np.uint8), output_size, interpolation=cv2.INTER_NEAREST)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        cv2.imwrite(str(video_dir / f"frame_{idx:04d}_gt.png"), frame_bgr)
        cv2.imwrite(str(video_dir / f"frame_{idx:04d}_mask.png"), mask_u8)


def save_prediction_video(video_name: str, frames: list[np.ndarray], pred_root: Path) -> None:
    video_dir = pred_root / video_name
    video_dir.mkdir(parents=True, exist_ok=True)

    for idx, frame in enumerate(frames):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(video_dir / f"frame_{idx:04d}_pred.png"), frame_bgr)


def _run_repo_module(
    repo_root: Path,
    python_executable: str,
    module_name: str,
    args: list[str],
) -> None:
    cmd = [python_executable, "-m", module_name, *args]
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Official evaluator step failed: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def _ensure_eval_features(
    repo_root: Path,
    gt_root: Path,
    eval_feats_root: Path | None,
    metrics: tuple[str, ...],
    python_executable: str,
    use_cpu: bool = False,
) -> Path:
    if eval_feats_root is not None:
        eval_feats_root = Path(eval_feats_root)
        if not eval_feats_root.exists():
            raise FileNotFoundError(f"Official evaluation feature root not found: {eval_feats_root}")
        return eval_feats_root

    eval_feats_root = gt_root.parent / "eval_feats"
    eval_feats_root.mkdir(parents=True, exist_ok=True)

    needed = set(metrics)
    if "warp_error" in needed or "warp_error_mask" in needed:
        flow_args = [
            f"--gt_root={gt_root}",
            f"--output_root={eval_feats_root}",
        ]
        if use_cpu:
            flow_args.append("-cpu")
        _run_repo_module(
            repo_root=repo_root,
            python_executable=python_executable,
            module_name="src.main.compute_flow_occlusion",
            args=flow_args,
        )
    if "fid" in needed:
        _run_repo_module(
            repo_root=repo_root,
            python_executable=python_executable,
            module_name="src.main.compute_fid_features",
            args=[
                f"--gt_root={gt_root}",
                f"--output_path={eval_feats_root / 'fid.npy'}",
            ],
        )
    if "vfid" in needed:
        vfid_args = [
            f"--gt_root={gt_root}",
            f"--output_path={eval_feats_root / 'vfid.npy'}",
        ]
        if use_cpu:
            vfid_args.append("--cpu")
        _run_repo_module(
            repo_root=repo_root,
            python_executable=python_executable,
            module_name="src.main.compute_vfid_features",
            args=vfid_args,
        )
    if "vfid_clips" in needed:
        _run_repo_module(
            repo_root=repo_root,
            python_executable=python_executable,
            module_name="src.main.compute_vfid_clips_features",
            args=[
                f"--gt_root={gt_root}",
                f"--output_path={eval_feats_root / 'vfid_clips.npy'}",
            ],
        )
    return eval_feats_root


def _run_official_metric(
    repo_root: Path,
    gt_root: Path,
    pred_root: Path,
    eval_feats_root: Path,
    metric_key: str,
    python_executable: str,
) -> float:
    output_path = gt_root.parent / f"{metric_key}.npz"
    cmd = [
        python_executable,
        "-m",
        "src.main.evaluate_inpainting",
        f"--gt_root={gt_root}",
        f"--pred_root={pred_root}",
        f"--eval_feats_root={eval_feats_root}",
        f"--output_path={output_path}",
        f"--include={metric_key}",
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)

    results_obj = np.load(output_path, allow_pickle=True)
    if isinstance(results_obj, np.lib.npyio.NpzFile):
        results = {k: results_obj[k] for k in results_obj.files}
    elif hasattr(results_obj, "item"):
        results = results_obj.item()
    else:
        raise RuntimeError(f"Unexpected official metric output format: {type(results_obj)}")

    if metric_key not in results:
        raise KeyError(f"Metric '{metric_key}' not found in official evaluation output")
    return float(results[metric_key])


def run_official_synthetic_eval(
    videos: list[object],
    pred_root: str | Path,
    repo_root: str | Path,
    output_size: tuple[int, int],
    eval_feats_root: str | Path | None = None,
    metrics: str | tuple[str, ...] = ("vfid", "warp_error_mask"),
    python_executable: str = sys.executable,
    use_cpu: bool = False,
) -> dict[str, float]:
    repo_root = Path(repo_root)
    if not repo_root.exists():
        raise FileNotFoundError(f"Official evaluation repo not found: {repo_root}")

    pred_root = Path(pred_root)
    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_root}")

    if isinstance(metrics, str):
        metrics = (metrics,)

    with tempfile.TemporaryDirectory(prefix="official_eval_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        gt_root = tmp_path / "gt_frames"
        gt_root.mkdir(parents=True, exist_ok=True)

        for video in videos:
            _write_ground_truth_video(
                video.name,
                video.frames,
                video.masks,
                gt_root,
                output_size=output_size,
            )

        eval_feats_root = _ensure_eval_features(
            repo_root=repo_root,
            gt_root=gt_root,
            eval_feats_root=eval_feats_root,
            metrics=metrics,
            python_executable=python_executable,
            use_cpu=use_cpu,
        )

        return {
            metric_key: _run_official_metric(
                repo_root=repo_root,
                gt_root=gt_root,
                pred_root=pred_root,
                eval_feats_root=eval_feats_root,
                metric_key=metric_key,
                python_executable=python_executable,
            )
            for metric_key in metrics
        }
