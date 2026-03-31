"""
Evaluation metrics for video inpainting.

Quality metrics:
    - compute_psnr: Peak Signal-to-Noise Ratio
    - compute_ssim: Structural Similarity Index
    - compute_vfid: Video Fréchet Inception Distance (requires I3D model)
    - compute_ewarp: Warping error for temporal consistency

Performance metrics:
    - measure_performance: FPS, latency, and peak memory usage
"""

import json
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import cv2
import numpy as np
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Quality Metrics
# ============================================================


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images.

    Args:
        img1: Ground truth image, shape (H, W, 3), dtype uint8, range [0, 255].
        img2: Inpainted image, same shape/dtype as img1.

    Returns:
        PSNR value in dB. Returns float('inf') if images are identical.
    """
    assert img1.shape == img2.shape, (
        f"Image shapes differ: {img1.shape} vs {img2.shape}"
    )
    return float(peak_signal_noise_ratio(img1, img2, data_range=255))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index between two images.

    Args:
        img1: Ground truth image, shape (H, W, 3), dtype uint8, range [0, 255].
        img2: Inpainted image, same shape/dtype as img1.

    Returns:
        SSIM value in range [-1, 1]. Higher is better.
    """
    assert img1.shape == img2.shape, (
        f"Image shapes differ: {img1.shape} vs {img2.shape}"
    )
    return structural_similarity(
        img1, img2, data_range=255, channel_axis=2, win_size=65
    )


# ============================================================
# E_warp — Temporal consistency via warping error
# ============================================================


def _write_eval_video_gt(
    video_name: str,
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    gt_root: Path,
    output_size: tuple[int, int] | None = None,
) -> None:
    """Write GT frames and masks in the MichiganCOG evaluation layout."""
    video_dir = gt_root / video_name
    video_dir.mkdir(parents=True, exist_ok=True)

    for idx, (frame, mask) in enumerate(zip(frames, masks)):
        if output_size is not None:
            frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask.astype(np.uint8), output_size, interpolation=cv2.INTER_NEAREST)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        cv2.imwrite(str(video_dir / f"frame_{idx:05d}_gt.png"), frame_bgr)
        cv2.imwrite(str(video_dir / f"frame_{idx:05d}_mask.png"), mask_u8)


def compute_ewarp(
    videos: list[object],
    pred_root: str | Path,
    eval_repo_root: str | Path,
    eval_feats_root: str | Path | None = None,
    metric_key: str = "warp_error_mask",
    output_size: tuple[int, int] | None = None,
    python_executable: str = sys.executable,
) -> float:
    """Compute official E_warp via MichiganCOG/video-inpainting-evaluation.

    The raw value returned here matches the external evaluator. To report
    the paper's ``E_warp (x10^-2)`` style number, multiply the raw result
    by 100.
    """
    eval_repo_root = Path(eval_repo_root)
    if not eval_repo_root.exists():
        raise FileNotFoundError(
            f"Official evaluation repo not found: {eval_repo_root}. "
            "Clone MichiganCOG/video-inpainting-evaluation and update "
            "OFFICIAL_EVAL_REPO before running synthetic evaluations."
        )

    pred_root = Path(pred_root)
    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_root}")

    with tempfile.TemporaryDirectory(prefix="ewarp_gt_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        gt_root = tmp_path / "gt_frames"
        gt_root.mkdir(parents=True, exist_ok=True)

        for video in videos:
            _write_eval_video_gt(
                video.name,
                video.frames,
                video.masks,
                gt_root,
                output_size=output_size,
            )

        if eval_feats_root is None:
            eval_feats_root = tmp_path / "eval_feats"
            preprocess_script = eval_repo_root / "scripts" / "preprocessing" / "compute-evaluation-features.sh"
            if not preprocess_script.exists():
                raise FileNotFoundError(
                    f"Feature preprocessing script not found: {preprocess_script}"
                )
            bash_exe = shutil.which("bash")
            if bash_exe is None:
                raise RuntimeError(
                    "Official E_warp requires precomputed evaluation features or "
                    "a bash executable to run the MichiganCOG preprocessing script."
                )
            subprocess.run(
                [bash_exe, str(preprocess_script), str(gt_root), str(eval_feats_root)],
                cwd=eval_repo_root,
                check=True,
            )
        else:
            eval_feats_root = Path(eval_feats_root)

        output_path = tmp_path / "ewarp_results.json"
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
        subprocess.run(cmd, cwd=eval_repo_root, check=True)

        with open(output_path, encoding="utf-8") as f:
            results = json.load(f)

    if metric_key not in results:
        raise KeyError(f"Metric '{metric_key}' not found in official evaluation output")
    return float(results[metric_key])


# ============================================================
# VFID — Video Fréchet Inception Distance
# ============================================================


def compute_vfid(
    real_videos: list[list[np.ndarray]],
    fake_videos: list[list[np.ndarray]],
    i3d_model_path: str,
) -> float:
    """Compute Video Fréchet Inception Distance between real and generated videos.

    Uses an I3D model pretrained on ImageNet to extract video features,
    then computes the Fréchet distance between the two feature distributions.

    Args:
        real_videos: List of ground truth videos. Each video is a list of
            (H, W, 3) uint8 frames.
        fake_videos: List of inpainted videos, same structure as real_videos.
        i3d_model_path: Path to I3D pretrained weights (i3d_rgb_imagenet.pt).

    Returns:
        VFID score. Lower is better.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i3d_model = _init_i3d_model(i3d_model_path, device)

    real_activations = []
    fake_activations = []

    for real_vid, fake_vid in zip(real_videos, fake_videos):
        real_tensor = _frames_to_tensor(real_vid).to(device)
        fake_tensor = _frames_to_tensor(fake_vid).to(device)

        real_feat = _get_i3d_activations(real_tensor, i3d_model)
        fake_feat = _get_i3d_activations(fake_tensor, i3d_model)

        real_activations.append(real_feat.cpu().numpy().flatten())
        fake_activations.append(fake_feat.cpu().numpy().flatten())

    real_activations = np.array(real_activations)
    fake_activations = np.array(fake_activations)

    return _calculate_vfid(real_activations, fake_activations)


def _frames_to_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    """Convert a list of uint8 numpy frames to a batched float tensor.

    Args:
        frames: List of (H, W, 3) uint8 arrays.

    Returns:
        Tensor of shape (1, 3, T, H, W) with values in [0, 1].
    """
    tensors = []
    for frame in frames:
        t = torch.from_numpy(frame).float() / 255.0
        t = t.permute(2, 0, 1)  # (3, H, W)
        tensors.append(t)
    # Stack to (T, 3, H, W), then rearrange to (1, 3, T, H, W) for I3D
    video = torch.stack(tensors, dim=0)  # (T, 3, H, W)
    video = video.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 3, T, H, W)
    return video


def _init_i3d_model(model_path: str, device: torch.device) -> "InceptionI3d":
    """Load a pretrained I3D model."""
    model = InceptionI3d(400, in_channels=3, final_endpoint="Logits")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def _get_i3d_activations(video: torch.Tensor, model: "InceptionI3d") -> torch.Tensor:
    """Extract flattened I3D features from a video tensor.

    Args:
        video: Tensor of shape (1, 3, T, H, W).
        model: Pretrained InceptionI3d model.

    Returns:
        Feature tensor of shape (1, D).
    """
    with torch.no_grad():
        feat = model.extract_features(video, target_endpoint="Logits")
    return feat.view(feat.size(0), -1)


def _calculate_vfid(
    real_activations: np.ndarray, fake_activations: np.ndarray
) -> float:
    """Compute FID score between two sets of I3D activations."""
    mu1 = np.mean(real_activations, axis=0)
    mu2 = np.mean(fake_activations, axis=0)
    sigma1 = np.cov(real_activations, rowvar=False)
    sigma2 = np.cov(fake_activations, rowvar=False)
    return _calculate_frechet_distance(mu1, sigma1, mu2, sigma2)


def _calculate_frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray,
    mu2: np.ndarray, sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute Fréchet distance between two multivariate Gaussians.

    Adapted from: https://github.com/mseitzer/pytorch-fid
    """
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
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    return float(
        diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    )


# ============================================================
# I3D Model (from https://github.com/piergiaj/pytorch-i3d)
# ============================================================


class _MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        pad = (pad_w // 2, pad_w - pad_w // 2,
               pad_h // 2, pad_h - pad_h // 2,
               pad_t // 2, pad_t - pad_t // 2)
        x = F.pad(x, pad)
        return super().forward(x)


class _Unit3D(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1), padding=0, activation_fn=F.relu,
                 use_batch_norm=True, use_bias=False, name="unit_3d"):
        super().__init__()
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self._output_channels,
            kernel_size=self._kernel_shape,
            stride=self._stride,
            padding=0,
            bias=self._use_bias,
        )
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        pad = (pad_w // 2, pad_w - pad_w // 2,
               pad_h // 2, pad_h - pad_h // 2,
               pad_t // 2, pad_t - pad_t // 2)
        x = F.pad(x, pad)
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class _InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super().__init__()
        self.b0 = _Unit3D(in_channels, out_channels[0], [1, 1, 1], padding=0,
                           name=name + "/Branch_0/Conv3d_0a_1x1")
        self.b1a = _Unit3D(in_channels, out_channels[1], [1, 1, 1], padding=0,
                            name=name + "/Branch_1/Conv3d_0a_1x1")
        self.b1b = _Unit3D(out_channels[1], out_channels[2], [3, 3, 3],
                            name=name + "/Branch_1/Conv3d_0b_3x3")
        self.b2a = _Unit3D(in_channels, out_channels[3], [1, 1, 1], padding=0,
                            name=name + "/Branch_2/Conv3d_0a_1x1")
        self.b2b = _Unit3D(out_channels[3], out_channels[4], [3, 3, 3],
                            name=name + "/Branch_2/Conv3d_0b_3x3")
        self.b3a = _MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1),
                                          padding=0)
        self.b3b = _Unit3D(in_channels, out_channels[5], [1, 1, 1], padding=0,
                            name=name + "/Branch_3/Conv3d_0b_1x1")

    def forward(self, x):
        return torch.cat([self.b0(x), self.b1b(self.b1a(x)),
                          self.b2b(self.b2a(x)), self.b3b(self.b3a(x))], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.

    From: Quo Vadis, Action Recognition? (Carreira & Zisserman, 2017)
    Code adapted from: https://github.com/piergiaj/pytorch-i3d
    """

    VALID_ENDPOINTS = (
        "Conv3d_1a_7x7", "MaxPool3d_2a_3x3", "Conv3d_2b_1x1", "Conv3d_2c_3x3",
        "MaxPool3d_3a_3x3", "Mixed_3b", "Mixed_3c", "MaxPool3d_4a_3x3",
        "Mixed_4b", "Mixed_4c", "Mixed_4d", "Mixed_4e", "Mixed_4f",
        "MaxPool3d_5a_2x2", "Mixed_5b", "Mixed_5c", "Logits", "Predictions",
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint="Logits", name="inception_i3d",
                 in_channels=3, dropout_keep_prob=0.5):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError(f"Unknown final endpoint {final_endpoint}")
        super().__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        self.end_points = {}
        ep = "Conv3d_1a_7x7"
        self.end_points[ep] = _Unit3D(in_channels, 64, [7, 7, 7], stride=(2, 2, 2),
                                       padding=(3, 3, 3), name=name + ep)
        if self._final_endpoint == ep: self.build(); return

        ep = "MaxPool3d_2a_3x3"
        self.end_points[ep] = _MaxPool3dSamePadding(kernel_size=[1, 3, 3],
                                                     stride=(1, 2, 2), padding=0)
        if self._final_endpoint == ep: self.build(); return

        ep = "Conv3d_2b_1x1"
        self.end_points[ep] = _Unit3D(64, 64, [1, 1, 1], padding=0, name=name + ep)
        if self._final_endpoint == ep: self.build(); return

        ep = "Conv3d_2c_3x3"
        self.end_points[ep] = _Unit3D(64, 192, [3, 3, 3], padding=1, name=name + ep)
        if self._final_endpoint == ep: self.build(); return

        ep = "MaxPool3d_3a_3x3"
        self.end_points[ep] = _MaxPool3dSamePadding(kernel_size=[1, 3, 3],
                                                     stride=(1, 2, 2), padding=0)
        if self._final_endpoint == ep: self.build(); return

        ep = "Mixed_3b"
        self.end_points[ep] = _InceptionModule(192, [64, 96, 128, 16, 32, 32],
                                                name + ep)
        if self._final_endpoint == ep: self.build(); return

        ep = "Mixed_3c"
        self.end_points[ep] = _InceptionModule(256, [128, 128, 192, 32, 96, 64],
                                                name + ep)
        if self._final_endpoint == ep: self.build(); return

        ep = "MaxPool3d_4a_3x3"
        self.end_points[ep] = _MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                                     stride=(2, 2, 2), padding=0)
        if self._final_endpoint == ep: self.build(); return

        ep = "Mixed_4b"
        self.end_points[ep] = _InceptionModule(480, [192, 96, 208, 16, 48, 64],
                                                name + ep)
        if self._final_endpoint == ep: self.build(); return

        ep = "Mixed_4c"
        self.end_points[ep] = _InceptionModule(512, [160, 112, 224, 24, 64, 64],
                                                name + ep)
        if self._final_endpoint == ep: self.build(); return

        ep = "Mixed_4d"
        self.end_points[ep] = _InceptionModule(512, [128, 128, 256, 24, 64, 64],
                                                name + ep)
        if self._final_endpoint == ep: self.build(); return

        ep = "Mixed_4e"
        self.end_points[ep] = _InceptionModule(512, [112, 144, 288, 32, 64, 64],
                                                name + ep)
        if self._final_endpoint == ep: self.build(); return

        ep = "Mixed_4f"
        self.end_points[ep] = _InceptionModule(528, [256, 160, 320, 32, 128, 128],
                                                name + ep)
        if self._final_endpoint == ep: self.build(); return

        ep = "MaxPool3d_5a_2x2"
        self.end_points[ep] = _MaxPool3dSamePadding(kernel_size=[2, 2, 2],
                                                     stride=(2, 2, 2), padding=0)
        if self._final_endpoint == ep: self.build(); return

        ep = "Mixed_5b"
        self.end_points[ep] = _InceptionModule(832, [256, 160, 320, 32, 128, 128],
                                                name + ep)
        if self._final_endpoint == ep: self.build(); return

        ep = "Mixed_5c"
        self.end_points[ep] = _InceptionModule(832, [384, 192, 384, 48, 128, 128],
                                                name + ep)
        if self._final_endpoint == ep: self.build(); return

        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = _Unit3D(1024, self._num_classes, [1, 1, 1], padding=0,
                               activation_fn=None, use_batch_norm=False,
                               use_bias=True, name="logits")
        self.build()

    def build(self):
        for k in self.end_points:
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        for ep in self.VALID_ENDPOINTS:
            if ep in self.end_points:
                x = self._modules[ep](x)
        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            x = x.squeeze(3).squeeze(3)
        return x

    def extract_features(self, x, target_endpoint="Logits"):
        for ep in self.VALID_ENDPOINTS:
            if ep in self.end_points:
                x = self._modules[ep](x)
                if ep == target_endpoint:
                    break
        if target_endpoint == "Logits":
            return x.mean(4).mean(3).mean(2)
        return x


# ============================================================
# Performance Metrics
# ============================================================


def measure_performance(
    model_fn: callable,
    num_warmup: int = 5,
    num_runs: int = 20,
    use_cuda: bool | None = None,
) -> dict:
    """Measure FPS, latency, and peak memory usage of a model function.

    Args:
        model_fn: A callable that performs one inference step.
            Should be a closure that captures its own inputs, e.g.:
            ``lambda: model(input_tensor)``
        num_warmup: Number of warmup iterations (not timed).
        num_runs: Number of timed iterations.
        use_cuda: Whether to use CUDA timing. If None, auto-detects from
            torch.cuda.is_available().

    Returns:
        Dict with keys:
            - "fps": Frames per second (inferences per second).
            - "latency_ms": Average latency per inference in milliseconds.
            - "peak_memory_mb": Peak memory usage in megabytes.
    """
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()

    # Warmup
    for _ in range(num_warmup):
        model_fn()

    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        for _ in range(num_runs):
            model_fn()
            torch.cuda.synchronize()
        end = time.perf_counter()

        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        tracemalloc.start()

        start = time.perf_counter()
        for _ in range(num_runs):
            model_fn()
        end = time.perf_counter()

        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory_mb = peak_memory / (1024 ** 2)

    total_time = end - start
    latency_ms = (total_time / num_runs) * 1000
    fps = num_runs / total_time

    return {
        "fps": round(fps, 2),
        "latency_ms": round(latency_ms, 2),
        "peak_memory_mb": round(peak_memory_mb, 2),
    }

def measure_video_run(
    video_fn: callable,
    num_frames: int,
    use_cuda: bool | None = None,
) -> tuple[object, dict]:
    """Run one full video inference and measure throughput/latency/memory."""
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        result = video_fn()
        torch.cuda.synchronize()
        end = time.perf_counter()

        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        tracemalloc.start()

        start = time.perf_counter()
        result = video_fn()
        end = time.perf_counter()

        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory_mb = peak_memory / (1024 ** 2)

    total_time = end - start
    latency_ms = (total_time / num_frames) * 1000 if num_frames else 0.0
    fps = num_frames / total_time if total_time > 0 else 0.0

    return result, {
        "fps": round(fps, 2),
        "latency_ms": round(latency_ms, 2),
        "peak_memory_mb": round(peak_memory_mb, 2),
        "total_time_s": round(total_time, 3),
    }
