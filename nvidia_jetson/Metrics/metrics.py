from __future__ import annotations

"""Performance measurement utilities used by local inference runs."""

import time
import tracemalloc

import torch


def _safe_cuda_mem_get_info() -> tuple[float | None, float | None]:
    """Return CUDA free/total memory in MB when supported, else (None, None)."""
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception:
        return None, None
    return free_bytes / (1024 ** 2), total_bytes / (1024 ** 2)


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
        baseline_allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        baseline_reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
        start_free_mb, total_cuda_mb = _safe_cuda_mem_get_info()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        for _ in range(num_runs):
            model_fn()
            torch.cuda.synchronize()
        end = time.perf_counter()

        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
        end_free_mb, _ = _safe_cuda_mem_get_info()

        # Backward-compatible headline metric: worst PyTorch-reported CUDA peak.
        peak_memory_mb = max(peak_allocated_mb, peak_reserved_mb)

        cuda_used_start_mb = None
        cuda_used_end_mb = None
        if start_free_mb is not None and total_cuda_mb is not None:
            cuda_used_start_mb = total_cuda_mb - start_free_mb
        if end_free_mb is not None and total_cuda_mb is not None:
            cuda_used_end_mb = total_cuda_mb - end_free_mb
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

    metrics = {
        "fps": round(fps, 2),
        "latency_ms": round(latency_ms, 2),
        "peak_memory_mb": round(peak_memory_mb, 2),
    }

    if use_cuda:
        metrics.update(
            {
                "baseline_allocated_mb": round(baseline_allocated_mb, 2),
                "baseline_reserved_mb": round(baseline_reserved_mb, 2),
                "peak_allocated_mb": round(peak_allocated_mb, 2),
                "peak_reserved_mb": round(peak_reserved_mb, 2),
                "cuda_total_mb": round(total_cuda_mb, 2) if total_cuda_mb is not None else None,
                "cuda_used_start_mb": round(cuda_used_start_mb, 2) if cuda_used_start_mb is not None else None,
                "cuda_used_end_mb": round(cuda_used_end_mb, 2) if cuda_used_end_mb is not None else None,
            }
        )

    return metrics

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
        baseline_allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        baseline_reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
        start_free_mb, total_cuda_mb = _safe_cuda_mem_get_info()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        result = video_fn()
        torch.cuda.synchronize()
        end = time.perf_counter()

        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
        end_free_mb, _ = _safe_cuda_mem_get_info()

        # Keep existing key while making it less likely to under-report.
        peak_memory_mb = max(peak_allocated_mb, peak_reserved_mb)

        cuda_used_start_mb = None
        cuda_used_end_mb = None
        if start_free_mb is not None and total_cuda_mb is not None:
            cuda_used_start_mb = total_cuda_mb - start_free_mb
        if end_free_mb is not None and total_cuda_mb is not None:
            cuda_used_end_mb = total_cuda_mb - end_free_mb
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

    metrics = {
        "fps": round(fps, 2),
        "latency_ms": round(latency_ms, 2),
        "peak_memory_mb": round(peak_memory_mb, 2),
        "total_time_s": round(total_time, 3),
    }

    if use_cuda:
        metrics.update(
            {
                "baseline_allocated_mb": round(baseline_allocated_mb, 2),
                "baseline_reserved_mb": round(baseline_reserved_mb, 2),
                "peak_allocated_mb": round(peak_allocated_mb, 2),
                "peak_reserved_mb": round(peak_reserved_mb, 2),
                "cuda_total_mb": round(total_cuda_mb, 2) if total_cuda_mb is not None else None,
                "cuda_used_start_mb": round(cuda_used_start_mb, 2) if cuda_used_start_mb is not None else None,
                "cuda_used_end_mb": round(cuda_used_end_mb, 2) if cuda_used_end_mb is not None else None,
            }
        )

    return result, metrics
