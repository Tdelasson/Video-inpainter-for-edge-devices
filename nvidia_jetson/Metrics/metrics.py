from __future__ import annotations

"""Performance measurement utilities used by local inference runs."""

import time
import tracemalloc

import torch


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
