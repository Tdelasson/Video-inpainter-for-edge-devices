"""
Segmentation stream pipeline for NVIDIA Jetson.

Captures live video from a CSI camera, runs YOLO segmentation to produce
binary masks, and (optionally) streams / displays the results.

The masks and frames are made available for a downstream inpainting model
via a shared queue that another thread or process can consume.

Usage examples
--------------
# Run with the small model, display locally:
    python segmentation_stream.py --model yolo26s-seg --display

# Run with the nano model, only person class (COCO class 0):
    python segmentation_stream.py --model yolo26n-seg --classes 0

# Use a USB camera instead of CSI:
    python segmentation_stream.py --source 0

# Use a video file for testing:
    python segmentation_stream.py --source path/to/video.mp4
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from Masking.yolo_segmenter import YOLOSegmenter, AVAILABLE_MODELS

# ── Camera defaults (Jetson CSI) ─────────────────────────────────────────────
WIDTH = 1280
HEIGHT = 720
FPS = 30


def gstreamer_pipeline_in(sensor_id: int = 0, w: int = WIDTH, h: int = HEIGHT, fps: int = FPS) -> str:
    """Build a GStreamer capture pipeline for the Jetson CSI camera."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={w}, height={h}, framerate={fps}/1, format=NV12 ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )


# ── Shared output queue for downstream inpainting ────────────────────────────
# Each item is a dict: {"frame": np.ndarray (BGR), "mask": np.ndarray (H,W)}
inpaint_queue: queue.Queue[dict[str, np.ndarray]] = queue.Queue(maxsize=2)


def open_camera(source: str | int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    """Open a camera source – CSI via GStreamer, USB index, or video file."""
    if source == "csi":
        pipeline = gstreamer_pipeline_in(sensor_id=0, w=width, h=height, fps=fps)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    elif source.isdigit():
        cap = cv2.VideoCapture(int(source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
    else:
        # Treat as a video file path
        cap = cv2.VideoCapture(source)

    return cap


def segmentation_loop(
    segmenter: YOLOSegmenter,
    cap: cv2.VideoCapture,
    *,
    display: bool = False,
    stream_port: int | None = None,
) -> None:
    """Main capture → segment → enqueue loop.

    Args:
        segmenter: Initialised YOLOSegmenter instance.
        cap: Opened cv2.VideoCapture.
        display: If True, show an OpenCV window with the annotated output.
        stream_port: (future) UDP port for streaming the result.
    """
    frame_count = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed – exiting.")
                break

            # Run segmentation
            mask, annotated = segmenter.segment(frame, return_annotated=True)

            # Publish frame + mask for downstream inpainting
            payload = {"frame": frame, "mask": mask}
            try:
                inpaint_queue.put_nowait(payload)
            except queue.Full:
                # Drop the oldest to keep latency low
                try:
                    inpaint_queue.get_nowait()
                except queue.Empty:
                    pass
                inpaint_queue.put_nowait(payload)

            # FPS counter
            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed >= 2.0:
                fps_measured = frame_count / elapsed
                print(f"Segmentation FPS: {fps_measured:.1f}")
                frame_count = 0
                t_start = time.time()

            # Optional local display
            if display:
                # Stack the annotated frame and mask side by side
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                combined = np.hstack([annotated, mask_bgr])
                # Resize for display if too large
                disp_w = 1280
                if combined.shape[1] > disp_w:
                    scale = disp_w / combined.shape[1]
                    combined = cv2.resize(combined, None, fx=scale, fy=scale)
                cv2.imshow("YOLO Segmentation | Mask", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted – stopping.")

    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live YOLO segmentation pipeline for Jetson"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n-seg",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Which YOLO segmentation model to use (default: yolo26n-seg)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional explicit model path (.pt/.onnx/.engine). Overrides --model.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="csi",
        help="Camera source: 'csi' for Jetson CSI, integer for USB cam index, "
             "or a path to a video file (default: csi)",
    )
    parser.add_argument("--width", type=int, default=WIDTH, help="Capture width")
    parser.add_argument("--height", type=int, default=HEIGHT, help="Capture height")
    parser.add_argument("--fps", type=int, default=FPS, help="Capture FPS")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input image size")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device, e.g. 'cpu', '0' for GPU (default: auto)",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="COCO class IDs to detect (default: all). "
             "E.g. --classes 0 for person only.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show an OpenCV window with the segmentation output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading {args.model} …")
    segmenter = YOLOSegmenter(
        model_name=args.model,
        model_path=args.model_path,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        target_classes=args.classes,
    )
    print(f"Model loaded from: {segmenter.model_source}")

    cap = open_camera(args.source, args.width, args.height, args.fps)
    if not cap.isOpened():
        print("Error: could not open camera / video source.")
        sys.exit(1)

    print(f"Source: {args.source}  Resolution: {args.width}x{args.height}  FPS: {args.fps}")
    print("Starting segmentation loop … (press Ctrl+C or 'q' in the window to stop)")
    segmentation_loop(segmenter, cap, display=args.display)


if __name__ == "__main__":
    main()
