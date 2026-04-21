from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Add the local ultralytics repo so it can be imported without pip-installing
_ULTRALYTICS_DIR = str(Path(__file__).resolve().parent / "ultralytics-main")
if _ULTRALYTICS_DIR not in sys.path:
    sys.path.insert(0, _ULTRALYTICS_DIR)

from ultralytics import YOLO

_MASKING_DIR = Path(__file__).resolve().parent

# Available model weight files – extend this map when you download more.
AVAILABLE_MODELS: dict[str, Path] = {
    "yolo26n-seg": _MASKING_DIR / "yolo26n-seg.pt",
    "yolo26s-seg": _MASKING_DIR / "yolo26s-seg.pt",
}


class YOLOSegmenter:
    """Thin wrapper around an Ultralytics YOLO segmentation model.

    Usage::

        seg = YOLOSegmenter(model_name="yolo26n-seg", conf=0.35)
        mask = seg.segment(frame_bgr)          # (H, W) uint8  0/255
        mask, annotated = seg.segment(frame_bgr, return_annotated=True)
    """

    def __init__(
        self,
        model_name: str = "yolo26n-seg",
        conf: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str | None = None,
        target_classes: list[int] | None = None,
    ):
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(AVAILABLE_MODELS.keys())}"
            )

        weights = AVAILABLE_MODELS[model_name]
        if not weights.exists():
            raise FileNotFoundError(f"Weight file not found: {weights}")

        self.model = YOLO(str(weights))
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.target_classes = target_classes  # None = all classes

    def segment(
        self,
        frame_bgr: np.ndarray,
        return_annotated: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Run segmentation on a single BGR frame.

        Args:
            frame_bgr: (H, W, 3) uint8 BGR image (OpenCV convention).
            return_annotated: If True, also return the annotated frame
                with bounding boxes and masks drawn by Ultralytics.

        Returns:
            mask: (H, W) uint8 binary mask – 255 where objects are detected,
                  0 elsewhere.  Ready to be used as an inpainting mask.
            annotated (optional): (H, W, 3) uint8 BGR annotated frame.
        """
        h, w = frame_bgr.shape[:2]

        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.target_classes,
            verbose=False,
        )

        result = results[0]

        # Build a combined binary mask from all detected instances
        mask = np.zeros((h, w), dtype=np.uint8)

        if result.masks is not None:
            for seg_mask in result.masks.data:
                # Each mask is a (model_h, model_w) tensor with values in [0, 1]
                m = seg_mask.cpu().numpy()
                m = (m > 0.5).astype(np.uint8) * 255
                # Resize to original frame dimensions if needed
                if m.shape[:2] != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                mask = cv2.bitwise_or(mask, m)

        if return_annotated:
            annotated = result.plot()  # BGR annotated image
            return mask, annotated

        return mask
