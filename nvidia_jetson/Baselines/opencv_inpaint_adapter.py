from __future__ import annotations

import cv2
import numpy as np

from .base_adapter import BaseVideoInpainter


class OpenCVInpaintAdapter(BaseVideoInpainter):
    """Per-frame OpenCV inpainting baseline (no temporal information)."""

    def __init__(self, method: str = "telea", radius: float = 3.0):
        method_key = method.lower()
        if method_key not in ("telea", "ns"):
            raise ValueError(f"Unsupported OpenCV inpaint method: {method}")

        self.method = method_key
        self.radius = float(radius)
        self._cv_flag = cv2.INPAINT_TELEA if method_key == "telea" else cv2.INPAINT_NS

    @property
    def name(self) -> str:
        return "OpenCV_Telea" if self.method == "telea" else "OpenCV_NS"

    def inpaint(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        resize_to_original: bool = True,
    ) -> list[np.ndarray]:
        _ = resize_to_original  # Kept for interface compatibility.
        outputs: list[np.ndarray] = []

        for frame, mask in zip(frames, masks):
            mask_u8 = (mask > 0).astype(np.uint8) * 255
            out = cv2.inpaint(frame, mask_u8, self.radius, self._cv_flag)
            outputs.append(out)

        return outputs
