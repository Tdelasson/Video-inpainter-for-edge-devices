from __future__ import annotations

import numpy as np

from .base_adapter import BaseVideoInpainter


class ConstantFillAdapter(BaseVideoInpainter):
    """Fill each masked region with the per-frame average RGB color."""

    @property
    def name(self) -> str:
        return "ConstantFill"

    def inpaint(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        resize_to_original: bool = True,
    ) -> list[np.ndarray]:
        _ = resize_to_original  # Kept for interface compatibility.
        outputs: list[np.ndarray] = []

        for frame, mask in zip(frames, masks):
            out = frame.copy()
            mask_bool = mask.astype(bool)

            if np.any(~mask_bool):
                mean_color = frame[~mask_bool].mean(axis=0)
            else:
                # Degenerate case: if everything is masked, fallback to frame mean.
                mean_color = frame.reshape(-1, 3).mean(axis=0)

            out[mask_bool] = mean_color.astype(np.uint8)
            outputs.append(out)

        return outputs
