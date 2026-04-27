from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BaseVideoInpainter(ABC):
    """Base class for video inpainting model adapters.

    Each adapter wraps a specific model and handles its own:
    - Model loading from checkpoint
    - Preprocessing (resize, normalize, mask dilation)
    - Inference (online sequential, sliding window, etc.)
    - Postprocessing (denormalize, composite, resize to original)

    Usage:
        adapter = SomeAdapter("path/to/weights.pth")
        inpainted = adapter.inpaint(video_sample.frames, video_sample.masks)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for logging and result directories."""
        ...

    @abstractmethod
    def inpaint(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        resize_to_original: bool = True,
    ) -> list[np.ndarray]:
        """Inpaint masked regions in a video sequence.

        Args:
            frames: List of (H, W, 3) uint8 RGB frames.
            masks: List of (H, W) binary uint8 masks (1=inpaint, 0=keep).
            resize_to_original: If True, resize the output back to the
                original frame resolution. If False, return frames at the
                model's native inference resolution.

        Returns:
            List of (H, W, 3) uint8 RGB frames with masked regions inpainted.
        """
        ...
