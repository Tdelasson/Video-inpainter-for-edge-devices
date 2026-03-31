from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import numpy as np
import json


# Supported image extensions for filtering out non-image files (.DS_Store, thumbs.db, etc.)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class VideoSample:
    """A single video sequence with its frames and masks."""
    name: str                    # e.g. "bear" or "0070461469"
    frames: list[np.ndarray]     # list of (H, W, 3) uint8 arrays (Height, Width, RGB)
    masks: list[np.ndarray]      # list of (H, W) binary uint8 arrays (0 or 1)
    dataset: str                 # "DAVIS" or "YouTube-VOS"
    mask_type: str               # "synthetic" or "RealObject"


class TestDataset:
    """
    Dataset loader for video inpainting evaluation.

    Supports:
        - DAVIS with SyntheticMasks (quantitative evaluation)
        - DAVIS with RealObjectMasks (qualitative evaluation / object removal)
        - YouTube-VOS with SyntheticMasks (quantitative evaluation)

    Usage:
        dataset = TestDataset("path/to/Test_Data", "DAVIS", "synthetic")
        for video in dataset:
            print(video.name, len(video.frames))
    """

    def __init__(self, data_root: str, dataset: str, mask_type: str = "synthetic"):
        self.data_root = Path(data_root)
        self.dataset = dataset
        self.mask_type = mask_type

        # --- Validate inputs ---
        if dataset not in ("DAVIS", "YouTube-VOS"):
            raise ValueError(f"dataset must be 'DAVIS' or 'YouTube-VOS', got '{dataset}'")
        if mask_type not in ("synthetic", "RealObject"):
            raise ValueError(f"mask_type must be 'synthetic' or 'RealObject', got '{mask_type}'")
        if mask_type == "RealObject" and dataset != "DAVIS":
            raise ValueError("RealObject masks are only available for the DAVIS dataset")

        # --- Set up directory paths ---
        self.frames_dir = self.data_root / dataset / "JPEGImages"
        if mask_type == "synthetic":
            self.masks_dir = self.data_root / dataset / "SyntheticMasks"
        else:
            self.masks_dir = self.data_root / dataset / "RealObjectMasks"

        # Verify the directories actually exist (catches missing downloads early)
        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {self.frames_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        # --- Discover which video sequences to load ---
        self.video_names = self._discover_videos()

    def _discover_videos(self) -> list[str]:
        """
        Determine which video sequences are available.

        For DAVIS synthetic: uses test.json to get the 50 test sequences.
        For DAVIS RealObject / YouTube-VOS: uses the intersection of frame and mask folders.
        """
        if self.dataset == "DAVIS" and self.mask_type == "synthetic":
            # test.json defines exactly which 50 sequences have synthetic masks
            test_json = self.data_root / "DAVIS" / "test.json"
            if not test_json.exists():
                raise FileNotFoundError(f"test.json not found: {test_json}")
            with open(test_json) as f:
                test_info = json.load(f)
            return sorted(test_info.keys())

        # For all other cases: find videos that exist in BOTH frames and masks dirs
        frame_videos = {d.name for d in self.frames_dir.iterdir() if d.is_dir()}
        mask_videos = {d.name for d in self.masks_dir.iterdir() if d.is_dir()}
        common = sorted(frame_videos & mask_videos)

        if not common:
            raise ValueError(
                f"No matching video sequences found between "
                f"{self.frames_dir} and {self.masks_dir}"
            )

        return common

    def _get_image_paths(self, directory: Path) -> list[Path]:
        """Get sorted image file paths from a directory, filtering out non-image files."""
        return sorted(
            p for p in directory.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

    def __len__(self) -> int:
        return len(self.video_names)

    def __getitem__(self, idx: int) -> VideoSample:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        video_name = self.video_names[idx]

        # Get sorted image paths (sorted order handles the filename format mismatch
        # between 4-digit synthetic masks and 5-digit JPEG frames)
        frame_paths = self._get_image_paths(self.frames_dir / video_name)
        mask_paths = self._get_image_paths(self.masks_dir / video_name)

        if len(frame_paths) != len(mask_paths):
            raise ValueError(
                f"Frame/mask count mismatch for '{video_name}': "
                f"{len(frame_paths)} frames vs {len(mask_paths)} masks"
            )

        # Load frames as RGB numpy arrays
        frames = []
        for p in frame_paths:
            img = Image.open(p).convert("RGB")
            frames.append(np.array(img))

        # Load masks as binary numpy arrays (1 = inpaint this region, 0 = keep)
        masks = []
        for p in mask_paths:
            mask_img = Image.open(p)
            mask_array = np.array(mask_img)

            # If mask is RGB/RGBA for some reason, reduce to one channel
            if mask_array.ndim == 3:
                mask_array = mask_array[..., 0]

            if self.mask_type == "synthetic":
                # Synthetic masks are ordinary black/white masks
                mask_array = (mask_array > 127).astype(np.uint8)
            else:
                # RealObject masks are label maps / palette masks
                # Any non-zero label means "remove this object region"
                mask_array = (mask_array > 0).astype(np.uint8)

            masks.append(mask_array)


        return VideoSample(
            name=video_name,
            frames=frames,
            masks=masks,
            dataset=self.dataset,
            mask_type=self.mask_type,
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return (
            f"TestDataset(dataset='{self.dataset}', "
            f"mask_type='{self.mask_type}', videos={len(self)})"
        )
