import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import json
import sys

# Add the nvidia_jetson directory to the path so we can import the dataloader
NVIDIA_JETSON_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NVIDIA_JETSON_DIR))

from Data.dataloader import TestDataset, VideoSample

# Path to real test data (for integration tests)
TEST_DATA_ROOT = NVIDIA_JETSON_DIR / "Data" / "Test_Data"


# ============================================================
# Helper to create fake test data in a temporary directory
# ============================================================

def create_fake_dataset(tmp_path, dataset_name, mask_folder, video_names, num_frames,
                        frame_ext=".jpg", mask_ext=".png",
                        frame_digits=5, mask_digits=5,
                        create_test_json=False):
    """
    Creates a minimal fake dataset structure for unit testing.

    tmp_path/
        <dataset_name>/
            JPEGImages/<video_name>/00000.jpg, 00001.jpg, ...
            <mask_folder>/<video_name>/00000.png, 00001.png, ...
            test.json (optional)
    """
    frames_dir = tmp_path / dataset_name / "JPEGImages"
    masks_dir = tmp_path / dataset_name / mask_folder

    test_json_data = {}

    for vname in video_names:
        # Create frame images (small 8x8 RGB)
        vid_frames = frames_dir / vname
        vid_frames.mkdir(parents=True)
        for i in range(num_frames):
            img = Image.fromarray(np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8))
            img.save(vid_frames / f"{i:0{frame_digits}d}{frame_ext}")

        # Create mask images (small 8x8 grayscale, either 0 or 255)
        vid_masks = masks_dir / vname
        vid_masks.mkdir(parents=True)
        for i in range(num_frames):
            mask = np.random.choice([0, 255], size=(8, 8)).astype(np.uint8)
            Image.fromarray(mask, mode="L").save(vid_masks / f"{i:0{mask_digits}d}{mask_ext}")

        test_json_data[vname] = num_frames

    if create_test_json:
        with open(tmp_path / dataset_name / "test.json", "w") as f:
            json.dump(test_json_data, f)

    return tmp_path


# ============================================================
# Unit Tests — work without real data, use temporary fake data
# ============================================================

class TestValidation:
    """Tests for input validation in __init__."""

    def test_invalid_dataset_name(self, tmp_path):
        with pytest.raises(ValueError, match="dataset must be"):
            TestDataset(str(tmp_path), dataset="InvalidName", mask_type="synthetic")

    def test_invalid_mask_type(self, tmp_path):
        with pytest.raises(ValueError, match="mask_type must be"):
            TestDataset(str(tmp_path), dataset="DAVIS", mask_type="invalid")

    def test_object_masks_youtube_vos(self, tmp_path):
        with pytest.raises(ValueError, match="Object masks are only available"):
            TestDataset(str(tmp_path), dataset="YouTube-VOS", mask_type="object")

    def test_missing_frames_directory(self, tmp_path):
        # Create masks dir but not frames dir
        (tmp_path / "DAVIS" / "SyntheticMasks").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="Frames directory not found"):
            TestDataset(str(tmp_path), dataset="DAVIS", mask_type="synthetic")

    def test_missing_masks_directory(self, tmp_path):
        # Create frames dir but not masks dir
        (tmp_path / "DAVIS" / "JPEGImages").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="Masks directory not found"):
            TestDataset(str(tmp_path), dataset="DAVIS", mask_type="synthetic")


class TestDiscovery:
    """Tests for video sequence discovery."""

    def test_davis_synthetic_uses_test_json(self, tmp_path):
        create_fake_dataset(tmp_path, "DAVIS", "SyntheticMasks",
                            ["bear", "bus", "car"], num_frames=3,
                            mask_digits=4, create_test_json=True)
        dataset = TestDataset(str(tmp_path), "DAVIS", "synthetic")
        assert len(dataset) == 3
        assert dataset.video_names == ["bear", "bus", "car"]

    def test_davis_synthetic_missing_test_json(self, tmp_path):
        create_fake_dataset(tmp_path, "DAVIS", "SyntheticMasks",
                            ["bear"], num_frames=3, create_test_json=False)
        with pytest.raises(FileNotFoundError, match="test.json not found"):
            TestDataset(str(tmp_path), "DAVIS", "synthetic")

    def test_davis_object_uses_folder_intersection(self, tmp_path):
        create_fake_dataset(tmp_path, "DAVIS", "ObjectMasks",
                            ["bear", "bus", "dog"], num_frames=3)
        dataset = TestDataset(str(tmp_path), "DAVIS", "object")
        assert len(dataset) == 3

    def test_youtube_vos_uses_folder_intersection(self, tmp_path):
        create_fake_dataset(tmp_path, "YouTube-VOS", "SyntheticMasks",
                            ["abc123", "def456"], num_frames=5)
        dataset = TestDataset(str(tmp_path), "YouTube-VOS", "synthetic")
        assert len(dataset) == 2

    def test_no_matching_videos_raises(self, tmp_path):
        # Create frames and masks with different video names
        (tmp_path / "DAVIS" / "JPEGImages" / "bear").mkdir(parents=True)
        (tmp_path / "DAVIS" / "ObjectMasks" / "dog").mkdir(parents=True)
        with pytest.raises(ValueError, match="No matching video sequences"):
            TestDataset(str(tmp_path), "DAVIS", "object")


class TestLoading:
    """Tests for loading video data."""

    def test_loads_correct_frame_shape_and_dtype(self, tmp_path):
        create_fake_dataset(tmp_path, "DAVIS", "ObjectMasks",
                            ["bear"], num_frames=4)
        dataset = TestDataset(str(tmp_path), "DAVIS", "object")
        video = dataset[0]

        assert len(video.frames) == 4
        assert video.frames[0].shape == (8, 8, 3)
        assert video.frames[0].dtype == np.uint8

    def test_loads_correct_mask_shape_and_dtype(self, tmp_path):
        create_fake_dataset(tmp_path, "DAVIS", "ObjectMasks",
                            ["bear"], num_frames=4)
        dataset = TestDataset(str(tmp_path), "DAVIS", "object")
        video = dataset[0]

        assert len(video.masks) == 4
        assert video.masks[0].shape == (8, 8)
        assert video.masks[0].dtype == np.uint8

    def test_masks_are_binary(self, tmp_path):
        create_fake_dataset(tmp_path, "DAVIS", "ObjectMasks",
                            ["bear"], num_frames=4)
        dataset = TestDataset(str(tmp_path), "DAVIS", "object")
        video = dataset[0]

        for mask in video.masks:
            unique_values = set(np.unique(mask))
            assert unique_values.issubset({0, 1})

    def test_video_sample_metadata(self, tmp_path):
        create_fake_dataset(tmp_path, "DAVIS", "ObjectMasks",
                            ["bear"], num_frames=3)
        dataset = TestDataset(str(tmp_path), "DAVIS", "object")
        video = dataset[0]

        assert isinstance(video, VideoSample)
        assert video.name == "bear"
        assert video.dataset == "DAVIS"
        assert video.mask_type == "object"

    def test_frame_mask_count_mismatch_raises(self, tmp_path):
        create_fake_dataset(tmp_path, "DAVIS", "ObjectMasks",
                            ["bear"], num_frames=4)
        # Delete one mask to create mismatch
        masks_dir = tmp_path / "DAVIS" / "ObjectMasks" / "bear"
        first_mask = sorted(masks_dir.iterdir())[0]
        first_mask.unlink()

        dataset = TestDataset(str(tmp_path), "DAVIS", "object")
        with pytest.raises(ValueError, match="Frame/mask count mismatch"):
            dataset[0]

    def test_handles_different_digit_formats(self, tmp_path):
        """Synthetic masks use 4-digit names, frames use 5-digit. Should still align."""
        create_fake_dataset(tmp_path, "DAVIS", "SyntheticMasks",
                            ["bear"], num_frames=3,
                            frame_digits=5, mask_digits=4,
                            create_test_json=True)
        dataset = TestDataset(str(tmp_path), "DAVIS", "synthetic")
        video = dataset[0]

        assert len(video.frames) == 3
        assert len(video.masks) == 3

    def test_filters_non_image_files(self, tmp_path):
        create_fake_dataset(tmp_path, "DAVIS", "ObjectMasks",
                            ["bear"], num_frames=3)
        # Add a non-image file that should be ignored
        (tmp_path / "DAVIS" / "JPEGImages" / "bear" / ".DS_Store").touch()
        (tmp_path / "DAVIS" / "ObjectMasks" / "bear" / "Thumbs.db").touch()

        dataset = TestDataset(str(tmp_path), "DAVIS", "object")
        video = dataset[0]
        assert len(video.frames) == 3
        assert len(video.masks) == 3


class TestIndexingAndIteration:
    """Tests for __getitem__, __len__, __iter__, __repr__."""

    def test_index_out_of_range(self, tmp_path):
        create_fake_dataset(tmp_path, "DAVIS", "ObjectMasks",
                            ["bear"], num_frames=2)
        dataset = TestDataset(str(tmp_path), "DAVIS", "object")
        with pytest.raises(IndexError):
            dataset[5]

    def test_iteration(self, tmp_path):
        create_fake_dataset(tmp_path, "DAVIS", "ObjectMasks",
                            ["bear", "bus"], num_frames=2)
        dataset = TestDataset(str(tmp_path), "DAVIS", "object")

        videos = list(dataset)
        assert len(videos) == 2
        assert videos[0].name == "bear"
        assert videos[1].name == "bus"

    def test_repr(self, tmp_path):
        create_fake_dataset(tmp_path, "DAVIS", "ObjectMasks",
                            ["bear", "bus"], num_frames=2)
        dataset = TestDataset(str(tmp_path), "DAVIS", "object")
        r = repr(dataset)
        assert "DAVIS" in r
        assert "object" in r
        assert "2" in r


# ============================================================
# Integration Tests — require real test data on disk
# ============================================================

HAVE_DAVIS = (TEST_DATA_ROOT / "DAVIS" / "JPEGImages").exists()
HAVE_YOUTUBE_VOS = (TEST_DATA_ROOT / "YouTube-VOS" / "JPEGImages").exists()


@pytest.mark.skipif(not HAVE_DAVIS, reason="DAVIS test data not found")
class TestDAVISIntegration:

    def test_davis_synthetic_finds_50_videos(self):
        dataset = TestDataset(str(TEST_DATA_ROOT), "DAVIS", "synthetic")
        assert len(dataset) == 50

    def test_davis_object_finds_90_videos(self):
        dataset = TestDataset(str(TEST_DATA_ROOT), "DAVIS", "object")
        assert len(dataset) == 90

    def test_davis_synthetic_load_first_video(self):
        dataset = TestDataset(str(TEST_DATA_ROOT), "DAVIS", "synthetic")
        video = dataset[0]

        assert isinstance(video, VideoSample)
        assert len(video.frames) == len(video.masks)
        assert len(video.frames) > 0
        # Check frames are valid 3-channel images
        fh, fw = video.frames[0].shape[:2]
        assert fh > 0 and fw > 0
        assert video.frames[0].shape == (fh, fw, 3)
        # Masks may be at a different resolution than frames
        # (DAVIS synthetic masks are 240p while frames are 480p)
        mh, mw = video.masks[0].shape
        assert mh > 0 and mw > 0

    def test_davis_object_load_first_video(self):
        dataset = TestDataset(str(TEST_DATA_ROOT), "DAVIS", "object")
        video = dataset[0]

        assert len(video.frames) == len(video.masks)
        for mask in video.masks:
            assert set(np.unique(mask)).issubset({0, 1})


@pytest.mark.skipif(not HAVE_YOUTUBE_VOS, reason="YouTube-VOS test data not found")
class TestYouTubeVOSIntegration:

    def test_youtube_vos_finds_541_videos(self):
        dataset = TestDataset(str(TEST_DATA_ROOT), "YouTube-VOS", "synthetic")
        assert len(dataset) == 541

    def test_youtube_vos_load_first_video(self):
        dataset = TestDataset(str(TEST_DATA_ROOT), "YouTube-VOS", "synthetic")
        video = dataset[0]

        assert isinstance(video, VideoSample)
        assert len(video.frames) == len(video.masks)
        assert len(video.frames) > 0
        assert video.frames[0].ndim == 3
        assert video.masks[0].ndim == 2
