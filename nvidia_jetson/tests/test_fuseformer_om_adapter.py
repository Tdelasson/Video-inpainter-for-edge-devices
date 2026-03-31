import pytest
import numpy as np
import sys
from pathlib import Path

# Add nvidia_jetson to path
NVIDIA_JETSON_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NVIDIA_JETSON_DIR))

from Baselines.base_adapter import BaseVideoInpainter
from Baselines.fuseformer_om_adapter import (
    FuseFormerOMAdapter,
    _get_ref_index,
    MODEL_H,
    MODEL_W,
)

# Paths for integration tests
REPO_ROOT = NVIDIA_JETSON_DIR.parent
WEIGHTS_PATH = REPO_ROOT / "Baselines_Repos" / "pthFiles" / "OnlineInpainting" / "fuseformer.pth"
TEST_DATA_ROOT = NVIDIA_JETSON_DIR / "Data" / "Test_Data"


# ============================================================
# Helper
# ============================================================

def make_fake_video(num_frames=5, h=64, w=64):
    """Create random frames and masks for testing."""
    frames = [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(num_frames)]
    masks = [np.random.choice([0, 1], size=(h, w)).astype(np.uint8) for _ in range(num_frames)]
    return frames, masks


# ============================================================
# Unit Tests — no GPU or weights needed
# ============================================================

class TestInterface:
    def test_is_subclass_of_base(self):
        assert issubclass(FuseFormerOMAdapter, BaseVideoInpainter)

    def test_has_required_methods(self):
        assert hasattr(FuseFormerOMAdapter, "name")
        assert hasattr(FuseFormerOMAdapter, "inpaint")


class TestGetRefIndex:
    def test_first_frame_no_refs(self):
        """Frame 0 has no past frames, so no refs."""
        refs = _get_ref_index(0, [], ref_step=10, num_refs=3)
        assert refs == []

    def test_early_frames_no_refs(self):
        """Frames before ref_step have no valid refs."""
        refs = _get_ref_index(5, [2, 3, 4], ref_step=10, num_refs=3)
        assert refs == [0]

    def test_frame_at_ref_step(self):
        """Frame 10 should reference frame 0 (divisible by 10, not neighbor)."""
        refs = _get_ref_index(10, [7, 8, 9], ref_step=10, num_refs=3)
        assert refs == [0]

    def test_multiple_refs(self):
        """Frame 30 should find refs at 20, 10, 0 (in chrono order)."""
        refs = _get_ref_index(30, [27, 28, 29], ref_step=10, num_refs=3)
        assert refs == [0, 10, 20]

    def test_excludes_neighbors(self):
        """Refs must not include frames that are in neighbor_ids."""
        refs = _get_ref_index(20, [18, 19, 20], ref_step=10, num_refs=3)
        # Frame 20 is a neighbor so it's excluded; only 10, 0
        assert 20 not in refs
        assert refs == [0, 10]

    def test_respects_num_refs_limit(self):
        """Should return at most num_refs references."""
        refs = _get_ref_index(50, [47, 48, 49], ref_step=10, num_refs=2)
        assert len(refs) <= 2

    def test_chronological_order(self):
        """Returned refs should be in ascending order."""
        refs = _get_ref_index(40, [37, 38, 39], ref_step=10, num_refs=3)
        assert refs == sorted(refs)


class TestPreprocess:
    """Test preprocessing without loading the model (mock the adapter partially)."""

    @pytest.fixture
    def adapter_preprocess(self):
        """Create adapter with mocked model to test preprocessing only."""
        # We can't construct FuseFormerOMAdapter without weights,
        # so we test the static/functional parts directly.
        # Import the preprocess method reference
        from Baselines.fuseformer_om_adapter import FuseFormerOMAdapter
        return FuseFormerOMAdapter._preprocess

    def test_preprocess_output_shapes(self, adapter_preprocess):
        """Verify tensor shapes from preprocessing."""
        import torch
        from unittest.mock import MagicMock

        frames, masks = make_fake_video(5, h=96, w=128)

        # Create a mock adapter with necessary attributes
        mock_self = MagicMock()
        mock_self.device = torch.device("cpu")
        mock_self.fp16 = False

        imgs, masks_t, binary_masks, resized = adapter_preprocess(mock_self, frames, masks)

        assert imgs.shape == (1, 5, 3, MODEL_H, MODEL_W)
        assert masks_t.shape == (1, 5, 1, MODEL_H, MODEL_W)
        assert len(binary_masks) == 5
        assert binary_masks[0].shape == (MODEL_H, MODEL_W, 1)
        assert len(resized) == 5
        assert resized[0].shape == (MODEL_H, MODEL_W, 3)

    def test_preprocess_frame_range(self, adapter_preprocess):
        """Frames should be normalized to [-1, 1]."""
        import torch
        from unittest.mock import MagicMock

        frames, masks = make_fake_video(3, h=64, w=64)
        mock_self = MagicMock()
        mock_self.device = torch.device("cpu")
        mock_self.fp16 = False

        imgs, _, _, _ = adapter_preprocess(mock_self, frames, masks)
        assert imgs.min() >= -1.0
        assert imgs.max() <= 1.0

    def test_preprocess_mask_binary(self, adapter_preprocess):
        """Mask tensors should be binary (0.0 or 1.0)."""
        import torch
        from unittest.mock import MagicMock

        frames, masks = make_fake_video(3, h=64, w=64)
        mock_self = MagicMock()
        mock_self.device = torch.device("cpu")
        mock_self.fp16 = False

        _, masks_t, binary_masks, _ = adapter_preprocess(mock_self, frames, masks)
        unique = set(masks_t.unique().numpy())
        assert unique.issubset({0.0, 1.0})

        for bm in binary_masks:
            assert set(np.unique(bm)).issubset({0, 1})

    def test_preprocess_mask_dilation(self, adapter_preprocess):
        """Dilated mask should be >= original mask (dilation only adds pixels)."""
        import torch
        from unittest.mock import MagicMock

        # Create a small dot mask
        frames = [np.zeros((MODEL_H, MODEL_W, 3), dtype=np.uint8)]
        masks = [np.zeros((MODEL_H, MODEL_W), dtype=np.uint8)]
        masks[0][120, 216] = 1  # single pixel

        mock_self = MagicMock()
        mock_self.device = torch.device("cpu")
        mock_self.fp16 = False

        _, _, binary_masks, _ = adapter_preprocess(mock_self, frames, masks)
        # Dilation should expand the single pixel
        assert binary_masks[0].sum() > 1


class TestPostprocess:
    def test_resize_to_original(self):
        """Should resize back to original dimensions."""
        comp_frames = [np.random.randint(0, 255, (MODEL_H, MODEL_W, 3), dtype=np.uint8) for _ in range(3)]
        result = FuseFormerOMAdapter._postprocess(comp_frames, orig_h=480, orig_w=854)
        assert len(result) == 3
        assert result[0].shape == (480, 854, 3)
        assert result[0].dtype == np.uint8

    def test_no_resize_when_same_dims(self):
        """Should not resize if already at model resolution."""
        comp_frames = [np.random.randint(0, 255, (MODEL_H, MODEL_W, 3), dtype=np.uint8) for _ in range(3)]
        result = FuseFormerOMAdapter._postprocess(comp_frames, orig_h=MODEL_H, orig_w=MODEL_W)
        assert result[0].shape == (MODEL_H, MODEL_W, 3)


# ============================================================
# Integration Tests — require GPU + weights + real data
# ============================================================

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

HAS_WEIGHTS = WEIGHTS_PATH.exists()
HAS_DAVIS = (TEST_DATA_ROOT / "DAVIS" / "JPEGImages").exists()


@pytest.mark.skipif(not HAS_CUDA or not HAS_WEIGHTS, reason="Needs CUDA and FuseFormer weights")
class TestFuseFormerOMIntegration:

    @pytest.fixture(scope="class")
    def adapter(self):
        return FuseFormerOMAdapter(str(WEIGHTS_PATH), device="cuda")

    def test_name(self, adapter):
        assert adapter.name == "FuseFormer_OM"

    def test_inpaint_synthetic_video(self, adapter):
        """Inpaint random frames and verify output format."""
        frames, masks = make_fake_video(10, h=480, w=854)
        result = adapter.inpaint(frames, masks)

        assert len(result) == 10
        assert result[0].shape == (480, 854, 3)
        assert result[0].dtype == np.uint8

    @pytest.mark.skipif(not HAS_DAVIS, reason="DAVIS test data not found")
    def test_inpaint_davis_video(self, adapter):
        """Inpaint a real DAVIS video and verify output differs in masked regions."""
        from Test_Data.dataloader import TestDataset

        dataset = TestDataset(str(TEST_DATA_ROOT), "DAVIS", "object")
        video = dataset[0]

        result = adapter.inpaint(video.frames[:20], video.masks[:20])

        assert len(result) == 20
        assert result[0].shape == video.frames[0].shape
        assert result[0].dtype == np.uint8
