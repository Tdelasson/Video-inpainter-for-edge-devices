import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

NVIDIA_JETSON_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NVIDIA_JETSON_DIR))

from Baselines.base_adapter import BaseVideoInpainter
from Baselines.vinet_adapter import (
    MODEL_SIZE,
    ViNETAdapter,
    _reflect_index,
)


def make_fake_video(num_frames=6, h=96, w=128):
    frames = [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(num_frames)]
    masks = [np.random.choice([0, 1], size=(h, w)).astype(np.uint8) for _ in range(num_frames)]
    return frames, masks


class TestInterface:
    def test_is_subclass_of_base(self):
        assert issubclass(ViNETAdapter, BaseVideoInpainter)

    def test_has_required_methods(self):
        assert hasattr(ViNETAdapter, "name")
        assert hasattr(ViNETAdapter, "inpaint")


class TestReflectIndex:
    @pytest.mark.parametrize(
        "index,length,expected",
        [
            (-1, 5, 1),
            (-2, 5, 2),
            (0, 5, 0),
            (4, 5, 4),
            (5, 5, 3),
            (6, 5, 2),
            (8, 5, 0),
        ],
    )
    def test_reflect_index(self, index, length, expected):
        assert _reflect_index(index, length) == expected


class TestPreprocess:
    @pytest.fixture
    def preprocess(self):
        return ViNETAdapter._preprocess

    def test_preprocess_shapes(self, preprocess):
        frames, masks = make_fake_video(num_frames=4, h=128, w=160)

        mock_self = MagicMock()
        mock_self.device = torch.device("cpu")
        mock_self.model_h = MODEL_SIZE
        mock_self.model_w = MODEL_SIZE

        masked_inputs, masks_t = preprocess(mock_self, frames, masks)

        assert masked_inputs.shape == (1, 3, 4, MODEL_SIZE, MODEL_SIZE)
        assert masks_t.shape == (1, 1, 4, MODEL_SIZE, MODEL_SIZE)

    def test_preprocess_ranges(self, preprocess):
        frames, masks = make_fake_video(num_frames=3, h=80, w=100)

        mock_self = MagicMock()
        mock_self.device = torch.device("cpu")
        mock_self.model_h = MODEL_SIZE
        mock_self.model_w = MODEL_SIZE

        masked_inputs, masks_t = preprocess(mock_self, frames, masks)

        assert masked_inputs.min() >= -1.0
        assert masked_inputs.max() <= 1.0
        unique_mask_values = set(np.unique(masks_t.numpy()))
        assert unique_mask_values.issubset({0.0, 1.0})


class TestPostprocess:
    def test_resize_to_original(self):
        comp_frames = [np.random.randint(0, 255, (MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8) for _ in range(3)]
        result = ViNETAdapter._postprocess(comp_frames, orig_h=480, orig_w=854)

        assert len(result) == 3
        assert result[0].shape == (480, 854, 3)
        assert result[0].dtype == np.uint8

    def test_no_resize_when_same_dims(self):
        comp_frames = [np.random.randint(0, 255, (MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8) for _ in range(2)]
        result = ViNETAdapter._postprocess(comp_frames, orig_h=MODEL_SIZE, orig_w=MODEL_SIZE)

        assert result[0].shape == (MODEL_SIZE, MODEL_SIZE, 3)
