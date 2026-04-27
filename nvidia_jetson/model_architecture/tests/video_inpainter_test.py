import torch
import pytest
from ..video_inpainter import VideoInpainter

class VideoInpainterTest:

    def test_video_inpainter_live(self):
        in_channels: 10
        base_channels: 32
        num_layers: 3

        model = VideoInpainter(in_channels=in_channels, base_channels=base_channels, num_layers=num_layers)

        input_tensor = torch.randn(1, in_channels, 64, 64)

        output = model(input_tensor)

        assert output.shape == (1, 4, 64, 64)


    def test_video_inpainter_sequence(self):
        in_channels: 10
        base_channels: 32
        num_layers: 3

        model = VideoInpainter(in_channels=in_channels, base_channels=base_channels, num_layers=num_layers)

        input_tensor = torch.randn(1, 100, in_channels, 64, 64)

        output = model(input_tensor)

        assert output.shape == (1, 100, 4, 64, 64)


