import torch
import pytest
from ..unet_cell import UNetCell

class TestUNetCell:

    def test_unet_cell(self):
        in_channels = 10
        base_channels = 32
        num_layers = 3

        model = UNetCell(in_channels, base_channels, num_layers)

        input_tensor = torch.randn(1, in_channels, 520, 520)
        input_h = torch.zeros(1, base_channels * (2 ** (num_layers - 1)), 520 // (2 ** num_layers), 520 // (2 ** num_layers))

        output_image, output_h = model(input_tensor, input_h)

        assert output_image.shape == (1, 4, 520, 520)
        assert output_h.shape == (1, base_channels * (2 ** (num_layers - 1)), 520 // (2 ** num_layers), 520 // (2 ** num_layers))

