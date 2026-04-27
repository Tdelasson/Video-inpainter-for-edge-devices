import torch
import pytest
from ..encoder import Encoder

@pytest.mark.parametrize("num_layers, base_channels", [
    (3,32),
    (4,64),
])

def test_encoder_variants(base_channels, num_layers):
    model = Encoder(10, base_channels, num_layers)
    input_tensor = torch.randn(1, 10, 64, 64)

    output = model(input_tensor)

    assert len(output) == num_layers
    for i in range(num_layers):
        expected_channels = base_channels * (2 ** i)

        expected_hw = int(64 * (0.5 ** (i + 1)))

        assert output[i].shape == (1, expected_channels, expected_hw, expected_hw)