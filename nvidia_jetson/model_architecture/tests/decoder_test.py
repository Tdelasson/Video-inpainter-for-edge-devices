import torch
import pytest
from ..decoder import Decoder


@pytest.mark.parametrize("num_layers, base_channels", [
    (3, 32),
    (4, 64),
])

def test_decoder_variants(base_channels, num_layers):
    final_hw = 512
    bottleneck_hw = final_hw // (2 ** num_layers)

    in_ch = base_channels * (2 ** (num_layers - 2))

    model = Decoder(in_ch, base_channels, num_layers, 10)
    input_tensor = torch.randn(1, in_ch, bottleneck_hw, bottleneck_hw)

    skips = []

    skips.append(torch.randn(1, 10, final_hw, final_hw))

    for i in range(num_layers - 1):
        ch = base_channels * (2 ** i)
        hw = final_hw // (2 ** (i + 1))
        skips.append(torch.randn(1, ch, hw, hw))

    output = model(input_tensor, skips)

    assert output.shape == (1, base_channels, final_hw, final_hw)