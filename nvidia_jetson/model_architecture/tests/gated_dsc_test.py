import torch
import pytest
from ..gated_dsc import GatedDSC


@pytest.mark.parametrize("stride, expected_hw", [
    (1, 64),  # Stride 1 preserves resolution
    (2, 32),  # Stride 2 halves resolution
])
@pytest.mark.parametrize("in_ch, out_ch", [
    (3, 16),
    (128, 64),
])

def test_gated_dsc_variants(stride, expected_hw, in_ch, out_ch):
    model = GatedDSC(in_channels=in_ch, out_channels=out_ch, stride=stride)
    input_tensor = torch.randn(1, in_ch, 64, 64)

    output = model(input_tensor)

    assert output.shape == (1, out_ch, expected_hw, expected_hw)