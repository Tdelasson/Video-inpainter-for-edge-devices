import torch
import pytest
from ..conv_gru import ConvolutionalGatedRecurrentUnits

def test_conv_gru():
    in_channels = 3
    hidden_dim = 16

    model = ConvolutionalGatedRecurrentUnits(in_channels, hidden_dim)

    input_tensor = torch.randn(1, in_channels, 64, 64)
    h_prev = torch.zeros(1, hidden_dim, 64, 64)

    output = model(input_tensor, h_prev)

    assert output.shape == (1, hidden_dim, 64, 64)
    assert torch.any(output != 0)


