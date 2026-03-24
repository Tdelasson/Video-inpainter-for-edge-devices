from unet_cell import UNetCell
from torch import nn
import torch

class VideoInpainter(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, num_layers: int, kernel_size: int=3, stride: int=1, padding: int=1):
        super(VideoInpainter, self).__init__()
        self.unet_cell = UNetCell(in_channels, base_channels, num_layers, kernel_size, stride, padding)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim == 5:
            return self._forward_sequence(x, h_prev)
        else:
            return self.unet_cell(x, h_prev)

    def _forward_sequence(self, x: torch.Tensor, h_prev: torch.Tensor=None):
        batch_size, seq_len, c, h, w = x.size()
        outputs = []

        # Loop through every frame in the sequence
        for t in range(seq_len):
            # use frame 't'
            current_frame = x[:, t, :, :, :]

            output_frame, h_prev = self.unet_cell(current_frame, h_prev)

            outputs.append(output_frame)

        return torch.stack(outputs, dim=1), h_prev