from torchinfo import summary
from unet_cell import UNetCell
from video_inpainter import VideoInpainter

model = VideoInpainter(in_channels=10, base_channels=32, num_layers=3)
summary(model, input_size=(1, 10, 520, 520)) # Batch, Channels, H, W