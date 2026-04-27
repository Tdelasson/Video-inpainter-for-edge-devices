from torchinfo import summary
from .. import VideoInpainter
from .. import unet_cell


model = VideoInpainter(in_channels=10, base_channels=128, num_layers=4)
summary(model, input_size=(1, 10, 512, 512)) # Batch, Channels, H, W