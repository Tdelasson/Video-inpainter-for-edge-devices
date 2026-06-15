from torchinfo import summary
from .. import Viper
from .. import unet_cell


model = Viper(in_channels=20, base_channels=128, num_layers=4)
summary(model, input_size=(1, 20, 432, 240)) # Batch, Channels, H, W