import torch
from model_architecture.video_inpainter import VideoInpainter

# 1. Setup parameters (must match your trained model)
seq_len = 5
in_channels = seq_len * 3 + seq_len
base_channels = 32 # Use your BASE_CHANNELS value
num_layers = 4     # Use your NUM_LAYERS value
device = torch.device("cuda")

# 2. Load the model
model = VideoInpainter(in_channels=in_channels, base_channels=base_channels, num_layers=num_layers)
model.load_state_dict(torch.load("results/MyModel_V1/phase_name/final_model.pth"))
model.to(device).eval()

# 3. Create dummy input (Batch, Channels, Height, Width)
# Assuming 448x448 based on your inference script
dummy_input = torch.randn(1, 20, 512, 512).to("cuda")

# 4. Export
torch.onnx.export(
    model,
    dummy_input,
    "video_inpainter_dynamic.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    # Specify which dimensions can change
    dynamic_axes={
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'height', 3: 'width'}
    }
)