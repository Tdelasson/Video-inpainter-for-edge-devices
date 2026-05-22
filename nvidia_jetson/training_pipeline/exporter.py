import torch
import argparse
import sys
import os
from model_architecture.viper import Viper

parser = argparse.ArgumentParser(description="Export Viper model to ONNX (static)")
parser.add_argument("--pth_path", type=str, required=True, help="Path to the .pth model file")
parser.add_argument("--seq_len", type=int, default=5)
parser.add_argument("--base_channels", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--height", type=int, default=432)
parser.add_argument("--width", type=int, default=240)
args = parser.parse_args()

in_channels = args.seq_len * 3 + args.seq_len
device = torch.device("cpu")

# Initialize architecture
model = Viper(in_channels=in_channels, base_channels=args.base_channels, num_layers=args.num_layers)

# Robustly load weights whether wrapped in a dict or raw
checkpoint = torch.load(args.pth_path, map_location=device)
if isinstance(checkpoint, dict):
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint)

model.to(device).eval()

# Programmatically calculate hidden state shape (Will yield 1024, 27, 15)
downsample_factor = 2 ** args.num_layers
hidden_channels = args.base_channels * (2 ** (args.num_layers - 1))
h, w = args.height // downsample_factor, args.width // downsample_factor

# Fixed, static dummy arrays matching your desired resolution
dummy_input = torch.randn(1, in_channels, args.height, args.width)
dummy_hidden = torch.zeros(1, hidden_channels, h, w)

onnx_path = os.path.splitext(args.pth_path)[0] + ".onnx"

# Exporting without dynamic_axes locks in the shapes permanently
torch.onnx.export(
    model,
    (dummy_input, dummy_hidden),
    onnx_path,
    export_params=True,
    opset_version=17,  # Recommended stability target for recurrent blocks in TensorRT
    do_constant_folding=True,
    input_names=['input', 'hidden_state'],
    output_names=['output', 'next_hidden_state'],
)
print(f"Static model successfully exported to {onnx_path}")