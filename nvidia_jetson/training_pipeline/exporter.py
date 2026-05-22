import torch
import argparse
import os
from model_architecture.viper import Viper

parser = argparse.ArgumentParser(description="Export Viper model to ONNX (static)")
parser.add_argument("--pth_path", type=str, required=True)
parser.add_argument("--seq_len", type=int, default=5)
parser.add_argument("--base_channels", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--height", type=int, default=432)
parser.add_argument("--width", type=int, default=240)
parser.add_argument("--fp16", action='store_true', help="Export in FP16 precision") # Added flag
args = parser.parse_args()

in_channels = args.seq_len * 3 + args.seq_len
device = torch.device("cpu")

# Initialize architecture
model = Viper(in_channels=in_channels, base_channels=args.base_channels, num_layers=args.num_layers)

# Load weights
checkpoint = torch.load(args.pth_path, map_location=device)
state_dict = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
model.load_state_dict(state_dict)

# Prepare shapes
downsample_factor = 2 ** args.num_layers
hidden_channels = args.base_channels * (2 ** (args.num_layers - 1))
h, w = args.height // downsample_factor, args.width // downsample_factor

# Initialize dummy tensors (Must exist before we cast the model!)
dummy_input = torch.randn(1, in_channels, args.height, args.width)
dummy_hidden = torch.zeros(1, hidden_channels, h, w)

# Apply precision
if args.fp16:
    model = model.half()
    dummy_input = dummy_input.half()
    dummy_hidden = dummy_hidden.half()

model.eval()

onnx_path = os.path.splitext(args.pth_path)[0] + ("_fp16.onnx" if args.fp16 else ".onnx")

# Export
torch.onnx.export(
    model,
    (dummy_input, dummy_hidden),
    onnx_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input', 'hidden_state'],
    output_names=['output', 'next_hidden_state'],
)
print(f"Model exported to {onnx_path}")