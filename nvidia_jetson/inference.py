import cv2
import torch
import os
import numpy as np
from model_architecture.video_inpainter import VideoInpainter
from torchinfo import summary
from model_architecture.optical_flow import get_optical_flow

def prepare_input_blocks(input_frames, window_size=3):
    """Handles all the image processing and sliding window logic."""
    optical_flow_tensors: list[torch.tensor] = []
    tensors = []

    optical_flow = get_optical_flow(input_frames)


    for flow in optical_flow:
        optical_flow_tensors.append(torch.from_numpy(flow))
        print(flow)

    stacked_optical_flow_tensor = torch.stack(optical_flow_tensors)
    print(stacked_optical_flow_tensor)

    for f in input_frames:
        print(f.shape)
        t = torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).float() / 255.0
        tensors.append(t.permute(2, 0, 1)) # We need shape [C,H,W], current shape is [H, W, C]

    blocks = []

    for i in range(len(tensors) - window_size + 1):
        # Slice the list to get 'window_size' frames and concat on Channel dim
        window_frames = tensors[i: i + window_size] # slices the tensors list to get a sublist
        # starting at index i and ending at index i + window_size, giving exactly window_size elements.
        flow_frames = input_frames[i: i + window_size]

        flow = get_optical_flow(flow_frames)
        flow = [torch.from_numpy(f).unsqueeze(0) for f in flow]  # (H, W) -> (1, H, W)
        flow_block = torch.cat(flow, dim=0)

        window_block = torch.cat(window_frames, dim=0)  # [window_size * 3, H, W]

        block = torch.cat([window_block, flow_block], dim=0)  # [window_size * 3 + (window_size - 1), H, W]
        blocks.append(block)

    if not blocks:
        raise ValueError(f"Not enough frames ({len(tensors)}) for window_size {window_size}")

    # Final shape: [1, Seq, Channels, H, W]
    # where Channels = window_size * 3
    return torch.stack(blocks).unsqueeze(0)

def run_model(video_inpainter, model_input_tensor):
    """inference: takes a tensor, returns a tensor."""
    video_inpainter.eval()
    with torch.no_grad():
        output_frames, hidden = video_inpainter(model_input_tensor)
    return output_frames


def save_and_show_results(output_tensor, original_frames, window_size):
    target_idx_in_window = window_size // 2

    for t in range(output_tensor.shape[1]):
        # 1. Process Model Output
        out_frame = output_tensor[0, t].cpu().permute(1, 2, 0).numpy()
        out_frame = (out_frame * 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)

        # 2. Get Corresponding Original Frame
        orig_bgr = original_frames[t + target_idx_in_window]
        orig_bgr = cv2.resize(orig_bgr, (out_bgr.shape[1], out_bgr.shape[0]))

        # 3. Stitch them together
        combined_view = np.hstack((orig_bgr, out_bgr))

        # 4. SCALE DOWN FOR DISPLAY
        display_scale = 0.5
        width = int(combined_view.shape[1] * display_scale)
        height = int(combined_view.shape[0] * display_scale)
        display_img = cv2.resize(combined_view, (width, height))

        # Add labels to the resized version so they stay legible
        cv2.putText(display_img, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_img, "Model Output", (int(width / 2) + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Side-by-Side Comparison", display_img)
        print(f"Showing Comparison {t}. Press any key...")

        # This allows the window to be moved/resized by the OS
        cv2.waitKey(0)

    cv2.destroyAllWindows()

WINDOW_SIZE = 5
IN_CHANNELS = WINDOW_SIZE * 3 + (WINDOW_SIZE - 1)

# --- Execution ---
root_dir = r"C:\Users\tobpu\Documents\aau\Semester 6\training_data\train"
jpeg_path = os.path.join(root_dir, "JPEGImages", "00a23ccf53")

# Load 5 frames
frame_indices = ["00000.jpg", "00005.jpg", "00010.jpg", "00015.jpg", "00020.jpg", "00025.jpg", "00030.jpg"]
frames = [cv2.imread(os.path.join(jpeg_path, name)) for name in frame_indices]

# 1. Prepare data
input_tensor = prepare_input_blocks(frames, window_size=WINDOW_SIZE)
print(f"Input Shape: {input_tensor.shape}")

# 2. Setup and Run Model
model = VideoInpainter(in_channels=IN_CHANNELS, base_channels=32, num_layers=3)
output = run_model(model, input_tensor)

# 3. Handle Results
save_and_show_results(output, frames, WINDOW_SIZE)

# 4. Summary
summary(model, input_data=input_tensor)