import cv2
import torch
import numpy as np
from model_architecture.video_inpainter import VideoInpainter
from model_architecture.optical_flow import get_optical_flow
import time
from collections import deque

WINDOW_SIZE = 2
IN_CHANNELS = WINDOW_SIZE * 3 + (WINDOW_SIZE - 1)
INPUT_RESOLUTION = (520, 520)

def preprocess_frame(frame):
    """Convert a raw BGR frame to a normalized RGB tensor."""
    frame = cv2.resize(frame, INPUT_RESOLUTION)
    t = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).float() / 255.0
    return t.permute(2, 0, 1)  # [C, H, W]

def prepare_window(frame_buffer, raw_buffer, window_size):
    """Convert a window of frames into a model input tensor."""
    window_frames = list(frame_buffer)

    window_block = torch.cat(window_frames, dim=0)  # [window_size * 3, H, W]

    if window_size > 1:
        # Resize raw frames to match INPUT_RESOLUTION before computing flow
        resized_raw = [cv2.resize(f, INPUT_RESOLUTION) for f in raw_buffer]
        flow = get_optical_flow(resized_raw)
        flow = [torch.from_numpy(f).unsqueeze(0) for f in flow]  # (H, W) -> (1, H, W)
        flow_block = torch.cat(flow, dim=0)
        block = torch.cat([window_block, flow_block], dim=0)
    else:
        block = window_block

    return block.unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]

def run_inference(model, input_tensor, device):
    """Run a single forward pass and return output + latency."""
    input_tensor = input_tensor.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        output_frames, hidden = model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    latency_ms = (end - start) * 1000
    return output_frames, latency_ms

def postprocess_frame(output_tensor):
    """Convert model output tensor to a displayable BGR frame."""
    frame = output_tensor[0, 0].cpu().permute(1, 2, 0).numpy()
    frame = (frame * 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    model = VideoInpainter(in_channels=IN_CHANNELS, base_channels=32, num_layers=3)
    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(0) # Insert video stream here
    if not cap.isOpened():
        raise RuntimeError("Could not open video stream.")

    # Rolling buffers to hold the last window_size frames
    frame_buffer = deque(maxlen=WINDOW_SIZE)   # preprocessed tensors
    raw_buffer = deque(maxlen=WINDOW_SIZE)     # raw BGR frames for optical flow

    print("Warming up...")
    warmed_up = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended.")
            break

        raw_buffer.append(frame)
        frame_buffer.append(preprocess_frame(frame))

        # Wait until we have a full window
        if len(frame_buffer) < WINDOW_SIZE:
            continue

        input_tensor = prepare_window(frame_buffer, raw_buffer, WINDOW_SIZE)

        # Warmup pass on first window
        if not warmed_up:
            with torch.no_grad():
                _ = model(input_tensor.to(device))
            if device.type == "cuda":
                torch.cuda.synchronize()
            warmed_up = True
            print("Warmup done. Starting inference...")
            continue

        output, latency_ms = run_inference(model, input_tensor, device)
        print(f"Frame latency: {latency_ms:.2f}ms  ({1000/latency_ms:.1f} FPS)")

        out_bgr = postprocess_frame(output)
        orig_resized = cv2.resize(frame, INPUT_RESOLUTION)
        combined = np.hstack((orig_resized, out_bgr))

        cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, f"Output | {latency_ms:.1f}ms", (INPUT_RESOLUTION[0] + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Video Inpainter", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()