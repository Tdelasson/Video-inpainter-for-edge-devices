import cv2
import numpy as np

def get_optical_flow(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Handles all the image processing and sliding window logic.
    Takes a sorted list of frames, where the current frame is the last element"""

    if len(frames) < 2:
        print("At least 2 frames are required to compute optical flow.")
        return []

    grayscale_frames = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grayscale_frames.append(gray)


    flow_list: list[np.ndarray] = []
    for i in range(1, len(grayscale_frames)):
        current_flow_frame = np.abs(grayscale_frames[i] - grayscale_frames[i - 1])
        flow_list.append(current_flow_frame)

    return flow_list