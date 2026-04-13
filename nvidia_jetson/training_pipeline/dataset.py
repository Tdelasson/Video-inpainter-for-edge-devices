import os
import cv2
from torch.utils.data import Dataset
import numpy as np
from  training_pipeline.config import *


class YouTubeVOSDataset(Dataset):
    def __init__(self, root_dir):
        self.jpeg_path = os.path.join(root_dir, "JPEGImages")
        self.target_res = TARGET_RES

        if not os.path.exists(self.jpeg_path):
            raise FileNotFoundError(f"Could not find the JPEG folder: {self.jpeg_path}")

        self.video_list = [f for f in os.listdir(self.jpeg_path)
                           if os.path.isdir(os.path.join(self.jpeg_path, f))]

        print(f"Dataset initialized: Found {len(self.video_list)} videos")

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_id = self.video_list[idx]
        specific_jpeg_path = os.path.join(self.jpeg_path, video_id)
        all_frames = sorted([f for f in os.listdir(specific_jpeg_path) if f.endswith('.jpg')])

        # Need at least SEQ_LEN frames to form one window
        if len(all_frames) < SEQ_LEN:
            return self.__getitem__(np.random.randint(0, len(self.video_list)))

        # Load ALL frames
        rgb_frames = []
        for frame_name in all_frames:
            frame_path = os.path.join(specific_jpeg_path, frame_name)
            jpeg_img = cv2.imread(frame_path)
            if jpeg_img is not None:
                jpeg_img = cv2.resize(jpeg_img, self.target_res)
                rgb_img = cv2.cvtColor(jpeg_img, cv2.COLOR_BGR2RGB)
                rgb_frames.append(rgb_img)

        if len(rgb_frames) < SEQ_LEN:
            return self.__getitem__(np.random.randint(0, len(self.video_list)))

        return np.array(rgb_frames)  # (T, H, W, C) where T is dynamic