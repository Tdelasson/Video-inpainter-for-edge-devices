import os
import cv2
from torch.utils.data import Dataset
import json
import numpy as np


class YouTubeVOSDataset(Dataset):
    def __init__(self, root_dir, seq_len: int = 15):
        self.jpeg_path = os.path.join(root_dir, "JPEGImages")
        self.seq_len = seq_len

        with open(os.path.join(root_dir, "meta.json"), "r") as f:
            self.meta = json.load(f)['videos']

        self.video_list = list(self.meta.keys())

        self.start_index = "00000"
        self.video_index = 0
        self.specific_jpeg_path = os.path.join(self.jpeg_path, self.video_list[self.video_index])



    def __len__(self):
        return len(self.video_list)

    def load_data(self):

        self.check_valid_start_index()

        rgb_frames = []
        selected_frames = []

        for frame in range(0, self.seq_len):
            selected_frames.append(self.start_index)
            self.start_index = str(int(self.start_index) + 5).zfill(5)

        print(f"selected frames: {selected_frames}")

        for frame in selected_frames:
            if os.path.exists(os.path.join(self.specific_jpeg_path, f"{frame}.jpg")):
                jpeg_img = cv2.imread(os.path.join(self.specific_jpeg_path, f"{frame}.jpg"))
                rgb_img = cv2.cvtColor(jpeg_img, cv2.COLOR_BGR2RGB)

                rgb_frames.append(rgb_img)

        if len(rgb_frames) < self.seq_len:
            print(f"Warning: Only {len(rgb_frames)} frames were loaded for video {self.video_list[self.video_index]}. Discarding this video.")
            rgb_frames = []

        return np.array(rgb_frames)

    def get_smallest_existing_image_index(self, path):
        start_index = "00000"
        while not os.path.exists(os.path.join(path, f"{start_index}.jpg")):
            start_index = str(int(start_index) + 5).zfill(5)

        return start_index

    def update_video_index(self, video_index):
        self.video_index = video_index
        self.specific_jpeg_path = os.path.join(self.jpeg_path, self.video_list[video_index])

    def check_valid_start_index(self):
        if not os.path.exists(os.path.join(self.specific_jpeg_path, f"{self.start_index}.jpg")):
            smallest_existing_image_index = self.get_smallest_existing_image_index(self.specific_jpeg_path)

            if int(self.start_index) > int(smallest_existing_image_index):
                self.update_video_index(self.video_index + 1)
                self.start_index = "00000"
            else:
                self.start_index = smallest_existing_image_index










