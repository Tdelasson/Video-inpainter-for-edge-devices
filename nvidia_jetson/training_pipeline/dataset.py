import os
import cv2
from torch.utils.data import Dataset
import numpy as np
from  training_pipeline.config import *
import torch
import json


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

class IrregularMaskDataset(Dataset):
    def __init__(self, root_dir):
        self.mask_path = root_dir
        self.target_res = TARGET_RES

        if not os.path.exists(self.mask_path):
            raise FileNotFoundError(f"Could not find the IrregularMasks folder: {self.mask_path}")

        self.mask_list = sorted([f for f in os.listdir(self.mask_path) if f.endswith('.png')])
        print(f"Mask dataset initialized: Found {len(self.mask_list)} masks")

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):
        mask_name = self.mask_list[idx]
        mask_path = os.path.join(self.mask_path, mask_name)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask_img is None:
            return self.__getitem__(0)

        mask_img = cv2.resize(mask_img, self.target_res)
        binary_mask = (mask_img < 127).astype(np.float32)

        return torch.from_numpy(binary_mask).unsqueeze(0)

class HumanMaskDataset(Dataset):
    def __init__(self, root_dir):
        self.mask_list = []
        self.mask_path = root_dir
        self.target_res = TARGET_RES

        if not os.path.exists(self.mask_path):
            raise FileNotFoundError(f"Could not find the human masking folder: {self.mask_path}")


        with open(os.path.join(self.mask_path, 'meta.json')) as f:
            meta = json.load(f)
            videos = meta["videos"]
            for vid_id, vid_data in videos.items():
                for obj in vid_data["objects"].values():
                    if obj["category"] == "person":
                        self.mask_list.append(vid_id)
                        break

        print(f"Human mask dataset initialized: Found {len(self.mask_list)} masks")

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):
        vid_id = self.mask_list[idx]
        vid_folder_path = os.path.join(self.mask_path, 'Annotations', vid_id)

        if not os.path.isdir(vid_folder_path):
            return self.__getitem__(0)

        mask_files = sorted([f for f in os.listdir(vid_folder_path) if f.endswith('.png')])

        if len(mask_files) == 0:
            return self.__getitem__(0)

        mask_sequence = []

        for frame_name in mask_files:
            frame_path = os.path.join(vid_folder_path, frame_name)
            mask_img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

            if mask_img is not None:
                mask_img = cv2.resize(mask_img, self.target_res)

                binary_mask = (mask_img > 0).astype(np.float32)
                mask_sequence.append(binary_mask)

        if len(mask_sequence) == 0:
            return self.__getitem__(0)

        # Stack into a tensor of shape (T, 1, H, W)
        mask_tensor = torch.from_numpy(np.stack(mask_sequence)).unsqueeze(1)

        return mask_tensor


class YouTubeVOSDatasetWithoutHumans(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.jpeg_path = os.path.join(root_dir, "JPEGImages")
        self.target_res = TARGET_RES
        self.video_list = []

        meta_path = os.path.join(root_dir, 'meta.json')
        with open(meta_path) as f:
            meta = json.load(f)
            videos = meta["videos"]

            for vid_id, vid_data in videos.items():
                has_person = False
                for obj in vid_data["objects"].values():
                    if obj["category"] == "person":
                        has_person = True
                        break

                # FIX: Only add if there is NOT a person
                if not has_person:
                    self.video_list.append(vid_id)

        print(f"Clean dataset initialized: Found {len(self.video_list)} videos without humans")

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_id = self.video_list[idx]
        specific_jpeg_path = os.path.join(self.jpeg_path, video_id)
        all_frames = sorted([f for f in os.listdir(specific_jpeg_path) if f.endswith('.jpg')])

        if len(all_frames) < SEQ_LEN:
            return self.__getitem__(np.random.randint(0, len(self.video_list)))

        rgb_frames = []
        for frame_name in all_frames:
            frame_path = os.path.join(specific_jpeg_path, frame_name)
            img = cv2.imread(frame_path)
            if img is not None:
                img = cv2.resize(img, self.target_res)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rgb_frames.append(img)

        # FIX: Return as numpy (T, H, W, C) so the train loop handles it consistently
        return np.array(rgb_frames)


class HumanInpaintingDataset(Dataset):
    def __init__(self, clean_dataset, human_mask_dataset):
        self.clean_dataset = clean_dataset
        self.human_mask_dataset = human_mask_dataset

    def __len__(self):
        # We use the clean background videos as our primary length
        return len(self.clean_dataset)

    def __getitem__(self, idx):
        # 1. Get a clean background video (T, H, W, C)
        bg_video = self.clean_dataset[idx]

        # 2. Get a random human mask sequence (T, 1, H, W)
        mask_idx = np.random.randint(0, len(self.human_mask_dataset))
        mask_video = self.human_mask_dataset[mask_idx]

        # 3. Synchronize Temporal Length

        t_len = min(bg_video.shape[0], mask_video.shape[0])

        # Crop both to same length
        bg_video = bg_video[:t_len]
        mask_video = mask_video[:t_len]

        # Return as a dictionary
        return {
            "video": bg_video,  # Ground Truth
            "mask": mask_video  # The human mask
        }