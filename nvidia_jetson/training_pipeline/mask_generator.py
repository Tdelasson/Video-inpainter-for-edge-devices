from training_pipeline.config import *
import torch
import numpy as np
from torch.utils.data import DataLoader
from training_pipeline.dataset import *

def generate_random_square_mask(video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, C, H, W = video.shape
    mask_size = np.random.randint(MASK_SIZE_RANGE[0], MASK_SIZE_RANGE[1])
    y1 = np.random.randint(0, H - mask_size)
    x1 = np.random.randint(0, W - mask_size)

    # 1 = masked, 0 = visible
    masks = torch.zeros((B, T, 1, H, W), device=video.device)
    masks[:, :, :, y1:y1 + mask_size, x1:x1 + mask_size] = 1.0

    masked_video = video * (1.0 - masks)
    return masked_video, masks

def generate_flying_square_mask(video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, C, H, W = video.shape
    mask_size = np.random.randint(MASK_SIZE_RANGE[0], MASK_SIZE_RANGE[1])
    y1 = np.random.randint(0, H - mask_size)
    x1 = np.random.randint(0, W - mask_size)

    # 1 = masked, 0 = visible
    masks = torch.zeros((B, T, 1, H, W), device=video.device)

    for t in range(T):
        masks[:, t, :, y1:y1 + mask_size, x1:x1 + mask_size] = 1.0
        x1 += MASK_PIXEL_MOVEMENT_SPEED
        if x1 >= W - mask_size:
            x1 = np.random.randint(0, W - mask_size)
            y1 = np.random.randint(0, H - mask_size)

    masked_video = video * (1.0 - masks)
    return masked_video, masks


def generate_arbitrary_shape_mask(video, mask_dataset: IrregularMaskDataset)-> tuple[torch.Tensor, torch.Tensor]:
    """
    video: torch.Tensor (B, T, C, H, W)
    mask_dataset: IrregularMaskDataset
    """
    B, T, C, H, W = video.shape
    device = video.device
    all_masks = []

    for b in range(B):
        max_start_idx = len(mask_dataset) - T

        if max_start_idx <= 0:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, max_start_idx)

        video_masks = []
        for t in range(T):
            mask = mask_dataset[start_idx + t]
            video_masks.append(mask)

        all_masks.append(torch.stack(video_masks))

    masks = torch.stack(all_masks).to(device)

    masked_video = video * (1.0 - masks)
    return masked_video, masks

def generate_video_object_mask(video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pass