from training_pipeline.config import *
import torch
import numpy as np

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

def generate_arbitrary_shape_mask(video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pass

def generate_video_object_mask(video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pass