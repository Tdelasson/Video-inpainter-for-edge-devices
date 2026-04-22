from training_pipeline.config import *
import torch
import numpy as np
from torch.utils.data import DataLoader
from training_pipeline.dataset import *
import torchvision.transforms.functional as TF

def generate_random_square_mask(video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, C, H, W = video.shape
    mask_size = np.random.randint(MASK_SIZE_RANGE[0], MASK_SIZE_RANGE[1])
    y1 = np.random.randint(0, H - mask_size)
    x1 = np.random.randint(0, W - mask_size)

    # 1 = masked, 0 = visible
    masks = torch.zeros((B, T, 1, H, W), device=video.device)
    masks[:, :, :, y1:y1 + mask_size, x1:x1 + mask_size] = 1.0

    return masks

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

    return masks


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

    return masks

def generate_video_object_mask(video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pass


def random_dilate_and_blur_mask(mask, max_radius=3, blur_kernel_size=5, blur_sigma=2.0):
    original_shape = mask.shape

    # Reshape to (B, C, H, W) for max_pool2d and blur
    if len(original_shape) == 5:
        B, T, C, H, W = original_shape
        m = mask.view(B * T, C, H, W)
    elif len(original_shape) == 4:
        B, T, H, W = original_shape
        m = mask.view(B * T, 1, H, W)
    else:
        m = mask

    # Dilation
    radius = torch.randint(1, max_radius + 1, (1,)).item()
    kernel_size = 2 * radius + 1
    dilated = torch.nn.functional.max_pool2d(
        m, kernel_size=kernel_size, stride=1, padding=radius
    )

    # Blur
    blurred = TF.gaussian_blur(dilated, kernel_size=[blur_kernel_size, blur_kernel_size], sigma=[blur_sigma, blur_sigma])

    # Clamp to ensure 0.0 to 1.0 range
    blurred = torch.clamp(blurred, 0.0, 1.0)

    return blurred.view(original_shape)
