from training_pipeline.config import *
import torch
import numpy as np
from torch.utils.data import DataLoader
from training_pipeline.dataset import *
import torchvision.transforms.functional as TF
import torch.nn.functional as F

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


def generate_arbitrary_shape_mask(video, mask_dataset: IrregularMaskDataset, size_range):
    B, T, C, H, W = video.shape
    device = video.device
    all_masks = []
    min_size, max_size = size_range

    for b in range(B):
        max_start_idx = len(mask_dataset) - T
        start_idx = 0 if max_start_idx <= 0 else np.random.randint(0, max_start_idx)

        # Pick a random size constraint for this specific batched sequence
        target_size = np.random.randint(min_size, max_size + 1)

        # Pick random placement coordinates
        y1 = np.random.randint(0, H - target_size + 1) if H > target_size else 0
        x1 = np.random.randint(0, W - target_size + 1) if W > target_size else 0

        video_masks = []
        for t in range(T):
            mask = mask_dataset[start_idx + t].unsqueeze(0)  # (1, 1, H, W)

            # Scale down the mask to the curriculum size
            scaled_mask = F.interpolate(mask, size=(target_size, target_size), mode='nearest')

            # Place the scaled mask onto an empty canvas of the original video size
            canvas = torch.zeros((1, 1, H, W), device=mask.device)
            canvas[:, :, y1:y1 + target_size, x1:x1 + target_size] = scaled_mask

            video_masks.append(canvas.squeeze(0))  # Back to (1, H, W)

        all_masks.append(torch.stack(video_masks))

    return torch.stack(all_masks).to(device)

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
