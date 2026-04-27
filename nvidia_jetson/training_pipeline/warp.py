import torch

def warp(x, flow):
    """
    Warps frame x using the flow field.
    x: [B, C, H, W], flow: [B, 2, H, W]
    """
    B, C, H, W = x.size()
    # Create a grid of coordinates
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).float().to(x.device)  # [H, W, 2]
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

    # Add flow to the grid and normalize to [-1, 1] for grid_sample
    v_grid = grid + flow.permute(0, 2, 3, 1)
    v_grid[:, :, :, 0] = (v_grid[:, :, :, 0] / (W - 1) * 2) - 1
    v_grid[:, :, :, 1] = (v_grid[:, :, :, 1] / (H - 1) * 2) - 1

    return torch.nn.functional.grid_sample(x, v_grid, align_corners=True)