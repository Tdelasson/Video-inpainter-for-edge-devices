import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from .base_adapter import BaseVideoInpainter

# Resolve paths relative to repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_DIR = str(_REPO_ROOT / "Baselines_Repos" / "towards-online-video-inpainting-main")

# Model constants (matching evaluate_FuseFormer_OM.py defaults)
MODEL_H = 240
MODEL_W = 432
NUM_NEIGHBORS = 3
REF_STEP = 10
NUM_REFS = 3


def _import_inpaint_generator():
    """Import InpaintGenerator from the towards-online-video-inpainting codebase."""
    sys.path.insert(0, _BASELINE_DIR)
    try:
        from model.FuseFormer_OM import InpaintGenerator
        return InpaintGenerator
    finally:
        sys.path.remove(_BASELINE_DIR)


def _get_ref_index(f: int, neighbor_ids: list[int], ref_step: int, num_refs: int) -> list[int]:
    """Get reference frame indices from the past, excluding neighbors.

    Walks backwards from f-1, collecting frames divisible by ref_step
    that are not in neighbor_ids, up to num_refs total.
    Returns indices in chronological order.
    """
    ref_index = []
    i = f - 1
    while i >= 0 and len(ref_index) < num_refs:
        if i not in neighbor_ids and i % ref_step == 0:
            ref_index.append(i)
        i -= 1
    return ref_index[::-1]


class FuseFormerOMAdapter(BaseVideoInpainter):
    """Adapter for FuseFormer Online-Memory video inpainting.

    Processes frames sequentially, accumulating transformer intermediate
    representations ("inpainting memory") from past frames. Uses the same
    pretrained weights as offline FuseFormer (zero new parameters).

    Args:
        weights_path: Path to the FuseFormer checkpoint (.pth file).
        device: Torch device string ("cuda", "cuda:0", "cpu").
        fp16: If True, run inference in float16 for reduced memory usage.
    """

    def __init__(self, weights_path: str, device: str = "cuda", fp16: bool = False):
        self.device = torch.device(device)
        self.fp16 = fp16

        InpaintGenerator = _import_inpaint_generator()
        self.model = InpaintGenerator().to(self.device)
        data = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(data, strict=False)
        self.model.eval()

        if fp16:
            self.model.half()

    @property
    def name(self) -> str:
        return "FuseFormer_OM"

    def inpaint(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        resize_to_original: bool = True,
    ) -> list[np.ndarray]:
        orig_h, orig_w = frames[0].shape[:2]
        imgs, masks_t, binary_masks, resized_frames = self._preprocess(frames, masks)
        comp_frames = self._online_infer(imgs, masks_t, binary_masks, resized_frames)
        if not resize_to_original:
            return [frame.astype(np.uint8) for frame in comp_frames]
        return self._postprocess(comp_frames, orig_h, orig_w)

    def _preprocess(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor, list[np.ndarray], list[np.ndarray]]:
        """Resize, normalize, dilate masks, build tensors."""
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        resized_frames = []
        frame_tensors = []
        mask_tensors = []
        binary_masks = []

        for f, m in zip(frames, masks):
            # Resize frame
            rf = cv2.resize(f, (MODEL_W, MODEL_H))
            resized_frames.append(rf)
            # Frame tensor: (3, H, W) float in [-1, 1]
            ft = torch.from_numpy(rf).permute(2, 0, 1).float() / 255.0 * 2 - 1
            frame_tensors.append(ft)

            # Resize mask
            rm = cv2.resize(m, (MODEL_W, MODEL_H), interpolation=cv2.INTER_NEAREST)
            # Binarize and dilate
            rm = (rm > 0).astype(np.uint8)
            rm = cv2.dilate(rm, kernel, iterations=4)
            # Binary mask for compositing: (H, W, 1)
            binary_masks.append(np.expand_dims(rm, 2))
            # Mask tensor: (1, H, W) float
            mt = torch.from_numpy(rm).unsqueeze(0).float()
            mask_tensors.append(mt)

        # Stack: (1, T, C, H, W)
        imgs = torch.stack(frame_tensors).unsqueeze(0).to(self.device)
        masks_t = torch.stack(mask_tensors).unsqueeze(0).to(self.device)

        if self.fp16:
            imgs = imgs.half()

        return imgs, masks_t, binary_masks, resized_frames

    def _online_infer(
        self,
        imgs: torch.Tensor,
        masks_t: torch.Tensor,
        binary_masks: list[np.ndarray],
        resized_frames: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Run sequential online inference with memory accumulation."""
        video_length = imgs.shape[1]
        comp_frames = [None] * video_length
        memory_bank_cpu: list[torch.Tensor] = []

        with torch.inference_mode():
            for f in range(video_length):
                neighbor_ids = [i for i in range(max(0, f - NUM_NEIGHBORS), f)]
                ref_ids = _get_ref_index(f, neighbor_ids, REF_STEP, NUM_REFS)
                selected_ids = neighbor_ids + ref_ids

                current_frame = imgs[:, [f], :, :, :]
                current_mask = masks_t[:, [f], :, :, :]
                masked_img = current_frame * (1 - current_mask)

                if selected_ids:
                    selected_memory = torch.stack(
                        [memory_bank_cpu[i] for i in selected_ids],
                        dim=0,
                    ).to(self.device, non_blocking=True)
                    if self.fp16:
                        selected_memory = selected_memory.half()
                else:
                    selected_memory = torch.empty(0, device=self.device, dtype=masked_img.dtype)

                if self.fp16:
                    with torch.cuda.amp.autocast():
                        pred_img, attn = self.model(masked_img, selected_memory)
                else:
                    pred_img, attn = self.model(masked_img, selected_memory)

                stored_attn = attn.detach().to("cpu")
                if self.fp16:
                    stored_attn = stored_attn.half()
                memory_bank_cpu.append(stored_attn)

                pred_img = (pred_img.float() + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                pred_uint8 = np.clip(pred_img[0], 0, 255).astype(np.uint8)

                img = pred_uint8 * binary_masks[f] + resized_frames[f] * (1 - binary_masks[f])
                comp_frames[f] = img.astype(np.uint8)
                del selected_memory, attn

        return comp_frames

    @staticmethod
    def _postprocess(
        comp_frames: list[np.ndarray],
        orig_h: int,
        orig_w: int,
    ) -> list[np.ndarray]:
        """Resize composited frames back to original resolution."""
        if orig_h == MODEL_H and orig_w == MODEL_W:
            return [f.astype(np.uint8) for f in comp_frames]

        result = []
        for frame in comp_frames:
            resized = cv2.resize(frame.astype(np.uint8), (orig_w, orig_h))
            result.append(resized)
        return result
