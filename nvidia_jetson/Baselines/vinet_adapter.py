from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from .base_adapter import BaseVideoInpainter

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_DIR = str(_REPO_ROOT / "Baselines_Repos" / "Deep-Video-Inpainting-master")

MODEL_SIZE = 512
SEARCH_RANGE = 4
TEMPORAL_STRIDE = 3
PRE_ROLL = 30


@dataclass
class _ViNetOptions:
    crop_size: int = MODEL_SIZE
    double_size: bool = True
    search_range: int = SEARCH_RANGE
    model: str = "vinet_final"
    batch_norm: bool = False
    no_cuda: bool = False
    no_train: bool = True
    test: bool = True
    t_stride: int = TEMPORAL_STRIDE
    loss_on_raw: bool = False
    prev_warp: bool = True


def _import_vinet_class():
    if _BASELINE_DIR not in sys.path:
        sys.path.insert(0, _BASELINE_DIR)
    try:
        from models.vinet import VINet_final

        return VINet_final
    except Exception as exc:
        raise RuntimeError(
            "Failed to import ViNET. Build Deep-Video-Inpainting CUDA extensions first: "
            "run Baselines_Repos/Deep-Video-Inpainting-master/install.sh"
        ) from exc


def _reflect_index(index: int, length: int) -> int:
    if length <= 1:
        return 0

    while index < 0 or index >= length:
        if index < 0:
            index = -index
        if index >= length:
            index = (2 * length - 2) - index
    return index


def _load_vinet_weights(model: torch.nn.Module, weights_path: str, device: torch.device) -> None:
    checkpoint = torch.load(weights_path, map_location=device)
    source_state = checkpoint.get("state_dict", checkpoint)
    target_state = model.state_dict()

    mapped_state = {}
    for key, value in source_state.items():
        normalized = key[7:] if key.startswith("module.") else key
        if normalized in target_state and target_state[normalized].shape == value.shape:
            mapped_state[normalized] = value

    missing, unexpected = model.load_state_dict(mapped_state, strict=False)
    if not mapped_state:
        raise RuntimeError(f"No compatible ViNET parameters were loaded from: {weights_path}")
    if unexpected:
        raise RuntimeError(f"Unexpected ViNET parameters found in checkpoint: {unexpected[:5]}")
    if len(missing) == len(target_state):
        raise RuntimeError("ViNET checkpoint mapping failed: all model parameters are missing.")


class ViNETAdapter(BaseVideoInpainter):
    def __init__(self, weights_path: str, device: str = "cuda", fp16: bool = False):
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("ViNET requires CUDA in this baseline implementation.")

        self.fp16 = False
        if fp16:
            print("Warning: ViNET adapter does not support fp16 reliably; using fp32.")

        self.model_h = MODEL_SIZE
        self.model_w = MODEL_SIZE
        self.temporal_stride = TEMPORAL_STRIDE
        self.pre_roll = PRE_ROLL

        self.opt = _ViNetOptions()
        ViNetClass = _import_vinet_class()
        self.model = ViNetClass(opt=self.opt).to(self.device)
        _load_vinet_weights(self.model, weights_path, self.device)
        self.model.eval()

    @property
    def name(self) -> str:
        return "ViNET"

    def inpaint(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        resize_to_original: bool = True,
    ) -> list[np.ndarray]:
        if not frames:
            return []

        orig_h, orig_w = frames[0].shape[:2]
        masked_inputs, masks_t = self._preprocess(frames, masks)
        pred_frames = self._infer(masked_inputs, masks_t)

        if not resize_to_original:
            return pred_frames
        return self._postprocess(pred_frames, orig_h, orig_w)

    def _preprocess(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        resized_frames = []
        resized_masks = []

        for frame, mask in zip(frames, masks):
            frame_resized = cv2.resize(frame, (self.model_w, self.model_h), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask, (self.model_w, self.model_h), interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask_resized > 0).astype(np.float32)

            resized_frames.append(frame_resized)
            resized_masks.append(mask_bin)

        frame_np = np.stack(resized_frames, axis=0).astype(np.float32) / 255.0
        mask_np = np.stack(resized_masks, axis=0).astype(np.float32)

        inputs = torch.from_numpy(frame_np).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)
        masks_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(self.device)

        inputs = 2.0 * inputs - 1.0
        masked_inputs = inputs * (1.0 - masks_t)

        return masked_inputs, masks_t

    def _infer(self, masked_inputs: torch.Tensor, masks_t: torch.Tensor) -> list[np.ndarray]:
        total_frames = masked_inputs.size(2)
        if total_frames == 0:
            return []

        pre_roll = min(self.pre_roll, total_frames)
        idx = torch.arange(pre_roll - 1, -1, -1, device=masked_inputs.device)

        pre_inputs = masked_inputs[:, :, :pre_roll].index_select(2, idx)
        pre_masks = masks_t[:, :, :pre_roll].index_select(2, idx)
        masked_inputs = torch.cat((pre_inputs, masked_inputs), dim=2)
        masks_t = torch.cat((pre_masks, masks_t), dim=2)

        num_frames = masked_inputs.size(2)
        outputs: list[np.ndarray] = []
        prev_mask = None
        lstm_state = None

        with torch.no_grad():
            for t in range(num_frames):
                masked_window, mask_window = self._build_temporal_window(masked_inputs, masks_t, t)

                if prev_mask is None:
                    prev_mask = mask_window[:, :, 2]
                ones = torch.ones_like(prev_mask)

                if t == 0:
                    prev_feed = torch.cat(
                        [masked_window[:, :, 2, :, :], ones, ones * prev_mask],
                        dim=1,
                    )
                else:
                    prev_feed = torch.cat(
                        [pred_t.detach().squeeze(2), ones, ones * prev_mask],
                        dim=1,
                    )

                pred_t, _, _, _, _ = self.model(masked_window, mask_window, lstm_state, prev_feed, t)

                # Keep behavior aligned with official demo, which resets recurrent state each step.
                lstm_state = None
                prev_mask = mask_window[:, :, 2] * 0.5

                if t >= pre_roll:
                    frame = self._to_uint8_image(pred_t)
                    outputs.append(frame)

        return outputs

    def _build_temporal_window(
        self,
        masked_inputs: torch.Tensor,
        masks_t: torch.Tensor,
        t: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = masked_inputs.size(2)
        offsets = (-2 * self.temporal_stride, -self.temporal_stride, 0, self.temporal_stride, 2 * self.temporal_stride)

        selected = [_reflect_index(t + off, length) for off in offsets]
        idx = torch.tensor(selected, device=masked_inputs.device, dtype=torch.long)

        masked_window = masked_inputs.index_select(2, idx)
        mask_window = masks_t.index_select(2, idx)
        return masked_window, mask_window

    @staticmethod
    def _to_uint8_image(pred: torch.Tensor) -> np.ndarray:
        img = (pred[0, :, 0].float().cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5
        img = np.clip(img, 0.0, 1.0)
        return (img * 255.0).astype(np.uint8)

    @staticmethod
    def _postprocess(comp_frames: list[np.ndarray], orig_h: int, orig_w: int) -> list[np.ndarray]:
        if not comp_frames:
            return []
        if comp_frames[0].shape[0] == orig_h and comp_frames[0].shape[1] == orig_w:
            return [f.astype(np.uint8) for f in comp_frames]

        return [cv2.resize(f.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) for f in comp_frames]