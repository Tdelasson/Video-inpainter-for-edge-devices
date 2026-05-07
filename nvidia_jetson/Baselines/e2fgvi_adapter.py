from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d

from .base_adapter import BaseVideoInpainter

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_DIR = str(_REPO_ROOT / "Baselines_Repos" / "E2FGVI-master")

MODEL_H = 240
MODEL_W = 432
REF_LENGTH = 10
NUM_REF = -1
NEIGHBOR_STRIDE = 5
MASK_DILATION = 4
MOD_SIZE_H = 60
MOD_SIZE_W = 108


def _import_inpaint_generator():
    _ensure_mmcv_compat()

    if _BASELINE_DIR not in sys.path:
        sys.path.insert(0, _BASELINE_DIR)
    try:
        from model.e2fgvi import InpaintGenerator

        return InpaintGenerator
    except Exception as exc:
        raise RuntimeError(
            "Failed to import E2FGVI model from Baselines_Repos/E2FGVI-master. "
            "Please verify dependencies for E2FGVI are installed."
        ) from exc


def _ensure_mmcv_compat() -> None:
    """Install a lightweight mmcv shim when mmcv is unavailable.

    E2FGVI uses only a small subset of mmcv APIs for inference. This shim keeps
    those imports working in pip-only environments where mmcv-full wheels are
    not available (e.g. newer Python versions).
    """
    try:
        import mmcv  # type: ignore

        _ = mmcv
        return
    except ModuleNotFoundError:
        pass

    mmcv_mod = types.ModuleType("mmcv")
    cnn_mod = types.ModuleType("mmcv.cnn")
    runner_mod = types.ModuleType("mmcv.runner")
    ops_mod = types.ModuleType("mmcv.ops")

    class ConvModule(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            norm_cfg=None,
            act_cfg: dict | None = None,
        ):
            super().__init__()
            del norm_cfg
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True)
            self.activate: nn.Module | None = None
            if act_cfg is not None:
                act_type = str(act_cfg.get("type", "ReLU"))
                if act_type.lower() == "relu":
                    self.activate = nn.ReLU(inplace=True)
                elif act_type.lower() == "leakyrelu":
                    neg = float(act_cfg.get("negative_slope", 0.1))
                    self.activate = nn.LeakyReLU(negative_slope=neg, inplace=True)
                else:
                    raise ValueError(f"Unsupported activation type in mmcv shim: {act_type}")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv(x)
            if self.activate is not None:
                x = self.activate(x)
            return x

    def constant_init(module: nn.Module, val: float = 0.0, bias: float = 0.0) -> None:
        if hasattr(module, "weight") and getattr(module, "weight") is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, "bias") and getattr(module, "bias") is not None:
            nn.init.constant_(module.bias, bias)

    def load_checkpoint(model: nn.Module, filename: str, strict: bool = True):
        if filename.startswith("http://") or filename.startswith("https://"):
            state = torch.hub.load_state_dict_from_url(filename, map_location="cpu", progress=False)
        else:
            state = torch.load(filename, map_location="cpu")

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        if isinstance(state, dict):
            remapped = {}
            target = model.state_dict()
            for k, v in state.items():
                nk = k[7:] if k.startswith("module.") else k
                if nk in target and target[nk].shape == v.shape:
                    remapped[nk] = v
            model.load_state_dict(remapped if remapped else state, strict=strict)
        else:
            model.load_state_dict(state, strict=strict)
        return state

    class ModulatedDeformConv2d(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            deform_groups=1,
            bias=True,
        ):
            super().__init__()
            k = _pair(kernel_size)
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = k
            self.stride = _pair(stride)
            self.padding = _pair(padding)
            self.dilation = _pair(dilation)
            self.groups = groups
            self.deform_groups = deform_groups
            self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, k[0], k[1]))
            self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
            nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
            if self.bias is not None:
                fan_in = (in_channels // groups) * k[0] * k[1]
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

        def forward(self, x: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return deform_conv2d(
                input=x,
                offset=offset,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                mask=mask,
            )

    def modulated_deform_conv2d(
        x: torch.Tensor,
        offset: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        stride,
        padding,
        dilation,
        groups: int,
        deform_groups: int,
    ) -> torch.Tensor:
        del groups, deform_groups
        return deform_conv2d(
            input=x,
            offset=offset,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            mask=mask,
        )

    cnn_mod.ConvModule = ConvModule
    cnn_mod.constant_init = constant_init
    runner_mod.load_checkpoint = load_checkpoint
    ops_mod.ModulatedDeformConv2d = ModulatedDeformConv2d
    ops_mod.modulated_deform_conv2d = modulated_deform_conv2d

    mmcv_mod.cnn = cnn_mod
    mmcv_mod.runner = runner_mod
    mmcv_mod.ops = ops_mod

    sys.modules["mmcv"] = mmcv_mod
    sys.modules["mmcv.cnn"] = cnn_mod
    sys.modules["mmcv.runner"] = runner_mod
    sys.modules["mmcv.ops"] = ops_mod


def _get_ref_index(
    frame_idx: int,
    neighbor_ids: list[int],
    length: int,
    ref_length: int,
    num_ref: int,
) -> list[int]:
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, frame_idx - ref_length * (num_ref // 2))
        end_idx = min(length, frame_idx + ref_length * (num_ref // 2))
        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) >= num_ref:
                    break
                ref_index.append(i)
    return ref_index


class E2FGVIAdapter(BaseVideoInpainter):
    def __init__(self, weights_path: str, device: str = "cuda", fp16: bool = False):
        self.device = torch.device(device)
        self.fp16 = fp16 and self.device.type == "cuda"
        self.model_h = MODEL_H
        self.model_w = MODEL_W
        self.ref_length = REF_LENGTH
        self.num_ref = NUM_REF
        self.neighbor_stride = NEIGHBOR_STRIDE

        InpaintGenerator = _import_inpaint_generator()
        self.model = InpaintGenerator().to(self.device)

        checkpoint = torch.load(weights_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"Unsupported E2FGVI checkpoint format: {weights_path}")

        target = self.model.state_dict()
        remapped = {}
        for key, value in checkpoint.items():
            normalized = key[7:] if key.startswith("module.") else key
            if normalized in target and target[normalized].shape == value.shape:
                remapped[normalized] = value

        missing, unexpected = self.model.load_state_dict(remapped, strict=False)
        if not remapped:
            raise RuntimeError(f"No compatible E2FGVI parameters were loaded from: {weights_path}")

        loaded_ratio = len(remapped) / max(1, len(target))
        if loaded_ratio < 0.90:
            raise RuntimeError(
                f"E2FGVI checkpoint compatibility is too low ({loaded_ratio:.1%}). "
                f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
            )
        self.model.eval()

        if self.fp16:
            self.model.half()

    @property
    def name(self) -> str:
        return "E2FGVI"

    def inpaint(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        resize_to_original: bool = True,
    ) -> list[np.ndarray]:
        if not frames:
            return []

        orig_h, orig_w = frames[0].shape[:2]
        imgs, masks_t, binary_masks, resized_frames = self._preprocess(frames, masks)
        comp_frames = self._infer(imgs, masks_t, binary_masks, resized_frames)

        if not resize_to_original:
            return [f.astype(np.uint8) for f in comp_frames]
        return self._postprocess(comp_frames, orig_h, orig_w)

    def _preprocess(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor, list[np.ndarray], list[np.ndarray]]:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        resized_frames = []
        frame_tensors = []
        mask_tensors = []
        binary_masks = []

        for frame, mask in zip(frames, masks):
            frame_resized = cv2.resize(frame, (self.model_w, self.model_h), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask, (self.model_w, self.model_h), interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask_resized > 0).astype(np.uint8)
            mask_bin = cv2.dilate(mask_bin, kernel, iterations=MASK_DILATION)

            resized_frames.append(frame_resized)
            binary_masks.append(np.expand_dims(mask_bin, 2))

            frame_t = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
            mask_t = torch.from_numpy(mask_bin).unsqueeze(0).float()

            frame_tensors.append(frame_t)
            mask_tensors.append(mask_t)

        imgs = torch.stack(frame_tensors, dim=0).unsqueeze(0).to(self.device)
        masks_t = torch.stack(mask_tensors, dim=0).unsqueeze(0).to(self.device)

        if self.fp16:
            imgs = imgs.half()
            masks_t = masks_t.half()

        return imgs, masks_t, binary_masks, resized_frames

    def _infer(
        self,
        imgs: torch.Tensor,
        masks_t: torch.Tensor,
        binary_masks: list[np.ndarray],
        resized_frames: list[np.ndarray],
    ) -> list[np.ndarray]:
        video_length = imgs.shape[1]
        _, _, _, h, w = imgs.shape
        comp_frames: list[np.ndarray | None] = [None] * video_length

        with torch.inference_mode():
            for f in range(0, video_length, self.neighbor_stride):
                neighbor_ids = [
                    i
                    for i in range(
                        max(0, f - self.neighbor_stride),
                        min(video_length, f + self.neighbor_stride + 1),
                    )
                ]
                ref_ids = _get_ref_index(f, neighbor_ids, video_length, self.ref_length, self.num_ref)
                selected_ids = neighbor_ids + ref_ids

                selected_imgs = imgs[:, selected_ids, :, :, :]
                selected_masks = masks_t[:, selected_ids, :, :, :]
                masked_imgs = selected_imgs * (1 - selected_masks)

                h_pad = (MOD_SIZE_H - h % MOD_SIZE_H) % MOD_SIZE_H
                w_pad = (MOD_SIZE_W - w % MOD_SIZE_W) % MOD_SIZE_W

                masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [3])], dim=3)[:, :, :, : h + h_pad, :]
                masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [4])], dim=4)[:, :, :, :, : w + w_pad]

                if self.fp16 and self.device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        pred_imgs, _ = self.model(masked_imgs, len(neighbor_ids))
                else:
                    pred_imgs, _ = self.model(masked_imgs, len(neighbor_ids))

                pred_imgs = pred_imgs[:, :, :h, :w]
                pred_imgs = (pred_imgs.float() + 1.0) / 2.0
                pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255.0

                for i, idx in enumerate(neighbor_ids):
                    pred_u8 = np.clip(pred_imgs[i], 0, 255).astype(np.uint8)
                    img = pred_u8 * binary_masks[idx] + resized_frames[idx] * (1 - binary_masks[idx])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = (
                            comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                        ).astype(np.uint8)

        result = []
        for idx, frame in enumerate(comp_frames):
            if frame is None:
                result.append(resized_frames[idx].astype(np.uint8))
            else:
                result.append(frame.astype(np.uint8))
        return result

    @staticmethod
    def _postprocess(comp_frames: list[np.ndarray], orig_h: int, orig_w: int) -> list[np.ndarray]:
        if not comp_frames:
            return []
        if comp_frames[0].shape[0] == orig_h and comp_frames[0].shape[1] == orig_w:
            return [f.astype(np.uint8) for f in comp_frames]

        return [cv2.resize(f.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) for f in comp_frames]