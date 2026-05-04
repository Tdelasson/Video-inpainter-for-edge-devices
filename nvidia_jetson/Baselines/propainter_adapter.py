from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from .base_adapter import BaseVideoInpainter

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_DIR = str(_REPO_ROOT / "Baselines_Repos" / "ProPainter-main")

MODEL_H = 240
MODEL_W = 432
MASK_DILATION = 4
REF_STRIDE = 10
NEIGHBOR_LENGTH = 10
SUBVIDEO_LENGTH = 80
RAFT_ITERS = 20


def _import_propainter_modules():
    sys.path.insert(0, _BASELINE_DIR)
    try:
        from model.modules.flow_comp_raft import RAFT_bi
        from model.recurrent_flow_completion import RecurrentFlowCompleteNet
        from model.propainter import InpaintGenerator

        return RAFT_bi, RecurrentFlowCompleteNet, InpaintGenerator
    finally:
        sys.path.remove(_BASELINE_DIR)


def _get_ref_index(mid_neighbor_id: int, neighbor_ids: list[int], length: int, ref_stride: int, ref_num: int) -> list[int]:
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index


class ProPainterAdapter(BaseVideoInpainter):
    def __init__(
        self,
        weights_path: str,
        raft_weights_path: str,
        flow_weights_path: str,
        device: str = "cuda",
        fp16: bool = False,
        ref_stride: int = REF_STRIDE,
        neighbor_length: int = NEIGHBOR_LENGTH,
        subvideo_length: int = SUBVIDEO_LENGTH,
        raft_iters: int = RAFT_ITERS,
    ):
        self.device = torch.device(device)
        self.use_half = fp16 and self.device.type == "cuda"
        self.model_h = MODEL_H
        self.model_w = MODEL_W
        self.ref_stride = max(1, int(ref_stride))
        self.neighbor_length = max(2, int(neighbor_length))
        self.subvideo_length = max(8, int(subvideo_length))
        self.raft_iters = max(1, int(raft_iters))

        RAFTBi, RecurrentFlowCompleteNet, InpaintGenerator = _import_propainter_modules()

        self.fix_raft = RAFTBi(model_path=raft_weights_path, device=self.device)
        self.fix_raft.eval()

        self.fix_flow_complete = RecurrentFlowCompleteNet(model_path=flow_weights_path).to(self.device)
        self.fix_flow_complete.eval()
        for p in self.fix_flow_complete.parameters():
            p.requires_grad = False

        self.model = InpaintGenerator(model_path=weights_path).to(self.device)
        self.model.eval()

        if self.use_half:
            self.fix_flow_complete.half()
            self.model.half()

    @property
    def name(self) -> str:
        return "ProPainter"

    def inpaint(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        resize_to_original: bool = True,
    ) -> list[np.ndarray]:
        orig_h, orig_w = frames[0].shape[:2]
        frames_t, flow_masks_t, masks_dilated_t, ori_frames = self._preprocess(frames, masks)
        comp_frames = self._infer(frames_t, flow_masks_t, masks_dilated_t, ori_frames)

        if not resize_to_original:
            return [f.astype(np.uint8) for f in comp_frames]
        return self._postprocess(comp_frames, orig_h, orig_w)

    def _preprocess(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[np.ndarray]]:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        resized_frames = []
        flow_masks = []
        masks_dilated = []

        for frame, mask in zip(frames, masks):
            rf = cv2.resize(frame, (self.model_w, self.model_h), interpolation=cv2.INTER_LINEAR)
            rm = cv2.resize(mask, (self.model_w, self.model_h), interpolation=cv2.INTER_NEAREST)
            rm = (rm > 0).astype(np.uint8)
            rm = cv2.dilate(rm, kernel, iterations=MASK_DILATION)

            resized_frames.append(rf)
            flow_masks.append(rm)
            masks_dilated.append(rm)

        frames_np = np.stack(resized_frames, axis=0)
        flow_masks_np = np.stack(flow_masks, axis=0)[:, :, :, None]
        masks_dilated_np = np.stack(masks_dilated, axis=0)[:, :, :, None]

        frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float().unsqueeze(0).to(self.device)
        frames_t = frames_t / 255.0 * 2.0 - 1.0

        flow_masks_t = torch.from_numpy(flow_masks_np).permute(0, 3, 1, 2).float().unsqueeze(0).to(self.device)
        masks_dilated_t = torch.from_numpy(masks_dilated_np).permute(0, 3, 1, 2).float().unsqueeze(0).to(self.device)

        return frames_t, flow_masks_t, masks_dilated_t, resized_frames

    def _infer(
        self,
        frames: torch.Tensor,
        flow_masks: torch.Tensor,
        masks_dilated: torch.Tensor,
        ori_frames: list[np.ndarray],
    ) -> list[np.ndarray]:
        video_length = frames.size(1)
        h = frames.size(-2)
        w = frames.size(-1)

        with torch.no_grad():
            if frames.size(-1) <= 640:
                short_clip_len = 12
            elif frames.size(-1) <= 720:
                short_clip_len = 8
            elif frames.size(-1) <= 1280:
                short_clip_len = 4
            else:
                short_clip_len = 2

            if video_length > short_clip_len:
                gt_flows_f_list = []
                gt_flows_b_list = []
                for f in range(0, video_length, short_clip_len):
                    end_f = min(video_length, f + short_clip_len)
                    if f == 0:
                        flows_f, flows_b = self.fix_raft(frames[:, f:end_f], iters=self.raft_iters)
                    else:
                        flows_f, flows_b = self.fix_raft(frames[:, f - 1:end_f], iters=self.raft_iters)
                    gt_flows_f_list.append(flows_f)
                    gt_flows_b_list.append(flows_b)
                    self._empty_cuda_cache()
                gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                gt_flows_bi = (gt_flows_f, gt_flows_b)
            else:
                gt_flows_bi = self.fix_raft(frames, iters=self.raft_iters)
                self._empty_cuda_cache()

            if self.use_half:
                # RAFT stays fp32; cast only after RAFT for fp16 ProPainter modules.
                frames = frames.half()
                flow_masks = flow_masks.half()
                masks_dilated = masks_dilated.half()
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())

            flow_length = gt_flows_bi[0].size(1)
            if flow_length > self.subvideo_length:
                pred_flows_f = []
                pred_flows_b = []
                pad_len = 5
                for f in range(0, flow_length, self.subvideo_length):
                    s_f = max(0, f - pad_len)
                    e_f = min(flow_length, f + self.subvideo_length + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(flow_length, f + self.subvideo_length)
                    pred_flows_bi_sub, _ = self.fix_flow_complete.forward_bidirect_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                        flow_masks[:, s_f : e_f + 1],
                    )
                    pred_flows_bi_sub = self.fix_flow_complete.combine_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                        pred_flows_bi_sub,
                        flow_masks[:, s_f : e_f + 1],
                    )
                    pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s : e_f - s_f - pad_len_e])
                    pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s : e_f - s_f - pad_len_e])
                    self._empty_cuda_cache()

                pred_flows_bi = (torch.cat(pred_flows_f, dim=1), torch.cat(pred_flows_b, dim=1))
            else:
                pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
                pred_flows_bi = self.fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
                self._empty_cuda_cache()

            masked_frames = frames * (1 - masks_dilated)
            subvideo_length_img_prop = min(100, self.subvideo_length)
            if video_length > subvideo_length_img_prop:
                updated_frames = []
                updated_masks = []
                pad_len = 10
                for f in range(0, video_length, subvideo_length_img_prop):
                    s_f = max(0, f - pad_len)
                    e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                    b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                    pred_flows_bi_sub = (
                        pred_flows_bi[0][:, s_f : e_f - 1],
                        pred_flows_bi[1][:, s_f : e_f - 1],
                    )
                    prop_imgs_sub, updated_local_masks_sub = self.model.img_propagation(
                        masked_frames[:, s_f:e_f],
                        pred_flows_bi_sub,
                        masks_dilated[:, s_f:e_f],
                        "nearest",
                    )
                    updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + (
                        prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
                    )
                    updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)

                    updated_frames.append(updated_frames_sub[:, pad_len_s : e_f - s_f - pad_len_e])
                    updated_masks.append(updated_masks_sub[:, pad_len_s : e_f - s_f - pad_len_e])
                    self._empty_cuda_cache()

                updated_frames = torch.cat(updated_frames, dim=1)
                updated_masks = torch.cat(updated_masks, dim=1)
            else:
                b, t, _, _, _ = masks_dilated.size()
                prop_imgs, updated_local_masks = self.model.img_propagation(
                    masked_frames,
                    pred_flows_bi,
                    masks_dilated,
                    "nearest",
                )
                updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
                updated_masks = updated_local_masks.view(b, t, 1, h, w)
                self._empty_cuda_cache()

            comp_frames = [None] * video_length
            neighbor_stride = max(1, self.neighbor_length // 2)
            if video_length > self.subvideo_length:
                ref_num = self.subvideo_length // self.ref_stride
            else:
                ref_num = -1

            for f in range(0, video_length, neighbor_stride):
                neighbor_ids = [
                    i
                    for i in range(
                        max(0, f - neighbor_stride),
                        min(video_length, f + neighbor_stride + 1),
                    )
                ]
                ref_ids = _get_ref_index(f, neighbor_ids, video_length, self.ref_stride, ref_num)

                selected_ids = neighbor_ids + ref_ids
                selected_imgs = updated_frames[:, selected_ids, :, :, :]
                selected_masks = masks_dilated[:, selected_ids, :, :, :]
                selected_update_masks = updated_masks[:, selected_ids, :, :, :]
                selected_pred_flows_bi = (
                    pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :],
                    pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :],
                )

                l_t = len(neighbor_ids)
                pred_img = self.model(
                    selected_imgs,
                    selected_pred_flows_bi,
                    selected_masks,
                    selected_update_masks,
                    l_t,
                )
                pred_img = pred_img.view(-1, 3, h, w)

                pred_img = (pred_img.float() + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)

                for i, idx in enumerate(neighbor_ids):
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] + ori_frames[idx] * (1 - binary_masks[i])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = (
                            comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                        ).astype(np.uint8)

                self._empty_cuda_cache()

        return [f.astype(np.uint8) for f in comp_frames]

    @staticmethod
    def _postprocess(comp_frames: list[np.ndarray], orig_h: int, orig_w: int) -> list[np.ndarray]:
        if comp_frames[0].shape[0] == orig_h and comp_frames[0].shape[1] == orig_w:
            return [f.astype(np.uint8) for f in comp_frames]

        return [cv2.resize(f.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) for f in comp_frames]

    def _empty_cuda_cache(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()