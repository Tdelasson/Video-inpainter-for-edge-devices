import cv2
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional as TF
from training_pipeline.config import NUM_LAYERS, BASE_CHANNELS


class ViperAdapter:
    def __init__(self, model_path, device="cuda", seq_len=5, fp16=True):
        self.device = device
        self.seq_len = seq_len
        self.fp16 = fp16

        # Load from config
        self.num_layers = NUM_LAYERS
        self.base_channels = BASE_CHANNELS
        self.downsample_factor = 2 ** self.num_layers
        # Dynamically calculate the channels at the bottleneck
        self.hidden_channels = self.base_channels * (2 ** (self.num_layers - 1))

        if model_path.endswith('.engine'):
            from torch2trt import TRTModule
            import tensorrt as trt

            with open(model_path, 'rb') as f:
                engine_data = f.read()

            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            engine = runtime.deserialize_cuda_engine(engine_data)

            self.model = TRTModule(
                engine=engine,
                input_names=["input", "hidden_state"],
                output_names=["output", "add_251"]
            )

        else:
            from model_architecture.viper import Viper
            in_channels = seq_len * 3 + seq_len
            self.model = Viper(
                in_channels=in_channels,
                base_channels=self.base_channels,
                num_layers=self.num_layers
            ).to(device)
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            if self.fp16:
                self.model = self.model.half()

        self.model.eval()
        self.hidden_state = None

    def inpaint(self, frame_list, mask_list, resize_to_original=True):
        if len(frame_list) < self.seq_len:
            return None

        orig_h, orig_w = frame_list[-1].shape[:2]

        # Align target resolution to multiples of the downsample factor
        target_w = ((orig_w + self.downsample_factor - 1) // self.downsample_factor) * self.downsample_factor
        target_h = ((orig_h + self.downsample_factor - 1) // self.downsample_factor) * self.downsample_factor
        target_res = (target_w, target_h)

        frames = [
            torch.from_numpy(cv2.cvtColor(cv2.resize(f, target_res), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            for f in frame_list[-self.seq_len:]
        ]

        masks = []
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        for m in mask_list[-self.seq_len:]:
            _, m_processed = cv2.threshold(m, 0.5, 1.0, cv2.THRESH_BINARY)

            m_processed = cv2.dilate(m_processed, kernel, iterations=1)
            m_processed = cv2.GaussianBlur(m_processed, (5, 5), 2.0, borderType=cv2.BORDER_REPLICATE)

            masks.append(torch.from_numpy(cv2.resize(m_processed, target_res, interpolation=cv2.INTER_NEAREST)).unsqueeze(0))

        video_tensor = torch.stack(frames).unsqueeze(0).to(self.device).float() / 255.0
        mask_tensor = torch.stack(masks).unsqueeze(0).to(self.device).float().clamp(0.0, 1.0)
        B, T, C, H, W = mask_tensor.shape

        if self.fp16:
            video_tensor = video_tensor.half()
            mask_tensor = mask_tensor.half()

        pixel_input = (video_tensor * (1.0 - mask_tensor)).reshape(B, T * 3, H, W)
        mask_input = mask_tensor.squeeze(2)
        full_input = torch.cat([pixel_input, mask_input], dim=1)

        if self.hidden_state is None:
            self.hidden_state = torch.zeros(
                (B, self.hidden_channels, H // self.downsample_factor, W // self.downsample_factor),
                dtype=full_input.dtype,
                device=self.device
            )

        with torch.no_grad():
            output, self.hidden_state = self.model(full_input, self.hidden_state)

            if self.hidden_state is not None:
                self.hidden_state = self.hidden_state.detach()

            target_mask = mask_tensor[:, -1]
            target_frame = video_tensor[:, -1]
            composited = output * target_mask + target_frame * (1 - target_mask)

            res = composited.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
            res = (res * 255).clip(0, 255).astype('uint8')

            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

            if resize_to_original and (res.shape[0] != orig_h or res.shape[1] != orig_w):
                res = cv2.resize(res, (orig_w, orig_h))

            return [res]