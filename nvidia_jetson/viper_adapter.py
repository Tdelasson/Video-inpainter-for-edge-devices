import cv2
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional as TF


class ViperAdapter:
    def __init__(self, model_path, device="cuda", seq_len=5, fp16=True):
        self.device = device
        self.seq_len = seq_len
        self.fp16 = fp16

        if model_path.endswith('.engine'):
            from torch2trt import TRTModule
            import tensorrt as trt

            with open(model_path,'rb') as f:
                engine_data = f.read()

            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            engine = runtime.deserialize_cuda_engine(engine_data)

            self.model = TRTModule(engine = engine, input_names=["input"], output_names=["output", "add_251"])

        else:
            from model_architecture.viper import Viper
            in_channels = seq_len * 3 + seq_len
            self.model = Viper(in_channels=in_channels, base_channels=128, num_layers=4).to(device)
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            if self.fp16:
                self.model = self.model.half()

        self.model.eval()
        self.hidden_state = None

    def prepare_mask(self, mask_tensor):
        """
        Deterministic mask processing: Dilation + Gaussian Blur.
        This removes the black 'seam' and matches training logic.
        """
        # 1. Dilation: Expand the mask slightly to ensure the object is fully covered
        kernel_size = 3
        padding = kernel_size // 2
        mask_tensor = F.max_pool2d(mask_tensor, kernel_size=kernel_size, stride=1, padding=padding)

        # 2. Feathering: Blur the edges for a smooth transition
        mask_tensor = TF.gaussian_blur(mask_tensor, kernel_size=[5, 5], sigma=[2.0, 2.0])

        return mask_tensor.clamp(0, 1)

    def inpaint(self, frame_list, mask_list, resize_to_original=True):
        if len(frame_list) < self.seq_len:
            return None

        # Get original dimensions for final upscaling
        orig_h, orig_w = frame_list[-1].shape[:2]
        target_res = (256, 256)  # Matches your TRT Engine optShapes

        # 1. Resize and Convert to tensors
        # We must resize here because TRT engines have fixed/optimized input sizes
        frames = [
            torch.from_numpy(cv2.resize(f, target_res)).permute(2, 0, 1)
            for f in frame_list[-self.seq_len:]
        ]
        masks = [
            torch.from_numpy(cv2.resize(m, target_res)).unsqueeze(0)
            for m in mask_list[-self.seq_len:]
        ]

        # Shape: [1, 5, 3, res, res] and [1, 5, 1, res, res]
        video_tensor = torch.stack(frames).unsqueeze(0).to(self.device).float() / 255.0
        mask_raw = torch.stack(masks).unsqueeze(0).to(self.device).float().clamp(0.0, 1.0)

        # 2. Process Masks (Dilation/Blur)
        B, T, C, H, W = mask_raw.shape
        mask_processed = mask_raw.view(B * T, C, H, W)
        mask_processed = self.prepare_mask(mask_processed)
        mask_tensor = mask_processed.view(B, T, C, H, W)

        if self.fp16:
            video_tensor = video_tensor.half()
            mask_tensor = mask_tensor.half()

        # 3. Prepare TRT Input (Total 20 channels: 15 RGB + 5 Mask)
        # video_tensor is [1, 5, 3, res, res] -> Reshape to [1, 15, res, res]
        pixel_input = (video_tensor * (1.0 - mask_tensor)).reshape(B, T * 3, H, W)

        # mask_tensor is [1, 5, 1, res, res] -> Squeeze to [1, 5, res, res]
        # This safely removes the 'C' dimension without confusing the total element count
        mask_input = mask_tensor.squeeze(2)

        # Now cat 15 channels + 5 channels = 20 channels
        full_input = torch.cat([pixel_input, mask_input], dim=1)

        with torch.no_grad():
            # Execute TensorRT Inference
            output, self.hidden_state = self.model(full_input, self.hidden_state)

            if self.hidden_state is not None:
                self.hidden_state = self.hidden_state.detach()

            # 4. Post-processing (Alpha Blending at resxres)
            target_mask = mask_tensor[:, -1]
            target_frame = video_tensor[:, -1]
            composited = output * target_mask + target_frame * (1 - target_mask)

            # Convert back to NumPy
            res = composited.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
            res = (res * 255).clip(0, 255).astype('uint8')

            # 5. Resize back to original camera resolution (if needed)
            if resize_to_original and (res.shape[0] != orig_h or res.shape[1] != orig_w):
                res = cv2.resize(res, (orig_w, orig_h))

            return [res]