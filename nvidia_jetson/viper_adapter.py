import torch.nn.functional as F
import torch


class ViperAdapter:
    def __init__(self, model_path, device="cuda", seq_len=5, fp16=False):
        from model_architecture.viper import Viper

        self.device = device
        self.seq_len = seq_len
        in_channels = seq_len * 3 + seq_len

        self.model = Viper(in_channels=in_channels, base_channels=128, num_layers=4).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.hidden_state = None

        if fp16:
            self.model = self.model.half()

    def inpaint(self, frame_list, mask_list, resize_to_original=True):
        if len(frame_list) < self.seq_len:
            return None

        # Convert lists to tensors
        # Expected shape: [T, H, W, C] -> [T, C, H, W]
        frames = [torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frame_list[-self.seq_len:]]
        masks = [torch.from_numpy(m).float().unsqueeze(0).clamp(0.0, 1.0) for m in mask_list[-self.seq_len:]]

        video_tensor = torch.stack(frames).unsqueeze(0).to(self.device)  # [1, T, 3, H, W]
        mask_tensor = torch.stack(masks).unsqueeze(0).to(self.device)  # [1, T, 1, H, W]

        B, T, C, H, W = video_tensor.shape

        # Prepare inputs
        pixel_input = (video_tensor * (1.0 - mask_tensor)).reshape(B, T * C, H, W)
        mask_input = mask_tensor.reshape(B, T, H, W)
        full_input = torch.cat([pixel_input, mask_input], dim=1)

        with torch.no_grad():
            # Pass hidden state
            output, self.hidden_state = self.model(full_input, self.hidden_state)

            if self.hidden_state is not None:
                self.hidden_state = self.hidden_state.detach()

            # Put the predicted pixels into the masked area
            target_mask = mask_tensor[:, -1]
            target_frame = video_tensor[:, -1]
            composited = output * target_mask + target_frame * (1 - target_mask)

            # Convert back to OpenCV format [H, W, 3]
            res = composited.squeeze(0).permute(1, 2, 0).cpu().numpy()
            res = (res * 255).clip(0, 255).astype('uint8')

            return [res]  # Return list to match adapter interface