from dataset import YouTubeVOSDataset
from model_architecture.video_inpainter import VideoInpainter
import cv2
import torch
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision.models import vgg16, VGG16_Weights

class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.mse = torch.nn.MSELoss()

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        return self.mse(x_vgg, y_vgg)

# We have 3471 files
number_of_seq = 50


dataset = YouTubeVOSDataset(root_dir=r"C:\Users\tobpu\Documents\aau\Semester 6\training_data\train", seq_len=5)

IN_CHANNELS = dataset.seq_len * 3
TARGET_RES = (256, 256)

rgb_data = []
training_tensors = []
rgb_img_tensor = []

for i in range(0, number_of_seq):
    data = dataset.load_data()
    if len(data) > 0:
        rgb_data.append(data)
        print(len(data))
        print("Shape")
        print(data[0].shape)

        rgb_img_tensor = torch.from_numpy(data).float() / 255.0
        rgb_img_tensor = rgb_img_tensor.permute(0, 3, 1, 2)  # First to (seq_len, C, H, W)
        rgb_img_tensor = torch.nn.functional.interpolate(
            rgb_img_tensor,
            size=TARGET_RES,
            mode='bilinear',
            align_corners=False
        )
        target = rgb_img_tensor[-1]
        print(f"target shape: {target.shape}")
        rgb_img_tensor = rgb_img_tensor.reshape(-1, 256, 256)  # (seq_len * C, 256, 256)

        print(rgb_img_tensor.shape)
        training_tensors.append(rgb_img_tensor)



print(len(rgb_data))

model = VideoInpainter(in_channels=IN_CHANNELS, base_channels=64, num_layers=3)

print("input")
print(training_tensors[0].shape)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
results = []

perceptual_criterion = PerceptualLoss()
l1_criterion = torch.nn.L1Loss()

for i in range(0, 1000):
    optimizer.zero_grad()

    output_frame, _ = model(training_tensors[0].unsqueeze(0).unsqueeze(0))
    output_frame = output_frame.squeeze(0).squeeze(0)

    out_vgg = output_frame.unsqueeze(0)
    target_vgg = target.unsqueeze(0)

    l1_loss = l1_criterion(output_frame, target)
    vgg_loss = perceptual_criterion(out_vgg, target_vgg)
    # Kombineret loss (prøv med en vægt på 0.1 til VGG i starten)
    total_loss = l1_loss + (0.1 * vgg_loss)


    total_loss.backward()
    optimizer.step()
    scheduler.step()

    print(f"Iteration {i} | Total: {total_loss.item():.4f} | L1: {l1_loss.item():.4f} | VGG: {vgg_loss.item():.4f}")

    if i % 50 == 0:
        print(f"Iteration {i} | Loss: {total_loss.item():.6f}")
        results.append((target, output_frame.detach()))


num_results = len(results)

for idx, (target_img, output_img) in enumerate(results):
    for i, image in enumerate([target_img, output_img]):
        img_np = image.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        cv2.imshow('image', img_np)
        cv2.waitKey(0)

    if idx == num_results - 1:
        target_np = target_img.permute(1, 2, 0).numpy()
        target_np = (target_np * 255).astype(np.uint8)
        target_np = cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR)

        output_np = output_img.permute(1, 2, 0).numpy()
        output_np = (output_np * 255).astype(np.uint8)
        output_np = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

        cv2.imwrite("target.png", target_np)
        cv2.imwrite("final_output.png", output_np)

        print("Færdig! Gemte både 'target.png' og 'final_output.png'")

#for i in range(0, len(rgb_data)):
#    for h in range(0, len(rgb_data[i])):
#        cv2.imshow('image', rgb_data[i][h])
#        cv2.waitKey(0)
