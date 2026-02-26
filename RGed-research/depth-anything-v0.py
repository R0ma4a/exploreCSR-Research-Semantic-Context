import sys
import torch
import matplotlib.pyplot as plt
import os
sys.path.append(r"D:\Research Projects\exploreCSR-Research-Semantic-Context\RGed-research\Depth-Anything-V2")

from depth_anything_v2.dpt import DepthAnythingV2
from image_encoder import upload_image, normalize_image_to_tensor

from image_encoder import upload_image, normalize_image_to_tensor  # import the separate module

# --- Paths ---
img_dir = r"D:\Research Projects\exploreCSR-Research-Semantic-Context\RGed-research\imgs"
img_file = os.path.join(img_dir, "test_img.jpeg")  # double-check extension
checkpoint_path = r"D:\Research Projects\exploreCSR-Research-Semantic-Context\RGed-research\Depth-Anything-V2\checkpoints\depth_anything_v2_vitl.pth"

# --- Load and preprocess image ---
raw_image = upload_image(img_file)
normalized_image = normalize_image_to_tensor(raw_image)

# --- Load model ---
model = DepthAnythingV2(checkpoint_path=checkpoint_path)
model.eval()

# --- Run depth prediction ---
with torch.no_grad():
    depth = model(normalized_image)

# --- Visualize depth map ---
depth_np = depth.squeeze().cpu().numpy()
plt.imshow(depth_np, cmap='plasma')
plt.axis('off')
plt.show()