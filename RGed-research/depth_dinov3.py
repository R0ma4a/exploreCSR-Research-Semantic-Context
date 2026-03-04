import torch
import timm
import cv2
import depth_anything
import math
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Load Image from DepthAnything Class --
depthAnything = depth_anything.DepthAnything()

image_path = r"D:\Research Projects\exploreCSR-Research-Semantic-Context\RGed-research\imgs\test_img.jpeg"
image = depthAnything.image_to_tensor(image_path)[0]
#print(image.shape)

#-- Initialize DINOv3 Model --
model = timm.create_model(
    "vit_small_patch16_dinov3_qkvb",
    pretrained=True
)

model = model.to(device)  # <- MOVE MODEL TO GPU
model.eval()
image = image.to(device)

print("Model:", next(model.parameters()).device)
print("Image:", image.device)

with torch.no_grad():
    features = model.forward_features(image)


patch_tokens = features[:, 1:, :]

N = patch_tokens.shape[1]
side = int(math.sqrt(N))        # H and W of patch grid
patch_tokens = patch_tokens[:, :side*side, :]  # crop extra tokens if needed

patch_grid = patch_tokens.reshape(1, side, side, -1)  # (B, H, W, D)

B, H, W, D = patch_grid.shape

# -- Flatten spatial dimensions --
patch_features = patch_grid.reshape(B, H*W, D)  # (1, N, D)
patch_features_np = patch_features[0].cpu().numpy()  # convert to numpy for sklearn


k = 5

kmeans = KMeans(n_clusters= k, random_state=0).fit(patch_features_np)
labels = kmeans.labels_  # shape: (H*W,)

segmentation_map = labels.reshape(H, W)  # (H, W)
print(segmentation_map.shape)  # (32, 32)

seg_tensor = torch.from_numpy(segmentation_map).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
seg_upsampled = F.interpolate(seg_tensor, size=(512, 512), mode='nearest')
seg_upsampled = seg_upsampled.squeeze().numpy().astype(np.int32)

#-- Visualize Segmentation Map --
#plt.imshow(seg_upsampled, cmap='tab20')
#plt.axis('off')
#plt.show()

#-- Overlay Segmentation on Original Image --
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (512,512))

# --- Overlay ---
plt.figure(figsize=(6,6))
plt.imshow(img)
plt.imshow(seg_upsampled, cmap='tab20', alpha=0.5)  # alpha for transparency
plt.axis('off')
plt.show()