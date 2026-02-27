import depth_anything_v2.dpt as dpt
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --Upload Image and Convert to Tensor--
image = cv2.imread(r"D:\Research Projects\exploreCSR-Research-Semantic-Context\RGed-research\imgs\test_img.jpeg")
rgb_image = cv2.imread(r"D:\Research Projects\exploreCSR-Research-Semantic-Context\RGed-research\imgs\test_img.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
original_height = image.shape[0]
original_width = image.shape[1]
image = cv2.resize(image, (518, 518))

# --Normalize and Prepare for Model--
image = image.astype(np.float32) / 255.0
image = np.transpose(image, (2, 0, 1))
image_tensor = torch.from_numpy(image).unsqueeze(0) 
# output shape: (1, 3, 518, 518) -> (batch_size, channels, height, width)
#print(image_tensor.shape)


# --Initialize Model--
model = dpt.DepthAnythingV2(encoder='vitb', features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False)
model.eval()

#--Run Inference--
depth = model(image_tensor)
#print(depth.shape)  # Output shape: (1, 518, 518)

depth = depth.squeeze(0).cpu().detach().numpy()  # Remove batch dimension and convert to numpy array

depth_min = depth.min()
depth_max = depth.max()

if depth_max - depth_min > 1e-6:
    depth_norm = (depth - depth_min) / (depth_max - depth_min)
else:
    depth_norm = np.zeros_like(depth)
depth_norm = cv2.resize(depth_norm, (original_width, original_height))

depth_norm = np.expand_dims(depth_norm, axis=2)
rgbd = np.concatenate((rgb_image, depth_norm), axis=2)

#--Visualize RGBD Image--
depth_vis = (depth_norm * 255).astype("uint8")
depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
cv2.imshow("Depth", depth_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

#-- Save Image--
cv2.imwrite(r"D:\Research Projects\exploreCSR-Research-Semantic-Context\RGed-research\imgs\rgbd_output.png", rgbd)
combined = np.hstack((rgb_image, depth_color))
cv2.imshow("RGB and Depth", combined)