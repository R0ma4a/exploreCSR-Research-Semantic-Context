import depth_anything_v2.dpt as dpt
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

class DepthAnything:
    
    # --Initialize Class Model--
    def __init__(self): 
        self.model = dpt.DepthAnythingV2(encoder='vitb', features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

    def image_to_tensor(self, image_path):
        image = cv2.imread(image_path)

        rgb_image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height = image.shape[0]
        original_width = image.shape[1]
        image = cv2.resize(image, (518, 518))

        # --Normalize and Prepare for Model--
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image).unsqueeze(0) 
        # output shape: (1, 3, 518, 518) -> (batch_size, channels, height, width)
        return image_tensor.to(self.device), rgb_image, original_width, original_height
    
    def predict_depth(self, image_tensor):
        with torch.no_grad():
            depth = self.model(image_tensor)
            depth = depth.squeeze(0).cpu().detach().numpy()  # Remove batch dimension and convert to numpy array
            
        return depth
    
    def process_rgbd(self, depth, rgb_image, original_width, original_height):
        depth_min = depth.min()
        depth_max = depth.max()

        if depth_max - depth_min > 1e-6:
            depth_norm = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth)
        depth_norm = cv2.resize(depth_norm, (original_width, original_height))

        depth_norm = np.expand_dims(depth_norm, axis=2)
        rgbd = np.concatenate((rgb_image, depth_norm), axis=2)

        return rgbd, depth_norm
    
    def visualize_depth(self, depth_norm):
        depth_vis = (depth_norm * 255).astype("uint8")
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        cv2.imshow("Depth", depth_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_rgbd(self, rgbd, output_path, name):
        output_file = f"{output_path}{name}.jpg"

        if rgbd.dtype != np.uint8:
            rgbd_norm = (rgbd - rgbd.min()) / (rgbd.max() - rgbd.min())
            rgbd = (rgbd_norm * 255).astype(np.uint8)

        cv2.imwrite(output_file, rgbd)
