import depth_anything_v2.dpt as dpt
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

class DepthAnything:
    
    # --Initialize Class Model--
    def __init__(self, checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = dpt.DepthAnythingV2(
            encoder='vitb',
            features=128,
            out_channels=[96, 192, 384, 768],
            use_bn=False,
            use_clstoken=False
        )

        # --- Load checkpoint ---
        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)
        self.model.eval()

    def load_rgb_image(self, image_path):
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_image

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
    
    def process_depth(self, depth, original_width, original_height):
        # --- Remove extreme outliers (huge improvement) ---
        low, high = np.percentile(depth, (2, 98))
        depth = np.clip(depth, low, high)

        # --- Normalize ---
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # --- Resize to original resolution ---
        depth_norm = cv2.resize(
            depth_norm,
            (original_width, original_height),
            interpolation=cv2.INTER_CUBIC
        )

        # --- Smooth patch artifacts slightly ---
        depth_norm = cv2.GaussianBlur(depth_norm, (5, 5), 0)

        return depth_norm
    
    def visualize_depth(self, depth_norm):
        print("min:", depth_norm.min())
        print("max:", depth_norm.max())
        print("dtype:", depth_norm.dtype)
        plt.figure(figsize=(8, 6))
        plt.imshow(depth_norm, cmap='plasma')
        plt.colorbar()
        plt.axis('off')
        plt.show()

    def save_depth(self, depth_norm, output_path, name):
        # Convert to 0–255
        depth_8bit = (depth_norm * 255).astype("uint8")

        # Apply colormap
        depth_color = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_PLASMA)

        # Save colored version
        cv2.imwrite(f"{output_path}{name}_depth_color.png", depth_color)

    def create_side_by_side(self, rgb_image, depth_norm):
        depth_vis = (depth_norm.squeeze() * 255).astype("uint8")
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

        combined = np.hstack((rgb_image, depth_color))
        return combined
