import torch
import timm
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch.nn.functional as F
import matplotlib.pyplot as plt


class dino:

    def __init__(self, model_name="vit_small_patch16_224.dino", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = timm.create_model(model_name, pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()


    # ------------------------------------------------
    # Extract raw transformer features
    # ------------------------------------------------
    def extract_features(self, image_tensor):

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            features = self.model.forward_features(image_tensor)

        return features


    # ------------------------------------------------
    # Convert tokens to patch grid
    # ------------------------------------------------
    def process_patch_tokens(self, features):

        patch_tokens = features[:, 1:, :]

        N = patch_tokens.shape[1]
        side = int(math.sqrt(N))

        patch_tokens = patch_tokens[:, :side * side, :]
        patch_grid = patch_tokens.reshape(1, side, side, -1)

        return patch_grid


    # ------------------------------------------------
    # Prepare features for clustering
    # ------------------------------------------------
    def prepare_features_for_clustering(self, patch_grid):

        B, H, W, D = patch_grid.shape

        patch_features = patch_grid.reshape(B, H * W, D)
        patch_features_np = patch_features[0].cpu().numpy()

        # Normalize features
        patch_features_np = patch_features_np / np.linalg.norm(
            patch_features_np, axis=1, keepdims=True
        )

        # PCA dimensionality reduction
        pca = PCA(n_components=50)
        patch_features_np = pca.fit_transform(patch_features_np)

        # Add spatial information
        coords = np.indices((H, W)).reshape(2, -1).T
        coords = coords / max(H, W)

        patch_features_np = np.concatenate([patch_features_np, coords], axis=1)

        return patch_features_np, H, W


    # ------------------------------------------------
    # Cluster features (Segmentation NON-CAPTRA)
    # ------------------------------------------------
    def cluster_features(self, patch_features_np, H, W, k=5):

        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(patch_features_np)

        segmentation_map = labels.reshape(H, W)

        segmentation_map = cv2.medianBlur(segmentation_map.astype(np.uint8), 5)

        return segmentation_map


    # ========================================================
    # CAPTRA-based OBJECT MASK GENERATION
    # ========================================================


    def get_attention_map(self, image_tensor):

        image_tensor = image_tensor.to(self.device)

        # --- Save original size ---
        _, _, H, W = image_tensor.shape

        # --- Resize to 224 for DINO ---
        image_resized = F.interpolate(
            image_tensor,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )

        with torch.no_grad():
            features = self.model.forward_features(image_resized)

        # remove CLS token
        patch_tokens = features[:, 1:, :]   # [1, N, D]

        N = patch_tokens.shape[1]
        side = int(N ** 0.5)

        patch_grid = patch_tokens.reshape(1, side, side, -1)

        # --- Feature norm ---
        attn_map = torch.norm(patch_grid, dim=-1).squeeze().cpu().numpy()

        # normalize
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        # --- 🔥 Upsample BACK to original size ---
        attn_map = torch.tensor(attn_map).unsqueeze(0).unsqueeze(0)
        attn_map = F.interpolate(attn_map, size=(H, W), mode='bilinear', align_corners=False)

        return attn_map.squeeze().numpy()

    def create_foreground_mask(self, attn_map, threshold=0.6):
        return (attn_map > threshold).astype(np.uint8)

    def fuse_depth(self, mask, depth_map, depth_weight=0.6):
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

        depth_mask = (depth_norm < depth_weight).astype(np.uint8)

        return mask & depth_mask

    def upsample_mask(self, mask, output_size):
        mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float()

        upsampled = F.interpolate(
            mask_tensor,
            size=output_size,
            mode='nearest'   # critical
        )

        return upsampled.squeeze().cpu().numpy().astype(np.uint8)

    def extract_largest_component(self, mask):
        num_labels, labels = cv2.connectedComponents(mask)

        if num_labels <= 1:
            return mask

        sizes = np.bincount(labels.flatten())
        sizes[0] = 0

        largest_label = sizes.argmax()

        return (labels == largest_label).astype(np.uint8)

    def refine_mask(self, mask):
        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        mask = cv2.GaussianBlur(mask.astype(float), (5, 5), 0)
        mask = (mask > 0.5).astype(np.uint8)

        return mask

    # ========================================================
    # CAPTRA-READY OBJECT MASK
    # ========================================================

    def generate_object_mask(self, image_tensor, depth_map, output_size):

        # 1. Attention
        attn_map = self.get_attention_map(image_tensor)

        # 2. Foreground
        mask = self.create_foreground_mask(attn_map, threshold=0.6)

        # Resize mask to match depth map
        mask = cv2.resize(
            mask.astype(np.uint8),
            (depth_map.shape[1], depth_map.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        # 3. Depth fusion
        mask = self.fuse_depth(mask, depth_map, depth_weight=0.6)

        # 4. Upsample
        mask = self.upsample_mask(mask, output_size)

        # 5. Keep ONE object
        mask = self.extract_largest_component(mask)

        # 6. Cleanup
        mask = self.refine_mask(mask)

        return mask
    
    def visualize_mask_overlay(self, image_path, mask):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (mask.shape[1], mask.shape[0]))

        plt.figure(figsize=(6,6))
        plt.imshow(img)
        plt.imshow(mask, cmap='jet', alpha=0.4)

        plt.title("Mask Overlay")
        plt.axis('off')
        plt.show()

    def visualize_masked_image(self, image_path, mask):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (mask.shape[1], mask.shape[0]))

        masked_img = img.copy()
        masked_img[mask == 0] = 0
        plt.imshow(masked_img)
        plt.title("Masked Image (CAPTRA Input)")            
        plt.axis('off')
        plt.show()