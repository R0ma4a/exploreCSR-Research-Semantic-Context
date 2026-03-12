import torch
import timm
import cv2
import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch.nn.functional as F
import matplotlib.pyplot as plt


class dino:

    def __init__(self, model_name="vit_small_patch16_dinov3_qkvb", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = timm.create_model(model_name, pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()


    # ------------------------------------------------
    # Step 1: Extract raw transformer features
    # ------------------------------------------------
    def extract_features(self, image_tensor):

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            features = self.model.forward_features(image_tensor)

        return features


    # ------------------------------------------------
    # Step 2: Convert tokens to patch grid
    # ------------------------------------------------
    def process_patch_tokens(self, features):

        patch_tokens = features[:, 1:, :]

        N = patch_tokens.shape[1]
        side = int(math.sqrt(N))

        patch_tokens = patch_tokens[:, :side * side, :]
        patch_grid = patch_tokens.reshape(1, side, side, -1)

        return patch_grid


    # ------------------------------------------------
    # Step 3: Prepare features for clustering
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
    # Step 4: Cluster features (segmentation)
    # ------------------------------------------------
    def cluster_features(self, patch_features_np, H, W, k=5):

        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(patch_features_np)

        segmentation_map = labels.reshape(H, W)

        segmentation_map = cv2.medianBlur(segmentation_map.astype(np.uint8), 5)

        return segmentation_map


    # ------------------------------------------------
    # Step 5: Upsample segmentation to image size
    # ------------------------------------------------
    def upsample_segmentation(self, segmentation_map, output_size=(512, 512)):

        seg_tensor = torch.from_numpy(segmentation_map).unsqueeze(0).unsqueeze(0).float()

        seg_upsampled = F.interpolate(
            seg_tensor,
            size=output_size,
            mode='bilinear'
        )

        seg_upsampled = seg_upsampled.squeeze().numpy().astype(np.int32)

        return seg_upsampled


    # ------------------------------------------------
    # Step 6: Visualization
    # ------------------------------------------------
    def visualize_overlay(self, image_path, seg_upsampled):

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))

        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.imshow(seg_upsampled, cmap='tab20', alpha=0.5)
        plt.axis('off')
        plt.show()

