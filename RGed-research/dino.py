import torch
import timm
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch.nn.functional as F


class dino:

    def __init__(self, model_name="vit_small_patch16_224.dino", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # dynamic_img_size=True removes the hardcoded 224×224 assertion in
        # patch_embed so we can pass any H/W that is a multiple of patch_size.
        self.model = timm.create_model(model_name, pretrained=True, dynamic_img_size=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Patch size from model name (default 16)
        self.patch_size = 16
        if "patch8" in model_name:
            self.patch_size = 8

        # Register a forward hook on the LAST attention block to capture raw attention weights
        self._attn_weights = None
        self._register_attention_hook()


    # ------------------------------------------------
    # Hook into the last transformer block's attention
    # ------------------------------------------------
    def _register_attention_hook(self):
        """
        timm ViT blocks expose self-attention via block.attn.
        We hook the last block so the CLS token's attention to
        all patches is captured *before* softmax averaging.
        """
        last_block = self.model.blocks[-1]

        def _hook(module, input, output):
            # output shape from Attention.forward is the projected result,
            # but we need the softmax weights — so we recompute them here.
            # timm's Attention stores qkv as a single linear; we read from input.
            x = input[0]  # [B, N, D]
            B, N, D = x.shape
            qkv = module.qkv(x)                          # [B, N, 3*D]
            qkv = qkv.reshape(B, N, 3, module.num_heads, D // module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)            # [3, B, heads, N, head_dim]
            q, k, _ = qkv.unbind(0)                      # each [B, heads, N, head_dim]
            scale = (D // module.num_heads) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale     # [B, heads, N, N]
            attn = attn.softmax(dim=-1)                  # [B, heads, N, N]
            self._attn_weights = attn.detach()           # save for later

        last_block.attn.register_forward_hook(_hook)


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

        patch_features_np = patch_features_np / np.linalg.norm(
            patch_features_np, axis=1, keepdims=True
        )

        pca = PCA(n_components=50)
        patch_features_np = pca.fit_transform(patch_features_np)

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
    # CAPTRA-based OBJECT MASK GENERATION  (improved)
    # ========================================================

    def get_attention_map(self, image_tensor):
        """
        Returns a foreground saliency map using the TRUE CLS self-attention
        from DINO's last transformer block, averaged across all heads.

        Key fix vs. original:
          - Uses real attention weights (CLS→patches) via forward hook
            instead of patch feature norms, which are NOT attention.
          - Input is padded to the nearest multiple of patch_size rather
            than squished to 224×224, preserving aspect ratio.
          - Threshold is computed adaptively (Otsu on the attention map)
            instead of a fixed scalar.
        """
        image_tensor = image_tensor.to(self.device)
        _, _, H_orig, W_orig = image_tensor.shape

        # --- Resize to nearest multiple of patch_size ---
        # dynamic_img_size=True (set in __init__) lifts the 224×224 assertion,
        # so we can use any multiple of patch_size.  We still need to snap to a
        # multiple so patch_embed divides evenly.
        p = self.patch_size
        H_pad = math.ceil(H_orig / p) * p
        W_pad = math.ceil(W_orig / p) * p
        image_padded = F.interpolate(
            image_tensor,
            size=(H_pad, W_pad),
            mode='bilinear',
            align_corners=False
        )

        # --- Forward pass (hook fires and stores self._attn_weights) ---
        with torch.no_grad():
            _ = self.model.forward_features(image_padded)

        # _attn_weights: [B, heads, N_tokens, N_tokens]
        # N_tokens = 1 (CLS) + H_patches * W_patches
        attn = self._attn_weights  # [1, heads, N, N]

        # CLS token attends to all patch tokens: row 0, columns 1:
        # shape → [heads, num_patches]
        cls_attn = attn[0, :, 0, 1:].cpu().numpy()   # [heads, N_patches]

        H_patches = H_pad // p
        W_patches = W_pad // p

        # Average over heads, then reshape to spatial grid
        attn_map = cls_attn.mean(axis=0)              # [N_patches]
        attn_map = attn_map.reshape(H_patches, W_patches)

        # Normalize to [0, 1]
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        # Upsample back to original image size
        attn_tensor = torch.tensor(attn_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        attn_tensor = F.interpolate(attn_tensor, size=(H_orig, W_orig), mode='bilinear', align_corners=False)

        return attn_tensor.squeeze().numpy()


    def get_attention_map_multihead(self, image_tensor):
        """
        Returns per-head attention maps (useful for debugging / head selection).
        Shape: [num_heads, H_orig, W_orig]
        """
        image_tensor = image_tensor.to(self.device)
        _, _, H_orig, W_orig = image_tensor.shape

        p = self.patch_size
        H_pad = math.ceil(H_orig / p) * p
        W_pad = math.ceil(W_orig / p) * p
        image_padded = F.interpolate(image_tensor, size=(H_pad, W_pad),
                                     mode='bilinear', align_corners=False)

        with torch.no_grad():
            _ = self.model.forward_features(image_padded)

        attn = self._attn_weights   # [1, heads, N, N]
        cls_attn = attn[0, :, 0, 1:].cpu().numpy()   # [heads, N_patches]

        H_patches = H_pad // p
        W_patches = W_pad // p
        num_heads = cls_attn.shape[0]

        head_maps = []
        for h in range(num_heads):
            m = cls_attn[h].reshape(H_patches, W_patches)
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)
            m_tensor = torch.tensor(m, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            m_up = F.interpolate(m_tensor, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
            head_maps.append(m_up.squeeze().numpy())

        return np.stack(head_maps, axis=0)   # [heads, H, W]


    # ------------------------------------------------
    # Adaptive threshold via Otsu on the attention map
    # ------------------------------------------------
    def _otsu_threshold(self, attn_map):
        """Compute Otsu threshold on a float [0,1] attention map."""
        attn_uint8 = (attn_map * 255).astype(np.uint8)
        thresh_val, _ = cv2.threshold(attn_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh_val / 255.0


    def create_foreground_mask(self, attn_map, threshold=None):
        """
        threshold=None  → use Otsu (recommended)
        threshold=float → use fixed value (legacy behaviour)
        """
        if threshold is None:
            threshold = self._otsu_threshold(attn_map)
        return (attn_map > threshold).astype(np.uint8)


    # ------------------------------------------------
    # Depth fusion  (less aggressive than before)
    # ------------------------------------------------
    def fuse_depth(self, mask, depth_map, depth_weight=0.5, depth_percentile=70):
        """
        Keep masked pixels whose depth is below the `depth_percentile`-th
        percentile of depth values inside the initial mask.
        Using a percentile instead of (median - 0.05) avoids the mask
        collapsing when the object spans a wide depth range.
        """
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

        masked_depth = depth_norm[mask.astype(bool)]
        if len(masked_depth) == 0:
            return mask

        depth_thresh = np.percentile(masked_depth, depth_percentile)
        depth_mask = depth_norm <= depth_thresh

        # Blend: require BOTH attention foreground AND depth constraint
        fused = mask.astype(bool) & depth_mask
        return fused


    def upsample_mask(self, mask, output_size):
        mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float()
        upsampled = F.interpolate(mask_tensor, size=output_size, mode='nearest')
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


    def keep_center_object(self, mask):
        mask_uint8 = mask.astype(np.uint8)
        dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        dist = dist / (dist.max() + 1e-8)
        return dist > 0.4


    # ========================================================
    # CAPTRA-READY OBJECT MASK  (main entry point)
    # ========================================================

    def generate_object_mask(self, image_tensor, depth_map, output_size,
                             use_depth=True, fixed_threshold=None):
        """
        Parameters
        ----------
        image_tensor    : [1, 3, H, W] float tensor (ImageNet-normalized)
        depth_map       : HxW numpy array (metric or relative depth)
        output_size     : (H_out, W_out) tuple for final mask resolution
        use_depth       : set False to skip depth fusion (useful when depth
                          map quality is poor)
        fixed_threshold : float or None.  None → adaptive Otsu (recommended)
        """

        # 1. TRUE CLS attention map
        attn_map = self.get_attention_map(image_tensor)

        # 2. Adaptive foreground mask
        mask = self.create_foreground_mask(attn_map, threshold=fixed_threshold)

        if use_depth and depth_map is not None:
            # Resize mask to match depth map
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (depth_map.shape[1], depth_map.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            # 3. Depth fusion (gentler than before)
            mask_fused = self.fuse_depth(mask_resized, depth_map, depth_weight=0.5)
            mask = mask_fused.astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        # 4. Morphological cleanup
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask * 255, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask,       cv2.MORPH_CLOSE, kernel)
        mask = (mask > 127).astype(np.uint8)

        # 5. Upsample to target resolution
        mask = self.upsample_mask(mask, output_size)

        # 6. Keep largest connected component
        mask = self.extract_largest_component(mask)

        # 7. Final cleanup
        mask = self.refine_mask(mask)

        return mask


    # ------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------

    def visualize_attention_heads(self, image_tensor, image_path=None):
        """
        Plot each attention head side-by-side — helpful for debugging
        which heads carry the foreground signal.
        """
        head_maps = self.get_attention_map_multihead(image_tensor)
        n = head_maps.shape[0]
        fig, axes = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 3))

        if image_path:
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            axes[0].imshow(img)
            axes[0].set_title("Input")
        else:
            axes[0].axis('off')

        for h in range(n):
            axes[h + 1].imshow(head_maps[h], cmap='inferno')
            axes[h + 1].set_title(f"Head {h}")
            axes[h + 1].axis('off')

        plt.tight_layout()
        plt.show()


    def visualize_mask_overlay(self, image_path, mask):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (mask.shape[1], mask.shape[0]))

        plt.figure(figsize=(6, 6))
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
