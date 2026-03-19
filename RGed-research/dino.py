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

    def get_attention_map(self, image_tensor, top_k_heads=None):
        """
        Returns a foreground saliency map using the TRUE CLS self-attention
        from DINO's last transformer block.

        Instead of naively averaging ALL heads (which dilutes the signal with
        diffuse background-attending heads), we select only the most "focused"
        heads by ranking on Shannon entropy:
          - Low entropy  -> concentrated attention -> object-like head  (keep)
          - High entropy -> diffuse attention      -> background head   (drop)

        Parameters
        ----------
        top_k_heads : int or None
            Number of lowest-entropy heads to use.
            None -> auto, keeps the bottom 50% of heads by entropy.
        """
        image_tensor = image_tensor.to(self.device)
        _, _, H_orig, W_orig = image_tensor.shape

        # --- Snap to nearest multiple of patch_size ---
        p = self.patch_size
        H_pad = math.ceil(H_orig / p) * p
        W_pad = math.ceil(W_orig / p) * p
        image_padded = F.interpolate(
            image_tensor, size=(H_pad, W_pad),
            mode='bilinear', align_corners=False
        )

        # --- Forward pass (hook fires -> self._attn_weights) ---
        with torch.no_grad():
            _ = self.model.forward_features(image_padded)

        attn     = self._attn_weights                    # [1, heads, N, N]
        cls_attn = attn[0, :, 0, 1:].cpu().numpy()      # [heads, N_patches]

        H_patches = H_pad // p
        W_patches = W_pad // p
        num_heads = cls_attn.shape[0]

        # --- Entropy-based head selection ---
        eps       = 1e-8
        entropies = -(cls_attn * np.log(cls_attn + eps)).sum(axis=1)  # [heads]

        if top_k_heads is None:
            top_k_heads = max(1, num_heads // 2)         # default: best 50%

        selected  = np.argsort(entropies)[:top_k_heads]  # lowest-entropy heads
        attn_map  = cls_attn[selected].mean(axis=0)      # [N_patches]
        attn_map  = attn_map.reshape(H_patches, W_patches)

        # Normalize to [0, 1]
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        # Upsample back to original image size
        attn_tensor = torch.tensor(attn_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        attn_tensor = F.interpolate(attn_tensor, size=(H_orig, W_orig),
                                    mode='bilinear', align_corners=False)

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
    # Percentile threshold  (replaces Otsu)
    # ------------------------------------------------
    def _percentile_threshold(self, attn_map, keep_fraction=0.35):
        """
        Return the value at the (1 - keep_fraction) percentile so that
        roughly `keep_fraction` of patches are classified as foreground.

        keep_fraction=0.35 means the top 35% of patches are kept.
        Tune upward (0.4-0.5) for larger objects, downward (0.2-0.3) for small ones.
        """
        return float(np.percentile(attn_map, (1.0 - keep_fraction) * 100))

    def create_foreground_mask(self, attn_map, threshold=None, keep_fraction=0.35):
        """
        threshold=None  -> use percentile threshold (recommended)
        threshold=float -> use fixed value (legacy)

        keep_fraction : fraction of patches to treat as foreground (default 35%).
        Ignored when threshold is supplied explicitly.
        """
        if threshold is None:
            threshold = self._percentile_threshold(attn_map, keep_fraction)
        return (attn_map > threshold).astype(np.uint8)


    # ------------------------------------------------
    # Depth fusion
    # ------------------------------------------------
    def fuse_depth(self, mask, depth_map, foreground_depth_fraction=0.45):
        """
        Keep masked pixels that fall within the closest
        `foreground_depth_fraction` of the FULL image depth range.

        Thresholding against the full image (not within the mask) is more
        reliable when the initial attention mask bleeds into the background:
        the object of interest is always a foreground object, occupying
        the lowest depth values in the scene.
        foreground_depth_fraction=0.45 keeps the closest 45% of the scene;
        tune down toward 0.30 for tighter object crops.
        """
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

        # DepthAnything outputs INVERSE depth: larger value = CLOSER to camera.
        # So foreground pixels have HIGH depth values — keep the top percentile.
        depth_thresh = np.percentile(depth_norm, (1.0 - foreground_depth_fraction) * 100)
        depth_mask = depth_norm >= depth_thresh
        return mask.astype(bool) & depth_mask


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

    def extract_center_component(self, mask):
        """
        Return the connected component whose centroid is closest to the image
        center, weighted by component size.

        Better than extract_largest_component for scenes where:
          - A thin object (lamppost, sign pole) connects to the ground through
            its base, creating one large L-shaped blob that is technically the
            biggest region but is mostly noise.
          - Multiple blobs exist and the target object is not the largest.

        Score = distance_from_center / sqrt(size)
          -> large components tolerate being slightly off-center
          -> small peripheral specks score worse than a large centered object
        """
        num_labels, labels = cv2.connectedComponents(mask)
        if num_labels <= 1:
            return mask

        H, W = mask.shape
        cy, cx = H / 2.0, W / 2.0

        best_label = 1
        best_score = float('inf')

        for lbl in range(1, num_labels):
            component = (labels == lbl)
            size = int(component.sum())
            if size < 50:           # skip tiny specks
                continue
            ys, xs = np.where(component)
            dist = ((ys.mean() - cy) ** 2 + (xs.mean() - cx) ** 2) ** 0.5
            score = dist / (size ** 0.5)
            if score < best_score:
                best_score = score
                best_label = lbl

        return (labels == best_label).astype(np.uint8)


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
                             use_depth=True, fixed_threshold=None,
                             keep_fraction=0.38, top_k_heads=None):
        """
        Parameters
        ----------
        image_tensor    : [1, 3, H, W] float tensor (ImageNet-normalized)
        depth_map       : HxW numpy array or None
        output_size     : (H_out, W_out) tuple for final mask resolution
        use_depth       : False by default — DINO attention alone generalises
                          better than attention intersected with depth, because
                          depth fusion can slice through an object that spans
                          multiple depth planes (e.g. a tall bag on a table).
                          Set True only when the scene has strong depth separation.
        fixed_threshold : float or None. None -> percentile threshold
        keep_fraction   : fraction of patches treated as foreground (default 0.50).
                          0.50 = top half of attention patches kept.
                          Lower (0.35) for tighter mask; higher (0.65) to fill gaps.
        top_k_heads     : number of low-entropy heads to use. None -> auto (50%)
        """

        # 1. TRUE CLS attention map (entropy-filtered heads)
        attn_map = self.get_attention_map(image_tensor, top_k_heads=top_k_heads)

        # 2. Percentile foreground mask — keep top 50% of attended patches
        mask = self.create_foreground_mask(attn_map, threshold=fixed_threshold,
                                           keep_fraction=keep_fraction)

        if use_depth and depth_map is not None:
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (depth_map.shape[1], depth_map.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            mask_fused = self.fuse_depth(mask_resized, depth_map,
                                         foreground_depth_fraction=0.65)
            mask = mask_fused.astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        # 3. Morphological cleanup — close first to fill holes inside the object,
        #    then open to remove small isolated background blobs
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask * 255, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask,       cv2.MORPH_OPEN,  kernel)
        mask = (mask > 127).astype(np.uint8)

        # 4. Upsample to target resolution
        mask = self.upsample_mask(mask, output_size)

        # 5. Keep the component closest to image center
        mask = self.extract_center_component(mask)

        # 6. Final smoothing
        mask = self.refine_mask(mask)

        # 7. Auto-invert: if >55% of pixels are masked we selected the background
        if mask.mean() > 0.55:
            mask = (1 - mask).astype(np.uint8)
            mask = self.extract_center_component(mask)
            mask = self.refine_mask(mask)

        # 8. Center-crop attention fallback for small objects
        # Triggered when pass 1 still covers >40% of the image.
        # Depth doesn't help for signs in snow because the ground is CLOSER
        # than the sign. Instead: crop center 40% of the attention map,
        # find the locally strongest attended region there, and use that.
        # The target object is always roughly centered, and is the most
        # locally-attended thing in the center crop even when it loses globally
        # to a brighter/larger background region like snow.
        if mask.mean() > 0.40:
            H_a, W_a = attn_map.shape
            r0, r1 = int(H_a * 0.30), int(H_a * 0.70)
            c0, c1 = int(W_a * 0.30), int(W_a * 0.70)

            # Zero outside center crop so peak search is constrained
            cropped_attn = np.zeros_like(attn_map)
            cropped_attn[r0:r1, c0:c1] = attn_map[r0:r1, c0:c1]

            # Keep top 20% of attention values inside the crop only
            crop_vals = cropped_attn[r0:r1, c0:c1]
            if crop_vals.max() > 0:
                local_thresh = np.percentile(crop_vals[crop_vals > 0], 80)
                center_mask = (cropped_attn >= local_thresh).astype(np.uint8)

                k = np.ones((7, 7), np.uint8)
                center_mask = cv2.morphologyEx(center_mask * 255, cv2.MORPH_CLOSE, k)
                center_mask = cv2.morphologyEx(center_mask,       cv2.MORPH_OPEN,  k)
                center_mask = (center_mask > 127).astype(np.uint8)

                center_mask = self.upsample_mask(center_mask, output_size)
                center_mask = self.extract_center_component(center_mask)
                center_mask = self.refine_mask(center_mask)

                # Accept only if compact and non-trivial
                if 0.01 < center_mask.mean() < 0.35:
                    mask = center_mask

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


    def visualize_mask_overlay(self, image_path, mask, color=(255, 140, 0), alpha=0.55):
        """
        Highlight the masked object in `color` (default orange).
        Background pixels are shown as a dimmed greyscale so the object pops.

        Parameters
        ----------
        color : RGB tuple — Orange=(255,140,0), Green=(0,200,80), Red=(220,50,50)
        alpha : blend strength of the color on the object (0=no tint, 1=solid color)
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = cv2.resize(img, (mask.shape[1], mask.shape[0]))

        fg = mask.astype(bool)
        bg = ~fg

        result = img.copy()

        # Foreground: blend original pixel with the highlight color
        result[fg] = (
            img[fg] * (1 - alpha) +
            np.array(color, dtype=np.float32) * alpha
        ).clip(0, 255)

        # Background: blue-grey tint
        grey = img[bg].mean(axis=1, keepdims=True)  # [N,1]
        blue_tint = np.array([0.55, 0.65, 1.0], dtype=np.float32)   # R,G,B multipliers
        result[bg] = (grey * blue_tint * 0.55).clip(0, 255)

        plt.figure(figsize=(6, 6))
        plt.imshow(result.astype(np.uint8))
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
