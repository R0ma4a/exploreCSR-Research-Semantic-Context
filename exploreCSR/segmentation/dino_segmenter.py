"""
DINO-based object segmentation.

Uses self-attention from a pretrained DINO ViT to produce object masks,
either unsupervised (attention + optional depth fusion) or prompt-guided
(attention + spatial/semantic priors from text).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class DINOSegmenter:
    """
    Object segmentation via DINO ViT self-attention.

    Two main entry points:
    - ``generate_object_mask``: unsupervised foreground extraction
    - ``segment_from_prompt``: prompt-guided segmentation (DINO-only, no CLIP)
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224.dino",
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = timm.create_model(
            model_name, pretrained=True, dynamic_img_size=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.patch_size = 8 if "patch8" in model_name else 16

        # Hook for capturing attention weights from the last block
        self._attn_weights: Optional[torch.Tensor] = None
        self._register_attention_hook()

    # ==================================================================
    # Attention hook
    # ==================================================================

    def _register_attention_hook(self) -> None:
        """Register a forward hook on the last transformer block's attention."""
        last_block = self.model.blocks[-1]

        def _hook(module, input, output):
            x = input[0]  # (B, N, D)
            B, N, D = x.shape
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, D // module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
            q, k, _ = qkv.unbind(0)
            scale = (D // module.num_heads) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            self._attn_weights = attn.detach()

        last_block.attn.register_forward_hook(_hook)

    # ==================================================================
    # Feature extraction
    # ==================================================================

    def extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extract raw transformer features from image tensor."""
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            return self.model.forward_features(image_tensor)

    def extract_patch_grid(self, features: torch.Tensor) -> torch.Tensor:
        """Reshape patch tokens into a spatial grid (1, H, W, D)."""
        patch_tokens = features[:, 1:, :]
        N = patch_tokens.shape[1]
        side = int(math.sqrt(N))
        patch_tokens = patch_tokens[:, : side * side, :]
        return patch_tokens.reshape(1, side, side, -1)

    # ==================================================================
    # Clustering-based segmentation (non-CAPTRA)
    # ==================================================================

    def cluster_features(
        self,
        image_tensor: torch.Tensor,
        k: int = 5,
        n_pca_components: int = 50,
    ) -> np.ndarray:
        """
        Unsupervised KMeans segmentation on DINO patch features.

        Returns a (H_patch, W_patch) label map.
        """
        features = self.extract_features(image_tensor)
        patch_grid = self.extract_patch_grid(features)
        B, H, W, D = patch_grid.shape

        feats = patch_grid.reshape(B, H * W, D)[0].cpu().numpy()
        feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)

        pca = PCA(n_components=n_pca_components)
        feats = pca.fit_transform(feats)

        coords = np.indices((H, W)).reshape(2, -1).T / max(H, W)
        feats = np.concatenate([feats, coords], axis=1)

        labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(feats)
        seg_map = labels.reshape(H, W)
        seg_map = cv2.medianBlur(seg_map.astype(np.uint8), 5)
        return seg_map

    # ==================================================================
    # Attention maps
    # ==================================================================

    def get_attention_map(
        self,
        image_tensor: torch.Tensor,
        top_k_heads: Optional[int] = None,
    ) -> np.ndarray:
        """
        Foreground saliency map from entropy-filtered CLS attention heads.

        Low-entropy heads attend to objects; high-entropy heads are diffuse.
        By default, the bottom 50% by entropy are kept.

        Returns (H_orig, W_orig) float map in [0, 1].
        """
        image_tensor = image_tensor.to(self.device)
        _, _, H_orig, W_orig = image_tensor.shape

        p = self.patch_size
        H_pad = math.ceil(H_orig / p) * p
        W_pad = math.ceil(W_orig / p) * p
        image_padded = F.interpolate(
            image_tensor, size=(H_pad, W_pad), mode="bilinear", align_corners=False
        )

        with torch.no_grad():
            self.model.forward_features(image_padded)

        attn = self._attn_weights  # (1, heads, N, N)
        cls_attn = attn[0, :, 0, 1:].cpu().numpy()  # (heads, N_patches)

        H_patches = H_pad // p
        W_patches = W_pad // p
        num_heads = cls_attn.shape[0]

        # Entropy-based head selection
        eps = 1e-8
        entropies = -(cls_attn * np.log(cls_attn + eps)).sum(axis=1)

        if top_k_heads is None:
            top_k_heads = max(1, num_heads // 2)

        selected = np.argsort(entropies)[:top_k_heads]
        attn_map = cls_attn[selected].mean(axis=0).reshape(H_patches, W_patches)

        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + eps)

        attn_tensor = torch.tensor(attn_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        attn_tensor = F.interpolate(
            attn_tensor, size=(H_orig, W_orig), mode="bilinear", align_corners=False
        )
        return attn_tensor.squeeze().numpy()

    def get_attention_map_multihead(
        self, image_tensor: torch.Tensor
    ) -> np.ndarray:
        """Per-head attention maps (heads, H_orig, W_orig) for debugging."""
        image_tensor = image_tensor.to(self.device)
        _, _, H_orig, W_orig = image_tensor.shape

        p = self.patch_size
        H_pad = math.ceil(H_orig / p) * p
        W_pad = math.ceil(W_orig / p) * p
        image_padded = F.interpolate(
            image_tensor, size=(H_pad, W_pad), mode="bilinear", align_corners=False
        )

        with torch.no_grad():
            self.model.forward_features(image_padded)

        attn = self._attn_weights
        cls_attn = attn[0, :, 0, 1:].cpu().numpy()

        H_patches = H_pad // p
        W_patches = W_pad // p

        head_maps = []
        for h in range(cls_attn.shape[0]):
            m = cls_attn[h].reshape(H_patches, W_patches)
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)
            m_tensor = torch.tensor(m, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            m_up = F.interpolate(
                m_tensor, size=(H_orig, W_orig), mode="bilinear", align_corners=False
            )
            head_maps.append(m_up.squeeze().numpy())

        return np.stack(head_maps, axis=0)

    # ==================================================================
    # Mask utilities
    # ==================================================================

    @staticmethod
    def _percentile_threshold(attn_map: np.ndarray, keep_fraction: float = 0.35) -> float:
        """Threshold value that keeps the top ``keep_fraction`` of patches."""
        return float(np.percentile(attn_map, (1.0 - keep_fraction) * 100))

    def create_foreground_mask(
        self,
        attn_map: np.ndarray,
        threshold: Optional[float] = None,
        keep_fraction: float = 0.35,
    ) -> np.ndarray:
        """Binary foreground mask from attention map."""
        if threshold is None:
            threshold = self._percentile_threshold(attn_map, keep_fraction)
        return (attn_map > threshold).astype(np.uint8)

    @staticmethod
    def fuse_depth(
        mask: np.ndarray,
        depth_map: np.ndarray,
        foreground_depth_fraction: float = 0.45,
    ) -> np.ndarray:
        """
        Retain only mask pixels in the closest ``foreground_depth_fraction``
        of the full-image depth range. DepthAnything uses inverse depth
        (higher = closer), so we keep the top percentile.
        """
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        depth_thresh = np.percentile(depth_norm, (1.0 - foreground_depth_fraction) * 100)
        return (mask.astype(bool) & (depth_norm >= depth_thresh)).astype(np.uint8)

    @staticmethod
    def upsample_mask(mask: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
        """Nearest-neighbor upsample a mask to ``output_size`` (H, W)."""
        t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        up = F.interpolate(t, size=output_size, mode="nearest")
        return up.squeeze().cpu().numpy().astype(np.uint8)

    @staticmethod
    def extract_largest_component(mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component."""
        num_labels, labels = cv2.connectedComponents(mask)
        if num_labels <= 1:
            return mask
        sizes = np.bincount(labels.flatten())
        sizes[0] = 0
        return (labels == sizes.argmax()).astype(np.uint8)

    @staticmethod
    def extract_center_component(mask: np.ndarray) -> np.ndarray:
        """
        Keep the connected component whose centroid is closest to the image
        center, weighted by size: score = distance / sqrt(size).
        """
        num_labels, labels = cv2.connectedComponents(mask)
        if num_labels <= 1:
            return mask

        H, W = mask.shape
        cy, cx = H / 2.0, W / 2.0

        best_label, best_score = 1, float("inf")
        for lbl in range(1, num_labels):
            component = labels == lbl
            size = int(component.sum())
            if size < 50:
                continue
            ys, xs = np.where(component)
            dist = ((ys.mean() - cy) ** 2 + (xs.mean() - cx) ** 2) ** 0.5
            score = dist / (size**0.5)
            if score < best_score:
                best_score = score
                best_label = lbl

        return (labels == best_label).astype(np.uint8)

    @staticmethod
    def refine_mask(mask: np.ndarray) -> np.ndarray:
        """Morphological cleanup + light blur."""
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask.astype(float), (5, 5), 0)
        return (mask > 0.5).astype(np.uint8)

    # ==================================================================
    # Unsupervised object mask (CAPTRA-ready)
    # ==================================================================

    def generate_object_mask(
        self,
        image_tensor: torch.Tensor,
        depth_map: Optional[np.ndarray],
        output_size: Tuple[int, int],
        use_depth: bool = True,
        fixed_threshold: Optional[float] = None,
        keep_fraction: float = 0.38,
        top_k_heads: Optional[int] = None,
    ) -> np.ndarray:
        """
        Unsupervised object mask from DINO attention + optional depth fusion.

        Parameters
        ----------
        image_tensor   : (1, 3, H, W) float tensor
        depth_map      : H×W depth array, or None
        output_size    : (H_out, W_out) target mask resolution
        use_depth      : fuse depth to filter background (can hurt multi-plane objects)
        fixed_threshold: explicit threshold (None → percentile-based)
        keep_fraction  : fraction of patches treated as foreground
        top_k_heads    : number of low-entropy heads (None → auto 50%)

        Returns
        -------
        mask : (H_out, W_out) uint8 binary mask
        """
        # 1. CLS attention map
        attn_map = self.get_attention_map(image_tensor, top_k_heads=top_k_heads)

        # 2. Percentile foreground mask
        mask = self.create_foreground_mask(
            attn_map, threshold=fixed_threshold, keep_fraction=keep_fraction
        )

        # 3. Optional depth fusion
        if use_depth and depth_map is not None:
            mask_resized = cv2.resize(
                mask,
                (depth_map.shape[1], depth_map.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = self.fuse_depth(mask_resized, depth_map, foreground_depth_fraction=0.65)

        # 4. Morphological cleanup
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask * 255, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = (mask > 127).astype(np.uint8)

        # 5. Upsample + keep center component + refine
        mask = self.upsample_mask(mask, output_size)
        mask = self.extract_center_component(mask)
        mask = self.refine_mask(mask)

        # 6. Auto-invert if background was selected
        if mask.mean() > 0.55:
            mask = (1 - mask).astype(np.uint8)
            mask = self.extract_center_component(mask)
            mask = self.refine_mask(mask)

        # 7. Center-crop fallback for large masks
        if mask.mean() > 0.40:
            mask = self._center_crop_fallback(attn_map, mask, output_size)

        return mask

    def _center_crop_fallback(
        self,
        attn_map: np.ndarray,
        current_mask: np.ndarray,
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        When the mask covers >40% of the image, try extracting the
        strongest-attended region in the center 40% crop.
        """
        H_a, W_a = attn_map.shape
        r0, r1 = int(H_a * 0.30), int(H_a * 0.70)
        c0, c1 = int(W_a * 0.30), int(W_a * 0.70)

        cropped_attn = np.zeros_like(attn_map)
        cropped_attn[r0:r1, c0:c1] = attn_map[r0:r1, c0:c1]

        crop_vals = cropped_attn[r0:r1, c0:c1]
        if crop_vals.max() <= 0:
            return current_mask

        local_thresh = np.percentile(crop_vals[crop_vals > 0], 80)
        center_mask = (cropped_attn >= local_thresh).astype(np.uint8)

        k = np.ones((7, 7), np.uint8)
        center_mask = cv2.morphologyEx(center_mask * 255, cv2.MORPH_CLOSE, k)
        center_mask = cv2.morphologyEx(center_mask, cv2.MORPH_OPEN, k)
        center_mask = (center_mask > 127).astype(np.uint8)

        center_mask = self.upsample_mask(center_mask, output_size)
        center_mask = self.extract_center_component(center_mask)
        center_mask = self.refine_mask(center_mask)

        if 0.01 < center_mask.mean() < 0.35:
            return center_mask
        return current_mask

    # ==================================================================
    # Prompt-guided segmentation (DINO-only, no CLIP)
    # ==================================================================

    def segment_from_prompt(
        self,
        image_tensor: torch.Tensor,
        prompt: str = "",
        output_size: Optional[Tuple[int, int]] = None,
        keep_fraction: float = 0.25,
        top_k_heads: Optional[int] = None,
        use_edge_refine: bool = True,
        use_bilateral: bool = True,
        use_image_edges: bool = True,
    ) -> np.ndarray:
        """
        SAM-like segmentation using DINO attention + edge-aware refinement
        + prompt-derived spatial priors.

        Parameters
        ----------
        image_tensor   : (1, 3, H, W) float tensor
        prompt         : text prompt for spatial/semantic priors
        output_size    : (H_out, W_out) or None (→ input size)
        keep_fraction  : fraction of attention values to keep as foreground
        top_k_heads    : number of low-entropy heads (None → auto)
        use_edge_refine: sharpen mask boundaries using image edges
        use_bilateral  : bilateral smoothing to reduce noise
        use_image_edges: use real image edges (True) vs mask edges (False)

        Returns
        -------
        mask : (H_out, W_out) uint8 binary mask
        """
        eps = 1e-8
        _, _, H_orig, W_orig = image_tensor.shape
        if output_size is None:
            output_size = (H_orig, W_orig)

        # 1. DINO attention
        attn_map = self.get_attention_map(image_tensor, top_k_heads=top_k_heads)

        # 2. Prompt spatial bias
        priors = self._parse_prompt_priors(prompt)
        H, W = attn_map.shape
        ys, xs = np.indices((H, W))
        cy, cx = H / 2.0, W / 2.0
        center_dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        center_dist = center_dist / (center_dist.max() + eps)
        center_bias = 1.0 - center_dist

        spatial_bias = priors["center_weight"] * center_bias + (1 - priors["center_weight"])
        guidance = attn_map * spatial_bias
        guidance = (guidance - guidance.min()) / (guidance.max() - guidance.min() + eps)

        # 3. Edge refinement
        if use_edge_refine:
            if use_image_edges:
                img = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
                img = (img - img.min()) / (img.max() - img.min() + eps)
                img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(img_gray, 100, 200)
            else:
                edges = cv2.Canny((guidance * 255).astype(np.uint8), 50, 150)

            edges = edges.astype(np.float32) / 255.0
            guidance = np.clip(guidance * (1 + 0.8 * edges), 0, 1)

        # 4. Bilateral + Gaussian smoothing
        if use_bilateral:
            guidance = cv2.bilateralFilter(guidance.astype(np.float32), 9, 75, 75)
        guidance = cv2.GaussianBlur(guidance, (5, 5), 0)

        # 5. Threshold
        threshold = np.percentile(guidance, (1.0 - keep_fraction) * 100)

        # 6. Upsample
        guidance_tensor = torch.tensor(guidance).unsqueeze(0).unsqueeze(0).float()
        upsampled = F.interpolate(
            guidance_tensor, size=output_size, mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()
        mask = (upsampled > threshold).astype(np.uint8)

        # 7. Final cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask.astype(float), (7, 7), 0)
        mask = (mask > 0.4).astype(np.uint8)

        mask = self.extract_center_component(mask)

        # Auto-invert safeguard
        if mask.mean() > 0.55:
            mask = (1 - mask).astype(np.uint8)
            mask = self.extract_center_component(mask)

        return mask

    # ==================================================================
    # Prompt priors parser
    # ==================================================================

    @staticmethod
    def _parse_prompt_priors(prompt: str) -> Dict[str, float]:
        """
        Extract visual priors from a natural-language prompt (no external model).

        Returns dict with keys: center_weight, compact_weight, size_fraction, attn_weight.
        """
        p = prompt.lower()

        # Compactness prior
        compact_objects = {
            "bag", "backpack", "suitcase", "purse", "handbag",
            "bottle", "cup", "mug", "glass", "can", "bowl",
            "chair", "stool", "laptop", "phone", "keyboard", "monitor", "screen",
            "book", "box", "carton", "dog", "cat", "bird", "animal",
            "person", "human", "face", "head", "ball", "toy",
            "plant", "pot", "vase", "shoe", "boot",
        }
        spread_objects = {
            "table", "desk", "floor", "ground", "ceiling",
            "wall", "shelf", "rack", "road", "sky", "grass",
            "sofa", "couch", "bed", "carpet", "rug",
        }

        compact_weight = 0.5
        for w in compact_objects:
            if w in p:
                compact_weight = 0.85
                break
        for w in spread_objects:
            if w in p:
                compact_weight = 0.20
                break

        # Size prior
        size_fraction = 0.30
        if any(w in p for w in ("small", "tiny", "little", "mini")):
            size_fraction = 0.12
        elif any(w in p for w in ("large", "big", "huge", "giant")):
            size_fraction = 0.55
        if any(w in p for w in ("cup", "mug", "glass", "bottle", "phone", "book")):
            size_fraction = min(size_fraction, 0.20)
        if any(w in p for w in ("sofa", "couch", "bed", "table", "desk")):
            size_fraction = max(size_fraction, 0.45)

        # Spatial position prior
        center_weight = 0.65
        if any(w in p for w in ("left", "right")):
            center_weight = 0.30
        if any(w in p for w in ("top", "above", "upper")):
            center_weight = 0.30
        if any(w in p for w in ("bottom", "below", "lower", "ground", "floor")):
            center_weight = 0.20
        if any(w in p for w in ("center", "centre", "middle")):
            center_weight = 0.90

        return {
            "center_weight": center_weight,
            "compact_weight": compact_weight,
            "size_fraction": size_fraction,
            "attn_weight": 0.70,
        }

    # ==================================================================
    # Visualization helpers
    # ==================================================================

    def visualize_attention_heads(
        self, image_tensor: torch.Tensor, image_path: Optional[str] = None
    ) -> None:
        """Plot each attention head side-by-side for debugging."""
        import matplotlib.pyplot as plt

        head_maps = self.get_attention_map_multihead(image_tensor)
        n = head_maps.shape[0]
        fig, axes = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 3))

        if image_path:
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            axes[0].imshow(img)
            axes[0].set_title("Input")
        else:
            axes[0].axis("off")

        for h in range(n):
            axes[h + 1].imshow(head_maps[h], cmap="inferno")
            axes[h + 1].set_title(f"Head {h}")
            axes[h + 1].axis("off")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_mask_overlay(
        image_path: str,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (255, 140, 0),
        alpha: float = 0.55,
    ) -> None:
        """Overlay mask on image with foreground tint and dimmed background."""
        import matplotlib.pyplot as plt

        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype(np.float32)
        img = cv2.resize(img, (mask.shape[1], mask.shape[0]))

        fg = mask.astype(bool)
        bg = ~fg
        result = img.copy()

        result[fg] = np.clip(
            img[fg] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha, 0, 255
        )
        grey = img[bg].mean(axis=1, keepdims=True)
        blue_tint = np.array([0.55, 0.65, 1.0], dtype=np.float32)
        result[bg] = np.clip(grey * blue_tint * 0.55, 0, 255)

        plt.figure(figsize=(6, 6))
        plt.imshow(result.astype(np.uint8))
        plt.title("Mask Overlay")
        plt.axis("off")
        plt.show()

    @staticmethod
    def visualize_masked_image(image_path: str, mask: np.ndarray) -> None:
        """Show only the masked region of the image."""
        import matplotlib.pyplot as plt

        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
        masked_img = img.copy()
        masked_img[mask == 0] = 0

        plt.imshow(masked_img)
        plt.title("Masked Image (CAPTRA Input)")
        plt.axis("off")
        plt.show()
